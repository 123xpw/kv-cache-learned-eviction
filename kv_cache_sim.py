import bisect
import warnings
from dataclasses import dataclass
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

FONT_CANDIDATES = [
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
if FONT_PATH:
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = FontProperties(fname=FONT_PATH).get_name()
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════════
# 1. 块重要性分布与访问迹生成
# ═══════════════════════════════════════════════════════════════

def make_importance(n_blocks: int, seed: int = 0) -> np.ndarray:
    """
    生成 KV Cache 块的固定重要性分布（供训练迹/测试迹共享）。

    Block 0 强制为 Attention Sink：重要性固定为其余块最高值的 5 倍，
    与 StreamingLLM [Xiao et al., ICLR 2024] 揭示的锚点 Token 现象一致。
    训练迹和测试迹共享同一份 importance，使块位置特征具有跨迹信息量。
    """
    rng = np.random.default_rng(seed)
    importance = rng.zipf(1.5, n_blocks).astype(float)
    importance[0] = importance[1:].max() * 5.0   # block 0 → Attention Sink
    importance /= importance.sum()
    return importance


def generate_trace(n_steps: int, n_blocks: int, seed: int = 42,
                   importance: np.ndarray = None):
    """
    生成模拟 LLM KV Cache 块级访问迹。

    importance 须由 make_importance() 预先生成并在训练/测试间共享，
    确保块编号与重要性的对应关系在跨迹评测时保持一致。

    访问模式：
      - 重尾访问：块重要性由外部提供（Zipf(1.5) + Attention Sink）
      - 时间局部性：最近 8 个活跃块访问权重加倍
      - 稀疏注意力：每步访问约 25% 的活跃块
    """
    rng = np.random.default_rng(seed)
    if importance is None:
        importance = make_importance(n_blocks, seed=seed)

    trace = []
    for t in range(n_steps):
        n_active = min(t + 1, n_blocks)
        n_accessed = max(1, int(n_active * 0.25))

        weights = importance[:n_active].copy()
        recency_start = max(0, n_active - 8)
        weights[recency_start:] *= 2.0
        weights /= weights.sum()

        accessed = rng.choice(n_active, size=min(n_accessed, n_active),
                              replace=False, p=weights)
        for blk in accessed:
            score = float(importance[blk]) * rng.exponential(1.0)
            trace.append({
                'step':       t,
                'block_id':   int(blk),
                'attn_score': score,
                'n_active':   n_active,
            })
    return trace


def compute_reuse_distances(trace, n_blocks: int):
    """计算每次访问的真实重用距离（下次访问该块的步数间隔，未来不访问则 inf）。"""
    blk_indices = defaultdict(list)
    for i, acc in enumerate(trace):
        blk_indices[acc['block_id']].append(i)

    rd = [float('inf')] * len(trace)
    for blk, indices in blk_indices.items():
        for j, idx in enumerate(indices):
            if j + 1 < len(indices):
                rd[idx] = indices[j + 1] - idx
    return rd


# ═══════════════════════════════════════════════════════════════
# 2. 特征提取（对应论文式 3）
# ═══════════════════════════════════════════════════════════════

FEATURE_NAMES = ['块位置', '累积注意力', '对数频次', '访问频率', '访问时效', '上下文占用率']


def extract_feature(block_id, n_blocks, cum_attn, acc_count,
                    last_acc_step, current_step, n_active,
                    feature_mask=None):
    """
    返回 6 维特征向量（或 feature_mask 指定的子集）。
    feature_mask: 特征索引列表，如 [1] 表示仅使用累积注意力。
    """
    freq = acc_count / (current_step + 1)
    recency = (current_step - last_acc_step) / (current_step + 1)
    all_feat = [
        block_id / n_blocks,   # 0  归一化块位置
        cum_attn,              # 1  累积注意力分数
        np.log1p(acc_count),   # 2  对数访问频次
        freq,                  # 3  访问频率
        recency,               # 4  访问时效（越大越久未访问）
        n_active / n_blocks,   # 5  上下文填充比
    ]
    if feature_mask is not None:
        return [all_feat[i] for i in feature_mask]
    return all_feat


# ═══════════════════════════════════════════════════════════════
# 3. 离线训练 MLP 预测器
# ═══════════════════════════════════════════════════════════════

def train_mlp(train_trace, n_blocks: int, feature_mask=None, verbose=True):
    """
    在离线迹上训练双层 MLP，学习预测重用距离。
    参照 PARROT [Liu et al., ICML 2020] 的模仿学习范式。
    feature_mask: 若提供，仅使用指定特征（用于消融实验）。
    """
    reuse_dists = compute_reuse_distances(train_trace, n_blocks)

    cum_attn  = defaultdict(float)
    acc_count = defaultdict(int)
    last_acc  = defaultdict(int)

    X, y = [], []
    for i, acc in enumerate(train_trace):
        blk = acc['block_id']
        cum_attn[blk]  += acc['attn_score']
        acc_count[blk] += 1

        feat = extract_feature(blk, n_blocks, cum_attn[blk], acc_count[blk],
                               last_acc.get(blk, 0), i, acc['n_active'],
                               feature_mask=feature_mask)
        X.append(feat)
        y.append(min(reuse_dists[i], 1000))
        last_acc[blk] = i

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    mlp = MLPRegressor(hidden_layer_sizes=(32,), activation='relu',
                       max_iter=400, random_state=42,
                       learning_rate_init=0.005, n_iter_no_change=30)
    mlp.fit(Xs, y)

    if verbose:
        r2 = mlp.score(Xs, y)
        n_params = (sum(c.size for c in mlp.coefs_)
                    + sum(c.size for c in mlp.intercepts_))
        feat_desc = (str([FEATURE_NAMES[i] for i in feature_mask])
                     if feature_mask is not None else '全部6维')
        print(f"  MLP 训练完成  特征={feat_desc}  "
              f"params≈{n_params}  train-R²={r2:.3f}")
    return mlp, scaler


# ═══════════════════════════════════════════════════════════════
# 4. 缓存仿真器
# ═══════════════════════════════════════════════════════════════

H2O_RECENT_WINDOW = 8   # H2O Recent Window 大小（保护最近 W 个访问块）


@dataclass(frozen=True)
class CostModel:
    """
    简化系统代价模型（估计值，不代表真实硬件测量）。

    miss 后若块曾经出现过，则视为从 CPU 侧换入 GPU；cache 满时驱逐一个块，
    视为换出至 CPU。Learned 额外计入每个候选块的 MLP 预测开销。
    """
    block_size_mb: float = 2.0
    transfer_bandwidth_gbps: float = 16.0
    transfer_fixed_latency_us: float = 10.0
    hit_cost_us: float = 0.05
    compulsory_miss_cost_us: float = 0.10
    mlp_predict_us_per_block: float = 0.02

    @property
    def block_transfer_us(self) -> float:
        return self.transfer_fixed_latency_us + (
            self.block_size_mb * 1000.0 / self.transfer_bandwidth_gbps
        )


DEFAULT_COST_MODEL = CostModel()


def simulate(trace, capacity: int, policy: str, n_blocks: int = 100,
             mlp=None, scaler=None, reuse_dists=None, feature_mask=None,
             cost_model: CostModel = DEFAULT_COST_MODEL,
             return_stats: bool = False):
    """
    单策略 Trace-driven 仿真。

    H2O-style 策略（块级近似，不是 Zhang et al. 原论文的 token 级端到端复现）：
      - 保护 Attention Sink：block 0, 1 不驱逐
      - 保护 Recent Window：最近 H2O_RECENT_WINDOW 次访问块不驱逐
      - 从剩余候选中驱逐累积注意力分数最低的块

    默认返回 (overall_hit_rate, window_rates)。
    若 return_stats=True，返回包含命中率和简化系统代价估计的 dict。
    """
    if policy == 'opt':
        blk_acc_indices = defaultdict(list)
        for i, acc in enumerate(trace):
            blk_acc_indices[acc['block_id']].append(i)

        def next_access_after(block_id, after_idx):
            lst = blk_acc_indices.get(block_id, [])
            pos = bisect.bisect_right(lst, after_idx)
            return lst[pos] if pos < len(lst) else float('inf')
    else:
        next_access_after = None

    cache           = {}
    cum_attn        = defaultdict(float)
    acc_count       = defaultdict(int)
    last_acc        = defaultdict(lambda: -1)
    recent_accessed = []   # 全局访问历史（H2O Recent Window 使用）

    hits = misses = 0
    compulsory_misses = 0
    swap_ins = 0
    swap_outs = 0
    mlp_predictions = 0
    estimated_cost_us = 0.0
    seen_blocks = set()
    win_hits = win_total = 0
    window_rates = []
    WINDOW = 200

    def evict(current_idx, n_active):
        nonlocal mlp_predictions
        candidates = [b for b in cache if b > 1]
        if not candidates:
            candidates = list(cache.keys())

        if policy == 'lru':
            victim = min(candidates, key=lambda b: last_acc[b])

        elif policy == 'h2o':
            recent_set = set(recent_accessed[-H2O_RECENT_WINDOW:])
            h2o_cands = [b for b in candidates if b not in recent_set]
            if not h2o_cands:
                h2o_cands = candidates
            victim = min(h2o_cands, key=lambda b: cum_attn[b])

        elif policy == 'opt':
            victim = max(candidates,
                         key=lambda b: next_access_after(b, current_idx))

        elif policy == 'learned':
            if mlp is not None:
                mlp_predictions += len(candidates)
                feats = np.array([
                    extract_feature(b, n_blocks, cum_attn[b], acc_count[b],
                                    last_acc[b], current_idx, n_active,
                                    feature_mask=feature_mask)
                    for b in candidates
                ])
                pred_rd = mlp.predict(scaler.transform(feats))
                victim = candidates[int(np.argmax(pred_rd))]
            else:
                victim = min(candidates, key=lambda b: last_acc[b])

        del cache[victim]
        return victim

    for i, acc in enumerate(trace):
        blk = acc['block_id']
        cum_attn[blk]  += acc['attn_score']
        acc_count[blk] += 1
        recent_accessed.append(blk)

        if blk in cache:
            hits     += 1
            win_hits += 1
            estimated_cost_us += cost_model.hit_cost_us
        else:
            misses += 1
            if blk in seen_blocks:
                swap_ins += 1
                estimated_cost_us += cost_model.block_transfer_us
            else:
                compulsory_misses += 1
                estimated_cost_us += cost_model.compulsory_miss_cost_us
            if len(cache) >= capacity:
                evict(i, acc['n_active'])
                swap_outs += 1
                estimated_cost_us += cost_model.block_transfer_us
            cache[blk] = i

        seen_blocks.add(blk)
        last_acc[blk] = i
        win_total += 1

        if win_total == WINDOW:
            window_rates.append(win_hits / win_total)
            win_hits = win_total = 0

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0
    estimated_cost_us += mlp_predictions * cost_model.mlp_predict_us_per_block

    if not return_stats:
        return hit_rate, window_rates

    return {
        'hit_rate': hit_rate,
        'window_rates': window_rates,
        'hits': hits,
        'misses': misses,
        'compulsory_misses': compulsory_misses,
        'swap_ins': swap_ins,
        'swap_outs': swap_outs,
        'mlp_predictions': mlp_predictions,
        'estimated_cost_us': estimated_cost_us,
        'avg_access_cost_us': estimated_cost_us / total if total > 0 else 0.0,
    }


def summarize_costs(results, baseline='lru'):
    """根据 simulate(return_stats=True) 的结果计算相对基线的估计代价下降。"""
    base = results[baseline]['estimated_cost_us']
    summary = {}
    for policy, stats in results.items():
        reduction = (base - stats['estimated_cost_us']) / base * 100.0 if base else 0.0
        summary[policy] = {
            'estimated_cost_ms': stats['estimated_cost_us'] / 1000.0,
            'avg_access_cost_us': stats['avg_access_cost_us'],
            'swap_ins': stats['swap_ins'],
            'swap_outs': stats['swap_outs'],
            'mlp_predictions': stats['mlp_predictions'],
            'cost_reduction_vs_lru_pct': reduction,
        }
    return summary


# ═══════════════════════════════════════════════════════════════
# 5. 多随机种子统计
# ═══════════════════════════════════════════════════════════════

def run_multi_seed_experiment(n_seeds=5, n_blocks=100, capacity=40,
                               train_steps=1500, test_steps=2000):
    """
    使用 n_seeds 组独立种子重复实验，报告各策略命中率均值 ± 标准差。
    每组种子独立控制块重要性分布、训练迹访问模式、测试迹访问模式。
    """
    policies = ['opt', 'h2o', 'learned', 'lru']
    all_hrs = {p: [] for p in policies}

    for s in range(n_seeds):
        imp       = make_importance(n_blocks, seed=s)
        train_tr  = generate_trace(train_steps, n_blocks,
                                    seed=s * 3 + 1, importance=imp)
        test_tr   = generate_trace(test_steps,  n_blocks,
                                    seed=s * 3 + 2, importance=imp)
        test_rd   = compute_reuse_distances(test_tr, n_blocks)
        mlp, scaler = train_mlp(train_tr, n_blocks, verbose=False)

        for p in policies:
            kw = {}
            if p == 'opt':     kw['reuse_dists'] = test_rd
            if p == 'learned': kw['mlp'] = mlp; kw['scaler'] = scaler
            hr, _ = simulate(test_tr, capacity, p, n_blocks, **kw)
            all_hrs[p].append(hr * 100)

    means = {p: float(np.mean(all_hrs[p])) for p in policies}
    stds  = {p: float(np.std(all_hrs[p]))  for p in policies}
    return means, stds, all_hrs


# ═══════════════════════════════════════════════════════════════
# 6. 多缓存预算实验
# ═══════════════════════════════════════════════════════════════

def run_budget_experiment(test_trace, n_blocks, mlp, scaler, test_rd,
                           budgets=None):
    """在不同缓存预算（总块数的 20%–80%）下对比各策略命中率。"""
    if budgets is None:
        budgets = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    policies = ['opt', 'h2o', 'learned', 'lru']
    budget_results = {p: [] for p in policies}

    for bgt in budgets:
        cap = max(1, int(n_blocks * bgt))
        for p in policies:
            kw = {}
            if p == 'opt':     kw['reuse_dists'] = test_rd
            if p == 'learned': kw['mlp'] = mlp; kw['scaler'] = scaler
            hr, _ = simulate(test_trace, cap, p, n_blocks, **kw)
            budget_results[p].append(hr * 100)

    return budget_results, [int(b * 100) for b in budgets]


# ═══════════════════════════════════════════════════════════════
# 7. 特征消融实验
# ═══════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    '全部6特征（原版）':           [0, 1, 2, 3, 4, 5],
    '去除累积注意力+访问频率':      [0, 2, 4, 5],
    '去除块位置+累积注意力+频率':   [4, 5],
    '仅访问时效（≈ LRU）':         [4],
    '仅累积注意力（≈ H2O）':       [1],
}


def run_ablation_experiment(train_trace, test_trace, n_blocks, capacity):
    """为 5 种特征配置分别训练 MLP 并评测命中率。"""
    results = {}
    for name, mask in ABLATION_CONFIGS.items():
        mlp_a, scaler_a = train_mlp(train_trace, n_blocks,
                                     feature_mask=mask, verbose=False)
        hr, _ = simulate(test_trace, capacity, 'learned', n_blocks,
                         mlp=mlp_a, scaler=scaler_a, feature_mask=mask)
        results[name] = hr * 100
    return results


# ═══════════════════════════════════════════════════════════════
# 8. 绘图
# ═══════════════════════════════════════════════════════════════

def plot_main_results(results, save_path='kv_cache_sim_results.png'):
    # 柱状图顺序：OPT → Learned(本文) → H2O → LRU，突出本文方案
    bar_order   = ['opt', 'learned', 'h2o', 'lru']
    bar_labels  = ['OPT\n(理论上界)', 'Learned\n(本文)', '$H_2O$-style\n(块级基线)', 'LRU\n(基准)']
    bar_colors  = ['#37474F', '#1B5E20', '#E65100', '#B71C1C']

    # 折线图配置
    line_order  = ['opt', 'learned', 'h2o', 'lru']
    line_labels = ['OPT (理论上界)', 'Learned (本文)', '$H_2O$-style (块级基线)', 'LRU (基准)']
    line_colors = ['#37474F', '#1B5E20', '#E65100', '#B71C1C']
    line_styles = ['--', '-', '-.', ':']
    line_widths = [1.5, 2.2, 1.8, 1.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                    gridspec_kw={'width_ratios': [1, 1.3]})

    # ── 左：柱状图 ──
    hrs = [results[p]['hit_rate'] * 100 for p in bar_order]
    bars = ax1.bar(bar_labels, hrs, color=bar_colors, width=0.55,
                   edgecolor='white', linewidth=1.2, zorder=3)
    ax1.set_ylabel('缓存命中率 (%)', fontsize=10)
    ax1.set_title('(a) 各策略整体命中率\n(缓存预算 = 总块数 40%)', fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_yticks(range(0, 101, 20))
    ax1.grid(axis='y', alpha=0.25, zorder=0)
    ax1.tick_params(axis='both', labelsize=9)
    for bar, v in zip(bars, hrs):
        ax1.text(bar.get_x() + bar.get_width() / 2, v - 2.5,
                 f'{v:.1f}%', ha='center', va='top',
                 fontsize=10, fontweight='bold', color='white')

    # ── 右：折线图（纵轴缩小到数据实际区间，使差异可辨）──
    for p, lab, col, ls, lw in zip(line_order, line_labels, line_colors,
                                     line_styles, line_widths):
        wr = [r * 100 for r in results[p]['window_rates']]
        ax2.plot(range(len(wr)), wr, label=lab, color=col,
                 linestyle=ls, linewidth=lw, zorder=3)
    ax2.set_xlabel('仿真窗口序号 (每点 200 次访问)', fontsize=10)
    ax2.set_ylabel('缓存命中率 (%)', fontsize=10)
    ax2.set_title('(b) 命中率随仿真过程变化趋势', fontsize=10)
    ax2.set_ylim(65, 100)
    ax2.set_yticks(range(65, 101, 5))
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax2.grid(True, alpha=0.25, zorder=0)
    ax2.tick_params(axis='both', labelsize=9)

    plt.tight_layout(w_pad=3)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  保存：{save_path}")


def plot_budget_results(budget_results, budget_pcts,
                         save_path='kv_cache_budget_results.png'):
    # 顺序：OPT → Learned(本文) → H2O → LRU
    order   = ['opt', 'learned', 'h2o', 'lru']
    labels  = {'opt': 'OPT (理论上界)', 'learned': 'Learned (本文)',
               'h2o': '$H_2O$-style (块级基线)', 'lru': 'LRU (基准)'}
    colors  = {'opt': '#37474F', 'learned': '#1B5E20',
               'h2o': '#E65100', 'lru': '#B71C1C'}
    lstyle  = {'opt': '--', 'learned': '-', 'h2o': '-.', 'lru': ':'}
    markers = {'opt': 's', 'learned': 'o', 'h2o': '^', 'lru': 'D'}
    lwidths = {'opt': 1.5, 'learned': 2.2, 'h2o': 1.8, 'lru': 1.5}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in order:
        ax.plot(budget_pcts, budget_results[p], label=labels[p],
                color=colors[p], linestyle=lstyle[p],
                linewidth=lwidths[p], marker=markers[p], markersize=5,
                zorder=3)
    ax.set_xlabel('缓存预算 (%)', fontsize=10)
    ax.set_ylabel('缓存命中率 (%)', fontsize=10)
    ax.set_title('不同缓存预算下各策略命中率对比', fontsize=10)
    ax.set_ylim(15, 102)
    ax.set_yticks(range(20, 101, 10))
    ax.set_xticks(budget_pcts)
    ax.set_xticklabels([f'{b}%' for b in budget_pcts], fontsize=9)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.tick_params(axis='both', labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  保存：{save_path}")


def plot_cost_results(cost_summary, save_path='kv_cache_cost_results.png'):
    """绘制简化系统代价估计图。"""
    order = ['opt', 'learned', 'h2o', 'lru']
    labels = ['OPT\n(理论上界)', 'Learned\n(本文)',
              '$H_2O$-style\n(块级基线)', 'LRU\n(基准)']
    colors = ['#37474F', '#1B5E20', '#E65100', '#B71C1C']

    total_ms = [cost_summary[p]['estimated_cost_ms'] for p in order]
    avg_us = [cost_summary[p]['avg_access_cost_us'] for p in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(labels, total_ms, color=colors, width=0.58,
            edgecolor='white', linewidth=1.0, zorder=3)
    ax1.set_ylabel('估计累计代价 (ms)', fontsize=10)
    ax1.set_title('(a) 简化系统代价估计', fontsize=10)
    ax1.grid(axis='y', alpha=0.25, zorder=0)
    ax1.tick_params(axis='both', labelsize=9)

    ax2.bar(labels, avg_us, color=colors, width=0.58,
            edgecolor='white', linewidth=1.0, zorder=3)
    ax2.set_ylabel('平均每次访问代价 (us)', fontsize=10)
    ax2.set_title('(b) 平均访问代价估计', fontsize=10)
    ax2.grid(axis='y', alpha=0.25, zorder=0)
    ax2.tick_params(axis='both', labelsize=9)

    plt.tight_layout(w_pad=2.4)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  保存：{save_path}")


# ═══════════════════════════════════════════════════════════════
# 9. 主程序
# ═══════════════════════════════════════════════════════════════

def main():
    N_BLOCKS    = 100
    CAPACITY    = 40    # 主实验缓存预算 40%
    TRAIN_STEPS = 1500
    TEST_STEPS  = 2000
    IMP_SEED    = 0     # 块重要性分布种子（训练/测试共享）
    TRAIN_SEED  = 42    # 训练迹访问模式种子
    TEST_SEED   = 99    # 测试迹访问模式种子

    print("=" * 62)
    print("  KV Cache 替换策略 Trace-driven 仿真（v2 修正版）")
    print("=" * 62)

    # ── [1] 生成共享重要性分布与访问迹 ──────────────────────────
    print("\n[1/6] 生成共享块重要性分布与访问迹 ...")
    importance  = make_importance(N_BLOCKS, seed=IMP_SEED)
    train_trace = generate_trace(TRAIN_STEPS, N_BLOCKS,
                                  seed=TRAIN_SEED, importance=importance)
    test_trace  = generate_trace(TEST_STEPS,  N_BLOCKS,
                                  seed=TEST_SEED,  importance=importance)
    test_rd     = compute_reuse_distances(test_trace, N_BLOCKS)
    top5 = np.argsort(importance)[::-1][:5]
    print(f"  训练迹 {len(train_trace)} 条  |  测试迹 {len(test_trace)} 条")
    print(f"  Top-5 重要块（训练/测试共享）: {top5.tolist()}")
    print(f"  Block 0 重要性占比: {importance[0]*100:.2f}%  (Attention Sink)")

    # ── [2] 训练 MLP ─────────────────────────────────────────────
    print("\n[2/6] 训练 MLP 预测器（全部 6 维特征）...")
    mlp, scaler = train_mlp(train_trace, N_BLOCKS)

    # ── [3] 主实验：40% 缓存预算 ─────────────────────────────────
    print("\n[3/6] 主实验：四种策略对比（缓存预算 40%）...")
    configs = [
        ('opt',     dict(reuse_dists=test_rd)),
        ('learned', dict(mlp=mlp, scaler=scaler)),
        ('h2o',     dict()),
        ('lru',     dict()),
    ]
    results = {}
    cost_stats = {}
    for policy, kw in configs:
        stats = simulate(test_trace, CAPACITY, policy, N_BLOCKS,
                         return_stats=True, **kw)
        hr, wr = stats['hit_rate'], stats['window_rates']
        results[policy] = {'hit_rate': hr, 'window_rates': wr}
        cost_stats[policy] = stats
        print(f"  {policy:8s}  命中率 = {hr * 100:.1f}%")

    lru = results['lru']['hit_rate']     * 100
    h2o = results['h2o']['hit_rate']     * 100
    lrn = results['learned']['hit_rate'] * 100
    opt = results['opt']['hit_rate']     * 100
    print(f"\n  Learned vs LRU : {lrn - lru:+.1f} pp")
    print(f"  Learned vs H2O : {lrn - h2o:+.1f} pp")
    print(f"  距 OPT 差距    : {opt - lrn:.1f} pp")

    print("\n  生成主实验图表 ...")
    plot_main_results(results)

    # ── [4] 简化系统代价模型 ───────────────────────────────────
    print("\n[4/6] 简化系统代价模型（估计值，非真实硬件测量）...")
    cm = DEFAULT_COST_MODEL
    print("  参数："
          f"block={cm.block_size_mb:.1f} MB, "
          f"bandwidth={cm.transfer_bandwidth_gbps:.1f} GB/s, "
          f"fixed_latency={cm.transfer_fixed_latency_us:.1f} us, "
          f"mlp={cm.mlp_predict_us_per_block:.2f} us/block")
    cost_summary = summarize_costs(cost_stats)
    plot_cost_results(cost_summary)

    print(f"\n  {'策略':<8}  {'swap-in':>8}  {'swap-out':>8}  "
          f"{'估计代价(ms)':>12}  {'平均访问(us)':>12}  {'较LRU下降':>10}")
    for p in ['opt', 'learned', 'h2o', 'lru']:
        row = cost_summary[p]
        print(f"  {p:<8}  {row['swap_ins']:8d}  {row['swap_outs']:8d}  "
              f"{row['estimated_cost_ms']:12.1f}  "
              f"{row['avg_access_cost_us']:12.2f}  "
              f"{row['cost_reduction_vs_lru_pct']:9.1f}%")

    # ── [4] 多缓存预算实验 ───────────────────────────────────────
    print("\n[5/6] 多缓存预算实验（20%–80%）...")
    budget_results, budget_pcts = run_budget_experiment(
        test_trace, N_BLOCKS, mlp, scaler, test_rd)
    plot_budget_results(budget_results, budget_pcts)

    print(f"\n  {'预算':>4}  {'OPT':>6}  {'H2O':>6}  {'Learned':>8}  "
          f"{'LRU':>6}  {'Δ(Lrn-LRU)':>11}  {'Δ(Lrn-H2O)':>10}")
    for i, b in enumerate(budget_pcts):
        o, h, l, r = (budget_results[p][i]
                      for p in ['opt', 'h2o', 'learned', 'lru'])
        print(f"  {b:3d}%  {o:6.1f}%  {h:6.1f}%  {l:8.1f}%  {r:6.1f}%  "
              f"{l - r:+10.1f}  {l - h:+9.1f}")

    # ── [6] 特征消融实验 ─────────────────────────────────────────
    print("\n[6/6] 特征消融实验 ...")
    ablation = run_ablation_experiment(
        train_trace, test_trace, N_BLOCKS, CAPACITY)
    base_hr = ablation['全部6特征（原版）']
    print(f"\n  {'特征配置':<36}  {'命中率':>6}  {'vs 全特征':>9}")
    for name, hr in ablation.items():
        diff = f"{hr - base_hr:+.1f} pp" if name != '全部6特征（原版）' else '（基准）'
        print(f"  {name:<36}  {hr:5.1f}%  {diff:>9}")

    # ── [补充] 多随机种子统计 ────────────────────────────────────
    print("\n[补充] 多随机种子统计（5 组）...")
    means, stds, _ = run_multi_seed_experiment(
        n_seeds=5, n_blocks=N_BLOCKS, capacity=CAPACITY,
        train_steps=TRAIN_STEPS, test_steps=TEST_STEPS)
    print(f"\n  {'策略':<10}  {'均值':>6}  {'标准差':>6}")
    for p in ['opt', 'h2o', 'learned', 'lru']:
        print(f"  {p:<10}  {means[p]:5.1f}%  ±{stds[p]:.1f}%")

    print("\n" + "=" * 62)
    print("  所有实验完成。论文数字请以本次运行输出为准。")
    print("=" * 62)


if __name__ == '__main__':
    main()
