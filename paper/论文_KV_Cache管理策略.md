# 大语言模型推理中的 KV Cache 管理策略

---

**摘要：** 大语言模型推理中的 KV Cache 会随上下文长度和并发请求数线性增长，逐渐成为限制 GPU 显存利用率和服务吞吐的关键瓶颈。现有 PagedAttention 主要解决显存碎片问题，$H_2O$ 等方法依赖固定启发式规则，难以覆盖多样化访问模式。围绕“稀疏或近似注意力场景下，显存受限时如何选择更值得保留的 KV 块”这一问题，本文提出 Learned：以 PagedAttention 物理块为管理粒度，利用块位置、累积注意力、访问频次、访问时效和上下文占用率等六维特征预测未来重用距离，并据此执行块级驱逐或换出决策。本文进一步讨论该策略与 vLLM 块管理接口的适配方式，并通过合成访问迹进行 Trace-driven 仿真验证。在缓存预算为总块数 40% 的设定下，Learned 较 LRU 命中率提高 6.6 个百分点，较包含 Heavy Hitter 与 Recent Window 保护的 $H_2O$-style 块级基线高 0.6 个百分点，距 OPT 理论上界差距约 2.0 个百分点；多随机种子统计（5 组）显示 Learned 平均不低于该强启发式基线，但小幅优势尚不足以作为真实系统收益的强结论。特征消融实验表明，累积注意力分数是当前合成迹中唯一起决定性作用的特征，Learned 相对 $H_2O$-style 基线的优势主要来源于 MLP 回归预测机制比固定排序规则更灵活，而非来自多维特征的联合贡献；多维特征融合的完整价值有待真实 LLM 访问迹验证。结果表明，学习型重用距离预测在合成块级访问迹上具有可行性，但尚不能证明其在真实 LLM 服务中的端到端延迟或生成质量收益。

**关键词：** 大语言模型；KV Cache；学习型缓存；重用距离预测；动态驱逐

---

**Abstract:** KV Cache memory in large language model inference grows linearly with context length and request concurrency, making it a key bottleneck for GPU memory utilization and serving throughput. PagedAttention mainly reduces fragmentation, while H2O-like methods rely on fixed heuristics that may not fit diverse access patterns. Focusing on block-level retention decisions under memory pressure in sparse or approximate attention settings, this paper proposes Learned, a lightweight block-level eviction/offloading strategy for PagedAttention-style KV Cache management. Learned predicts future reuse distance from six features, including block position, cumulative attention, access frequency, recency, and context occupancy, and evicts blocks with larger predicted reuse distance. The paper also discusses how the policy can be plugged into a vLLM-like block manager. Synthetic trace-driven simulation shows that Learned improves hit rate by 6.6 percentage points over LRU and outperforms an H2O-style block-level baseline with Heavy Hitter and Recent Window protection by 0.6 percentage points under a 40% cache budget; multi-seed statistics (5 runs) show that Learned is not worse on average, but this small margin should not be over-interpreted as real-system evidence. Feature ablation reveals that cumulative attention score is the dominant feature in the current synthetic traces, and the observed advantage mainly comes from the MLP regression mechanism rather than from multi-feature fusion. These results suggest that learned reuse-distance prediction is feasible on synthetic block-level traces, while validation on real LLM traces, end-to-end latency, and generation quality remains as future work.

**Keywords:** Large Language Model; KV Cache; Learned Cache; Reuse Distance Prediction; Dynamic Eviction

---

## 目录

1. 引言
   - 1.1 研究背景与意义
   - 1.2 KV Cache 面临的性能挑战
   - 1.3 本文主要研究工作
2. 相关研究与背景分析
   - 2.1 Transformer 推理机制与 KV Cache 访存特征
   - 2.2 KV Cache 优化方法与本文定位
   - 2.3 传统缓存替换算法的局限性
3. 基于学习型缓存的 KV Cache 动态管理机制设计
   - 3.1 总体框架设计
   - 3.2 特征提取与轻量级预测模型设计
   - 3.3 基于重用距离预测的驱逐策略
   - 3.4 轻量级预测器的训练数据采集与在线更新扩展
4. 系统软件栈与硬件协同优化探讨
   - 4.1 与主流推理框架的接口适配
   - 4.2 预测开销与软硬件协同约束
5. 仿真实验与结果分析
   - 5.1 仿真环境与评测指标
   - 5.2 实验结果与分析
6. 结论与未来展望
   - 6.1 全文总结
   - 6.2 未来研究方向

参考文献

---

## 1. 引言

### 1.1 研究背景与意义

近年来，以 GPT 系列、LLaMA [1] 为代表的大语言模型（Large Language Models, LLM）在多轮对话、长文档理解与检索增强生成（RAG）等场景中得到广泛部署。这些应用的共同特点是输入上下文长、对话轮次多，模型需要在数千乃至数万个 Token 的历史信息上进行推理。与训练阶段不同，推理阶段对**延迟**与**吞吐量**同时提出了严苛要求：用户期望低延迟的逐 Token 响应，而服务提供商则需要在有限 GPU 资源上并发处理尽可能多的请求。

然而，Transformer 自回归解码的内在机制导致推理系统必须在 GPU 显存中维护规模庞大的 KV Cache。随着上下文长度与并发请求数量的增长，KV Cache 对显存的消耗迅速超越模型权重本身，成为制约系统扩展能力的核心瓶颈。这一矛盾在冯·诺依曼体系下尤为突出：GPU 片上高带宽存储（HBM）容量有限，而片外 CPU 内存与磁盘的访问延迟又远高于 GPU 计算速度，形成显著的"存储墙"效应 [5]。

### 1.2 KV Cache 面临的性能挑战

KV Cache 的显存开销随模型规模与序列长度同步增长。对于含 $L$ 层、$H$ 个注意力头、头维度为 $d_h$ 的模型，单请求 KV Cache 的显存占用为：

$$M = 2 \times L \times H \times d_h \times |\text{seq}| \times \text{sizeof(dtype)} \tag{1-1}$$

以 LLaMA-7B（FP16 精度）[1] 为例，上下文长度 4096 时单请求 KV Cache 接近 2 GB；若同时服务 16 个并发请求，仅 KV Cache 一项即需约 32 GB 显存。考虑到模型权重、激活缓冲区和运行时内存也需占用 HBM，这类开销会显著挤压单卡可用显存预算，并限制可服务的并发请求数。

现有应对策略存在明显局限。vLLM [2] 的 PagedAttention 解决了显存碎片化问题，但未能从根本上降低 KV Cache 的总量需求；$H_2O$ [3]、Scissorhands [4] 等启发式驱逐方法利用注意力分数保留重要 Token，但规则形式相对固定；传统的 LRU、LFU 等通用缓存替换策略则完全忽视注意力机制的结构化特征，在大模型场景下命中率损失显著。需要强调的是，在标准 dense attention 推理中，每一步理论上都要读取全部历史 KV，因此本文的“访问”和“命中率”并不表示 dense attention 对完整历史 KV 的逐步读取，而是表示在稀疏注意力、KV 压缩或分层 offload 场景中，被选中参与计算或需要驻留 GPU 的块级访问事件。由此可见，本文关注的核心问题不是“如何重新设计完整推理系统”，而是一个更具体的块级决策问题：**当系统只能保留部分候选 KV 块在 GPU 中时，应如何根据已有访问历史判断哪些块未来更晚被重用？**

### 1.3 本文主要研究工作

针对上述问题，本文研究学习型缓存（Learned Cache）机制在大模型 KV Cache 管理中的适用性，提出一种基于轻量级多层感知机（Multi-Layer Perceptron, MLP）的 KV Cache 动态驱逐方案。后文将该离线训练的 MLP 重用距离预测策略简称为 **Learned**。Learned 的意义不在于替代 PagedAttention、$H_2O$ 或 KV Cache 压缩技术，而在于把“按固定规则判断重要性”的驱逐过程，转化为“根据历史访问特征预测未来重用距离”的动态决策过程。Learned 的框架设计允许把注意力分数、位置、访问频次和访问时效等多维信号统一到一个可训练的预测器中；但需要指出，合成迹上的特征消融实验表明，在当前实验设定下累积注意力分数是唯一起决定性作用的特征，Learned 对 $H_2O$-style 基线的优势主要来自 MLP 回归预测机制的灵活性，而非多维特征的联合贡献；多维特征融合的完整价值有待在更复杂的真实 LLM 访问模式中验证。

全文围绕一条主线展开：首先说明 KV Cache 为什么会成为显存瓶颈；其次分析固定替换规则为何难以适配稀疏或近似注意力下的 LLM 块级访问模式；然后将缓存替换问题转化为“预测未来重用距离”的监督学习问题；最后通过合成 Trace 仿真检验该预测策略相对 LRU 和 $H_2O$-style 基线的效果。

本文主要工作如下。

（1）从计算机系统结构视角梳理 KV Cache 的显存扩张机制和访存特征，指出分页管理、启发式压缩与传统缓存替换各自解决的问题和不足，明确本文聚焦于**显存受限下的块级驱逐决策**。

（2）提出 Learned 块级驱逐策略。该策略以 PagedAttention 物理块为粒度，从块位置、累积注意力分数、访问频次、访问时效和上下文占用率等维度构建六维特征向量，训练双层 MLP 预测各块未来重用距离，并优先换出预测重用距离较大的块。与基于单一规则的启发式方法相比，Learned 的创新点在于把多个缓存状态信号统一到同一个可训练的重用距离模型中，使驱逐依据从人工规则扩展为数据驱动的预测结果。

（3）给出与 vLLM 类推理框架的接口适配方案，说明 Learned 只接管“选择哪些块换出”的策略层，不改变 PagedAttention 的块表格式、物理块分配逻辑和 `swap_out/swap_in` 基础机制。该设计使 Learned 可以作为可插拔策略模块存在，既能利用现有分页 KV Cache 管理能力，又避免重新实现完整推理系统。

（4）基于合成访问迹开展 Trace-driven 仿真，对比 OPT、LRU、$H_2O$-style 基线与 Learned 的缓存命中率。受限于实验条件，本文不声称完成端到端推理系统实现，也不评估生成质量或 PCIe 换入换出导致的停顿；实验重点是验证学习型重用距离预测在稀疏块级 Trace 上是否具备可行性。

---

## 2. 相关研究与背景分析

本章的作用是为第 3 章的方案设计建立问题边界。2.1 节说明 KV Cache 的重要性并非普通数组访问可概括，而是由注意力分布、上下文位置和生成阶段共同决定；2.2 节按“已有方法能解决什么、仍留下什么决策空间”的方式梳理 KV Cache 优化技术；2.3 节进一步指出，在稀疏注意力、压缩或分层驻留场景下，传统缓存替换算法缺乏对注意力结构的感知，因此需要引入可学习的动态预测机制。

### 2.1 Transformer 推理机制与 KV Cache 访存特征

Transformer 架构由 Vaswani 等 [6] 于 2017 年提出，其核心计算单元是多头自注意力机制。给定长度为 $n$ 的输入序列，注意力模块将每个 Token 映射为查询向量 $Q$、键向量 $K$ 与值向量 $V$，并按式（2-1）计算加权输出：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \tag{2-1}$$

在自回归生成阶段，模型逐 Token 产生输出：生成第 $t$ 个 Token 时，标准 dense attention 需读取前 $t-1$ 个 Token 的 $K$、$V$ 矩阵以完成注意力计算。为避免对历史序列的重复计算，推理系统将各层的 $K$、$V$ 矩阵缓存于 GPU 显存，称为 KV Cache。本文后续讨论的块级“访问”特指在稀疏注意力、KV 压缩或 GPU/CPU 分层驻留策略下，某个 KV 块被选择保留、读取或调入 GPU 的事件，而不是 dense attention 对全部历史 KV 的完整扫描。

从访存规律来看，KV Cache 常呈现两类结构化特征。**时间局部性**方面，近期 Token 在许多生成场景中具有较高访问概率，但系统提示、实体定义和 Attention Sink 等远距离 Token 也可能长期保持重要性；Zhang 等 [3] 的统计分析表明，注意力权重呈明显重尾分布，少数"重击者"Token 贡献了绝大部分注意力质量。**注意力稀疏性**方面，Liu 等 [4] 提出"重要性持久化"假设：在生成过程中，对某步骤具有高注意力权重的 Token，在后续步骤中往往维持较高重要性；这意味着 KV Cache 中存在大量可安全驱逐的低重要性冗余块。上述特征为本文的学习型预测方案提供了关键的结构先验。

### 2.2 KV Cache 优化方法与本文定位

围绕 KV Cache 显存压力，已有工作大致解决了三个层面的问题：第一，如何把 KV Cache 管理成可分配、可交换、低碎片的系统资源；第二，如何判断哪些 Token 或块在注意力计算中更重要；第三，如何在真实服务中平衡显存节省、数据搬移、额外计算和生成质量。本文的 Learned 策略并不取代这些工作，而是在它们之间补充一个更窄的决策环节：当块管理机制已经存在、注意力统计也可获得时，用可学习模型判断哪些块未来更晚被重用。

**块级内存管理提供了操作基础。** vLLM [2]（Kwon 等，2023）提出 PagedAttention 机制，借鉴操作系统虚拟内存的分页思想，将 KV Cache 拆分为固定大小的物理块，并通过块表维护逻辑地址到物理地址的映射，从而消除内部碎片与外部碎片。实验表明，传统系统的显存浪费率高达 60%–80%，而 vLLM 将其降至 4% 以下，在相同硬件上吞吐量最高提升 24 倍。FlexGen [8]（Sheng 等，2023）面向单 GPU 受限场景，通过线性规划搜索模型权重与 KV Cache 在 GPU 显存、CPU 内存及磁盘间的最优分配策略，说明 KV Cache 可以被纳入异构内存调度。上述工作解决了“缓存块如何被组织、迁移和复用”的问题，但当显存不足时，仍需要策略决定优先换出哪些块。

**注意力重要性方法提供了驱逐依据。** $H_2O$ [3]（Zhang 等，2023）将 KV Cache 驱逐建模为动态子模优化问题，保留累积注意力分数最高的 Token（Heavy Hitter）及最近窗口内的 Token，在受限缓存预算下显著提升吞吐。Scissorhands [4]（Liu 等，2023）进一步利用“重要性持久化”先验，在固定预算内周期性保留高重要性 Token。StreamingLLM [9]（Xiao 等，2024）发现初始 Token 存在“注意力汇聚”（Attention Sink）效应，因此需要保护首部锚点 Token；SnapKV [10]（Li 等，2024）则在生成开始前利用观察窗口识别关键 Prompt 区域。RocketKV [16] 和 Wang 等 [17] 的后续工作也表明，驱逐决策可以进一步区分输入阶段、解码阶段以及层/头级注意力差异。近期 LookaheadKV [23] 进一步从“预见未来重要性”的角度训练轻量模块估计未来响应对 KV 的需求，以降低显式草稿生成的成本。与 LookaheadKV 直接面向真实长上下文任务的 token 重要性预测不同，本文只研究 PagedAttention 块粒度上的重用距离回归，并且目前仅在合成块级访问迹中验证，因此本文不声称在真实 LLM 任务质量或端到端延迟上超过这类最新方法。这类工作共同说明：KV Cache 驱逐不能只依赖最近访问时间，还应利用注意力分数、位置和结构信息。本文的特征设计正是吸收这些信号，但将固定规则改写为重用距离预测问题。

**压缩与服务系统研究限定了方案边界。** GQA [7] 从模型架构层面减少 $K$、$V$ 头数量，CacheGen [11] 通过张量编码压缩跨轮会话中的 KV Cache，RetroInfer [18] 将 KV Cache 视为可检索的向量存储对象并按需加载，异步 KV Cache 预取工作 [19] 则从 L2 Cache、HBM 带宽和计算-加载重叠角度降低数据搬移开销。这些方法提示，实际部署中的瓶颈不只是“命中率”，还包括压缩/预测开销、高速串行总线（Peripheral Component Interconnect Express, PCIe）传输、预取时机和调度耦合。Gao 等 [20] 对生产部署的分析进一步指出，KV Cache 压缩并不总能转化为端到端收益；综述和比较性工作 [21][22] 也强调，不同请求长度、模型规模和稀疏度下不存在单一最优策略。因此，本文将 Learned 定位为可插拔的驱逐决策模块，而不是完整的 KV Cache 服务框架；第 5 章只验证块级命中率，端到端延迟和生成质量留待后续真实系统实验。

### 2.3 传统缓存替换算法的局限性

计算机体系结构领域的经典缓存替换策略以 LRU（最近最少使用）与 LFU（最低访问频率）为代表。两者的共同理论参照是 Belady [12] 于 1966 年提出的 OPT 最优替换算法：在未来访问序列已知的前提下，OPT 每次驱逐"下次被访问时间最远"的缓存块，给出了任意替换策略所能达到的命中率上界。

将固定替换策略直接应用于大模型 KV Cache 时存在三方面本质局限。**其一，LRU 的时序假设与注意力分布不符。** LRU 假设近期访问的数据具有更高的未来访问概率，但大模型多轮对话中存在"沉睡唤醒"模式——早期注入的系统提示或关键实体，在若干轮沉寂后可能因用户追问而被大量引用，此类块按 LRU 早已被驱逐。**其二，LFU 在稀疏对话中易产生频率污染。** 早期高频访问的 Token 即便已失去语义相关性，仍会因历史频率优势长期占据缓存，挤占真正活跃块的空间。**其三，多头注意力的结构异质性超出全局策略的建模能力。** 不同层、不同头对 Token 重要性的判断存在显著差异，部分头高度稀疏，部分头则呈全局均匀分布 [13]；单一的全局驱逐策略难以感知这种层-头级差异，可能导致系统性误驱逐。

综合以上分析，KV Cache 管理需要超越固定规则、转向**基于预测的动态决策机制**。本文并不否定 PagedAttention、$H_2O$ 或 KV Cache 压缩的价值，而是在它们已经给出块管理和重要性统计基础之后，进一步研究一个可插拔问题：能否利用历史访问特征预测块的未来重用距离，从而在显存不足时做出更接近 Belady 思想的驱逐决策。第 3 章即围绕这一问题展开方案设计。

---

## 3. 基于学习型缓存的 KV Cache 动态管理机制设计

第 2 章已经说明，在稀疏注意力、KV 压缩或分层驻留场景中，KV Cache 驱逐的关键难点在于未来被选中访问或调入 GPU 的时间未知。本文采用的处理方式是将“未来是否还重要”转化为一个更具体的监督学习目标：预测每个块距离下一次块级访问事件还需要等待多少步，即重用距离。若某块预测重用距离较大，则在当前显存压力下优先换出或降级该块；若预测重用距离较小，则优先保留。

因此，Learned 的核心创新并不是简单地在缓存策略中加入一个 MLP，而是重新定义驱逐决策的学习目标和信息来源：以 Belady 最优替换思想中的“下次访问时间”为理论参照，以注意力统计和缓存元数据为可观测特征，以轻量级预测器近似未来重用距离。相比 LRU，Learned 不再只看最近访问时间；相比 $H_2O$-style 固定规则，Learned 不只把累积注意力分数作为排序依据，而是允许注意力分数与位置、频次、时效、上下文占用率共同进入预测模型。需要指出，第 5 章消融结果表明，当前合成迹中的实际收益主要来自 MLP 对累积注意力信号的回归映射，多特征融合的价值尚未被充分证明。

### 3.1 总体框架设计

![图 1　学习型 KV Cache 管理系统总体架构](fig1.png)

**图 1　学习型 KV Cache 管理系统总体架构**

本文提出的学习型 KV Cache 管理方案以 vLLM 的 PagedAttention [2] 为基础，在其分页内存管理层之上引入轻量级预测模块。为避免将系统误解为单一 MLP 模型，图 1 按**数据平面、控制平面、反馈平面**三个层次重新组织架构。

**数据平面**负责真实 KV 张量的存放与迁移。GPU HBM 中的活跃块直接参与注意力计算；当显存压力超过阈值时，部分物理块通过 `swap_out` 迁移至 CPU 双倍数据率内存（Double Data Rate, DDR），并在块表中标记为已换出。后续若稀疏注意力或上层调度再次选择该块参与计算，则通过 `swap_in` 回调至 GPU。该机制的真实收益还取决于 PCIe 传输、预取时机和 GPU stall，本文仿真只统计块级命中率，不等价于端到端加速。

**控制平面**负责作出驱逐决策。LLM 推理引擎在每步解码后产生注意力统计量，特征提取器将其与块位置、访问时效、访问频次等元数据合成为六维特征向量；轻量级 MLP 输出每个块的预测重用距离 $\hat{d}_i$；驱逐控制器按 $\hat{d}_i$ 从大到小选择候选块，同时保护首部 Attention Sink 锚点块，最后向底层块管理器发出换出或降级指令。

**反馈平面**负责记录模型自适应所需的监督信号。当已换出块被重新调入 GPU 时，系统可以计算其真实重用距离，并将该样本写入经验回放缓冲区。在线更新模块可作为异步线程周期性微调 MLP 参数，以尽量减少对主推理流水线的影响。需要说明的是，本文第 5 章的仿真实验仅验证离线训练预测器的驱逐效果，反馈平面主要作为工程扩展路径提出。

从验证范围看，本文区分两条路径。**实验路径**为：合成稀疏块级访问迹 $\rightarrow$ 特征提取 $\rightarrow$ 离线训练 MLP $\rightarrow$ Learned 驱逐决策 $\rightarrow$ 命中率统计。**工程扩展路径**为：`swap_in` 反馈 $\rightarrow$ 真实重用距离记录 $\rightarrow$ 经验回放 $\rightarrow$ 可选在线更新。第 5 章只覆盖前一条实验路径，后一条路径作为后续系统实现方向。

系统的核心理念来自 Liu 等 [14] 在 CPU 缓存领域的工作 PARROT：通过模仿 Belady 最优策略 [12]，训练一个只依赖**过去访问历史**的轻量级模型来近似预测块的**未来重用距离**，并据此做出驱逐决策。本文将这一思路迁移至大模型 KV Cache 场景，并针对注意力机制的结构特性对特征工程进行专项设计。

### 3.2 特征提取与轻量级预测模型设计

**特征向量设计。** 对于 KV Cache 中的每一个物理块 $b_i$（存储序列位置区间 $[s,\, s+B-1]$ 对应的 K、V 张量，$B$ 为块大小），本文仿真原型在每次访问后提取以下六维特征向量 $\mathbf{f}_i \in \mathbb{R}^6$：

$$\mathbf{f}_i = \bigl[\, p_i,\; A_i,\; \log(1+c_i),\; r_i,\; \Delta_i,\; \rho_t \,\bigr] \tag{3-1}$$

各分量含义如表 1 所示。

**表 1　特征向量各分量说明**

| 分量 | 符号 | 含义 | 归一化方式 |
|:-----|:----:|:-----|:----------|
| 块相对位置 | $p_i$ | 块编号在总块数中的相对位置 | $i / N$ |
| 累积注意力分数 | $A_i$ | 块自出现以来累计获得的注意力分数 | 随其余特征一并标准化 |
| 对数访问频次 | $\log(1+c_i)$ | 块累计访问次数的对数变换 | $\log(1+c_i)$ |
| 访问频率 | $r_i$ | 块累计访问次数占当前访问步数比例 | $c_i / (t+1)$ |
| 访问时效 | $\Delta_i$ | 当前步距上次访问该块的相对间隔 | $(t-t_i^{last})/(t+1)$ |
| 上下文占用率 | $\rho_t$ | 当前已激活块数占总块数比例 | $N_t / N$ |

其中累积注意力分数的统计思路借鉴自 $H_2O$ [3]，体现了块对历史生成步骤的综合贡献；块相对位置刻画块编号与块重要性的结构关系（在本文仿真中，块重要性由固定分布决定，块编号是其天然代理量）；访问频率和访问时效则反映时间局部性。实现时，式（3-1）的全部输入特征先经 `StandardScaler` 做零均值、单位方差标准化后再送入 MLP。需要指出，块位置特征的有效性依赖于训练迹与测试迹共享同一块重要性分布这一前提；若跨迹分布存在漂移，块位置特征的泛化信息量将显著下降。真实 LLM 系统中还可进一步加入层索引、头索引、块内最大/平均注意力等结构特征，但这些扩展未纳入本文的仿真原型。

**预测模型选型。** 考虑到推理阶段的延迟敏感性，预测器须满足参数量小、推理延迟低两项约束。本文选用**双层 MLP** 作为默认预测器，结构为：

$$\hat{d}_i = W_2\,\text{ReLU}(W_1 \mathbf{f}_i + b_1) + b_2 \tag{3-2}$$

其中 $W_1 \in \mathbb{R}^{32 \times 6}$，$W_2 \in \mathbb{R}^{1 \times 32}$，参数总量约 257 个，单次推理浮点运算量可忽略不计。输出 $\hat{d}_i$ 为对块 $b_i$ 下次被访问前所需等待步数（即**预测重用距离**）的估计值。对于资源极端受限的场景，亦可替换为深度不超过 5 的分类回归树（Classification and Regression Tree, CART），以牺牲少量精度换取更低的推理开销。

### 3.3 基于重用距离预测的驱逐（Eviction）策略

在真实系统中，驱逐控制器可由 GPU 显存水位阈值 $\theta$ 触发，例如当 KV Cache 占用超过工程设定阈值时批量选择若干候选块换出。本文仿真并未模拟真实显存水位，而是采用固定缓存容量：当一次块级访问未命中且缓存已满时，触发一次驱逐并释放 1 个块。对应到真实系统时，下面算法中的驱逐目标块数 $k$ 应由当前显存压力、预留安全余量和调度器批量换出策略共同确定。

**算法 1：基于预测重用距离的 KV Cache 驱逐**

```
输入：当前 GPU 中所有缓存块集合 B = {b_1, ..., b_N}
      驱逐目标块数 k（由当前内存压力决定）
输出：被驱逐块集合 E，|E| = k

1.  FOR each block b_i in B:
2.      提取特征向量 f_i（见式 3-1）
3.      计算预测重用距离 d_i = MLP(f_i)
4.  END FOR
5.  按 d_i 降序排序，选取前 k 个块构成候选集 E_cand
6.  FOR each b_i in E_cand:
7.      IF b_i 属于注意力汇聚锚点块（序列首部 2 块）:
8.          从 E_cand 移除 b_i，补入下一候选
9.      END IF
10. END FOR
11. 将 E_cand 中各块换出至 CPU DDR 内存，或在压缩式实现中降级为低精度/低优先级状态
12. 更新块表中对应条目状态为"已换出"
13. RETURN E_cand 作为最终驱逐集合 E
```

第 6–9 行引入了对 StreamingLLM [9] 所揭示的注意力汇聚现象的保护：序列首部少量 Token 即便预测重用距离较大，也不应被驱逐，因其承担稳定注意力分布的锚点作用。被驱逐的块可以换出至 CPU 内存而非直接丢弃，保留未来按需回调的可能性，这与 FlexGen [8] 的多层卸载思路一致；也可以在压缩式实现中转为低精度或低优先级状态。本文仿真只抽象出“GPU 预算内是否命中”的块级行为，不模拟具体数据搬移延迟。

### 3.4 轻量级预测器的训练数据采集与在线更新扩展

**离线训练阶段。** 在真实系统实现中，预测器的初始模型可通过 Trace-driven 方式训练：在若干代表性工作负载（如长文本问答、多轮对话、代码补全）上运行目标 LLM，记录稀疏注意力选择、KV 压缩保留或分层 offload 调入所形成的块级访问序列，并依据访问序列计算每一时刻各块的**真实重用距离**（即当前访问事件至下次同块访问事件的间隔）。在本文仿真中，由于尚未采集真实 LLM 访问日志，上述 Trace 由第 5 章描述的合成访问迹生成器替代。训练样本按访问事件逐条构造：每条样本使用该次访问后更新的累积注意力、访问次数和上次访问间隔等状态作为输入；若该块在未来不再访问，则将重用距离标签截断为 1000，避免无穷大标签破坏回归训练。训练目标为最小化预测重用距离与真实重用距离之间的均方误差：

$$\mathcal{L} = \frac{1}{|\mathcal{D}|}\sum_{(\mathbf{f}_i,\, d_i) \in \mathcal{D}} \bigl(\hat{d}_i - d_i\bigr)^2 \tag{3-3}$$

这一训练范式借鉴了 Liu 等 [14] 对 Belady 策略的模仿学习思路：虽然在线推理时未来访问序列未知，但离线 Trace 提供了精确的监督信号，使预测器可以学习接近最优驱逐策略的重用距离估计。本文原型使用 `scikit-learn` 的 `MLPRegressor` 实现 6-32-1 ReLU 网络，学习率为 0.005，最大迭代次数为 400，连续 30 轮无明显改进后提前停止；本文未使用独立验证集进行超参数搜索，相关设置仅服务于合成迹可行性验证。

**在线增量更新扩展。** 在真实部署中，系统还可以维护一个固定容量的**经验回放缓冲区** $\mathcal{B}_{\text{online}}$（例如容量 $C = 10^4$ 条样本）。每当一个换出块被重新调回 GPU 时，系统记录其实际重用距离，并将对应的（特征向量, 实际重用距离）对写入缓冲区。每处理 $K$ 个新样本后，可对 MLP 执行一次小批量梯度下降更新，并混入少量离线训练样本以缓解灾难性遗忘。该机制为负载分布漂移提供了可行的自适应路径，但本文当前实验未实现在线训练闭环，因此第 5 章只报告离线训练预测器的仿真结果。

---

## 4. 系统软件栈与硬件协同优化探讨

第 3 章给出了 Learned 的算法逻辑，但若要把它放入真实推理系统，还必须回答两个工程问题：策略层接在什么位置，以及预测器开销是否会抵消驱逐收益。本章只讨论这两个问题，作为第 3 章算法到真实系统之间的接口说明；第 5 章的实验仍然只验证离线 Trace 上的驱逐效果。

### 4.1 与主流推理框架的接口适配

vLLM 的块管理与调度路径负责物理块的分配、释放与跨请求共享。为将第 3 章的学习型驱逐策略接入此类分页 KV Cache 管理框架，本文设计如下接口适配方案。

**可插拔策略接口。** 在 `BlockSpaceManager` 的驱逐调用路径上抽象出策略接口 `BlockEvictionPolicy`，如图 2 所示：

![图 2　LearnedEvictionPolicy 在 vLLM 中的接口位置与组件关系](fig2.png)

**图 2　LearnedEvictionPolicy 在 vLLM 中的接口位置与组件关系**

图 2 的重点不是重新实现 vLLM 的块管理器，而是在现有 `BlockSpaceManager` 与底层 `swap_out/swap_in` 能力之间插入一个可替换的策略层。`LRUEvictionPolicy` 保留为基线和异常回退；`LearnedEvictionPolicy` 只接管"选择哪些块换出"这一决策，不改变 PagedAttention 的块表格式和物理块分配逻辑。

`FeatureExtractor` 在每次注意力计算完成后，从注意力元数据中读取各块的累积注意力分数，与块表中已有的位置、时效信息合并为式（3-1）的特征向量。为避免 MLP 推理占用 GPU 计算流，`MLPPredictor` 可采用开放神经网络交换格式运行时（Open Neural Network Exchange Runtime, ONNX Runtime）在 CPU 上执行；鉴于预测器参数量仅约 257 个，其计算量相对于 GPU 注意力计算较小。`OnlineUpdater` 可作为独立线程异步执行增量更新，但该部分属于工程扩展，不计入本文仿真实验结果。

**与 PagedAttention 的块粒度对齐。** vLLM 默认块大小为 16 个 Token，与本文方案的块级操作粒度天然一致，无需额外的粒度转换逻辑。当驱逐控制器决定换出物理块时，通过 vLLM 已有的 `swap_out` 接口将块内容经 PCIe 总线写入 CPU 内存，并在块表中将该块状态置为 `SWAPPED`；后续若预取需求触发，则调用对称的 `swap_in` 接口将块回调至 GPU，整个过程与 vLLM 原有的 Beam Search 换入换出机制共享同一套底层基础设施，改动量最小。

### 4.2 预测开销与软硬件协同约束

从计算机体系结构角度审视，第 3 章的预测器本质上是一个小规模推理任务：6 维输入、两次矩阵-向量乘法、一次 ReLU 激活、标量输出。单个块的预测开销很小，但在长上下文和高并发场景下，若每生成一个 Token 均需为大量缓存块执行特征提取与预测，累计开销仍可能影响端到端延迟。因此，Learned 的系统实现应遵循两个约束。

**第一，预测应批量化并尽量避开 GPU 主计算流。** 将 $N$ 个块的特征向量拼接为批量矩阵后，可利用 CPU 单指令多数据流（Single Instruction Multiple Data, SIMD）指令或基础线性代数子程序库（Basic Linear Algebra Subprograms, BLAS）完成整批矩阵乘法，减少逐块调用带来的调度开销。FlashAttention [5] 和 FlashAttention-2 [15] 的核心启示在于，注意力相关计算的性能往往受内存访问和并行划分影响；对 Learned 而言，同样应避免频繁的小粒度同步，将预测放在 CPU 侧异步执行，并只把最终驱逐候选返回给 GPU 块管理器。

**第二，硬件加速只能作为长期扩展。** 理论上，可以在内存控制器旁增设轻量级加速单元，直接读取块元数据并输出预测重用距离，从而减少 CPU/GPU 之间的数据搬移。但该方案需要定制硬件支持，工程复杂度远高于本文提出的软件策略。因此，本文不把近内存预测器作为当前方案的一部分，只将其视为面向专用推理芯片的后续研究方向。

---

## 5. 仿真实验与结果分析

前文提出的 Learned 策略需要回答两个基本问题：第一，相比完全不理解注意力结构的 LRU，学习型重用距离预测是否能够提升命中率；第二，相比专门面向注意力重尾分布设计的 $H_2O$-style 启发式规则，Learned 是否至少能够达到接近的水平。由于本文尚未接入真实 LLM 推理框架，本章采用合成稀疏块级访问迹进行离线仿真，实验结论只对应给定访问迹上的 GPU 驻留命中率，不直接代表端到端延迟、PCIe 换入换出开销或生成质量。

### 5.1 仿真环境与评测指标

**仿真框架。** 本文采用 **Trace-driven 离线仿真**方案，仿真器以 Python 实现（源码文件 `kv_cache_sim.py` 与论文文件同目录）。由于缺少真实 LLM 推理框架中的注意力访问日志，本文使用合成访问迹模拟 KV Cache 的块级访问行为，在此基础上对比各替换策略的缓存命中率。这里的一次“访问”表示某个 KV 块在稀疏注意力、压缩保留或分层 offload 调度中被选择为需要驻留 GPU 的事件，并不表示 dense attention 每步读取全部历史 KV 的访存过程。合成迹的生成模型融合了两类经过文献验证的访问规律：①**注意力重尾分布**——块重要性服从 Zipf(1.5) 分布，约 20% 的块承担约 80% 的累积注意力分数，与 $H_2O$ [3] 对真实 LLM 注意力权重的统计结论一致；②**时间局部性偏置**——最近 8 个活跃块的访问权重加倍，模拟自回归解码中近期 Token 的高频访问特征。每步随机选取约 25% 的活跃块进行访问，用于近似稀疏块级选择过程。

**代码可用性。** 为便于课程评阅和复现实验，本文使用的仿真脚本 `kv_cache_sim.py` 可作为论文附件随同提交；若后续将代码发布到公开仓库，可在本处补充真实仓库地址。当前版本不填写虚构或不可访问的源码链接。

**实验设置。** 仿真使用两条共享重要性分布的访问迹：训练迹（1,500 步，生成 36,228 条访问记录）用于离线训练 MLP 预测器；测试迹（2,000 步，生成 48,728 条访问记录，不同访问模式种子）用于策略对比评估。两条迹的块重要性分布由同一随机种子（seed=0）生成并共享，确保块编号与重要性的对应关系在跨迹评测时保持一致，使块位置特征具备有效的泛化信息量。Block 0 被显式固定为 Attention Sink（重要性设为其余块最高值的 5 倍，占比约 66.5%），与 StreamingLLM [9] 揭示的锚点 Token 现象对齐。仿真器还对所有策略统一设置首部块保护：默认不驱逐 block 0 和 block 1。需要强调，只有 block 0 被额外提高重要性；block 1 的保护只是保守的首部锚点保护规则，并且对 LRU、OPT、$H_2O$-style 与 Learned 全部一致，因此不会给 Learned 单独带来优势。

$H_2O$-style 策略采用块级近似实现：在统一首部块保护之外，维护最近 8 次访问形成的 Recent Window；当需要驱逐时，先排除 Recent Window 中的块，再从剩余候选中选择累积注意力分数最低的块。该规则等价于在块级仿真中优先保留累积注意力较高的 Heavy Hitter，但并非 $H_2O$ 原论文 token 级算法在真实模型中的端到端复现。缓存容量设定为总块数（100 块）的 **40%**（即 40 块），模拟 GPU 显存受压场景；多缓存预算实验额外覆盖 20%–80% 区间。共对比四种策略：OPT（Belady 最优，需预知未来，作为理论上界）、LRU（基准）、$H_2O$-style 块级基线、Learned（本文提出的双层 MLP 预测重用距离策略）。主要参数如表 2 所示。本文还增加了特征消融实验（5 种特征配置）和多随机种子统计（5 组独立实验）以增强结论的可靠性。

**表 2　仿真实验参数设置**

| 参数 | 取值 | 说明 |
|:-----|:----:|:-----|
| 总块数 | 100 | 合成 KV Cache 块总数 |
| 缓存容量（主实验） | 40 块 | 占总块数 40% |
| 重要性分布种子 | 0 | 训练迹与测试迹共享同一块重要性分布 |
| Attention Sink | block 0 | 重要性固定为其余块最高值的 5 倍（占比约 66.5%） |
| 首部块保护 | block 0、block 1 | 所有策略统一不驱逐，用于模拟首部锚点保护 |
| 训练步数 | 1,500 | 生成 36,228 条训练访问记录 |
| 测试步数 | 2,000 | 生成 48,728 条测试访问记录 |
| 块重要性分布 | Zipf(1.5) + Sink | 模拟注意力重尾分布及锚点 Token |
| 近期窗口 | 8 块 | 最近活跃块访问权重加倍 |
| 每步访问比例 | 25% | 模拟稀疏注意力访问 |
| $H_2O$-style Recent Window | 8 次访问事件 | 保护最近 8 次访问涉及的块集合 |
| Learned 预测器 | 6-32-1 MLP | 参数量约 257 |
| 多种子实验组数 | 5 | 各组独立控制重要性分布与访问模式 |

**评测指标。** 主要评测指标为**缓存命中率**（Hit Rate）——命中次数占总访问次数的比例，直接反映显存压力的缓解效果。此外，图 3 折线子图展示了每 200 次访问窗口内命中率的变化趋势，用于分析各策略在不同仿真阶段的稳定性。

### 5.2 实验结果与分析

**整体命中率。** 四种策略在测试迹上的仿真结果如表 3 所示。

**表 3　各替换策略缓存命中率对比（缓存预算 40%，测试迹 48,728 次访问）**

| 策略 | 命中率 | 相对 LRU 增益 |
|:-----|:------:|:------------:|
| OPT（理论上界） | **92.7%** | **+8.5 pp** |
| Learned | <u>90.8%</u> | <u>+6.6 pp</u> |
| $H_2O$-style [3]（块级基线） | 90.2% | +6.0 pp |
| LRU（基准） | 84.2% | — |

Learned 相比 LRU 基准提升 **6.6 个百分点**，说明学习型重用距离预测在本文合成访问迹上能够有效改善块级驻留决策。Learned 相比 $H_2O$-style 基线（含 Recent Window 保护）高 **0.6 个百分点**，表明 MLP 回归预测在该合成迹上能够略优于固定排序规则；与 OPT 理论上界差距约 **2.0 个百分点**，预测器参数量仅约 **257 个**，符合第 3 章的轻量化设计目标。由于 0.6 pp 差距较小，该结果更适合解释为“Learned 在当前设定下不弱于强启发式基线”，而非真实系统中必然显著优于 $H_2O$。

![图 3　各替换策略缓存命中率对比](kv_cache_sim_results.png)

**图 3　各替换策略缓存命中率对比。** 左侧柱状图给出整条测试迹上的平均命中率；右侧折线图按 200 次访问为窗口统计局部命中率，用于观察策略在访问模式演化时的稳定性。

**多缓存预算分析。** 表 4 给出 20%–80% 预算区间内各策略的命中率，图 4 为对应折线图。

**表 4　多缓存预算下各替换策略命中率（%）**

| 缓存预算 | OPT | $H_2O$-style | Learned | LRU | Δ(Lrn−LRU) | Δ(Lrn−H2O) |
|:-------:|:---:|:------:|:-------:|:---:|:----------:|:----------:|
| 20% | 69.1 | 52.5 | 62.5 | 21.6 | **+40.9** | **+10.0** |
| 30% | 85.6 | 79.5 | 82.3 | 67.2 | +15.1 | +2.9 |
| 40% | 92.7 | 90.2 | 90.8 | 84.2 | +6.6 | +0.6 |
| 50% | 95.9 | 94.3 | 94.5 | 91.3 | +3.2 | +0.2 |
| 60% | 97.5 | 96.4 | 96.4 | 94.7 | +1.7 | ±0.0 |
| 70% | 98.5 | 97.6 | 97.6 | 96.7 | +0.8 | −0.1 |
| 80% | 99.2 | 98.4 | 98.4 | 98.0 | +0.3 | −0.1 |

![图 4　不同缓存预算下各策略命中率](kv_cache_budget_results.png)

**图 4　不同缓存预算下各策略命中率**

Learned 在 20%–50% 的紧缩预算区间内持续优于 $H_2O$-style 基线，其中 20% 预算时优势为 **10.0 个百分点**。这一结果表明，在本文合成迹中，当 GPU 驻留预算最紧、每次驱逐决策代价最高时，MLP 回归预测机制具有更明显的边际收益。在 60% 及以上的宽松预算下，各策略差距收敛，接近 OPT 上界，Learned 与 $H_2O$-style 基线趋于持平。Learned 相比 LRU 在全部预算区间内均取得正增益，说明其相对通用缓存策略的优势不依赖特定预算点。

**特征消融实验。** 为分析各特征维度对 Learned 的贡献，表 5 给出 5 种特征配置下的命中率。

**表 5　特征消融实验结果（缓存预算 40%）**

| 特征配置 | 使用特征 | 命中率 | 相对全特征 |
|:--------|:--------|:------:|:--------:|
| 全部 6 特征（原版） | 块位置、累积注意力、对数频次、访问频率、访问时效、上下文占用率 | **90.8%** | 基准 |
| 去除累积注意力+访问频率 | 块位置、对数频次、访问时效、上下文占用率 | 90.6% | −0.2 pp |
| 去除块位置+累积注意力+频率 | 访问时效、上下文占用率 | 84.2% | −6.6 pp |
| 仅访问时效（退化为 LRU） | 访问时效 | 84.1% | −6.7 pp |
| 仅累积注意力 | 累积注意力 | **90.8%** | 0.0 pp |

消融结果揭示以下规律。**第一，累积注意力是主导特征：** 单独使用累积注意力可复现全特征 MLP 的完整性能（90.8%），说明注意力分数信号足以区分高/低重用距离块。值得注意的是，“仅累积注意力”的 MLP（90.8%）比 $H_2O$-style 基线（90.2%）高 0.6 pp——两者核心信息来源相近，差距来自 MLP 的回归预测机制比“Recent Window 保护 + 最小累积注意力驱逐”的固定规则更灵活；这也是全特征 Learned 超越该启发式基线的主要原因。**第二，块位置在共享重要性分布下具有独立贡献：** “去除累积注意力+访问频率”后仍保留了块位置、对数频次、访问时效和上下文占用率，命中率仅降低 0.2 pp（90.6%）；而进一步去除块位置后命中率降至 84.2%（−6.6 pp），说明块位置特征在块重要性与块编号存在固定对应关系的合成设定下，能够独立承担接近累积注意力的预测能力。该现象不应直接推广到真实 LLM Trace，因为真实负载中块编号与重要性的对应关系可能随请求内容变化。**第三，仅用访问时效近似 LRU：** 命中率 84.1%，与 LRU（84.2%）几乎相同，印证了单一时效特征退化为 LRU 策略的理论预期。

上述结果还揭示了 Learned（全特征，90.8%）轻微超越 $H_2O$-style 基线（90.2%）的原因。消融表明，仅用累积注意力一个特征的 MLP 就已达到 90.8%，与全特征完全持平。这说明 0.6 pp 的优势**不是来自多特征的非线性组合**，而是来自 **MLP 与固定启发式决策机制的差异**：$H_2O$-style 基线使用“保护 Recent Window + 驱逐 min(cum_attn)”这两条固定规则，MLP 则将累积注意力分数回归映射为重用距离预测，在排序精度上比固定排序规则更灵活，从而在边缘情况下做出更接近 Belady 最优的驱逐决策。块位置特征的独立预测价值体现在“去除累积注意力后仍能维持 90.6%”的消融结果中，但它并非 Learned 超越 $H_2O$-style 基线的直接原因。

**多随机种子统计。** 为评估实验结论的稳健性，表 6 给出 5 组独立种子（每组独立控制块重要性分布、训练迹和测试迹）实验的均值与标准差。

**表 6　多随机种子统计结果（缓存预算 40%，5 组）**

| 策略 | 均值 | 标准差 |
|:----|:---:|:-----:|
| OPT（理论上界） | 90.9% | ±2.1% |
| Learned | **88.1%** | ±3.0% |
| $H_2O$-style | 87.3% | ±3.2% |
| LRU（基准） | 80.7% | ±4.6% |

Learned 在 5 组独立实验中均值（88.1%）高于 $H_2O$-style 基线（87.3%），差距为 0.8 pp，与单次运行结论方向一致。LRU 标准差最大（±4.6%），说明其性能对具体访问模式较敏感；Learned 标准差（±3.0%）略低于 $H_2O$-style 基线（±3.2%）。需要指出，由于实验组数有限，0.8 pp 的差距在统计上不显著，实验结论应理解为“Learned 在当前合成迹上不弱于 $H_2O$-style 基线，并在低缓存预算下表现出更明显优势”。

**命中率趋势分析。** 从图 3 折线子图可观察到，Learned 在整个测试过程中稳定接近或超过 $H_2O$-style 基线，仅在少数窗口出现小幅交叉。LRU 命中率曲线波动相对较大，在访问模式切换时出现明显下跌，反映其对非均匀访问分布的适应能力较弱。

**局限性。** 本仿真存在以下需正视的局限。**其一**，合成迹无法完全还原真实 LLM 推理中多层、多头注意力的异质性，也不代表 dense attention 每步读取全部历史 KV 的完整访存过程，真实场景的性能可能与仿真结果存在偏差。**其二**，本文仿真未实现第 3.4 节讨论的在线增量训练闭环，因此无法证明在线自适应机制的实际收益。**其三**，本仿真未度量 PCIe 传输、预取失败和 GPU stall，也未能度量驱逐策略对生成质量（如 Perplexity、任务准确率）的影响，这需要在完整的端到端 LLM 推理框架中进行。**其四**，消融实验在合成迹上所揭示的特征重要性可能与真实 LLM Trace 存在差异：真实场景中访问模式更复杂，块位置与其他特征的交互效应可能更为显著。

---

## 6. 结论与未来展望

### 6.1 全文总结

本文围绕大语言模型推理中的 KV Cache 显存瓶颈，研究了一个具体问题：在稀疏注意力、KV 压缩或分层 offload 场景中，当 GPU 显存不足、只能在 PagedAttention 块粒度上保留部分候选 KV Cache 时，能否利用历史访问特征预测未来重用距离，从而做出比通用缓存替换策略更合理的驱逐或换出决策。为此，本文提出 Learned 策略，将块位置、累积注意力分数、访问频次、访问时效与上下文占用率组合为六维特征，并使用双层 MLP 预测块的未来重用距离，再优先换出预测重用距离较大的块。

Learned 的主要意义在于为 KV Cache 管理提供了一种介于固定启发式和完整系统重构之间的轻量级方案：它不要求改变模型结构，也不要求重写底层分页内存管理，而是在现有块管理机制之上增加一个可训练的驱逐决策层。其创新点可以概括为三方面：一是以重用距离作为学习目标，将 Belady 式缓存替换思想迁移到 KV Cache 块级管理场景；二是将注意力分数、位置、频次、时效和上下文占用率统一为可学习特征，而不是依赖单一人工规则；三是将该策略设计为可插拔模块，使其能够与 PagedAttention、`swap_out/swap_in` 和后续在线更新机制协同工作。

从系统角度看，Learned 的定位是可插拔的驱逐决策模块。基于修正后的合成稀疏块级访问迹 Trace-driven 仿真实验（共享块重要性分布、$H_2O$-style 块级基线、多预算与多种子统计）表明，在缓存预算为总块数 40% 时，Learned 相比 LRU 命中率提升 **6.6 个百分点**，相比 $H_2O$-style 基线高 **0.6 个百分点**，预测器参数量仅 257 个；多随机种子统计（5 组）显示 Learned 平均不低于该启发式基线，但小幅差距不宜过度解释。在 20% 极紧预算下，Learned 相对 $H_2O$-style 基线的优势扩大至 **10.0 个百分点**，说明在本文合成迹中，显存压力最高时 MLP 回归预测机制的边际收益更明显。特征消融实验表明：单独使用累积注意力特征的 MLP 即可达到全特征水平（均为 90.8%），说明 Learned 超越 $H_2O$-style 基线（90.2%）的 0.6 pp 优势来源于 MLP 对注意力分数的回归预测比固定排序规则更灵活，而非来自多特征的联合效应；块位置特征在共享重要性分布下具有独立预测能力（去除累积注意力后仍维持 90.6%），但这与 Learned 对 $H_2O$-style 基线的优势无直接关联。

由此可以得到本文的主要结论：**学习型重用距离预测在稀疏块级 KV Cache 驱逐问题上具有可行性，在合成迹场景下能够明显优于不感知注意力结构的通用替换策略，并通过 MLP 的回归预测机制轻微超过 $H_2O$-style 强启发式基线**；特征消融表明，当前优势来源于机制灵活性而非多维特征融合——多维特征的真正价值有待真实 LLM 访问迹验证。现阶段结果仍停留在合成 Trace 仿真层面，尚不能证明其在真实 LLM 服务中的端到端延迟或生成质量收益。

### 6.2 未来研究方向

本文的探索尚留有若干值得深入的问题。**真实 Trace 与端到端质量评估**：后续应在 vLLM 等推理框架中采集真实注意力访问日志，并评估驱逐策略对 Perplexity、任务准确率和端到端延迟的影响。**在线更新机制验证**：经验回放与增量微调目前仍是设计扩展，需要通过真实负载漂移实验验证其收益和开销。**预测器的硬件卸载**：将 MLP 推理下沉至近内存计算单元，从体系结构层面降低预测开销，是面向专用推理芯片设计的重要方向。**多租户场景下的 QoS 协同**：在共享 GPU 集群中，不同请求的 KV Cache 驱逐决策相互耦合，如何在保障各租户服务质量的同时最大化整体命中率，需要与调度器协同设计。**与模型量化的联合优化**：KV Cache 量化与驱逐策略存在互补效应 [13]，两者的联合优化空间值得系统性探索。

---

## 参考文献

[1] Touvron H, Lavril T, Izacard G, 等：LLaMA: Open and Efficient Foundation Language Models，arXiv预印本，2023，arXiv:2302.13971

[2] Kwon W, Li Z, Zhuang S, 等：Efficient Memory Management for Large Language Model Serving with PagedAttention，Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)，2023，611-626

[3] Zhang Z, Sheng Y, Zhou T, 等：$H_2O$: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models，Advances in Neural Information Processing Systems，2023，36

[4] Liu Z, Desai A, Liao F, 等：Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time，Advances in Neural Information Processing Systems，2023，36

[5] Dao T, Fu D Y, Ermon S, 等：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness，Advances in Neural Information Processing Systems，2022，35 16344-16359

[6] Vaswani A, Shazeer N, Parmar N, 等：Attention Is All You Need，Advances in Neural Information Processing Systems，2017，30 5998-6008

[7] Ainslie J, Lee-Thorp J, de Jong M, 等：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints，Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing，2023，4895-4910

[8] Sheng Y, Zheng L, Yuan B, 等：FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU，Proceedings of the 40th International Conference on Machine Learning (ICML)，2023，31094-31116

[9] Xiao G, Tian Y, Chen B, 等：Efficient Streaming Language Models with Attention Sinks，Proceedings of the 12th International Conference on Learning Representations (ICLR)，2024

[10] Li Y, Huang Y, Yang B, 等：SnapKV: LLM Knows What You Are Looking for Before Generation，Advances in Neural Information Processing Systems，2024，37

[11] Liu Y, Li H, Chen X, 等：CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving，Proceedings of the ACM SIGCOMM 2024 Conference，2024，38-56

[12] Belady L A：A Study of Replacement Algorithms for a Virtual-Storage Computer，IBM Systems Journal，1966，5(2) 78-101

[13] Li H, Li Y, Tian A, 等：A Survey on Large Language Model Acceleration based on KV Cache Management，arXiv预印本，2024，arXiv:2412.19442

[14] Liu E Z, Hashemi M, Swersky K, 等：An Imitation Learning Approach for Cache Replacement，Proceedings of the 37th International Conference on Machine Learning (ICML)，2020，119 6237-6247

[15] Dao T：FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning，Proceedings of the 12th International Conference on Learning Representations (ICLR)，2024

[16] Behnam P, Fu Y, Zhao R, 等：RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression，Proceedings of the 42nd International Conference on Machine Learning (ICML)，2025，3358-3392

[17] Wang G, Upasani S, Wu C, 等：LLMs Know What to Drop: Self-Attention Guided KV Cache Eviction for Efficient Long-Context Inference，arXiv预印本，2025，arXiv:2503.08879

[18] Chen Y, Zhang J K, Lu B, 等：RetroInfer: A Vector Storage Engine for Scalable Long-Context LLM Inference，Proceedings of the VLDB Endowment，2025，19(5)

[19] Dong Y, Miao Y, Li W, 等：Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching，Proceedings of the AAAI Conference on Artificial Intelligence，2026，40(25) 20844-20851

[20] Gao W, Zhou X, Sun P, 等：Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving，Proceedings of Machine Learning and Systems (MLSys)，2025

[21] Xu Y, Khaira N K, Singh T：KV Cache Optimization Strategies for Scalable and Efficient LLM Inference，arXiv预印本，2026，arXiv:2603.20397

[22] Mamo O, Kogiou O, Yi H, 等：Comparative Characterization of KV Cache Management Strategies for LLM Inference，arXiv预印本，2026，arXiv:2604.05012

[23] Ahn J, Seong I, Kedia A, 等：LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation，arXiv预印本，2026，arXiv:2603.10899
