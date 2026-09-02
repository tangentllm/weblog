# 深入理解 KV Cache：Transformer 自回归推理的核心优化

如果你已经熟悉 Transformer 的 Self-Attention 结构，却不太清楚 LLM **推理**为什么比训练慢那么多、框架里常说的 KV Cache 究竟在缓存什么——这篇文章基于 [带kv cache的Transformer.ipynb](带kv%20cache的Transformer.ipynb) 中的手撕实现与性能对比，从动机、原理、实现到权衡，把这一推理优化的核心机制讲清楚。

---

## 一、背景与问题动机

### 1.1 自回归生成：每一步都在「重算历史」

大语言模型的文本生成采用**自回归（autoregressive）**方式：给定 prompt，模型逐个预测下一个 token，每步把新 token 拼到序列末尾，再预测再一个——循环直到结束。

在 Decoder-only Transformer 中，每生成一个新 token，都要做一次完整的 Self-Attention：当前 token 的 Query 需要与**所有历史 token**（含刚生成的）的 Key 做内积，再对 Value 加权求和。问题在于：**历史 token 的 Key 和 Value 在之前的步骤里已经算过了**，朴素实现却每步从头再算一遍。

可以把它类比成记账：每记一笔新账，你不必把过去所有账本条目重新誊抄一遍；但朴素 Transformer 推理恰恰在做这件事——每步都对全长序列重新做 Linear 投影、重新算 K/V、重新做注意力。序列越长，重复劳动越多。

### 1.2 朴素实现的复杂度

设当前序列长度为 $n$，注意力头数为 $h$，每头维度 $d_k$，层数为 $L$。

**不带 KV Cache** 的每步流程：

1. 对全部 $n$ 个 token 做 Embedding + 位置编码；
2. 每层对全长序列计算 $Q, K, V \in \mathbb{R}^{n \times d}$；
3. 计算注意力分数 $\text{softmax}(QK^\top / \sqrt{d_k})V$，再经 FFN。

其中 Self-Attention 的核心计算量为 $O(n^2 \cdot d)$（$QK^\top$ 是 $n \times n$ 矩阵）。若从长度 $s$ 的 prompt 起连续生成 $m$ 个 token，第 $i$ 步序列长 $s+i$，单步代价 $O((s+i)^2)$，**总计算量**约为：

$$
\sum_{i=1}^{m} O((s+i)^2) = O(m \cdot s^2 + m^2 s + m^3)
$$

当 $m$ 较大时，推理延迟随生成长度**近似平方级乃至更高次增长**——这正是长文本生成越来越慢的根本原因。

### 1.3 KV Cache 解决什么？训练 vs 推理

**KV Cache** 的核心思路：把每层、每个注意力头已经算过的 **Key** 和 **Value** 缓存起来；新 token 到来时，只计算它的 $q, k, v$，把新的 $k, v$ **追加**到缓存末尾，再用当前 $q$ 与完整缓存做注意力。

| 阶段 | 输入形态 | KV Cache 角色 |
|------|----------|---------------|
| **训练** | 整段序列并行送入 | 通常**不使用**；Teacher Forcing 下所有位置同时计算，无增量解码需求 |
| **推理** | 逐 token 增量生成 | **核心优化**：避免重复计算历史 K/V，将每步复杂度从 $O(n^2)$ 降为 $O(n)$ |

本质上，KV Cache 是**用显存换计算时间**——缓存历史「账本条目」，新步只追加一行，不再重算旧账。

---

## 二、核心原理

### 2.1 为什么只缓存 K 和 V，不缓存 Q？

Self-Attention 的计算流程（单头）：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- **Query** 表示「当前 token 要查什么」——只有**当前这一步正在生成的新 token** 需要 Query，历史 token 的 Q 在生成它们时已经用过，后续步骤不会再被引用。
- **Key / Value** 表示「历史 token 提供了什么可被检索的信息」——每生成一个新 token，它都要与**全部历史** K 做匹配、对全部历史 V 加权。这些历史 K/V 在后续每一步都会被复用。

因此缓存对象是 K 和 V；每步只需为新 token 算一个 $q$（及对应的 $k, v$ 用于更新缓存）。

### 2.2 增量推理（Incremental Decoding）

参考资料的实现区分两种前向模式：

1. **Prefill（首次 / 无缓存）**：prompt 整段送入，计算全部 token 的 Q/K/V，并将各层 K/V 写入 `kv_cache`；
2. **Decode（增量 / 有缓存）**：只取序列**最后一个 token** 送入模型，算新 $q, k, v$，与缓存拼接后再做注意力。

注意力在增量步的形状变化（batch=1，单头示意）：

| 张量 | 增量步 shape | 说明 |
|------|-------------|------|
| $Q$ | `[1, 1, d_k]` | 仅新 token |
| $K_{\text{cache}}$ | `[1, n, d_k]` | 历史 + 新 token，$n$ 随步数 +1 |
| $V_{\text{cache}}$ | `[1, n, d_k]` | 同上 |
| scores | `[1, 1, n]` | 一次矩阵乘，$O(n)$ |

多层 Decoder 每层各维护一份独立 cache，因此总 cache 条目数为 `2 × n_layers × n_heads`（K 和 V 各一份）。

### 2.3 显存占用

Cache 在第 $t$ 步的体积（浮点元素数）约为：

$$
\text{Cache Size} \propto 2 \times L \times H_{\text{kv}} \times d_k \times t
$$

其中 $L$ 为层数，$H_{\text{kv}}$ 为 KV 头数，$t$ 为当前序列长度。序列越长、层越多、头越多，显存线性增长——这是 KV Cache 的主要代价。

> **补充（知识背景）**：标准 MHA 中 $H_{\text{kv}} = H_q$；MQA 令 $H_{\text{kv}}=1$，GQA 取中间值，使 Cache 体积成倍下降。参考资料本身采用标准 MHA（`n_heads=8`），未展开 GQA/MQA 实现，但上述公式说明了为何工业界常将 GQA 与 KV Cache 组合使用。

---

## 三、实现细节

以下代码提炼自参考资料，保留核心逻辑，便于对照理解。

### 3.1 MultiHeadAttention：缓存的拼接与更新

```python
def forward(self, query, key, value, mask=None, kv_cache=None):
    Q = self.w_q(query)
    K = self.w_k(key)
    V = self.w_v(value)

    Q = Q.view(B, seq_len, n_heads, d_k).transpose(1, 2)
    K = K.view(B, -1, n_heads, d_k).transpose(1, 2)
    V = V.view(B, -1, n_heads, d_k).transpose(1, 2)

    if kv_cache is not None:
        if 'k' in kv_cache and 'v' in kv_cache:
            K = torch.cat([kv_cache['k'], K], dim=2)  # 沿 seq 维拼接
            V = torch.cat([kv_cache['v'], V], dim=2)
        kv_cache['k'] = K
        kv_cache['v'] = V

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    context = torch.matmul(F.softmax(scores, dim=-1), V)
    # ... reshape + w_o
```

要点：`dim=2` 是 sequence 维；每步 append 新 K/V，原地更新 `kv_cache` 供下一层/下一步复用。

### 3.2 带 / 不带 Cache 的生成循环

**朴素实现**——每步处理完整序列：

```python
def generate_without_kv_cache(model, input_ids, max_new_tokens=20):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated)              # 全长前向
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated
```

**带 KV Cache**——Prefill 后只送最后一个 token：

```python
def generate_with_kv_cache(model, input_ids, max_new_tokens=20):
    generated = input_ids.clone()
    kv_caches = None
    for _ in range(max_new_tokens):
        logits, kv_caches = model(generated, kv_caches)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated
```

模型内部根据 `kv_caches` 是否为空切换模式：无缓存时处理 `input_ids` 全文并初始化每层 `{}`；有缓存时 `input_ids_new = input_ids[:, -1:]`，位置编码取 `seq_len - 1`。

### 3.3 实现关键点与常见坑

| 问题 | 说明 |
|------|------|
| **Causal Mask** | Prefill 阶段用下三角 `tril` 掩码；增量步 $Q$ 仅 1 个位置，对全长 $K$ 的注意力均合法，参考资料简化为全 1 mask `[1,1,1,full_seq_len]` |
| **位置编码** | 增量步须用**绝对位置** `pos_ids = seq_len - 1`，而非从 0 重计 |
| **Cache 初始化** | 每层独立 dict；Prefill 结束后各层 `kv_cache['k'/'v']` 已含 prompt 长度 |
| **显存预分配 vs 动态增长** | 本实现用 `torch.cat` 动态拼接，简单但每步可能触发 realloc；生产环境常预分配 `[max_len, ...]` 缓冲区 |
| **Batch 变长 / Padding** | 多序列并行时 cache 按 slot 管理，需配合 attention mask 忽略 padding 位 |
| **多轮对话** | 新对话或 context 切换须**清空** cache，否则历史 K/V 会串台 |
| **Cache 与采样** | 同一 prompt 的 cache 可在 beam search 分支间复制（fork），但不可跨无关请求混用 |

---

## 四、对比分析

### 4.1 复杂度与资源对比

| 维度 | 不带 KV Cache | 带 KV Cache |
|------|--------------|-------------|
| **每步 Attention 计算** | $O(n^2)$，$n$ 为当前序列长 | $O(n)$，$Q$ 仅 1 token |
| **每步 K/V 投影** | $O(n)$，全长重算 | $O(1)$，仅新 token |
| **生成 $m$ 步总计算趋势** | $\sum (s+i)^2$，随 $m$ 快速增长 | $\sum (s+i) \approx O(m^2)$，线性累加 |
| **额外显存** | 无 persistent cache | $O(L \cdot H_{\text{kv}} \cdot d_k \cdot t)$，随 $t$ 线性增 |
| **推理延迟趋势** | 越长越慢，近似平方级 | 单步随长度线性增，长生成优势明显 |

### 4.2 参考资料实测数据

在参考资料的 benchmark 配置下（来源：notebook 运行输出）：

- 模型：`d_model=512, n_heads=8, n_layers=6, d_ff=2048`
- 初始序列长度 10，生成 50 个新 token（最终长度 60）

| 指标 | 不带 KV Cache | 带 KV Cache |
|------|--------------|-------------|
| 耗时 | **0.4460 秒** | **0.2461 秒** |
| 加速比 | — | **1.81×**（性能提升 81.2%） |

参考资料同时指出：在实际应用中，KV Cache 通常可带来 **2–10 倍**推理加速（来源：notebook 说明文字）；序列越长，优势越明显——与本实验 1.81× 的结果方向一致，绝对加速比随序列长度和模型规模上升而增大。

### 4.3 核心权衡（Trade-off）

```
         计算时间 ←—— KV Cache ——→ 显存占用
              ↓                           ↓
    避免重复 K/V 投影              缓存随序列线性增长
    注意力从 O(n²) → O(n)          多层 × 多头 × 长上下文
```

**收益递增**：生成长度 $m$ 越大，朴素实现的重复计算量以立方级累积，KV Cache 的节省越可观。

**新瓶颈——显存墙**：当上下文达 32K、128K 甚至更长，KV Cache 本身可能占数十 GB，成为 batch size 和并发度的限制因素。这也是 PagedAttention、KV 量化、GQA/MQA、MLA（Multi-head Latent Attention）等后续方向的动机——在「不回到 $O(n^2)$ 重算」的前提下继续压缩 cache  footprint。

---

## 五、总结

**KV Cache 是什么**：推理阶段在各层缓存历史 token 的 Key/Value，增量解码时只算新 token 的 Q/K/V，复用历史 K/V 完成注意力。

**适用场景**：几乎所有自回归 LLM 在线推理（Chat、代码补全、流式生成）；训练阶段 Teacher Forcing 并行计算，通常不需要。

**局限性**：

- 显存随序列长度线性增长，长上下文 + 大 batch 易触显存上限；
- 需正确管理 cache 生命周期（对话切换、分支搜索、多租户隔离）；
- 本身不减少单次注意力对历史长度的 $O(n)$ 读取，极长序列下 memory bandwidth 仍可能成为瓶颈。

**展望**（简要）：工业界在 KV Cache 之上叠加 **PagedAttention**（vLLM，分页管理显存碎片）、**量化 Cache**（INT8/INT4 KV）、**GQA/MQA**（减少 KV 头数）、**MLA**（低秩压缩 KV）等，目标一致——在保持增量解码优势的同时，缓解显存墙。

---

> 本文代码与性能数据均来自 [带kv cache的Transformer.ipynb](带kv%20cache的Transformer.ipynb)。建议动手运行 notebook 中的 `benchmark_models()`，调整 `max_new_tokens` 与 `initial_seq_len`，直观感受序列长度对加速比的影响。
