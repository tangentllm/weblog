---
title: 正余弦位置编码（Sinusoidal Positional Encoding）深度解析
slug: sinusoidal-positional-encoding
date: 2026-05-02
readTime: 14 分钟
category: 基础原理
tags: Transformer, Positional Encoding, RoPE
cover: ./content/assets/posts/covers/position.svg
excerpt: 从“Attention 天然不感知顺序”出发，推导正余弦位置编码公式与线性变换性质，并结合 PyTorch 实现逐行拆解。
---

# 正余弦位置编码（Sinusoidal Positional Encoding）深度解析

> 基于《Attention Is All You Need》(Vaswani et al., 2017)，结合 PyTorch 代码实现

---

## 1. 问题背景

### Attention 机制为什么天然不感知位置顺序？

Self-Attention 的计算本质是：对每个 query，计算它与所有 key 的点积打分，再加权求和 value：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

整个操作对输入顺序是**排列不变的（permutation invariant）**——把序列的 token 打乱顺序再送入，每个位置的输出完全相同（只是对应关系乱了）。

**反例**："猫咬了狗" 与 "狗咬了猫" 送入纯 Attention，模型看到的权重矩阵完全一样，无法区分语义差异。

### 位置编码要解决的核心问题

在不破坏 Attention **并行计算**优势的前提下，向模型注入序列的**绝对位置信息**，使得"第 3 个 token"和"第 7 个 token"的表示有本质区别。

---

## 2. 数学定义与公式推导

### 2.1 完整公式

$$PE(pos,\ 2i)   = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE(pos,\ 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

| 符号 | 含义 | 取值范围 |
|------|------|----------|
| `pos` | token 在序列中的位置 | 0, 1, …, seq\_len − 1 |
| `i` | 维度对的索引（每对包含 sin + cos） | 0, 1, …, d\_model/2 − 1 |
| `d_model` | 词向量维度（如 512） | 由模型超参数决定 |
| `10000` | 基底，控制波长范围 | 论文选定的经验常数 |
| `2i / d_model` | 指数，决定每个维度的频率 | 0 → 1（从高频到低频） |

### 2.2 为什么选 10000 作为基底？

波长由基底决定：

$$\lambda_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}$$

- 当 $i = 0$ 时：$\lambda_0 = 2\pi \approx 6.28$（最短，高频，区分近邻位置）
- 当 $i = d/2 - 1$ 时：$\lambda_{\max} = 2\pi \times 10000 \approx 62{,}832$（最长，低频，覆盖极长序列）

选 10000 的理由：
- 最长波长远超常见序列长度（通常 512 ~ 4096），确保高维分量能区分序列头尾
- 保留足够的高频分量，使模型能区分相邻位置
- 若基底过小，高维波长不够长，远距离位置编码会发生"混叠"

### 2.3 线性变换性质的证明

**目标**：证明对任意偏移量 $k$，$PE(pos + k)$ 可由 $PE(pos)$ 线性变换得到。

设 $\varphi = pos / 10000^{2i/d}$，$\psi = k / 10000^{2i/d}$，则：

$$\sin(\varphi + \psi) = \sin(\varphi)\cos(\psi) + \cos(\varphi)\sin(\psi)$$

$$\cos(\varphi + \psi) = \cos(\varphi)\cos(\psi) - \sin(\varphi)\sin(\psi)$$

写成矩阵形式：

$$\begin{bmatrix} \sin(\varphi + \psi) \\ \cos(\varphi + \psi) \end{bmatrix} = \underbrace{\begin{bmatrix} \cos(\psi) & \sin(\psi) \\ -\sin(\psi) & \cos(\psi) \end{bmatrix}}_{M(\psi),\ \text{仅依赖偏移量}\ k} \begin{bmatrix} \sin(\varphi) \\ \cos(\varphi) \end{bmatrix}$$

**关键结论**：旋转矩阵 $M(\psi)$ 仅依赖偏移量 $k$，与 $pos$ 无关。这意味着模型可以通过学习线性层来推断相对位置关系，这正是正余弦编码的核心数学性质。

---

## 3. PyTorch 代码逐行解析

### 完整代码（含详细注释）

```python
import torch
import torch.nn as nn
import math

class Position_Embedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(Position_Embedding, self).__init__()

        # ── Step 1 ──────────────────────────────────────────────
        # 创建全零矩阵，存储所有位置的编码向量
        # 维度变化: [max_len, d_model]  e.g. [5000, 512]
        pe = torch.zeros(max_len, d_model)

        # ── Step 2 ──────────────────────────────────────────────
        # 创建位置索引列向量
        # 维度变化: [max_len] → unsqueeze(1) → [max_len, 1]
        # 增加列维度是为了后续与 div_term 广播相乘
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # ── Step 3 ──────────────────────────────────────────────
        # 计算缩放因子 div_term
        # 维度: [d_model/2]  值域: exp(0)=1.0 → exp(-log10000)≈1e-4
        #
        # 数学等价：
        #   exp(-2i * log(10000) / d) = 10000^(-2i/d) = 1 / 10000^(2i/d)
        #
        # 工程原因（为什么用 exp(log(...)) 而不是直接 ** 幂运算）：
        #   1. 数值稳定性：避免大数幂运算的浮点精度损失
        #   2. 计算效率：exp/log 在 GPU 上有高度优化的 CUDA kernel
        #   3. 梯度规则：链式法则下梯度形式更简洁
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # ── Step 4 ──────────────────────────────────────────────
        # 广播机制：
        #   position  [max_len, 1]
        #   div_term  [d_model/2]
        #   乘积      [max_len, d_model/2]  （自动广播）
        #
        # 偶数列 0, 2, 4, ..., d-2 填入 sin 值
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数列 1, 3, 5, ..., d-1 填入 cos 值
        pe[:, 1::2] = torch.cos(position * div_term)

        # ── Step 5 ──────────────────────────────────────────────
        # 增加 batch 维度，使位置编码可与输入直接相加
        # 维度变化: [max_len, d_model] → [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # ── Step 6 ──────────────────────────────────────────────
        # 注册为 buffer（而非 nn.Parameter）
        # 详见下方"register_buffer vs nn.Parameter"说明
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的维度:      [B, seq_len, d_model]
        # self.pe 维度:  [1, max_len, d_model]
        # 切片后:        [1, seq_len, d_model]
        # 广播后:        [B, seq_len, d_model]
        #
        # [:, :x.size(1)] 取前 seq_len 列，自动适配变长序列
        # batch 维度 1 → B，每个样本共享同一套位置编码
        output = x + self.pe[:, :x.size(1)]
        return output
```

### register_buffer vs nn.Parameter 的区别

| 特性 | `nn.Parameter` | `register_buffer` |
|------|----------------|-------------------|
| 参与反向传播 | ✅ 是 | ❌ 否（梯度为 None） |
| `optimizer.step()` 更新 | ✅ 会更新 | ❌ 不更新 |
| `model.parameters()` 包含 | ✅ 包含 | ❌ 不包含 |
| `state_dict()` 保存 | ✅ 保存 | ✅ 保存 |
| `.to(device)` 自动迁移 | ✅ 迁移 | ✅ 迁移 |
| 适用场景 | 需要训练的权重 | 固定常量、缓存 |

**结论**：位置编码是固定公式生成的常量，不需要训练，但需要跟随模型移动到 GPU 并随 checkpoint 保存。`register_buffer` 是语义最准确的选择。

---

## 4. 直觉理解

### 4.1 二进制计数器类比

观察二进制计数：

```
位置  bit2  bit1  bit0（最低位）
  0    0     0     0
  1    0     0     1   ← 最低位每步翻转（高频）
  2    0     1     0
  3    0     1     1
  4    1     0     0   ← 最高位很少翻转（低频）
  5    1     0     1
  6    1     1     0
  7    1     1     1
```

正余弦编码做的完全相同：
- **最低位 → 低维（i=0）**：波长约 6，每隔几个 token 完成一个周期，**区分相邻位置**
- **最高位 → 高维（i=d/2-1）**：波长约 62,800，整个序列只走半个周期，**区分序列前半段 vs 后半段**

d\_model 个维度联合起来，形成近似唯一的"坐标"，理论上可区分任意两个不同位置。

### 4.2 低维与高维分别负责什么？

| 维度范围 | 频率特征 | 负责区分 | 类比 |
|----------|----------|----------|------|
| 低维（i 小） | 高频，短波长 | 相邻位置（1~10 步内） | 秒针 |
| 中维 | 中频 | 中等距离（10~100 步） | 分针 |
| 高维（i 大） | 低频，长波长 | 远距离位置（段落级别） | 时针 |

### 4.3 为什么相加而不是拼接？

**拼接**需要额外维度：词向量 512 + 位置编码 512 = 1024 维，整个模型所有权重矩阵翻倍，参数量和计算量都翻倍。

**相加**让位置信息直接融入词的语义空间，后续 Attention 层在同一个 d\_model 维空间工作，计算代价不变。论文实验表明两种方案效果相近，相加更经济。

---

## 5. 与其他位置编码的对比

| 对比项 | 正余弦（固定） | 可学习位置编码 | RoPE |
|--------|--------------|--------------|------|
| 是否训练 | 否，固定公式 | 是，随模型训练 | 否，旋转矩阵固定 |
| 能否外推 | 有限，超长序列性能下降 | 不能，超出训练长度则无效 | 较好，天然支持相对位置 |
| 计算方式 | sin/cos 解析式 | 查表（Embedding） | 旋转 Q、K 向量 |
| 与输入关系 | 相加（加性） | 相加（加性） | 乘性（旋转融合） |
| 建模位置信息 | 绝对位置 | 绝对位置 | 相对位置（内积天然编码） |
| 参数量 | 0 参数 | max\_len × d\_model | 0 参数 |
| 代表模型 | 原版 Transformer（2017） | BERT、GPT-2 | LLaMA、Qwen、Mistral |

---

## 6. 常见误解与易错点

### ❌ 误解1：直接用 pos/max_len 归一化不就行了？

归一化值域为 [0, 1]，**同一相对位置在不同序列长度下编码不同**：

- pos=5，max\_len=10 → 编码值 0.5
- pos=5，max\_len=100 → 编码值 0.05

模型无法泛化到不同长度的序列。正余弦方案的编码值只依赖绝对位置 `pos`，与 `max_len` 无关，天然具备长度无关性。

### ❌ 误解2：奇偶维度可以互换（sin↔cos）？

`sin` 在 pos=0 时值为 0，`cos` 在 pos=0 时值为 1。若互换，pos=0（序列起始位置）的编码会发生改变，影响模型对"起始位置"的感知。

论文并未严格规定哪个维度必须用哪个函数，但**一旦约定，不可随意互换**，否则破坏了编码的内部一致性。

### ❌ 误解3：位置编码应该用 nn.Parameter，这样可以微调

正余弦位置编码的设计初衷是**固定常量，不参与训练**。

- 若误用 `nn.Parameter`，优化器会为其分配梯度缓冲区，浪费显存
- 位置编码对 loss 的梯度理论上存在（通过残差连接传回），但更新它违背了设计意图
- `register_buffer` 既能保存到 checkpoint，又能随模型迁移到 GPU，语义最准确

---

## 7. 延伸思考

### 7.1 超长序列（> max_len）时会怎样？

正余弦编码在**数学上无限可延伸**（公式没有上界），但实际问题来自模型的泛化能力：

- 训练时 max\_len 决定了模型"见过"的位置范围
- 超出范围时，高维（低频）分量进入从未训练过的区域
- 模型的注意力分布会退化，最终性能下降

这不是编码本身的问题，而是模型未能从超长位置编码中学到泛化规律。

### 7.2 RoPE 解决了正余弦编码的哪个根本缺陷？

正余弦编码将位置信息**加**到 token 表示里（加性方案）。计算 Attention 的点积 $Q \cdot K$ 时，位置信息和语义信息混合在一起，模型需要"间接"推断相对位置。

**RoPE（Rotary Position Embedding）** 的核心思路：

- 直接旋转 Q 和 K 向量本身
- 使得 $Q(pos_m) \cdot K(pos_n)$ 的点积自然只依赖相对距离 $m - n$
- 相对位置关系直接编码进注意力分数，无需间接推断
- 外推能力因此更强，被现代大模型（LLaMA、Qwen 等）广泛采用

### 7.3 学习路径建议

```
正余弦 PE（本文）
    ↓
可学习 PE（BERT 的实现方式）
    ↓
相对位置编码（Shaw et al., 2018）
    ↓
RoPE（Su et al., 2021）
    ↓
ALiBi（Press et al., 2021，线性偏置方案）
```

---

*参考文献：Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.*