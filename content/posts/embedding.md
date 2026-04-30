---
title: 重新理解了一个「只有十几行」的 Embedding 层
slug: embedding-from-scratch
date: 2025-04-28
readTime: 12 分钟
category: 基础原理
tags: Transformer, Embedding, PyTorch
cover: ./content/assets/posts/covers/embedding.png
excerpt: 从初始化、缩放与 padding 细节出发，复盘 Embedding 层在训练稳定性中的关键作用。
---

# 重新理解了一个「只有十几行」的 Embedding 层

去年在复现 Transformer 的时候，我以为 Embedding 是最没有技术含量的部分。不就是一张查找表吗？`nn.Embedding` 包一套，完事。

然后我就掉坑里了。

---

事情是这样的。模型跑起来了，loss 也在下降，但收敛曲线抖得厉害，尤其是训练前期，loss 动不动就飙出一个离谱的尖峰，然后才慢慢回来。我当时以为是学习率的问题，调了半天调度器，没用。后来怀疑是 batch 里有问题样本，加了 gradient clipping，稍微好了一点，但根本没解决。

直到我把梯度可视化出来，才发现问题出在 Embedding 层的初始化上。

PyTorch 的 `nn.Embedding` 默认初始化是标准正态分布 $N(0,1)$。听起来人畜无害，但在 `d_model = 512` 的情况下，每个分量方差为 1，高维下嵌入向量的 L2 范数期望近似为 $\sqrt{512} \approx 22.6$。而 sinusoidal 位置编码的每个分量被严格约束在 $[-1, 1]$ 之间，向量范数大概是 $\sqrt{d\_model / 2} \approx 16$。

两者量级其实差不多，位置编码甚至稍小。但加上论文里那个 `* math.sqrt(d_model)` 的缩放之后，嵌入向量范数直接飙到 $22.6 \times \sqrt{512} \approx 512$，位置编码的信号被彻底淹没。

解决方案我最后选的是用标准差为 $1/\sqrt{d\_model}$ 的正态分布初始化：

```python
nn.init.normal_(self.embedding.weight, mean=0, std=1 / math.sqrt(d_model))
```

这样初始化之后，每个分量的方差为 $1/d\_model$。乘上 $\sqrt{d\_model}$ 之后，每个分量的方差变为 $1$，与 sinusoidal 位置编码各分量的方差（约 $0.5$）处于同一数量级。更重要的是，这保证了后续 Attention 计算中 $Q \cdot K^T$ 点积的方差不会因为 Embedding 初始值过大而爆炸。直觉上也好理解：缩放后嵌入向量范数约为 $\sqrt{512} \approx 22.6$，位置编码范数约为 $16$，相加时两者的信号强度是可以共存的，而不是一边倒。

训练曲线稳了很多。那些诡异的 loss 尖峰基本消失了。

---

关于那个 `* math.sqrt(self.d_model)` 的缩放因子，我想多说几句，因为原始论文（Vaswani et al., 2017）就一句话带过，很多复现直接跳过了这一步，然后发现效果差一截，却不知道为什么。

这个缩放不是孤立存在的，它和初始化方案是一体两面。用 `std=1/sqrt(d)` 初始化但不加 `* sqrt(d)` 的缩放，嵌入向量范数约为 $1$，而位置编码范数约为 $16$，相加时位置信息会完全压制词义信号，模型前期会极度依赖位置而忽视词义。反过来用默认 $N(0,1)$ 初始化不加缩放，嵌入范数约 $22.6$，位置编码约 $16$，嵌入反过来压制位置。原论文的方案——特定初始化配合显式缩放——是把这个稳定性在数学上锁死了。

有人问我为什么不直接对位置编码也做缩放、让它降下来。我试过，效果差不多，但有一个隐患：sinusoidal 位置编码的绝对值已经有明确的数学含义（不同频率的正余弦），如果缩小它，你等于在改变频率成分的相对权重，这个影响在长序列上会慢慢累积出来。保持位置编码不动，只 scale 嵌入向量，逻辑上更干净。

---

`padding_idx` 这个细节我也想提一下，因为我见过好几个开源实现里漏掉了。

问题不是"填充 token 的嵌入向量会不会影响注意力"——那个你可以在 attention mask 里处理，损失函数里也可以设 `ignore_index=pad_token_id` 跳过填充位置。问题是：如果你没有设 `padding_idx`，填充 token（通常是 ID=0）对应的嵌入行会在训练过程中被梯度更新。它会慢慢学到"我是填充"这个信号，听起来无害，但实际上这意味着模型在用一个"真实词"的槽位学了一个伪信号，在词汇表本来就紧张的情况下（比如中文分词场景，vocab 才三四万），这个浪费挺可惜的。设了 `padding_idx` 之后，那一行始终是零向量，梯度不会回传，干净很多。

```python
self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
```

注意一个坑：如果你后续要做权重共享，把 embedding 矩阵转置后当 output projection，`padding_idx` 那一行是零，softmax 之后那个位置的概率会系统性偏低。我在做 weight tying 的时候就被这个坑了，debug 了两个小时才发现。

---

横向比一下其他方案。BERT 用的是可学习的位置编码，但它不需要显式的 `* sqrt(d_model)` 缩放。BERT 官方实现对所有嵌入层用的是截断正态分布，均值 0，标准差 0.02（`initializer_range=0.02`）。在 `d_model=768` 下，这个设定给出的嵌入范数约为 $0.02 \times \sqrt{768} \approx 0.55$，位置编码和 token 嵌入都在同一量级，相加后不存在谁压倒谁的问题。更根本的原因是，BERT 的整体架构里 LayerNorm 承担了大量的尺度平抑工作，Post-LN 的位置使得嵌入层的初始尺度影响没有 Transformer 原版那么敏感。把 BERT 不需要 `* sqrt(d)` 简单归因于某种"隐式缩放"是不准确的，它更多是整体设计协同的结果。

坏处是位置泛化能力差——BERT 在预训练时是 512 长度，到了更长的序列上直接歇菜，需要各种插值或者外推技巧。

RoPE（旋转位置编码，LLaMA 系列在用）则把位置信息完全从嵌入层里剥离出来，通过旋转矩阵作用在注意力计算阶段，embedding 层就是纯粹的词义查找表。从工程角度看，这个设计最干净，embedding 层的职责单一，调试也容易，代价是注意力计算复杂度略有上升，实现也比 sinusoidal 麻烦不少。

我目前的代码停在 sinusoidal 这个阶段，主要是为了把原始 Transformer 的每一个设计决策都搞清楚再往前走。

---
## 完整代码
<pre> 
import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    """
    词嵌入层（Embedding Layer）
    
    将词ID序列转换为密集向量表示，并按照原始Transformer论文
    对嵌入向量进行 sqrt(d_model) 缩放，使其量级与位置编码对齐。
    """
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        """
        Args:
            vocab_size:  词汇表大小
            d_model:     嵌入向量维度
            padding_idx: 填充token的ID，对应的嵌入向量将被置零且不参与梯度更新
        """
        super().__init__()
        self.d_model = d_model
        
        # 创建嵌入矩阵，形状为 (vocab_size, d_model)
        # padding_idx 指定的token对应行始终为全零向量
        # 如果序列中有 padding token（通常 ID 为 0），应该告知 Embedding 层忽略它：
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # 使用均值为0、标准差为 1/sqrt(d_model) 的正态分布初始化权重
        # 避免训练初期嵌入向量方差过大导致的不稳定
        nn.init.normal_(self.embedding.weight, mean=0, std=1 / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入词ID序列，形状为 (batch_size, seq_len)
        Returns:
            嵌入向量序列，形状为 (batch_size, seq_len, d_model)
        """
        # 缩放嵌入向量：使嵌入向量的量级与位置编码保持一致（参考论文 Section 3.4）
        return self.embedding(x) * math.sqrt(self.d_model)

</pre>

## 更多参考 
* https://github.com/tangentllm
