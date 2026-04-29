---
title: 我是怎么从零写一个多头注意力层的，以及中间踩的那些坑
slug: attention-from-scratch
date: 2025-03-25
readTime: 15 分钟
category: 基础原理
tags: Transformer, Attention, PyTorch
cover: ./content/assets/posts/covers/attention.svg
excerpt: 从实现细节而非公式推导出发，复盘手写多头注意力时在 mask、shape、内存布局与性能取舍上的关键踩坑与经验。
---

# 我是怎么从零写一个多头注意力层的，以及中间踩的那些坑

前一段时间，我在做一个序列建模任务，数据量不大，但对延迟很敏感，用 HuggingFace 套壳总觉得哪里不对劲——不是说 Transformers 库不好，是那种用别人家厨房做饭的感觉，食材放在哪里你不清楚，火候也不太可控。后来决定把注意力层自己写一遍，理由很朴实：遇到问题要能改。

这篇文章不是教你多头注意力的原理，如果你还不清楚 Attention 是什么，去看 Vaswani 2017 那篇 paper 就够了。我想讲的是**实现过程中真正让人头疼的地方**，那些论文和教程不会告诉你的细节。

---

## 从一个最简单的问题开始：scale 放在哪里

注意力的计算公式大家都背得出来：

![Scaled dot-product attention computation pipeline showing QK transpose multiplication, scale by sqrt(d_k), softmax normalization, and weighted V aggregation.](./content/assets/posts/covers/scaled_dot_product.png)

*图1：Scaled Dot-Product Attention 的计算流程示意*

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


但真正写代码的时候，第一个让我停下来想的问题不是矩阵乘法，而是：这个 $\sqrt{d_k}$ 到底在运行时算还是预计算？

我最开始的实现是每次 forward 都在线算：

```python
scale = math.sqrt(self.head_dim)
attn_scores = (q @ k.transpose(-2, -1)) / scale
```

虽然这个开销在 GPU 上几乎可以忽略（benchmark 了一下差不到 0.1ms），但从工程习惯上讲把它放进 `__init__` 更清晰。这件事看起来很小，但它让我意识到自己写代码时的一个坏习惯：在 forward 里做不必要的常量计算，对于更复杂的项目这会是个真实的坑。

```python
# __init__ 里
self.scale = math.sqrt(self.head_dim)
```

这不是性能优化，是代码意图的表达——scale 是模型的静态属性，不是每次推理都在变的东西。

---

## reshape vs view，contiguous 是什么玄学

如果你第一次写这段代码：

```python
q = self.w_q(q).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
```

大概率会遇到这个报错：

```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

我当时是在做 `transpose` 之后接了一个 `view`，然后就炸了。解决方法是在 view 之前加 `.contiguous()`，或者把 view 换成 reshape。

但我花了一点时间去搞清楚这两者的区别，不然下次遇到还是蒙的。

PyTorch 里的 tensor 有个 `strides` 属性，描述内存里相邻元素之间的跨步。一个形状为 `(B, T, D)` 的张量，默认 stride 是 `(T*D, D, 1)`——行优先存储，很直觉。但 `transpose` 之后，内存排布没动，只是 stride 换了顺序，这时候 tensor 不再是"连续"的，`view` 要求内存必须连续，所以报错。

`contiguous()` 会重新分配内存把数据排好，`reshape` 内部做了同样的事但对你透明。从性能角度来说两者几乎等价，但在多头注意力这种全是矩阵变换的代码里，我倾向于显式写 `contiguous()`，因为这样你知道"这里有一次内存拷贝"，在 profiling 的时候不会漏掉。

最后那段 reshape 回去的代码：

```python
output = output.transpose(1, 2).contiguous().view(B, T_q, D)
```

transpose 把 `(B, H, T_q, d_k)` 变成 `(B, T_q, H, d_k)`，然后 contiguous 保证内存连续，view 把最后两维合并成 `D = H * d_k`。这是整个实现里最容易写错维度的地方，我建议把每一步的 shape 都注释上，不是给别人看的，是给三个月后的自己看的。

---

## mask 的处理：我在这里翻车了两次

![Lower-triangular causal mask matrix where each token only attends to itself and previous tokens, visualized as an autoregressive attention mask.](./content/assets/posts/covers/causal_mask.png)

*图2：自回归场景下的因果掩码（下三角）示意图*

掩码（mask）可能是整个实现里设计决策最多的地方，也是我改动最多的地方。

**第一次翻车**：mask 的语义。

我最开始把 mask 定义成"True 表示被屏蔽的位置"，因为直觉上感觉这样自然——"这个位置被掩掉了，所以是 True"。但用了一段时间之后发现这个约定跟 PyTorch 原生的 `nn.MultiheadAttention` 是反的——它用 `key_padding_mask` 里 True 表示"需要被忽略"，但在内部的 `attn_mask` 里又是用加法掩码。HuggingFace 的实现里 attention_mask 又是"1 表示有效，0 表示无效"。

各家约定不一，我最后选了"True/1 表示需要关注，False/0 表示被屏蔽"，理由是这样在创建因果掩码的时候更自然——下三角全是 True，表示当前位置能看到的历史。

```python
def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
```

但这个选择的代价是，每次从外部接入 mask 的时候都要核对一遍语义，因为调用方可能有不同的假设。我在代码里加了详细的文档注释，但还是有队友用反过来传进来，导致模型训练前几个 step loss 异常高，排查了半个小时才发现。

**第二次翻车**：布尔 mask 和加法 mask 混用。

实践中 mask 有两种形式：一种是布尔 mask，通过 `masked_fill` 把无效位置填成 `-inf`（softmax 后变成 0）；另一种是加法 mask，直接加到 attention scores 上（比如 ALiBi 位置编码）。这两种不能混用，但接口上很难区分，所以我在 forward 里加了显式的 dtype 判断：

```python
if mask.dtype == torch.bool:
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
else:
    attn_scores = attn_scores + mask
```

注意这里 `~mask` 是取反——因为我们的约定是 True 表示"可以看"，所以 `~mask` 里 True 的位置才是要填 `-inf` 的地方。这行代码我自己测试时写反过一次，排查时对着 attention weight 矩阵看了很久才发现因果掩码完全是倒的——高层 token 可以看到低层，低层反而看不到自己。

---

## 三种注意力模式：同一个类，三种用法

多头注意力的一个设计上让我比较满意的地方是它天然支持三种使用场景，只是输入不同：

**自注意力（encoder 风格）**：q、k、v 都是同一个序列，没有掩码。这是标准的 BERT 风格，每个位置都能看到全序列。

```python
output, _ = attention(X, X, X)
```

**带因果掩码的自注意力（decoder 风格）**：同样是 q、k、v 相同，但加上下三角掩码。自回归生成时用这个，保证 position t 只能看到 0 到 t 的内容。

```python
causal_mask = create_causal_mask(seq_len, X.device)
output, attn_weights = attention(X, X, X, mask=causal_mask, need_weights=True)
```

**交叉注意力（encoder-decoder）**：q 来自 decoder，k 和 v 来自 encoder 输出。这时候 q 和 k/v 的序列长度可以不同，这也是实现里 T_q 和 T_k 分开处理的原因。

```python
encoder_output = torch.randn(batch_size, 32, d_model)  # 不同长度
output, _ = attention(X, encoder_output, encoder_output)
```

最开始我是把这三种写成三个独立的函数，后来发现逻辑上 90% 是重复的，合并成一个带参数的类反而更干净。但这也引入了一个认知负担：调用方需要知道"q、k、v 该怎么传"，尤其是新来的同学不一定清楚 self-attention 时三个参数传同一个 tensor。

这是一个接口设计上的取舍，我现在仍然觉得这种方式是对的，但文档和注释必须写清楚，不然很容易用错。

---

## 和 PyTorch 原生实现的对比

写完自己的版本之后，我花时间对比了 `torch.nn.MultiheadAttention` 的实现，差异比我预期的大。

PyTorch 官方实现里有几个我没有做的事情：

**融合的线性投影（in_proj_weight）**：官方把 Q、K、V 三个投影矩阵合并成一个大矩阵 `in_proj_weight`，形状是 `(3*d_model, d_model)`，然后一次矩阵乘法完成三个投影，再 chunk 开。理论上这样对 cuBLAS 更友好，实际上在我们的场景下（batch_size=16，seq_len=64）几乎没有区别，但在更大的 batch 和更长的序列下应该是有收益的。

**`F.scaled_dot_product_attention`**：从 PyTorch 2.0 开始，官方引入了这个融合算子，内部会根据条件选择 Flash Attention 或者普通实现。我的版本是手写的 attention，没有用这个。好处是实现完全透明可控，坏处是在长序列上内存占用会显著高于 Flash Attention——Flash Attention 的核心优化是用分块计算避免把完整的 attention matrix 写回 HBM，我的实现没有这个。

对于我当时的任务（seq_len ≤ 512），显存不是瓶颈，所以没有引入这个依赖。但如果你在做长文档处理或者多轮对话，手写的 attention 在 seq_len > 1024 之后会开始显著慢于 Flash Attention。

还有一个差异是 **`need_weights` 的开销**。我的实现里，当 `need_weights=False` 时 `attn_weights` 仍然被计算出来（只是不返回），实际上应该在 `need_weights=False` 的路径上用 `F.scaled_dot_product_attention` 直接跳过权重的显式计算。这是个已知的优化点，还没改。

---

## dropout 加在哪：一个让我想了很久的问题

Dropout 有两个可以加的位置：attention weights 上，或者输出上。我选择了放在 attention weights 上：

```python
attn_weights = F.softmax(attn_scores, dim=-1)
attn_weights = self.dropout(attn_weights)
output = attn_weights @ v
```

这个位置的物理意义是：随机把某些 attention 头对某些 token 的关注"切掉"，让模型不过度依赖某个固定的注意力模式。Vaswani 原始论文就是这么做的。

但有些人认为应该在最后的线性投影之后再加 dropout，理由是这样的正则化更稳定，不会因为 softmax 之后的随机置零导致 attention distribution 变得不归一（dropout 之后权重和不再是 1，理论上这是个问题）。

我查了一些资料，两种做法在实践上差异很小，但从理论完备性上讲，他的观点有道理——softmax 输出是概率分布，dropout 之后它就不是了，虽然下游的 `@ v` 操作仍然能用，但语义上有点奇怪。不过这里有个细节值得说一下：PyTorch 的 `F.dropout` 在训练时会自动按 `1/(1-p)` 对保留的元素做缩放（inverted dropout），所以在期望意义上，注意力分布的均值仍然是无偏的。这也是为什么实践中这样做没有引发灾难性后果的原因——理论上权重和不为 1，但期望上仍然守恒。Transformers 库里不同版本的实现也前后换过位置。

我最后还是留在 attention weights 上，主要是为了和原始论文对齐，减少解释成本。这不是一个让我特别有把握的决定。

---

## `need_weights` 这个参数的设计

返回 attention weights 有时候是有用的——可视化、调试、或者做 attention-based 的特征提取。但在正常的训练和推理中它是额外开销：一个 `(B, H, T_q, T_k)` 的张量，seq_len=512 时大概是 `16 * 8 * 512 * 512 * 4 bytes ≈ 128MB`，这不是小数。

所以我加了 `need_weights` 参数，默认 False，只有显式要求时才返回。返回值设计成 tuple `(output, attn_weights)`，其中 attn_weights 在 False 时是 None。

这个设计有一个稍微反直觉的地方：即使 `need_weights=False`，attention weights 在计算图里仍然存在（只是没有被返回），如果要彻底避免它的存储，需要在 False 路径下走完全不同的计算路径。具体来说，在 PyTorch 2.0+ 中可以直接调用：

```python
output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0)
```

这个算子是 C++ 实现的融合内核，只有在显式需要权重时才会把完整的 attention matrix 写回显存，走 Flash Attention 路径时甚至根本不会产生这个中间张量。现在的实现是个"half measure"——接口上告诉调用方不需要权重，但内存上仍然分配了。这是下一个版本要改的地方。

---

## 下一步想做的事

这个实现现在在我的项目里跑得还不错，但有几个地方我一直想改但还没动手：

**GQA（Grouped Query Attention）**：LLaMA 2 和 Mistral 都在用这个——把 K 和 V 的头数减少，Q 的头数保持不变，每组 Q 共享一对 K/V。好处是显著减少 KV cache 的大小，代价是表达能力略有损失。现在的代码假设 Q、K、V 头数相同，改起来需要动 reshape 的部分，不复杂但需要仔细。

**RoPE 位置编码**：现在的实现没有内置位置编码，需要在外层处理。但 RoPE 的特点是直接作用在 Q 和 K 上、在 attention 计算之前旋转嵌入，和 attention 模块是紧密耦合的，拆开的话反而让外部调用变复杂。还在想怎么设计这个接口。

**Flash Attention 的接入**：如上面说的，对于长序列场景是必须的。但 flash-attn 库的安装很麻烦（依赖特定的 CUDA 版本），在我的部署环境里还没有跑通，先搁置着。

还有一件事让我最近在思考：我自己维护这个实现，意味着每次 PyTorch 升级都要重新验证一遍，`F.scaled_dot_product_attention` 的行为是否变了，掩码的 API 是否有 breaking change。这个维护成本是真实存在的。如果当初用 HuggingFace，这些事情他们帮你做了；自己写，你拿到了可控性，但也要承担相应的维护责任。

到底值不值，取决于你的项目有多定制化。对我来说暂时是值的，但我不会无条件推荐别人这么做。

---
## *完整代码参考，如果你有更好的实现方式或者发现了我描述有误的地方，欢迎在评论区讨论。*
<pre>
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """多头注意力机制。
    
    参数:
        d_model: 模型的维度（必须能被 n_heads 整除）
        n_heads: 注意力头的数量
        dropout: Dropout 概率 (默认: 0.1)
        bias: 线性投影中是否使用偏置 (默认: True)
    
    形状:
        - 输入: (batch_size, seq_len, d_model)
        - 输出: (batch_size, seq_len, d_model)
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        # 输入验证
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"
            )
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        # 输出线性层
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        参数:
            q: 查询张量 (B, T_q, D)
            k: 键张量 (B, T_k, D)
            v: 值张量 (B, T_v, D)
            mask: 可选的掩码张量，形状为 (B, 1, T_q, T_k) 或 (T_q, T_k)
                  True/1 表示需要关注的位置，False/0 表示被屏蔽的位置
            need_weights: 是否返回注意力权重
            
        返回:
            output: (B, T_q, D)
            attn_weights: 如果 need_weights 为 True 则返回 (B, H, T_q, T_k)，否则为 None
        """
        B, T_q, D = q.size()
        T_k = k.size(1)
        
        # 线性投影并拆分成多个头
        # (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k)
        q = self.w_q(q).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(k).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(v).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        # (B, H, T_q, d_k) @ (B, H, d_k, T_k) -> (B, H, T_q, T_k)
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale
        
        # 如果提供了掩码则应用
        if mask is not None:
            # 确保掩码在同一个设备上
            if mask.dim() == 2:
                # (T_q, T_k) -> (1, 1, T_q, T_k)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (B, T_q, T_k) -> (B, 1, T_q, T_k)
                mask = mask.unsqueeze(1)
            
            # 处理布尔类型的掩码
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            else:
                attn_scores = attn_scores + mask
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 将注意力权重应用到值上
        # (B, H, T_q, T_k) @ (B, H, T_k, d_k) -> (B, H, T_q, d_k)
        output = attn_weights @ v
        
        # 拼接多个头
        # (B, H, T_q, d_k) -> (B, T_q, H, d_k) -> (B, T_q, D)
        output = output.transpose(1, 2).contiguous().view(B, T_q, D)
        
        # 最后的线性投影
        output = self.w_o(output)
        
        if need_weights:
            return output, attn_weights
        return output, None


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """为自回归注意力创建因果（下三角）掩码。
    
    参数:
        seq_len: 序列长度
        device: 创建掩码的设备
        
    返回:
        mask: 形状为 (seq_len, seq_len) 的布尔张量
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


if __name__ == "__main__":
    # 配置参数
    d_model = 512
    n_heads = 8
    batch_size = 16
    seq_len = 64
    
    # 创建模型
    attention = MultiHeadAttention(d_model, n_heads, dropout=0.1)
    
    # 测试 1: 无掩码的自注意力（编码器风格）
    X = torch.randn(batch_size, seq_len, d_model)
    output, _ = attention(X, X, X)
    print(f"编码器自注意力输出形状: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    
    # 测试 2: 带因果掩码的自注意力（解码器风格）
    causal_mask = create_causal_mask(seq_len, X.device)
    output, attn_weights = attention(X, X, X, mask=causal_mask, need_weights=True)
    print(f"解码器自注意力输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 测试 3: 交叉注意力（编码器-解码器）
    encoder_output = torch.randn(batch_size, 32, d_model)  # 不同的序列长度
    output, _ = attention(X, encoder_output, encoder_output)
    print(f"交叉注意力输出形状: {output.shape}")
    
    print("\n✅ 所有测试通过！")

 
</pre>

## 更多参考 
* https://github.com/tangentllm
