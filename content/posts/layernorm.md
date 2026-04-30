---
title: 我为什么要手写一遍 LayerNorm
slug: layernorm-from-scratch
date: 2025-04-26
readTime: 11 分钟
category: 基础原理
tags: Transformer, LayerNorm, PyTorch
cover: ./content/assets/posts/covers/layernorm.png
excerpt: 围绕 eps、方差估计与精度问题，拆解 LayerNorm 的实现细节与工程踩坑经验。
---

# 我为什么要手写一遍 LayerNorm

在做一个推理服务的性能优化时，我第一次认真盯着 `nn.LayerNorm` 的源码看了很久。当时遇到一个诡异的问题：同样的模型，在 A100 上用 fp16 推理，偶尔会出现 loss spike，换回 fp32 就消失了。排查了两天，最后定位到 eps 的问题——我自己改过一个训练脚本，把 eps 从 `1e-5` 改成了 `1e-6`，觉得"精度更高"。结果在 fp16 下，`var + eps` 算出来的值有时候直接下溢，归一化结果爆掉了。

那次之后我决定自己手写一遍，不是因为要替换官方实现，而是想真正搞清楚每一行在干什么。

---

## 它在解决什么问题

Layer Normalization 的出现其实是为了解决 Batch Normalization 在某些场景下的局限。BatchNorm 在 batch 维度做归一化，这意味着在推理时你需要维护一个 running mean/var，而且 batch size 很小的时候统计量会很不稳定。如果batch size 经常是 1，用 BatchNorm 基本等于放弃。

LayerNorm 的思路是换个维度——不在 batch 上统计，在每个样本自己的特征维度上做。公式写出来很简单：

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 μ 和 σ² 是对最后一维（也就是 d_model 那个维度）算出来的均值和方差。γ 和 β 是可学习的缩放和偏移，初始化成 1 和 0。

理解这个公式很容易，但魔鬼藏在实现细节里。

---

## 手写过程中的那些坑

先说 `unbiased=False`。PyTorch 的 `torch.var()` 默认是无偏估计，也就是分母用 `N-1`。但 LayerNorm 里应该用有偏估计，分母是 `N`。这不是我自己想出来的——第一版代码里我直接写 `x.var(dim=-1, keepdim=True)`，和官方实现对比时发现有细微差异，在 d_model 比较小的时候（比如 64）差异会被放大，max diff 能到 `1e-4` 级别。翻了一下 PyTorch 的 LayerNorm 实现才发现这里有区别。

eps 的选择是第二个坑。`1e-5` 是 PyTorch 官方默认值，看起来随意，但其实有考量。fp16 的最小正规数大概是 `6e-5`，subnormal 从 `6e-8` 开始。如果你把 eps 设得太小，在 fp16 下 `var + eps` 可能还是一个很小的数，开根号之后精度损失会很严重。我们当时出问题就是把 eps 调到 `1e-6`，觉得能提升数值精度，结果适得其反。

还有一个当时没注意、后来回头看才意识到的地方：input shape 的校验。最早的版本没有这个检查，直接做计算。新旧权重的 d_model 不一样（一个是 512，另一个是 768），按理说 PyTorch 这里会直接抛 `RuntimeError`——512 和 768 从右对齐根本没法 broadcast。但当时的权重加载代码有个隐患：加载后做了一次错误的 `unsqueeze`，把 weight 从 `[768]` 变成了 `[1, 1, 768]`，输入 `x` 又因为某处转置 bug 变成了 `[B, 512, 1]`，两者在 PyTorch 的 broadcast 规则下居然能"对上"，算出来的结果维度都正确，但数值完全是垃圾。这种情况最可怕，因为没有任何报错，模型还在跑，只是悄悄地输出错误结果。

加了显式的 shape 检查之后，这类问题在加载阶段就能暴露：

```python
if x.shape[-1] != self.d_model:
    raise ValueError(
        f"输入最后一维 {x.shape[-1]} 与 d_model {self.d_model} 不匹配"
    )
```

细节很重要

---

## 性能：手写版 vs 官方实现

手写版在数值上和 `nn.LayerNorm` 的差异可以控制在 `1e-6` 以内，但性能差很多。我在 A100 上做了个简单的 benchmark，batch=32、seq_len=512、d_model=768 的配置，跑 1000 次取平均：

| 实现 | 平均耗时 |
|------|----------|
| 手写版（纯 Python + PyTorch 算子拼接） | ~0.87ms |
| `nn.LayerNorm`（fused CUDA kernel） | ~0.11ms |

差了将近 8 倍。原因很直接：我的手写版会产生多个中间 tensor（mean、var、x_norm），每个都要做一次显存读写。官方实现用的是 fused kernel，把这些操作合并在一次 GPU 计算里，减少了大量的内存带宽消耗。在 Transformer 里 LayerNorm 会被调用很多次，这 8 倍的差距叠加下来非常可观。

所以结论很明确：手写版用来学习，生产环境老老实实用 `nn.LayerNorm`。

---

## 横向看一下其他实现

Apex 的 `FusedLayerNorm` 是另一个选择，在早期 PyTorch 的 LayerNorm 还没那么成熟的时候用得比较多。现在 PyTorch 原生实现已经足够好，Apex 的优势在特定场景（比如混合精度训练的极端优化）下才体现出来，一般项目没必要引入额外依赖。

Flash Attention 里也有自己的 LayerNorm 实现，和 attention 计算做了更深度的融合。如果你在用 Flash Attention，那块的 LayerNorm 已经是最优路径之一了，不用另外折腾。

RMSNorm 是 LayerNorm 的一个变体，去掉了均值中心化这一步，只做方差归一化。LLaMA 系列用的就是这个。理论上少算一步，速度略快，而且实践中效果差不多。我后来在一个小项目里把 LayerNorm 换成 RMSNorm，推理延迟大概降了 3-5%，差距不算大，但积少成多。

---

## 验证的思路

手写实现最重要的是验证。我用了三层检查：

**第一层**，shape 验证，确保输入输出一致。

**第二层**，数值验证——在 gamma=1、beta=0 的初始状态下，归一化后的输出均值应该接近 0，方差应该接近 1。这是 LayerNorm 的定义保证的，如果这里不过，基本是算子用错了。

```python
mean_err = y.mean(dim=-1).abs().max().item()
var_err = (y.var(dim=-1, unbiased=False) - 1.0).abs().max().item()
assert mean_err < 1e-4
assert var_err < 1e-4
```

**第三层**，和官方实现对比。直接把 gamma、beta 复制过去，算同一个输入，max diff 要在 `1e-5` 以下。这一层是最硬的指标，如果这里有问题，说明实现本身有 bug。

写这些验证的过程本身也很有价值，它迫使你把每个假设都显式地写出来，而不是凭感觉觉得"应该没问题"。

---

## 后面的迭代

有一个我之前一直有误解的地方，后来才搞清楚：`keepdim=True` 产生的 `[B, S, 1]` 形状张量，在和 `[B, S, D]` 做运算时，并不会在显存里把数据物理展开。PyTorch 的 broadcast 底层是通过把那个维度的 stride 设为 0 来实现的，GPU 读数据时反复读同一个地址，不需要也不会分配额外显存。所以我之前担心超长序列（seq_len > 8192）下 keepdim 的内存开销，其实是瞎操心。如果真的在长序列下遇到内存压力，锅在 `x` 本身或者反向传播时保存的 activation（比如 `x`、`mean`、`var` 都要留下来算梯度），不在 keepdim。

真正还没解决的问题是 bf16 下的梯度行为。bf16 的动态范围和 fp32 一样宽（不容易溢出），但尾数只有 7 位，比 fp16 的 10 位还少，精度极低。LayerNorm 反向传播的梯度公式里有 $\frac{1}{\sqrt{\sigma^2 + \epsilon}}$ 和 $\frac{x - \mu}{(\sigma^2 + \epsilon)^{3/2}}$ 这类项，当方差很小的时候，这些除法在 bf16 下的相对误差会被急剧放大。目前业界的做法是在混合精度训练时强制让 LayerNorm 的 forward 和 backward 都在 fp32 下跑，算完再 cast 回去。Megatron-LM 和 Hugging Face 里不少 LLaMA 实现都对 Norm 层做了特殊处理，不让它参与 `autocast` 的降精度。这个方向我还没有在自己的项目里系统验证过，但直觉上这是个比 keepdim 内存更值得花时间的问题。

<pre> 
import torch
import torch.nn as nn


class Layer_Norm(nn.Module):
    """
    手动实现的 Layer Normalization。
    理解其内部原理
    
    论文：《Layer Normalization》Ba et al., 2016
    公式：LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    
    注意：若无自定义需求，直接使用 nn.LayerNorm 即可（有 fused kernel 加速）。

    Args:
        d_model: 归一化的最后一维大小
        eps: 防止除零的稳定项，默认 1e-5（对齐 PyTorch 官方，fp16 更安全）
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入校验：最后一维必须与 d_model 匹配
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"输入最后一维 {x.shape[-1]} 与 d_model {self.d_model} 不匹配"
            )
        # 除以 N（有偏估计，unbiased=False）：纯粹的数学计算。我有 64 个数字，算方差就是把它们平方和除以 64。
        # 除以 N-1（无偏估计，unbiased=True）：统计学里的概念。如果你用这 64 个数字去估算一个无限大总体的方差，除以 64 算出来的结果会系统性地偏小。为了修正这个偏差，统计学发明了除以 N-1（这叫贝塞尔校正）。
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ── 验证 ────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, d_model = 2, 5, 512
    x = torch.randn(B, T, d_model)

    ln = Layer_Norm(d_model)
    y = ln(x)

    # shape 验证
    assert y.shape == x.shape, "输出 shape 不符"

    # 数值验证（初始状态 gamma=1, beta=0，归一化结果均值≈0，方差≈1）
    mean_err = y.mean(dim=-1).abs().max().item()
    var_err = (y.var(dim=-1, unbiased=False) - 1.0).abs().max().item()
    assert mean_err < 1e-4, f"均值偏差过大: {mean_err}"
    assert var_err < 1e-4, f"方差偏差过大: {var_err}"
    print(f"均值最大误差: {mean_err:.2e}  ✅")
    print(f"方差最大误差: {var_err:.2e}  ✅")

    # 与 nn.LayerNorm 对比
    ref = nn.LayerNorm(d_model, eps=1e-5)
    with torch.no_grad():
        ref.weight.copy_(ln.gamma)
        ref.bias.copy_(ln.beta)
    max_diff = (y - ref(x)).abs().max().item()
    assert max_diff < 1e-5, f"与官方实现差异过大: {max_diff}"
    print(f"与 nn.LayerNorm 最大误差: {max_diff:.2e}  ✅")

    # 异常输入测试
    try:
        ln(torch.randn(2, 5, 256))  # 错误的 d_model
        assert False, "应该抛出异常"
    except ValueError as e:
        print(f"shape 校验正常触发: {e}  ✅")

</pre>

## 更多参考 
* https://github.com/tangentllm
