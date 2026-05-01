---
title: 一个 FFN 实现里藏着的那些事
slug: ffn-hidden-details
date: 2026-05-02
readTime: 9 分钟
category: 基础原理
tags: Transformer, FFN, Dropout, GELU, SwiGLU
cover: ./content/assets/posts/covers/ffn.svg
excerpt: 从两次 Dropout 的真实踩坑出发，重新审视 FFN 的激活函数选择、正则位置与“记忆存储”的解释框架。
---

# 一个FFN实现里藏着的那些事

去年在复现一篇关于稀疏注意力机制的论文时，我花了大量时间死磕一个莫名其妙的精度问题，最后发现根源不在注意力层，而在我对FFN的一个"理所当然"的实现上——在同一个输出上做了两次Dropout。从那以后我开始重新审视这个看起来人畜无害的两层MLP，发现里面确实有些值得聊的东西。

---

FFN的结构简单到近乎无聊：一个线性层把维度炸开，过个激活函数，再用一个线性层压回来。Transformer原始论文里用的是 $d_{model}=512$，$d_{ff}=2048$，也就是4倍扩展。这个4倍从哪来的？原论文作者（Ashish Vaswani 等）其实是通过实验发现：4× 在机器翻译任务上效果很好，再大收益递减，再小性能明显下降。4倍不是从某个严格理论推导出来的常数，而是一个在表达能力、计算成本和实验效果之间取得最佳平衡的经验设计。

我现在手头的这段实现大概长这样：

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

    def forward(self, x):
        output = self.linear1(x)
        output = self.activation(output)
        output = self.dropout(output)   # 放在中间，外层残差连接不再额外Dropout
        output = self.linear2(output)
        return output
```

先说激活函数这块。最初我跟很多人一样，反射性地用ReLU——因为熟悉，因为快，因为"ReLU不就是Transformer的标配嘛"。但实际上原始论文用的是ReLU，BERT才改成了GELU，后来几乎所有主流LLM都跟着用GELU了。有时候用ReLU的时候训练过程有点跳，loss曲线在某些epoch会突然抖一下，切到GELU之后就平滑了很多。

GELU和ReLU的区别说起来也不复杂。ReLU是个硬截断，输入小于0直接给你归零，梯度也跟着死了。GELU走的是概率路线——输出近似于 $x \cdot \Phi(x)$，其中 $\Phi$ 是标准正态分布的CDF，意思是对输入按"它有多大概率是正的"来做加权。在实际计算里用的是一个近似：

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

这个公式乍看很唬人，本质上就是用tanh拟合了正态CDF。好处是在0附近是平滑的，负值区域也保留了一点梯度，不会死得那么彻底。代价是计算比ReLU稍慢，但在GPU上这点差异基本可以忽略——我实测过，在batch_size=64、seq_len=512的设置下，单次forward的时间差在0.3ms以内。

然后是Dropout，这是我踩得最深的一个坑，但坑的位置和我最初以为的不一样。

原始Transformer论文对Dropout的描述是：*We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.* 按字面意思，sub-layer的输出就是整个FFN跑完linear2之后的向量，Dropout加在这里，然后才与残差相加、过LayerNorm。BERT和PyTorch官方Transformer实现走的正是这条路。

另一种做法是把Dropout放在激活函数之后、linear2之前，作用在高维的中间表示上。Tensor2Tensor等早期实现用的是这个方案，linear2的输出直接参与残差连接，不再单独Dropout。两种方式各有出处，并不存在谁对谁错的问题。

我当时的问题不是"位置选错了"，而是两头都加了——FFN模块内部linear2后面加了一次Dropout，外层残差连接时又加了一次，等于对同一个输出做了双重正则，实际强度翻倍。表现出来就是模型在训练集上还过得去，验证集上莫名欠拟合。当时排查了将近两天，以为是学习率的问题，还跑了一套学习率搜索，白白浪费时间。

后来重新捋模块结构的时候才意识到问题所在。修复方式很简单：选一个位置，去掉另一个。我最终选了中间Dropout这个方案——把它放在激活后、linear2前，外层残差连接那里不再额外加。选这个的原因一部分是习惯，一部分是觉得在高维空间上做掩码、逼模型不依赖特定神经元，逻辑上更直觉。但说实话，如果改成linear2后Dropout、外层干净，效果应该是等价的，我没有做过系统的对比实验。

注释里那句"外层残差连接不再额外Dropout"是我后来补上去的，算是给自己留个提醒，免得下次又在外面多加一层。

关于这段代码还有一个设计决策是我有意为之的：激活函数做成了可配置的参数，而不是硬编码。这听起来像是工程上的过度设计，但我当时有实际需求——同一套代码要在两个项目里复用，一个用GELU，另一个有个老前辈坚持用ReLU（他的理由是"模型小，ReLU已经够了，GELU算那个近似有开销"，不是没有道理）。与其维护两个版本，不如给一个`activation`参数。

我现在回头看，用字符串来指定激活函数其实不是最优雅的做法。更Python的方式应该是接受一个callable，或者直接传一个`nn.Module`实例进来，让调用方自己决定。字符串的问题是扩展性差，每加一个新的激活函数都要改这里面的if-elif，违反了开闭原则。不过当时两个项目同时在推进，顾不上，先把功能做出来再说，这是工程里的常态。

最后想聊一个更底层的问题。FFN在Transformer里承担的角色到底是什么，到现在理论界也没有完全搞清楚。有一种说法是FFN在做"记忆存储"——Attention负责把相关的token关联起来，FFN负责把知识存进去。这个解释来自Mor Geva等人2021年发表在EMNLP的论文《Transformer Feed-Forward Layers Are Key-Value Memories》，他们发现FFN里的权重矩阵可以被解读为键值对记忆：$W_1$的每一行像一个"键"，$W_2$对应的列像一个"值"，中间的激活值决定哪些记忆被检索出来。这个视角相当有启发性，能解释为什么LLM的FFN层参数量那么大、为什么MoE把专家放在FFN的位置而不是Attention上。

但这个解释能成立的前提是激活函数要有一定的稀疏性——ReLU在这里反而比GELU更符合"记忆检索"的直觉，因为它真的会把大量神经元置零，只激活少数几个。GELU的激活更稠密，记忆的边界更模糊。这也是为什么LLaMA系列最后用了SwiGLU，它有一个门控机制，能在平滑激活的同时保持一定的稀疏性——某种程度上算是在GELU和ReLU之间找了个折中。

我自己还没有实现过SwiGLU版本的FFN，这是接下来想做的事。理论上SwiGLU需要三个线性层（gate、up、down），结构和上面这段代码会有比较大的差异，参数量也会有变化。等实现完了看看在同等参数量下跟GELU版本的精度差，应该是个挺有意思的实验。