---
title: 大模型分词器完全指南：从原理到实践
slug: tokenization-guide
date: 2025-04-24
readTime: 18 分钟
category: 基础原理
tags: Tokenization, BPE, WordPiece, SentencePiece
cover: ./content/assets/posts/covers/tokenization.png
excerpt: 系统讲清词级、字符级与子词级分词，以及 BPE、WordPiece、SentencePiece 的差异与实践要点。
---

# 大模型分词器完全指南：从原理到实践

> 本文面向有一定技术基础的读者，系统讲解大语言模型（LLM）中分词（Tokenization）的核心概念、主流算法、多语言策略差异，以及对模型效率和语义理解的影响。

---

## 一、为什么需要分词？

大模型无法直接处理原始文本字符串，它需要先把文本转换成一串整数 ID，再喂给神经网络。这个转换过程就是**分词（Tokenization）**，其中每个最小处理单元叫做**词元（Token）**。

分词器的设计直接决定了：
- 模型"看到"的信息粒度
- 词汇表（Vocabulary）的大小
- 输入序列的长度
- 训练和推理的计算成本

---

## 二、三种基础分词粒度

在介绍具体算法之前，先理解三种最基本的切分粒度，它们构成了所有分词策略的出发点。

### 2.1 词级（Word-level）

最直觉的方式：**每个完整的词 = 一个 token**。

```
"I love cats"       →  ["I", "love", "cats"]          （3个token）
"我爱自然语言处理"   →  ["我", "爱", "自然语言", "处理"]  （4个token）
```

**致命缺陷：**
- 词表爆炸性膨胀（英语几十万词，加上变形更多）
- `run` / `runs` / `ran` / `running` 各占一个词表位置，浪费严重
- 遇到未登录词（OOV）只能用 `[UNK]` 替代，信息直接丢失

### 2.2 字符级（Character-level）

另一个极端：**每个字母 / 每个汉字 = 一个 token**，不做任何合并。

```
"hello"             →  ["h", "e", "l", "l", "o"]               （5个token）
"我爱自然语言处理"   →  ["我", "爱", "自", "然", "语", "言", "处", "理"]  （8个token）
```

BERT 中文版就采用了这种策略。

**优点：** 词表极小（几百个），绝对不会遇到未登录词。  
**缺点：** 序列极长，Transformer 的计算量与序列长度的平方成正比，训练和推理成本剧增。

### 2.3 子词级（Subword-level）——现代 LLM 的选择

介于词级和字符级之间：**把词拆成常见的子片段**，高频词保留完整，低频词拆开处理。

```
"newer"        →  ["new", "er"]
"unknowingly"  →  ["un", "know", "ing", "ly"]
```

三种粒度的对比：

| 方式 | Token 数量 | 词表大小 | 未登录词问题 |
|---|---|---|---|
| 词级 | 最少 | 极大（几十万+） | 有，且严重 |
| **子词级** | **适中** | **适中（3～25万）** | **基本没有** |
| 字符级 | 最多 | 极小（几百） | 完全没有 |

现代大模型几乎全部选择子词级分词，原因正是它在三者之间取得了最佳平衡。

---

## 三、三种主流子词分词算法

### 3.1 BPE（Byte-Pair Encoding，字节对编码）

**代表模型：** GPT-2/3/4、LLaMA、Mistral

**核心思想：** 自底向上合并，从字符集出发，反复统计当前语料中所有相邻符号对的全局频次，选出频次最高的那一对合并成新符号，直到达到预设词表大小。

> **准确说明：** BPE 统计的是符号对的**全局频次**，计算公式为：
> $$\text{freq}(a, b) = \sum_{w} \text{词频}(w) \times \text{符号对}(a,b)\text{在词}w\text{中的出现次数}$$
> 例如 `e+s` 出现 9 次，是因为 `newest`（词频 6）× 1处 + `widest`（词频 3）× 1处 = 9。统计的不是"包含该符号对的不同词的种类数"，而是加权后的全局出现总量。

**训练过程示例：**

假设语料包含：`low`（5次）、`lower`（2次）、`newest`（6次）、`widest`（3次）

> **说明：** 为了区分词内部的子串和词的结尾，BPE 通常在每个词末尾添加特殊词尾符号 `</w>`（End of Word）。例如 `low` 初始化后表示为 `l o w </w>`，`newest` 表示为 `n e w e s t </w>`。这样，合并规则就能区分"词中间的 `st`"和"词尾的 `st</w>`"，不会产生跨词边界的错误合并。

```
初始：所有单字符（含词尾符）  →  {l, o, w, e, r, n, s, t, i, d, </w>}

第1次合并：'e'+'s' 全局频次 = 6×1 + 3×1 = 9（newest贡献6，widest贡献3），频次最高
           词表新增 'es'
第2次合并：'es'+'t' 全局频次 = 9，词表新增 'est'
第3次合并：'est'+'</w>' 全局频次 = 9，词表新增 'est</w>'
第4次合并：'l'+'o' 全局频次 = 5×1 + 2×1 = 7（low贡献5，lower贡献2），词表新增 'lo'
第5次合并：'lo'+'w' 全局频次 = 7，词表新增 'low'
...
```

最终词表包含：基础字符 + 若干高频子词（如 `est`、`low`、`er` 等）。

**字节级 BPE（Byte-level BPE）：**

BPE 的重要变体，起点不是字符，而是 256 个原始字节。任意 Unicode 文本（包括中文、emoji）都先转成 UTF-8 字节序列再做合并，从根本上消灭了未登录词问题。

这里有一个巧妙的工程设计：原始字节序列是人类不可读的乱码（如空格对应 `\x20`），GPT-2 把 256 个字节**一一映射到可打印的 Unicode 字符**，例如空格字节 `\x20` 映射成 `Ġ`。这就是为什么我们用 GPT 系列分词器时，token 长这样：

```
"I love NLP"  →  ["I", "Ġlove", "ĠN", "LP"]
```

开头的 `Ġ` 并不是乱码，而是"这个 token 前面有一个空格"的视觉编码。这个设计让字节级 token 保持可读性，同时不损失任何信息。

GPT-2 采用此方案，词表共 50,257 个条目，由三部分构成：
- **256 个基础字节 token**（对应所有可能的单字节取值）
- **50,000 个合并 token**（通过 50,000 次 BPE 合并习得）
- **1 个特殊 token** `<|endoftext|>`（用于标记文档边界，性质与基础字节不同，是人为添加的特殊符号）

**优点：**
- 可逆无损，能准确还原原始文本
- 通用性强，任何新文本都能处理（最坏退回到字节级）
- 平均每个 token 约对应 4 个字节，压缩效率高

### 3.2 WordPiece

**代表模型：** BERT、DistilBERT

结构上与 BPE 相似，但**训练准则和推理策略是两个独立的步骤**，不能混淆：

- **训练时：** 不选频次最高的符号对，而选"合并后能最大化语料整体对数似然"的对（直觉上倾向于合并那些"只有在一起时才高频、单独出现频率低"的符号对，与点间互信息 PMI 的直觉相近，但两者在数学上并不严格等价）。
- **推理时：** 对新文本使用**从左到右的贪心最长前缀匹配**策略，在词表中找到能匹配当前位置最长子词，依次向右推进。

WordPiece 用 `##` 前缀标记"非词首"子词：

```
"playing"    →  ["play", "##ing"]
"jetsetter"  →  ["jet", "##set", "##ter"]
```

### 3.3 SentencePiece + Unigram

**代表模型：** T5、XLNet、ALBERT、Gemma

**SentencePiece** 是 Google 开源的分词工具框架，核心创新是：**把空格视为普通字符**（用 `▁` 表示），直接对原始文本流操作，不依赖任何语言特定的预切分步骤，天然适用于中日韩等无空格语言。

```
"I love NLP."  →  ["I", "▁love", "▁N", "LP", "."]
```

**Unigram 算法**（Kudo 2018）是 SentencePiece 支持的算法之一，方向与 BPE 相反——**自顶向下剪枝**：

```
从大词典出发 → 反复删除"对语言模型损失贡献最小"的子词 → 直到词表缩减到目标大小
```

每个子词带有概率值。**推理时**使用 **Viterbi 算法**在所有可能切分中找到概率最高的路径，结果是**确定性的**，并非随机。

> **重要区分：** Unigram 支持一种叫做"子词正则化（Subword Regularization）"的技术，即在训练阶段按概率随机采样不同切分方式，作为数据增强手段提升模型鲁棒性。这与推理时的确定性 Viterbi 解码是两回事，不能混淆。

---

## 四、三种算法横向对比

| 维度 | BPE | WordPiece | SentencePiece/Unigram |
|---|---|---|---|
| 方向 | 自底向上（合并） | 自底向上（合并） | 自顶向下（剪枝） |
| 训练准则 | 全局频次最高的符号对（词频加权） | 最大化语料对数似然 | 删除贡献最小的子词 |
| 推理策略 | 按合并规则顺序执行 | 从左到右贪心最长前缀匹配 | Viterbi 最高概率路径 |
| 推理结果 | 唯一确定 | 唯一确定 | 唯一确定（随机采样仅用于训练增强） |
| 空格处理 | 依赖空格预切分 | 依赖空格预切分 | 空格作为普通符号，语言无关 |
| 代表模型 | GPT 系列、LLaMA | BERT | T5、XLNet、Gemma |

---

## 五、英文 vs. 中文：为什么分词策略不同？

### 5.1 英文：词边界天然存在

英文用**空格**把词隔开，词的边界是客观存在、无歧义的：

```
"I love natural language processing"
 ↓ 按空格切开（免费获得词界）
["I", "love", "natural", "language", "processing"]
 ↓ 再对每个词内部做子词切分
完成
```

分词器只需要在词的内部工作，不需要猜测词边界。

### 5.2 中文：词边界需要推断

中文是连续字符流，天然没有分隔符：

```
"他在学科技公司工作"
```

这句话至少有两种合理切分：
- 他 / 在 / **学科技**（公司名）/ 公司 / 工作
- 他 / 在 / 学 / **科技公司** / 工作

词边界**不是客观存在的**，取决于上下文语义。如果直接把 BPE 套上去，算法只能看到"哪些字符经常相邻"，而"频繁相邻"不等于"属于同一个词"，极易合并出跨词边界的无意义组合。

### 5.3 各方案的应对策略

| 策略 | 代表模型 | 做法 | 代价 |
|---|---|---|---|
| 退化为字符级 | BERT 中文版 | 每个汉字一个 token | 序列变长，丢失词级语义 |
| 先切词再做 BPE | 部分中文 LLM | jieba/thulac 先切词 | 依赖外部工具，引入额外错误 |
| SentencePiece | T5、Gemma 等 | 数据驱动合并，语言无关 | 可能合并出跨词边界片段 |
| 字节级 BPE | LLaMA、GPT 系列 | 一切退到字节级 | 中文 token 效率低，同等内容需更多 token |

### 5.4 工业界的"终极妥协"：扩充中文词表

早期以英文为主的模型（如 LLaMA-1）在处理中文时效率极低——一个汉字往往需要 2～3 个 token 表示。这并非因为它"不能处理"中文（字节级 BPE 本身具备处理任意 Unicode 的能力），而是因为训练分词器时中文语料占比极少，导致中文相关的合并规则几乎没有被习得，大量汉字只能退回到字节级表示。

目前国内主流中文大模型（如 Qwen、GLM、Baichuan 等）采用的工业解法是：**在字节级 BPE 的基础上，用海量中文语料训练分词器，强制让 BPE 把高频汉字和常见词组直接合并进词表**。具体表现为词表大幅扩充：

```
LLaMA-1 词表：32,000（中文相关合并规则极少，汉字退回字节级表示，效率低）
Qwen 词表：约 150,000（数万位置专门留给中文 token）
```

这个方案的精妙之处在于：
- **保留字节级的通用性**：任何字符都能处理，不存在 OOV
- **解决中文效率问题**：常见汉字和词组作为独立 token，不再需要多个字节拼凑
- **代价**：词表变大，embedding 层参数增加，但这个代价在工程上完全可控

这也直接回答了前面的两个思考题——中文最优分词策略，以及词表大小与模型能力的关系：词表扩充不是为了提升"理解深度"，而是提升**token 效率**（同样内容消耗更少 token），从而降低推理成本、提升上下文利用率。

---

## 六、分词对效率和语义的影响

### 6.1 效率：词表大小 vs. 序列长度的核心矛盾

这是分词设计最本质的权衡：

- **词表过大** → embedding 层参数爆炸。词表 10 万、embedding 维度 768，仅 embedding 层就有 7,680 万参数。
- **词表过小** → 序列过长。Transformer 的 attention 计算量与序列长度的**平方**成正比，序列翻倍，计算量翻四倍。

类比理解：词表越厚（词表大），查单词越快（序列短），但字典本身很重（参数多）；字典越薄，书写越啰嗦（序列长），计算越慢。

### 6.2 语义：切分方式直接影响模型理解

**合理切分的好处：**

子词切分能帮助模型学到词根词缀的语义：
```
"unlock"  →  ["un", "lock"]
```
模型在大量单词中反复看到 `un-`，自然学会它"否定"的含义。

**错误切分的危害：**

如果把"不相"合并成一个 token，或把"的事"错当一个词元，模型接收到的就是语义上混乱的输入，直接误导下游理解。

---

## 七、值得深入思考的问题

1. **中文分词的效率与语义如何进一步权衡？** 扩充词表解决了 token 效率问题，但数据驱动合并出的中文子词（如"科技的"跨词边界组合）是否仍会引入语义噪声？有没有办法在 BPE 合并阶段引入语言学约束？

2. **词表扩充对模型能力有哪些间接影响？** 5.4 节说明词表扩充主要提升 token 效率而非理解深度，但更短的序列意味着模型在同等上下文窗口内能看到更多内容——这种"间接效应"在长文本任务中究竟有多大？

3. **分词器的训练集偏差会如何传播？** 某类文本在分词器训练集中占比过多，对应子词序列更短（更"紧凑"），模型是否会因此对这类内容产生"捷径"？偏差会叠加放大吗？

4. **分词策略与模型架构之间有没有最优匹配？** BERT 双向 attention 对分词粒度是否有特别要求？自回归模型（GPT）和编码器模型（BERT）对分词的需求是否根本不同？

5. **分词本身会成为推理瓶颈吗？** SentencePiece 的 Viterbi 解码在高吞吐场景下是否会成为独立的延迟瓶颈？是否有在 GPU 上并行化分词的可行方案？

---

## 八、总结

| 核心结论 | 说明 |
|---|---|
| 子词分词是现代 LLM 的标配 | 平衡词表规模、序列长度与未登录词处理 |
| BPE 频次统计是词频加权的全局总量 | 不是"包含该对的词的种类数"，是 ∑(词频 × 对在该词中出现次数) |
| 字节级 BPE 的 Ġ 不是乱码 | 是空格字节的可打印映射，设计上保留了空格信息又维持可读性 |
| 特殊 token 与基础词汇性质不同 | `<\|endoftext\|>` 等是人为添加的符号，不属于 BPE 合并产物 |
| WordPiece 训练与推理策略需区分 | 训练用最大似然，推理用从左到右贪心最长前缀匹配，两者独立 |
| WordPiece 与 PMI 直觉相近但不严格等价 | 最大化语料似然 ≠ 最大化点间互信息，不可画等号 |
| Unigram 推理是确定性的 | Viterbi 解码结果唯一；随机采样仅用于训练阶段数据增强 |
| 中文大模型的工业解法是扩充词表 | 海量中文语料训练 BPE，词表从 32k 扩至 150k，解决 token 效率问题 |
| 词表扩充提升效率而非理解深度 | 更短序列让模型在同等窗口内看到更多内容，是间接收益 |
| 分词质量直接影响模型上限 | 错误切分会从输入层就引入语义噪声 |

> 分词不是模型训练的"前处理杂活"，而是直接决定模型能看到什么、理解什么的关键设计决策。理解分词，是真正理解大模型的第一步。

## 简易bpe实现，学习用
<pre> 
简化版 BPE（Byte Pair Encoding）分词器
=====================================
from collections import defaultdict, Counter

class SimpleBPE:
    def __init__(self, corpus: str, vocab_size: int = 50):
        """
        初始化 BPE 模型。

        Args:
            corpus:     训练语料（字符串，词之间用空格分隔）
            vocab_size: 目标词汇表大小
        """
        # 训练并保存词汇表列表
        self.vocab = self._train_bpe(corpus, vocab_size)
        # 构建 token → id 映射
        self.token_to_id: dict[str, int] = {
            token: idx for idx, token in enumerate(self.vocab)
        }
        # 构建 id → token 映射（用于解码）
        self.id_to_token: dict[int, str] = {
            idx: token for idx, token in enumerate(self.vocab)
        }

    # ------------------------------------------------------------------ #
    #  训练                                                              #
    # ------------------------------------------------------------------ #

    def _train_bpe(self, corpus: str, target_size: int) -> list[str]:
        """
        训练 BPE，反复合并最高频字符对，直到词汇表达到目标大小。

        Args:
            corpus:      训练语料
            target_size: 目标词汇表大小

        Returns:
            词汇表列表（按 id 顺序排列）
        """
        # 统计每个词出现的次数，避免同一词存多份列表
        word_freq: Counter = Counter(corpus.split())

        # 初始化词汇表：特殊标记 + 词边界标记 + 所有基础字符（排序保证确定性）
        base_chars = sorted(set("".join(word_freq.keys())))
        vocab = ["<pad>", "<unk>", "_"] + base_chars

        # 初始化每个词的字符分割（词边界用 '_' 表示）
        # 格式：{ ('_', 'h', 'e', 'l', 'l', 'o'): 3, ... }
        words_splits: dict[tuple, int] = {
            tuple(["_"] + list(word)): freq
            for word, freq in word_freq.items()
        }

        # 迭代合并，直到词汇表达到目标大小
        while len(vocab) < target_size:
            # 统计所有相邻字符对的加权频率（乘以词频）
            pairs: dict[tuple, int] = defaultdict(int)
            for token_seq, freq in words_splits.items():
                for i in range(len(token_seq) - 1):
                    pairs[(token_seq[i], token_seq[i + 1])] += freq

            if not pairs:
                # 语料中已无可合并的字符对，提前结束
                break

            # 找到频率最高的字符对（同频时按字典序保证确定性）
            best_pair = max(pairs, key=lambda p: (pairs[p], p))
            merged_token = "".join(best_pair)

            # 将新合并的 token 加入词汇表
            vocab.append(merged_token)

            # 在所有词的分割序列中替换该高频对
            words_splits = {
                self._merge_pair(token_seq, best_pair): freq
                for token_seq, freq in words_splits.items()
            }

        return vocab

    @staticmethod
    def _merge_pair(tokens: tuple, pair: tuple) -> tuple:
        """
        将 tokens 序列中所有出现的 pair 合并为单个 token。

        Args:
            tokens: 当前词的 token 序列（tuple 形式）
            pair:   需要合并的字符对 (a, b)

        Returns:
            合并后的新序列（tuple 形式）
        """
        merged = []
        i = 0
        while i < len(tokens):
            # 如果当前位置与下一位置匹配目标对，则合并
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append("".join(pair))
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return tuple(merged)

    # ------------------------------------------------------------------ #
    #  编码                                                              #
    # ------------------------------------------------------------------ #

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 序列。
        Args:
            text: 待编码文本

        Returns:
            token ID 列表
        """
        if not text.strip():
            return []

        all_ids: list[int] = []
        for word in text.split():
            all_ids.extend(self._encode_word(word))
        return all_ids

    def _encode_word(self, word: str) -> list[int]:
        """
        对单个词进行贪心最长匹配编码。

        流程：
            1. 将词拆分为字符列表，并在开头添加词边界标记 '_'
            2. 从左到右扫描，每次尽可能匹配词汇表中最长的 token
            3. 匹配不到时输出 <unk>

        Args:
            word: 单个词（不含空格）

        Returns:
            该词对应的 token ID 列表
        """
        # 初始化为字符列表，'_' 表示词边界
        tokens: list[str] = ["_"] + list(word)

        result: list[str] = []
        i = 0
        # 贪心最长匹配：单次线性扫描即可，无需外层循环
        while i < len(tokens):
            matched = False
            # 从最长可能的子串开始尝试（最长不超过 10，防止极端情况）
            max_len = min(len(tokens) - i, 10)
            for length in range(max_len, 0, -1):
                candidate = "".join(tokens[i : i + length])
                if candidate in self.token_to_id:
                    result.append(candidate)
                    i += length
                    matched = True
                    break
            if not matched:
                # 词汇表中找不到该字符，标记为 <unk>
                result.append("<unk>")
                i += 1

        # 将 token 转换为对应 ID
        return [self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in result]

    # ------------------------------------------------------------------ #
    #  解码（新增）                                                        #
    # ------------------------------------------------------------------ #

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 序列解码回原始文本（尽力还原，供调试使用）。

        Args:
            ids: token ID 列表

        Returns:
            解码后的文本字符串
        """
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        # 将词边界标记 '_' 还原为空格，去掉首个多余空格
        text = "".join(tokens).replace("_", " ").strip()
        return text


# ------------------------------------------------------------------ #
#  使用示例                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # 定义训练语料
    corpus = "hello world hello there world of code code hello"

    # 创建 BPE 模型，词汇表大小设为 30
    bpe = SimpleBPE(corpus, vocab_size=30)

    print("=" * 50)
    print(f"词汇表大小: {len(bpe.vocab)}")
    print(f"词汇表内容: {bpe.vocab}")
    print("=" * 50)

    # 测试单词编码
    test_cases = ["hello", "hello world", "code hello world", "unknown"]
    for text in test_cases:
        encoded = bpe.encode(text)
        tokens  = [bpe.id_to_token[i] for i in encoded]
        decoded = bpe.decode(encoded)
        print(f"\n原文   : {text!r}")
        print(f"Token  : {tokens}")
        print(f"ID     : {encoded}")
        print(f"解码   : {decoded!r}")
</pre>

## 更多参考 
* https://github.com/tangentllm
