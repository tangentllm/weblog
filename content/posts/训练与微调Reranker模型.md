# 从召回到精排：用 Sentence Transformers 训练你自己的 Reranker 模型

> 作者：编译整理自 Tom Aarsen（Hugging Face）的技术分享
> 原文标题：*Training and Finetuning Reranker Models with Sentence Transformers*

---

## 引言：为什么召回之后还需要"精排"

想象一个场景：用户在你的搜索系统里输入了"如何训练 Reranker 模型"，Embedding 模型（Bi-encoder）从百万级文档库中，通过向量相似度快速召回了 Top-30 篇候选文档。这一步很快，但也很"粗糙"——因为 Embedding 模型在编码 query 和 document 时是**各自独立**进行的，二者的语义交互只能在最后的向量点积/余弦相似度这一步完成，信息损失不可避免。

于是问题来了：这 30 篇文档里，哪几篇才是真正最相关的？这正是 **Reranker（Cross-Encoder）** 登场的时刻——它把 query 和每一篇候选 document **拼接在一起**送入同一个模型，让两者在每一层 Transformer 中充分"attend"彼此，从而给出远比 Bi-encoder 精细的相关性打分。这就是检索系统中经典的 **Retrieve-then-Rerank（先召回、后精排）** 两段式架构。

本文将系统讲解如何使用 [Sentence Transformers](https://www.sbert.net/) 库，从零训练或微调一个 Reranker 模型，并通过一个真实实验证明：**一个 150M 参数的小模型，经过针对性微调后，完全可以超越参数量高出 10 倍的通用 Reranker。**

---

## 目录

- [一、Reranker 与 Embedding 模型：架构本质的差异](#一reranker-与-embedding-模型架构本质的差异)
- [二、为什么要微调 Reranker](#二为什么要微调-reranker)
- [三、训练四大组件详解](#三训练四大组件详解)
  - [3.1 数据集准备与 Hard Negatives Mining](#31-数据集准备与-hard-negatives-mining)
  - [3.2 损失函数（Loss）](#32-损失函数loss)
  - [3.3 评估器（Evaluator）](#33-评估器evaluator)
  - [3.4 训练器（CrossEncoderTrainer）](#34-训练器crossencodertrainer)
- [四、完整实战：微调 ModernBERT-base 在 GooAQ 上](#四完整实战微调-modernbert-base-在-gooaq-上)
- [五、实验结果与洞察](#五实验结果与洞察)
- [六、训练技巧与最佳实践](#六训练技巧与最佳实践)
- [七、延伸资源](#七延伸资源)
- [总结：微调的 ROI](#总结微调的-roi)

---

## 一、Reranker 与 Embedding 模型：架构本质的差异

在深入训练细节之前，有必要先厘清两类模型的本质区别。

| 维度 | Embedding 模型（Bi-encoder） | Reranker（Cross-Encoder） |
|---|---|---|
| 输入方式 | query 和 document **分别独立编码** | query 和 document **拼接后一起编码** |
| 输出 | 各自的向量表示，再计算相似度 | 直接输出一个相关性分数（score） |
| 交互程度 | 弱交互，仅在向量空间的最后一步比较 | 深度交互，Transformer 每一层都能相互 attend |
| 计算成本 | 低，文档向量可预先离线计算并索引 | 高，每次都要对 query-document 对做一次完整前向计算 |
| 适用场景 | 大规模初筛（百万级候选） | 小规模精排（几十到几百候选） |

**Cross-Encoder 的核心优势**在于：由于 query 和 document 的 token 可以在自注意力机制中互相"看到"彼此，模型能够捕捉到更细粒度的语义匹配关系（比如否定词、指代消解、细微的语义差别),因此打分精度通常显著优于 Bi-encoder。

**代价也同样明显**：Cross-Encoder 无法像 Bi-encoder 那样预先计算并缓存文档向量，每次查询都必须对每个候选 pair 做一次完整的模型推理，计算开销随候选数量线性增长。这意味着它**不可能**直接应用于百万级文档库的全量检索。

正因如此，业界形成了经典的 **两段式架构**：

1. **Retrieve（召回）**：用 Embedding 模型从海量文档中快速召回 Top-K（如 30~100）候选；
2. **Rerank（精排）**：用 Cross-Encoder 对这 K 篇候选逐一打分，重新排序，取 Top-N 展示给用户。

这样既保证了效率（Bi-encoder 负责规模），又保证了精度（Cross-Encoder 负责质量）。

---

## 二、为什么要微调 Reranker

开源社区已经有不少现成的通用 Reranker（如 BGE-reranker、Jina-reranker、mxbai-rerank 等），那为什么还要自己训练？

答案是：**通用模型在垂直领域往往表现有限**。通用 Reranker 是在大规模、跨领域的通用语料上训练的，面对法律、医疗、电商、代码搜索等特定领域的查询-文档匹配模式时，往往"隔靴搔痒"——它可能不理解领域术语的细微差异，也无法学到你的业务数据中特有的相关性判断标准。

更重要的是一个反直觉但极具价值的结论：**针对特定领域微调后的小模型，完全可以超越参数量大得多的通用模型**。原文作者做了一个很有说服力的实验：

> 将一个仅 150M 参数的 **ModernBERT-base** 模型，在 GooAQ 问答数据集上做领域微调后，其 **NDCG@10 达到 77.14**，超过了包括一个 **1.54B 参数**的大模型在内的 **13 个常用开源 Reranker**。

这说明什么？在实际业务场景中，"更懂你的数据"往往比"参数量更大"更重要。而且微调的成本极低——下文会看到，在一张消费级显卡（RTX 3090）上，整个训练过程只需要约 30 分钟到 1 小时。

---

## 三、训练四大组件详解

使用 Sentence Transformers 训练 Reranker，核心围绕四个组件展开：**数据集、损失函数、评估器、训练器**。下面逐一拆解。

### 3.1 数据集准备与 Hard Negatives Mining

Sentence Transformers 支持直接从 Hugging Face Hub 加载数据集，也支持加载本地的 CSV、JSON、Parquet 等格式文件，非常灵活：

```python
from datasets import load_dataset

# 方式一：直接从 Hugging Face Hub 加载
dataset = load_dataset("sentence-transformers/gooaq", split="train")

# 方式二：加载本地数据文件（以 CSV 为例）
# dataset = load_dataset("csv", data_files="my_qa_data.csv", split="train")
```

大多数真实场景下，我们手头的数据往往只有"正样本对"（比如一个 query 对应一个正确答案 passage），却没有负样本。而 Cross-Encoder 的训练**恰恰需要大量高质量的负样本**才能学会真正的判别能力——如果负样本太"简单"（比如随机采样的无关文档），模型很容易就能分辨，学不到有价值的判别边界。

这就是 **Hard Negatives Mining（难负样本挖掘）** 的用武之地。其核心思路是：用一个 Embedding 模型对 query 做召回，把那些"看起来相关但实际不是标注正例"的文档挖出来，作为难负样本，逼迫 Cross-Encoder 学习更精细的判别能力。

```python
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer

# 使用一个轻量级 Embedding 模型来挖掘难负样本
embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")

hard_train_dataset = mine_hard_negatives(
    dataset=dataset,
    model=embedding_model,
    num_negatives=5,          # 每个正样本挖掘 5 个难负样本
    margin=0,                 # 相似度 margin，用于过滤过于接近正例的"假负样本"
    range_min=0,               # 从相似度排序的第几名开始采样负样本
    range_max=100,              # 采样范围上限,过大会引入过难/噪声负样本
    sampling_strategy="top",    # 采样策略：top 表示优先选相似度最高的负样本
    batch_size=4096,
    use_faiss=True,            # 使用 FAISS 加速最近邻检索，降低内存占用
)
```

挖掘出来的结果会被组织成 **labeled-pair** 格式：`(query, passage, label)`，其中正例 label 为 1，负例 label 为 0。这种格式正好适配下面要讲的 `BinaryCrossEntropyLoss`。

### 3.2 损失函数（Loss）

Sentence Transformers 为 Cross-Encoder 训练提供了多种损失函数，选择哪一种取决于你的数据组织形式：

- **`BinaryCrossEntropyLoss`**：适用于已经标注好的正负样本对 `(query, passage, label)`，这是最常见、最直观的场景，也是本文实战案例采用的方案。

```python
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

# pos_weight 通常设置为负样本数量与正样本数量的比例，
# 用于缓解正负样本不均衡问题（负样本往往远多于正样本）
loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(5))
```

- **`CachedMultipleNegativesRankingLoss`**：适用于三元组格式的数据 `(anchor, positive, negative)`，通过对比学习的方式训练，且带有 in-batch negatives 的缓存机制，可以在有限显存下使用更大的有效 batch size。

一般来说：如果你的数据已经是"打好标签的正负样本对"，优先用 `BinaryCrossEntropyLoss`；如果数据是三元组或者只有正样本对（需要依靠 in-batch negatives），则更适合用 `CachedMultipleNegativesRankingLoss`。

### 3.3 评估器（Evaluator）

训练过程中，我们需要实时监控模型的排序质量，Sentence Transformers 提供了多个针对 Cross-Encoder 设计的评估器：

- **`CrossEncoderRerankingEvaluator`**：模拟真实的"召回 + 精排"流程，给定一批 query 及其候选文档列表（含正例和负例），评估模型重排序后的 NDCG、MRR 等指标。

```python
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=eval_samples,   # 每个 sample 包含 query、正例文档、候选负例文档列表
    at_k=10,
    name="gooaq-dev",
)
```

- **`CrossEncoderNanoBEIREvaluator`**：基于 NanoBEIR（BEIR 基准的轻量化版本）的评估器，可以快速在多个通用英文检索任务上评估模型的泛化表现，无需下载庞大的原始 BEIR 数据集。

```python
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator

nano_beir_evaluator = CrossEncoderNanoBEIREvaluator()
```

- **`CrossEncoderCorrelationEvaluator`**：用于语义相似度类任务（如 STSb 数据集），评估模型打分与人工标注相似度之间的相关系数（如 Spearman/Pearson）。

- **`SequentialEvaluator`**：可以把上面多个评估器组合起来，在训练过程中依次执行，得到综合的评估报告。

```python
from sentence_transformers.evaluation import SequentialEvaluator

evaluator = SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])
```

### 3.4 训练器（CrossEncoderTrainer）

万事俱备后，`CrossEncoderTrainer` 负责把模型、训练参数、数据集、损失函数、评估器整合到一起，驱动整个训练流程，其 API 风格与 Hugging Face `Trainer` 高度一致，非常容易上手：

```python
from sentence_transformers.cross_encoder import CrossEncoderTrainer

trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=hard_train_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()
```

值得一提的是，`CrossEncoderTrainer` 还支持 **多数据集训练（Multi-Dataset Training）**：你可以传入一个数据集字典，甚至针对不同数据集使用不同的 loss 函数，训练器会自动处理混合训练的调度。数据采样策略上提供两种选择：

- **`ROUND_ROBIN`**：轮询式采样，各数据集依次轮流取一个 batch，保证每个数据集被均匀访问；
- **`PROPORTIONAL`**：按数据集大小比例采样，数据量大的数据集会被更频繁地采样。

```python
from sentence_transformers.training_args import BatchSamplers

training_args = CrossEncoderTrainingArguments(
    # ... 其他参数
    multi_dataset_batch_sampler="proportional",  # 或 "round_robin"
)
```

---

## 四、完整实战：微调 ModernBERT-base 在 GooAQ 上

理论讲完，我们来走一遍完整的端到端训练流程。本案例的目标是：在 **GooAQ**（一个大规模英文问答数据集）上微调 **ModernBERT-base**，训练出一个专精于问答场景排序的 Reranker。

**第一步：加载基础模型**

```python
from sentence_transformers.cross_encoder import CrossEncoder

model = CrossEncoder("answerdotai/ModernBERT-base", num_labels=1)
# num_labels=1 表示模型输出一个连续的相关性分数（而非分类 logits）
```

**第二步：加载数据集**

```python
from datasets import load_dataset

full_dataset = load_dataset("sentence-transformers/gooaq", split="train")
# 原始数据规模：约 99k 条 query-answer 正样本对
dataset_dict = full_dataset.train_test_split(test_size=1000, seed=12)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]
```

**第三步：Hard Negative Mining**

```python
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")

hard_train_dataset = mine_hard_negatives(
    dataset=train_dataset,
    model=embedding_model,
    num_negatives=5,       # 每条正样本挖掘 5 条难负样本
    margin=0,
    range_min=0,
    range_max=100,
    sampling_strategy="top",
    batch_size=4096,
    use_faiss=True,
    output_format="labeled-pair",  # 输出为 (query, passage, label) 格式
)
# 挖掘后：99k 条正例 + 约 479k 条负例，共约 578k 条 labeled pairs
```

**第四步：定义损失函数**

```python
import torch
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

# pos_weight 设置为负样本数量 / 正样本数量的比例（此处约为 5，因为挖掘了 5 个负样本）
loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(5))
```

**第五步：配置评估器**

```python
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderNanoBEIREvaluator,
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.evaluation import SequentialEvaluator

# 领域内评估：在 GooAQ dev 集上评估重排序效果
reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=eval_dataset,  # 需组织成含正例与候选负例的样本格式
    at_k=10,
    name="gooaq-dev",
)

# 领域外泛化评估：NanoBEIR 基准
nano_beir_evaluator = CrossEncoderNanoBEIREvaluator()

evaluator = SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])
```

**第六步：配置训练参数**

```python
from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments

training_args = CrossEncoderTrainingArguments(
    output_dir="models/modernbert-base-gooaq-reranker",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    bf16=True,                          # 若 GPU 支持 bf16，可显著加速训练
    load_best_model_at_end=True,
    metric_for_best_model="eval_gooaq-dev_ndcg@10",  # 以验证集 NDCG@10 作为最优模型判断依据
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=200,
)
```

**第七步：训练、评估、保存**

```python
from sentence_transformers.cross_encoder import CrossEncoderTrainer

trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=hard_train_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()

# 训练结束后，在完整评估集上做最终评估
evaluator(model)

# 保存模型到本地
model.save_pretrained("models/modernbert-base-gooaq-reranker/final")

# 也可以直接推送到 Hugging Face Hub 与社区共享
model.push_to_hub("your-username/modernbert-base-gooaq-reranker")
```

整个流程走下来会发现：**代码量并不大**，绝大部分复杂度都被 Sentence Transformers 库封装掉了，开发者只需要专注于数据质量和超参数选择。

---

## 五、实验结果与洞察

原文作者在 GooAQ dev 集上，对比了"无重排序基线"、13 个主流开源 Reranker，以及自己微调的两个模型（base 和 large 版本），得到如下 NDCG@10 结果（数值越高越好）：

| 模型 | 参数量 | NDCG@10 |
|---|---|---|
| 无重排序（仅 Embedding 召回后的原始顺序） | — | 明显低于所有 Reranker |
| 多个通用开源 Reranker（含 1.54B 大模型等 13 个） | 参数范围广泛，最大达 1.54B | 均低于微调后的 base 模型 |
| **微调后 ModernBERT-base（本文实验）** | **150M** | **77.14** |
| **微调后 ModernBERT-large（本文实验）** | **396M** | **79.42** |

> 说明：具体的 13 个基线模型逐一分数因篇幅原因未在此逐条列出，读者可参阅原文获取完整对比表；核心结论是：**微调后的 150M 模型全面超越了包括 1.54B 参数模型在内的所有通用 Reranker**。

几个关键洞察值得特别强调：

1. **小模型微调后可以"以小博大"**。150M 参数的 ModernBERT-base 微调后的表现，超过了参数量是它 10 倍的通用大模型。这说明"领域适配"带来的收益，很多时候比单纯堆参数量更立竿见影。

2. **ModernBERT-large 达到了"独一档"的水平**。396M 参数的 large 版本微调后 NDCG@10 达到 79.42，进一步拉开了与所有基线模型的差距。

3. **训练成本极低**。在一张消费级的 RTX 3090 显卡上，base 版本训练约 30 分钟，large 版本训练不到 1 小时即可完成。相比模型带来的效果提升，这个训练成本几乎可以忽略不计。

4. **需要清醒认识微调的"代价"**：微调后的模型是**领域特化**的——它在 GooAQ 这类问答场景上表现极强，但未必能在其他领域（比如代码搜索或法律文书检索）上保持同样的优势。不过，这恰恰是我们想要的效果：**为特定业务场景训练一个"专才"，而不是追求一个样样通、样样松的"通才"**。

---

## 六、训练技巧与最佳实践

结合上述实战流程，这里总结几条在实际训练中值得注意的经验：

- **用 FAISS 加速难负样本挖掘**：在 `mine_hard_negatives` 中开启 `use_faiss=True`，可以显著降低大规模最近邻检索时的内存占用，加快挖掘速度，尤其在候选文档库较大时非常必要。

- **合理设置 `pos_weight`**：`BinaryCrossEntropyLoss` 中的 `pos_weight` 建议设置为负样本数量与正样本数量的比例（本例中挖掘了 5 个负样本，对应设置为 5），以缓解正负样本分布不均衡带来的训练偏差。

- **开启 `bf16` 混合精度训练**：如果你的 GPU 支持（如 Ampere 架构及以上），开启 `bf16=True` 可以在几乎不损失精度的情况下大幅提升训练速度、降低显存占用。

- **善用 `load_best_model_at_end` + `metric_for_best_model`**：将这两个参数配合评估器一起使用，训练器会自动在验证集表现最好的 checkpoint 处保存模型，无需人工挑选。

- **多数据集训练时谨慎选择采样策略**：如果多个数据集规模差异较大，`PROPORTIONAL` 策略能确保大数据集的信号不被小数据集"稀释"；但如果你希望每个数据集都被均匀、充分地学习（比如小数据集质量很高但数量少），则 `ROUND_ROBIN` 更合适。

- **难负样本的采样范围（`range_min`/`range_max`）要调优**：范围设置过小容易挖到"假负样本"（即语义上其实是正确答案，只是没被标注为正例）；设置过大则可能引入完全无关、模型一眼就能分辨的"简单负样本"，降低训练效率。建议根据具体数据集做一些小范围的超参搜索。

---

## 七、延伸资源

如果你希望进一步深入 Reranker 训练与相关话题，以下资源值得参考：

- **相关技术博客**：
  - Embedding 模型训练指南
  - Sparse Embedding 模型训练指南
  - Multimodal Reranker 训练指南

- **官方文档**：
  - Sentence Transformers 官方 Training Overview
  - Loss 函数总览（Loss Overview）
  - 完整 API Reference

- **训练示例**（可作为其他场景的参考模板）：
  - MS MARCO 段落检索排序训练示例
  - Quora 重复问题检测（Duplicate Questions）训练示例
  - 语义文本相似度（Semantic Textual Similarity, STS）训练示例

---

## 总结：微调的 ROI

回到最初的问题：召回之后，为什么还要精排？因为 Bi-encoder 的"弱交互"注定了它只能做到"差不多相关"，而 Cross-Encoder 的"深度交互"才能真正分辨出"哪个最相关"。

而本文最核心的结论是：**微调 Reranker 是一笔投入产出比极高的"买卖"**。

- **投入端**：几百到几万条标注的 query-document 正样本对（甚至可以只有正样本，配合 Hard Negatives Mining 自动补全负样本）；一张消费级 GPU；30 分钟到 1 小时的训练时间；不到百行的训练代码。
- **产出端**：一个在你自己的业务数据上，排序精度**超越通用大模型**的专属 Reranker，直接提升搜索、推荐、RAG 系统中最终展示给用户的结果质量。

对于任何已经上线了检索或 RAG 系统的团队来说，这几乎是"稳赚不赔"的优化方向——你不需要重新训练 Embedding 模型，也不需要改动整个检索架构,只需要在现有的"召回 + 精排"链路中,把通用 Reranker 换成一个用自己数据微调过的版本，就有很大概率获得实打实的精度提升。

如果你手头正好有一批标注好的 query-document 相关性数据（哪怕只有正样本），不妨按照本文的流程，花一个下午的时间跑一遍完整的微调流程——很可能会得到一个超出预期的结果。

---

**参考链接**

- Sentence Transformers 官方文档：https://www.sbert.net/
- Sentence Transformers GitHub 仓库：https://github.com/UKPLab/sentence-transformers
- Hugging Face Hub 模型与数据集：https://huggingface.co/
