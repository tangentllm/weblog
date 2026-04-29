---
title: LLM 微调实战笔记：SFT + LoRA + 评估闭环
slug: llm-sft-note
date: 2025-03-14
readTime: 16 分钟
category: 微调与对齐
tags: SFT, LoRA, DPO
cover: ./content/assets/posts/covers/sft.svg
excerpt: 从数据构造、训练参数到离线评估，记录一次可复现的指令微调流程与常见坑。
---

## 任务定义

目标：让基础模型在“技术问答 + 代码解释”场景下提升可控性与稳定性。

```mermaid
flowchart LR
    A[Instruction Data] --> B[SFT]
    B --> C[LoRA Adapter]
    C --> D[Offline Eval]
    D --> E[Online A/B]
    E --> F[Failure Replay Set]
    F --> A
```

*图1：SFT + LoRA 的训练与评估回流闭环*

## 数据准备

```json
{
  "instruction": "解释梯度裁剪的意义",
  "input": "",
  "output": "梯度裁剪可以限制梯度范数，避免训练早期出现梯度爆炸。"
}
```

## 训练配置（LoRA）

```bash
python train.py \
  --model_name_or_path meta-llama/Llama-3-8b \
  --lora_r 64 --lora_alpha 128 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16
```

<details>
<summary>参数选择经验（展开查看）</summary>

- 对小数据集，优先减小学习率，避免过拟合到模板化表达。  
- 对长回答任务，增大 max_seq_length 并配合 packing。  
- 先做烟雾测试，再做完整训练，能节省大量排障时间。  

</details>

## 评估闭环

1. 离线：抽样高价值问题做人审打分。  
2. 在线：灰度 A/B 对比平均回复长度与用户停留时间。  
3. 回流：将低分样本与拒答失败样本纳入下一轮训练。  

