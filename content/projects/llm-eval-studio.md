---
title: LLM Eval Studio
slug: llm-eval-studio
subtitle: 模型评测与提示词回归平台
status: 迭代中
period: 2026 Q1
tags: Evaluation, PromptOps, A/B Test
summary: 围绕真实业务样本做离线评测，建立 Prompt 与模型变更的质量闸门。
cover: https://picsum.photos/seed/eval-studio/1200/700
github: https://github.com/tangentllm/weblog.git
demo: https://example.com/
docs: https://example.com/docs
---

## 为什么做这个平台

模型和 Prompt 改动频繁，但评估缺少统一标准，导致线上质量波动。
这个平台的目标是把“上线前评估”工程化、可复现化。

## 核心能力

- 一键复跑历史版本，自动生成差异报告
- 多维评分：事实性、格式稳定性、指令遵循
- 失败样本自动聚类，辅助定位系统性问题

## 评分策略

```python
def final_score(rule_score: float, judge_score: float) -> float:
    # 规则分和 LLM-as-Judge 组合打分
    return round(rule_score * 0.6 + judge_score * 0.4, 4)
```

## 阶段性结果

| 指标 | 数值 |
|---|---:|
| 评测耗时降低 | 63% |
| 上线前缺陷拦截率 | +34% |
| 回归定位时长 | 2h -> 25min |

