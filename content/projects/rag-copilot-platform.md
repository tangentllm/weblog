---
title: 企业知识库 RAG Copilot
slug: rag-copilot-platform
subtitle: 私域知识问答平台，支持权限检索与引用溯源
status: 已上线
period: 2025 Q4
tags: RAG, LangGraph, Supabase, 权限系统
summary: 将文档、FAQ、工单统一索引，构建可审计的企业问答助手。
cover: ./content/assets/projects/covers/rag-copilot.svg
github: https://github.com/tangentllm/weblog
demo: https://example.com/
docs: https://example.com/docs
---

将文档、FAQ、工单统一索引，构建带引用溯源、可按部门权限过滤的企业问答助手。摄取 → 分块嵌入 → 检索重排 → 生成的流水线由 LangGraph 编排，并内置自我纠错与对话审计闭环，Top-3 检索命中率 82%。

下面会拆数据层、检索层与权限策略，说明我们如何在保证可审计的前提下，让回答必须带可点击引用、置信度不足时主动拒答。

## 项目背景

企业知识分散在文档、工单和内部 Wiki 中，信息检索成本高，且回答不可追溯。
我将该系统目标定义为：**高准确、可追溯、可控风险** 的企业问答平台。

## 架构拆解

### 数据层

- Supabase Postgres 统一存储元数据
- `pgvector` 负责向量索引
- 租户隔离 + 文档级 ACL 权限

### 检索层

$$Score = \alpha \cdot BM25 + (1-\alpha)\cdot Cosine$$

通过混合检索与重排融合，降低单一向量召回偏差。

```python
def hybrid_score(bm25: float, cosine: float, alpha: float = 0.45) -> float:
    return alpha * bm25 + (1 - alpha) * cosine
```

## 关键优化

1. 低置信度自动澄清，不强答  
2. 证据引用与原文跳转  
3. 失败样本自动回流评测集

## 线上结果

| 指标 | 改造前 | 改造后 |
|---|---:|---:|
| Top-5 召回率 | 74.2% | 89.7% |
| 首字响应时间 | 2.9s | 1.3s |
| 人工评测准确率 | 61% | 82% |

