---
title: 多智能体运营自动化系统
slug: multi-agent-ops
subtitle: 内容规划、投放分析、复盘建议的 Agent 协同
status: 概念验证
period: 2026 Q2
tags: Multi-Agent, Tool Calling, Workflow
summary: 通过角色化 Agent 协作，把运营分析流程从天级缩短到小时级。
cover: ./content/assets/projects/covers/agent-ops.svg
github: https://github.com/tangentllm/weblog
demo: https://example.com/
docs: https://example.com/docs
---

通过 Planner / Analyst / Writer / Supervisor 四角色分工，把运营分析从目标拆解、跨系统取数到复盘报告串成一条可观测链路。LangGraph 状态机负责编排与人工接管，统一 Tool Registry 对接 BI、埋点与文档系统，把原本天级的分析流程压缩到小时级。

这篇记录会把架构设计、各 Agent 职责划分、关键指标，以及从 PoC 到可运维部署时的取舍与踩坑一并展开。

## 问题定义

运营分析链路跨多个系统，人工汇总耗时且容易遗漏上下文。
我希望用多 Agent 协同把流程拆成可观测、可人工接管的工作流。

## Agent 角色设计

1. Planner：拆目标与任务  
2. Analyst：调用 BI 与埋点接口  
3. Writer：产出复盘建议  
4. Supervisor：结构校验与风险兜底

## 执行编排

```python
def run_pipeline(goal: str):
    plan = planner(goal)
    analytics = analyst(plan)
    draft = writer(analytics)
    return supervisor(draft)
```

## 当前效果

- 周报产出时间从 `8h` 降至 `1.5h`
- 异常归因覆盖率 `78%`
- 人工修改率下降 `42%`

