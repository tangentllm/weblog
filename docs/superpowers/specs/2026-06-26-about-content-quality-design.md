# About Page — Content Quality Redesign (1+2+3 Fusion)

**Date:** 2026-06-26  
**Route:** `/about` (hero + 10-card grid)  
**Scope:** Copy and content wiring only — no grid/layout/image changes  
**Status:** Implemented in `josh-site.js` (not committed unless user requests)

---

## Problem

Previous enrichment added volume but felt **padded**: repeated brand slogans, dashboard stats, generic principles, and a Josh English hero disconnected from Chinese cards.

## Strategy: 1 + 2 + 3 fusion

Each card carries **one primary lens**:

| Lens | Role | Cards |
|------|------|-------|
| **① 工程师档案** | Verifiable facts, career, current work | Hero, `tall`, `job` |
| **② 写作样本** | Problem → takeaway from real posts | `conf`, `desk`, `gay` (article-derived judgments) |
| **③ 个人随笔** | Tone, habit, personality | `map`, `side`, `cat`, `name`, `pkg` |

**Rules:** Say each theme once; drop tag counts / read-minute stats; principles must trace to articles or incidents.

## Identity (approved)

- Location: 深圳
- Path: 十年客户端 → 2024 LLM 应用
- Now: **A** 个人工作室接 LLM 落地 + **C** 博客/side project 为主要公开输出（低调表述）

---

## Hero (Chinese)

- Title: 你好，我是 Tangentllm。
- P1: 深圳 · 十年客户端 · 2024 转向 RAG/Agent/推理链路
- P2: 工作室接落地；**博客是我主要的公开输出**，side project 写在这里
- P3: 不写十分钟上手；留上线判断与踩坑复盘

---

## Card copy (final polished)

### `map`
- 我来自深圳。按 IP 粗算，我们相距约 {distance}。
- 我在这里写代码、记笔记。不表演成长，只留以后还能翻出来核对的东西。

### `tall` (swap ×3)
| Value | Line | Detail |
|-------|------|--------|
| 10+ | 10+年，都在跟交付打交道。 | 性能、架构、告警——先问会不会在用户那边爆 |
| 2015 | 2015年入行，从客户端开始。 | UI、稳定性、版本节奏 |
| 2024 | 2024年转向 LLM 应用。 | Demo 不算数；灰度 P95 曲线 |

### `job`
- 工作室做 LLM 落地；博客和 side project 是对外说话的地方
- 写给会调模型、想知道为什么这样改的人
- {postCount} 篇长文 + 三个分类入口（label → category）

### `gay` (credo ×4, article judgments)
1. 相关性上去，延迟未必好看 → 生成/Rerank 瓶颈  
2. 默认初始化会使坏 → Embedding 失稳  
3. 纯向量漏字面匹配 → BM25 补位  
4. 没 Trace 优化打在错觉上 → 先量 latency  

### `conf` (hot Top 3, dynamic)
- Intro: 想先感受我写的东西，从下面三段开始
- Format: hook → takeaway → title (link)
- `JOSH_ABOUT_POST_HOOKS` for curated slugs; fallback `joshAboutPostHook()` from excerpt

### `pkg`
- 写得最多的是这几类问题——点一下，听个响
- Q / W / E / R

### `desk`
- 演示 1.9s vs 灰度 P95 4.2s
- Faithfulness vs 延迟曲线反差

### `side`
- 交互图 > 段落；《Transformer 原理》《Attention 从零实现》
- 老虎机纯装饰

### `cat` / `name`
- Paper 复现 / 咖啡因；Tangent 切线释义 + TTS

---

## Implementation notes

| File | Changes |
|------|---------|
| `josh-site.js` | Hero, all card markup, `JOSH_ABOUT_POST_HOOKS`, `JOSH_ABOUT_JOB_CATEGORY_LINKS`, credo text, hook list |
| `josh-about.css` | Minor: hooks list alignment |

## Non-goals

- Josh placeholder images (conf, desk, job mascots)
- Grid restructure, drum pad navigation
- Git commit until user asks
