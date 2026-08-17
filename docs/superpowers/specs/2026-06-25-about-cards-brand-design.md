# About Page Cards — Brand Narrative Optimization

**Date:** 2026-06-25  
**Route:** `/about` content card section (below hero)  
**Decision Priority:** Brand narrative > content conversion > visual fidelity

---

## 1) Background

This redesign focuses on making the About card area communicate a clear identity in the first two screens:

- 10-year client engineer background
- transition to LLM application engineering
- practical, principle-driven writing style

The primary goal is not aggressive click-through. Conversion remains secondary and should only appear as low-interference extension paths.

---

## 2) Target Impression

If a visitor only scans the card area quickly, the intended takeaway is:

> 我是一个十年客户端工程师，已经转向 LLM 应用。  
> 我通过拆解真实踩坑过程和背后原理，为想扎实搞懂 LLM 工程、又厌倦营销腔教程的中文读者创造价值。

---

## 3) Content Tone

Tone is intentionally constrained to:

- rational and evidence-oriented
- gentle but with clear position

Execution rules:

- explain how and why decisions are made
- avoid hype words and "shortcut" claims
- show boundaries and trade-offs instead of one-size-fits-all conclusions

---

## 4) Narrative Backbone

The narrative backbone is fixed and should remain visible in card ordering:

1. Who I am
2. Transition milestones
3. Working method
4. Current practical focus
5. Personality and playful finish

Three approved milestones:

- 2015: 入行客户端开发
- 2024: 动手做一些 LLM demo
- 现在: 为工作室落地 LLM 项目

---

## 5) Chosen Direction (A)

Selected approach: **Transition narrative axis**, with light extension links.

Why:

- best aligned with priority model (brand first)
- strongest memory structure (timeline + method)
- still allows secondary conversion without turning About into a navigation dashboard

Not chosen:

- method-first architecture (strong expertise signal, weaker personal story)
- balanced conversion-heavy architecture (risks diluting identity expression)

---

## 6) Final Card Inventory (8 cards)

Use and keep this sequence as the canonical order:

1. `map` (large) — origin + visitor distance, lightweight opening
2. `arc` (large) — transition timeline with 3 milestone states
3. `foot` (large) — mission, audience, and writing intention
4. `credo` (medium-large) — method principles (3-4 rotating statements)
5. `pick` (medium) — 3 representative posts, extension-only
6. `topic` (medium) — topic distribution / domain anchors
7. `play` (small) — personality relief, low weight
8. `name` (small) — pronunciation and personal signature

---

## 7) Layout and Reading Order

### Desktop

Perceived reading flow should be:

`map -> arc -> foot -> credo -> pick -> topic -> play -> name`

### Mobile

Must preserve the exact same narrative order:

`map -> arc -> foot -> credo -> pick -> topic -> play -> name`

No playful card should appear before narrative-critical cards on narrow viewports.

---

## 8) Card-Level Content Guidance

### `map`

- Keep as emotional icebreaker
- Keep copy short; avoid heavy biography here

### `arc`

- Use the 3 approved milestones exactly
- Reuse existing swap interaction pattern for continuity

### `foot`

- Main brand statement card
- Clarify audience: readers who want practical LLM engineering depth without marketing tone

Suggested copy skeleton:

> 我是十年客户端工程师，现转向 LLM 应用工程。  
> 这里不讲速成神话，主要记录可复现的实践、踩坑复盘与工程取舍。

### `credo`

- 3-4 method principles, concise and concrete
- Emphasize reproducibility, constraints, and trade-offs

Suggested principle set:

1. 先讲原理，再谈方案
2. 记录真实踩坑，不做结果倒推教程
3. 给出边界条件，不把结论绝对化
4. 优先可上线与可维护，而非演示效果

### `pick`

- Keep 3 representative posts
- Function is "continue reading if interested", not a hard funnel

### `topic`

- Show practical topic clusters (for example: RAG, Embedding, RLHF, Agent)
- Present as capability landscape, not taxonomy overload

### `play` and `name`

- Keep existing Josh-style playful DNA
- Reduce copy dominance so these remain supporting accents

---

## 9) Keep / Replace / Remove Rules

Keep:

- Josh-like interaction rhythm and micro-feedback patterns
- existing map, name pronunciation, and light playful mechanisms

Replace:

- any copy that reads like Josh placeholder identity
- generic statements that do not prove Tangentllm-specific narrative

Remove:

- content blocks that look externally borrowed and do not support the brand arc
- visual noise that competes with timeline and method cards

---

## 10) Success Criteria

Brand narrative criteria (primary):

- visitors can restate identity + transition in one sentence after quick scan
- timeline and method are both discoverable without scrolling deeply

Secondary conversion criteria:

- representative posts are discoverable but not visually dominant
- topic card aids orientation without becoming navigation-first UI

Visual criteria:

- Josh rhythm preserved in spacing and interactions
- no decorative motion that competes with core narrative cards

---

## 11) QA Checklist

- [ ] Desktop first view clearly communicates "who + transition + method"
- [ ] Mobile order matches desktop narrative sequence
- [ ] `arc` presents the exact three milestone states
- [ ] `foot` copy explicitly names intended audience and stance
- [ ] `credo` statements are principle-level, non-marketing, and concise
- [ ] `pick` remains secondary (extension path only)
- [ ] `play` and `name` stay supportive, not dominant
- [ ] Tone remains rational + gentle with clear boundaries

---

## 12) Implementation Boundary

This spec defines content strategy, card hierarchy, and narrative sequencing for `/about` cards only.

Out of scope:

- hero rewrite
- global navigation changes
- unrelated route redesign
