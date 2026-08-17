# About Page Cards — Content Enrichment (Layered)

**Date:** 2026-06-26  
**Route:** `/about` — 10-card bento grid (below hero)  
**Scope:** Content filling only — no grid/layout/CSS restructure, no hero, no image swaps  
**Approach:** B+C hybrid — layered disclosure (L1/L2/L3) + blog data woven into copy

---

## Goals

1. Make each card feel richer on scan and on interaction without changing the 10-card Josh layout.
2. Balance narrative depth, information density, and discovery (user chose **D — all three**).
3. Keep tone: rational, evidence-oriented, gentle with clear boundaries; no marketing hype.

## Non-goals

- Changing `grid-template-areas`, card count, or hero copy
- Replacing Josh placeholder images (conf cover, desk photo, job mascots)
- Drum pads linking to `/tag/*` (labels + sound only)
- Git commit until user explicitly requests

---

## Layer model

| Layer | Role | Visibility |
|-------|------|------------|
| **L1** | One-line hook | Always visible |
| **L2** | Facts, metadata, second paragraph | `.small` or list sub-lines |
| **L3** | Discovery / interaction unlock | Swap, credo cycle, map route connected |

Mobile: L2 may show by default on key cards (`map` stats, `job` numbers); L3 on `map` reveals when IP route connects.

---

## Card inventory (approved A/B/C)

### A — `map`

- **L1:** 我来自深圳。根据 IP 估算，我们相距约 *{distance}*。
- **L2:** 博客是我系统沉淀工程实践与思考的地方。
- **L2 dynamic:** `{postCount}` 篇笔记 · `{categoryCount}` 个分类 · 最近更新 *{latestPost.title}* → `/post/{slug}`
- **L3:** 路线已连上——拖地图看看我们隔了多远。（`#josh-about-map-route-hint`, shown when route connects）

### A — `tall` (arc swap ×3)

| Value | L1 | L3 detail (swaps with button) |
|-------|-----|-------------------------------|
| `10+` | 我有 **10+** 年工程经验。 | 客户端、性能、架构、交付——先学会把东西做出来。 |
| `2015` | **2015** 年入行客户端开发。 | 从 UI 细节到系统稳定性，习惯了问「上线后会怎样」。 |
| `2024` | **2024** 年起专注大模型应用。 | 同一套工程思维，换了一条更陡的学习曲线。 |

- Tail: 职业路径的几个节点——点按钮切换看看。

### A — `job`

- **L1:** 我是十年客户端工程师，现转向 LLM 应用工程。
- **L2:** 不讲速成神话；记录可复现实践、踩坑复盘与工程取舍。面向想扎实搞懂 LLM 工程、又厌倦营销腔教程的读者。
- **L3 dynamic:** 博客已发布 **{postCount}** 篇（count-up）· **{tagCount}** 个标签 · 累计约 **{totalReadMinutes}** 分钟阅读
- **List:** `{category} · {n} 篇 →` for 基础原理 / 工程实践 / 微调与对齐

### B — `gay` (credo ×4)

| L1 | L3 subtitle (cycles with click) |
|----|----------------------------------|
| 先讲原理，再谈方案。 | 原理没懂，方案只是复制粘贴。 |
| 记录真实踩坑，不做结果倒推教程。 | 先有问题，再有答案——顺序不能反。 |
| 给出边界条件，不把结论绝对化。 | 「看场景」比「信结论」更接近工程现实。 |
| 优先可上线与可维护，而非演示效果。 | Demo 能骗人，日志不会。 |

- Small: 点击切换写作原则。

### B — `conf`

- **L1:** 我常回顾的代表作：
- **L2:** 按站内热度挑选；想快速建立图景，从这三篇开始。
- **List:** Title + `{category} · {readTime} · {YYYY-MM}` per hot post (top 3 from `JOSH_POPULAR_SLUGS` / view fallback)

### B — `pkg`

- **L1:** 博客主题分布——点一下听听看。
- **Drums:** Top 4 tag labels (sound only, no navigation)
- **L2:** `RAG · 7 篇　Transformer · 6 篇 …` (dynamic counts)
- **L3 small:** 全站最热标签：**{topTag}**（{count} 篇）
- **Small:** 也可以用键盘 Q / W / E / R 触发。

### C — `desk`

- **L1:** 有一次排查 RAG 召回率骤降，我在工位上对着日志干坐了一整个下午。
- **L2:** 最后发现是 chunk 边界切碎了表格——模型没问题，是预处理在悄悄使坏。
- **L3 small:** 这类问题不太会写进 README，但最值得记下来。

### C — `side`

- **L1:** 喜欢用代码做有趣的东西——交互图解、小工具、这个 About 页本身。
- **L2:** 例如这篇《Transformer 原理解析》和《从零写多头注意力》——图比字更好懂。（站内链接）
- **L3 small:** 老虎机纯装饰；真正好玩的是文章里的图。

### C — `cat`

- **L1/L2:** 新 paper 半夜敲代码 / 咖啡因超标（unchanged tone）
- **L3 small:** 撸猫没有实际功能。和我一样，很多动画也没有。

### C — `name`

- **L1:** IPA + Tangent-L-L-M
- **L2:** Tangent = 切线——在曲线上找局部最诚实的近似。
- **L3:** 点击听合成发音。

---

## Data helpers

- `joshAboutParseReadMinutes(readTime)` — extract minutes from `readTime` string
- `joshAboutTotalReadMinutes(postList)`
- `joshAboutCategoryPostCount(name, postList)`
- `joshAboutFormatPostMonth(date)`
- `joshAboutTopTags` — extend with `count` per tag
- `joshAboutHotPostSlugs` — unchanged (popular slugs → local views → latest)

## Files to change

| File | Changes |
|------|---------|
| `josh-site.js` | Copy layers, constants (`subtitle`/`detail`), markup helpers, swap/credo/map hint handlers |

**Unchanged:** `josh-about.css` grid, hero, images, interaction mechanics (except syncing new text nodes on swap).

---

## QA checklist

- [ ] All 10 cards show richer L2 without horizontal overflow on mobile
- [ ] Arc swap updates main line + detail line
- [ ] Credo click updates main + subtitle
- [ ] Map route hint appears after IP geolocation succeeds
- [ ] Hot posts show category · readTime · month
- [ ] Job category list shows per-category post counts
- [ ] Drum pads still play sounds only (no navigation)
- [ ] No git commit until user requests
