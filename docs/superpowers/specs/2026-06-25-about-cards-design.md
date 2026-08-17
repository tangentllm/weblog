# About Page Cards — Personal Identity Redesign

**Date:** 2026-06-25  
**Scope:** `/about` bento grid cards (body section below hero)  
**Approach:** A+C hybrid — keep Josh-style playful interactions, reorganize to 8 cards prioritized A (career) > B (thinking/writing) > C (personality).

---

## Goals

1. Express Tangentllm’s identity: career pivot (client → LLM), engineering values, and writing style.
2. Replace Josh homage placeholders (external talks, courses, stock photos) with site-native content.
3. Keep bento layout and micro-interactions (map, swap button, drums/topics, slot machine, TTS name).
4. Align visually with site-wide **monochrome base + rainbow accent** on emphasis only (credo stripes, topic keys, pick titles on hover).

## Non-goals

- Rewriting hero intro paragraphs (already personal).
- Adding skills section to about page (`joshAboutSkillsSectionMarkup` stays unused).
- User-editable picks UI — use `JOSH_ABOUT_PICKS` constant.
- Credo long-press theme toggle (theme remains in header nav).

## Priority model

| Tier | Focus | Card weight |
|------|-------|-------------|
| **A** | Career trajectory | Largest areas (map, arc, footprint) |
| **B** | Thinking & writing | Medium (credo, topics, picks) |
| **C** | Personality easter eggs | Smallest (name, play) |

---

## Card inventory (8 cards)

### A1 — `map` (keep area name)

| Item | Detail |
|------|--------|
| Interaction | Drag map, IP distance estimate (unchanged) |
| Primary copy | 我来自深圳。根据 IP 估算，我们相距约 *{distance}*。 |
| Secondary copy | 十年客户端工程 → 2024 起专注大模型。博客是我系统沉淀这条路径的地方。 |
| Data | `JOSH_ABOUT_AUTHOR_CITY`, existing geolocation helpers |
| Remove | 「交互创意致敬 Josh About 页。」 |

### A2 — `arc` (replaces `tall`)

| Item | Detail |
|------|--------|
| Interaction | Swap button cycles 3 states (`josh-about-swap-text`) |
| States |见下表 |
| Remove | Height figure SVG (`joshAboutFocusFigureMarkup`), height metaphor |

| State | Button value | Sentence |
|-------|--------------|----------|
| 1 | `10+` | 我有 **10+** 年工程经验。 |
| 2 | `2024` | **2024** 年起专注大模型。 |
| 3 | `{postCount}` | 博客已沉淀 **N** 篇长文笔记。 |

Config: `JOSH_ABOUT_ARC_UNITS` replaces `JOSH_ABOUT_HEIGHT_UNITS`.

### A3 — `foot` (replaces `job`)

| Item | Detail |
|------|--------|
| Interaction | `#josh-about-stat` count-up animation → target `postCount` |
| Primary copy | 这个博客面向同路人开放：用工程师视角写**能落地**的笔记。 |
| Dynamic line | **{postCount}** 篇笔记 · **{categoryCount}** 个分类 · **{tagCount}** 个标签 |
| Secondary | 最近更新：*{latestPost.title}* → link `/post/{slug}` |
| Remove | `JOSH_ABOUT_COURSES`, mascots, course list, `JOSH_ABOUT_COURSE_REGISTRATIONS` |

### B1 — `credo` (replaces `gay` / pride flag)

| Item | Detail |
|------|--------|
| Interaction | Click cycles credos; striped bar uses spectrum accent colors |
| Data | `JOSH_ABOUT_CREDOS` array: `{ label, text, colors[] }` |

| # | Text | Stripe hue |
|---|------|------------|
| 1 | 先搞懂原理，再写能上线的代码 | Blue `214deg` |
| 2 | 每篇笔记尽量可复现、可验证 | Green `152deg` |
| 3 | 生产环境里的权衡，比论文结论更重要 | Amber `36deg` |
| 4 | 复杂概念拆成小实验，建立直觉 | Coral `6deg` |

Reuse `#josh-about-flag-btn` DOM pattern; update `aria-label` per credo. Site theme toggle stays in header (not this card).

### B2 — `topic` (replaces `pkg` / drums)

| Item | Detail |
|------|--------|
| Interaction | Drum pads + keyboard shortcuts; click navigates to tag page + sound |
| Labels | Top 4 tags by post frequency, fallback: RAG, Transformer, RLHF, Agent |
| Data | Runtime `joshAboutTopTags(posts, 4)` |
| Copy below pads | 博客主题分布——点一下听听看（或类似一句） |

### B3 — `pick` (replaces `conf`)

| Item | Detail |
|------|--------|
| Title | 我常回顾的代表作 |
| List | 3 posts from `JOSH_ABOUT_PICKS` slugs |
| Images | Post `cover` via `resolveAssetUrl` |
| Remove | `josh-grabby-hands.jpg`, `JOSH_ABOUT_TALKS` |

Default picks:

1. `rag-production-performance-optimization`
2. `ppo-proximal-policy-optimization-pytorch`
3. `embedding-finetune-domain-rag`

### C1 — `name` (unchanged)

- IPA `/ˈtæn.dʒənt/`, TTS on click (`#josh-about-name-btn`).

### C2 — `play` (replaces `side`)

- Keep slot machine + Toggle Power.
- Copy: 喜欢用代码做有趣的东西——交互图解、小工具、这个 About 页本身。
- Remove duplicate “博客记录…” (covered in `foot`).

### Removed cards

| Old area | Reason |
|----------|--------|
| `desk` | Josh stock photo, no personal asset |
| `cat` | Generic easter egg, low identity signal |
| `job` | Replaced by `foot` |
| `gay` | Replaced by `credo` |
| `pkg` | Replaced by `topic` |
| `conf` | Replaced by `pick` |
| `tall` | Replaced by `arc` |

---

## Data helpers (new in `josh-site.js`)

```text
joshAboutPostStats() → { postCount, categoryCount, tagCount, latestPost }
joshAboutTopTags(posts, limit) → [{ name, count }]
joshAboutPicksMarkup(slugs) → HTML for pick list with covers
```

Sources: global `posts`, `categories`; no new API.

---

## Grid layout

### Desktop (8 columns)

```css
grid-template-areas:
  "map   map   map   map   arc   arc   name  name"
  "map   map   map   map   arc   arc   play  play"
  "foot  foot  foot  credo credo play  play  play"
  "pick  pick  pick  pick  topic topic topic topic";
```

CSS custom property per card: `--josh-about-area: map | arc | foot | credo | pick | topic | play | name`.

### Tablet / mobile

Re-stack with **A cards first**: map → arc → foot → credo → pick → topic → play → name. Update `@media` blocks in `josh-about.css` (≤51.875rem, ≤39rem, ≤24.5rem).

---

## Files to change

| File | Changes |
|------|---------|
| `josh-site.js` | Grid markup, constants, helpers, `initJoshAboutInteractions` |
| `josh-about.css` | `grid-template-areas`, remove desk/cat/job/tall styles, optional pick cover inset |

**Unchanged:** Hero (`joshAboutHeroMarkup`), about body sticky nav, footer, routes.

---

## Interaction migration

| Legacy | New |
|--------|-----|
| `#josh-about-flag-btn` | Credo cycle |
| `#josh-about-tall-btn` | Arc 3-state cycle |
| `#josh-about-stat` | `postCount` target |
| Drum pads | Tag navigation |
| `#josh-about-machine-btn`, `#josh-about-name-btn`, map | Unchanged |
| Cat, courses list | Removed with DOM |

---

## Visual tokens

- Card surface: existing `--josh-about-card-bg` (cloud gray), no full-card rainbow.
- Accents: credo stripes, topic key labels, pick link hover — use site spectrum tokens where available.
- Images: no `joshwcomeau.com/images` for content photos; local covers/assets only.

---

## Implementation order

1. Data helpers + constants (`JOSH_ABOUT_CREDOS`, `JOSH_ABOUT_PICKS`, `JOSH_ABOUT_ARC_UNITS`)
2. `joshAboutGridMarkup()` HTML (8 cards)
3. CSS grid + responsive areas
4. JS interaction updates (credo, arc, topic navigation)
5. Delete dead constants and unused markup functions
6. Manual QA: light/dark, mobile, reduced motion

---

## Test checklist

- [ ] 8 cards render on desktop; no overlap or horizontal scroll
- [ ] Tablet/mobile order: A blocks before B/C
- [ ] Arc cycles 3 states; stat animates to `postCount`
- [ ] Credo cycles 4 beliefs; stripes visible in light/dark
- [ ] Topic pads link to correct `/tag/*`
- [ ] Picks link to correct `/post/*` with cover images
- [ ] Map distance + drag still work
- [ ] Name TTS + slot machine still work
- [ ] No remaining `JOSH_ABOUT_TALKS` / `JOSH_ABOUT_COURSES` / Josh grabby-hands image
- [ ] `prefers-reduced-motion`: swap/stat animations respect reduced motion (existing patterns)

---

## Constants to delete

- `JOSH_ABOUT_TALKS`
- `JOSH_ABOUT_COURSES`
- `JOSH_ABOUT_COURSE_REGISTRATIONS`
- `JOSH_ABOUT_HEIGHT_UNITS` (replaced by `JOSH_ABOUT_ARC_UNITS`)
- `JOSH_ABOUT_PRIDE_FLAGS` (replaced by `JOSH_ABOUT_CREDOS`) — update flag click handler accordingly
