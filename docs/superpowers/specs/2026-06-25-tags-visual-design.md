# Tags Pages Visual Redesign — Tint Ladder (方案 C)

**Date:** 2026-06-25  
**Scope:** `/tags` index + `/tag/*` detail  
**Constraint:** DOM structure and content layout unchanged; visual tokens only (color, size, radius, shadow, motion).  
**Approach:** Tint Ladder — express tag heat through primary/cloud color steps, not opacity or layout changes.

---

## Goals

1. Make tag frequency readable at a glance without reading counts.
2. Unify `/tags` and `/tag/*` under one tint vocabulary.
3. Fix hot-card invisibility on first paint (scroll-reveal side effect).
4. Preserve Josh site palette (`--josh-color-primary`, `--josh-color-cloud-*`).

## Non-goals

- Reordering or removing stat grid, tag cloud, or hot section.
- Replacing pill/card/grid structure with a different IA.
- New routes, backend, or English copy changes.

---

## Design system: Tint Ladder tokens

Add scoped custom properties under `.josh-page--tags` and `.josh-page--tag` (defined in `josh-components.css`):

| Token | Light | Purpose |
|-------|-------|---------|
| `--josh-tint-step-5-bg` | `color-mix(in srgb, var(--josh-color-primary) 22%, white)` | Hottest surface |
| `--josh-tint-step-4-bg` | `color-mix(in srgb, var(--josh-color-primary) 14%, white)` | Hot |
| `--josh-tint-step-3-bg` | `color-mix(in srgb, var(--josh-color-cloud-400) 38%, white)` | Warm |
| `--josh-tint-step-2-bg` | `color-mix(in srgb, var(--josh-color-cloud-300) 62%, white)` | Cool |
| `--josh-tint-step-1-bg` | `color-mix(in srgb, var(--josh-color-cloud-300) 38%, white)` | Coldest |
| `--josh-tint-step-5-fg` | `var(--josh-color-primary)` | Hottest label |
| `--josh-tint-step-1-fg` | `var(--josh-color-gray-700)` | Coldest label |
| `--josh-tint-radius-card` | `0.75rem` (12px) | Cards & shells |
| `--josh-tint-radius-pill` | `999px` | Pills & cloud items |
| `--josh-tint-shadow-card` | Same stack as `.josh-blog-card` (light) | Floating cards |
| `--josh-tint-shadow-hover` | Blog-card stack + `translateY(-2px)` | Hover lift |

**Dark mode** (`html.josh-site.dark`, `[data-color-mode="dark"]`):

| Step | Background | Foreground / outline |
|------|------------|----------------------|
| 5 | `primary @ 28%` on transparent | `primary` |
| 4 | `primary @ 18%` | `text` |
| 3 | `cloud-400 @ 22%` | `text` |
| 2 | `cloud-300 @ 18%` | `gray-700` |
| 1 | `cloud-300 @ 10%` | `gray-700` |

Cards use `outline: 1px solid gray-700` instead of light-mode shadow. Hover deepens background one step.

---

## `/tags` index page

### 1. Page intro (unchanged markup)

- Keep `josh-tags-page__title` / `__desc`.
- Optional visual only: title `color: var(--josh-color-text)`; desc unchanged.

### 2. Stat grid (`josh-stat-card`)

| Property | Value |
|----------|-------|
| Background | `--josh-tint-step-2-bg` |
| Border-radius | `--josh-tint-radius-card` |
| Padding | `0.875rem 1.125rem` (slightly flatter) |
| Value color | `var(--josh-color-primary)` |
| Label | `gray-700`, `0.8125rem` |
| Shadow | none (light); dark: subtle outline |

Four stats share one step; numbers carry primary tint — heat metaphor without competing with cloud tiers.

### 3. Tag cloud container (`josh-tag-cloud`)

| Property | Value |
|----------|-------|
| Background | `--josh-tint-step-1-bg` |
| Border-radius | `0.75rem` |
| Padding | `1.5rem` |
| Shadow | none light; `outline: 1px` cloud-mix dark |
| Inner alignment | `justify-content: flex-start` (visual polish only) |

### 4. Cloud pills (`josh-tag-cloud__item--1` … `--5`)

**Remove opacity-based sizing hierarchy.** Keep font-size steps; drive heat via background + badge:

| Class | Font size | Background | Text | Count badge |
|-------|-----------|------------|------|-------------|
| `--5` | `1.25rem` | step-5-bg | step-5-fg | primary 24% pill |
| `--4` | `1.125rem` | step-4-bg | text | primary 16% pill |
| `--3` | `1rem` | step-3-bg | text | cloud badge |
| `--2` | `0.9375rem` | step-2-bg | text | cloud badge |
| `--1` | `0.875rem` | step-1-bg | step-1-fg | gray badge |

- Pill padding: `0.45rem 0.9rem`; radius `999px`.
- Hover: bump one tint step (CSS `color-mix` or predefined `--josh-tint-step-N-hover-bg`).
- Focus-visible: `outline: 2px solid var(--josh-color-primary)`.

### 5. Hot section (`josh-tags-page__hot`)

**Title (`josh-tags-page__hot-title`):** match `josh-section-label` — `0.8125rem`, bold, uppercase, `letter-spacing: 0.08em`, `gray-700`.

**Hot cards (`josh-tag-card` in grid):** rank by DOM order (already sorted by count). Use `nth-child` tints, no markup change:

| Child | Card background | Badge | Name weight |
|-------|-----------------|-------|-------------|
| 1–2 | step-5-bg | primary 24% | bold |
| 3–4 | step-4-bg | primary 16% | bold |
| 5–6 | step-3-bg | cloud badge | medium |

| Property | Value |
|----------|-------|
| Border-radius | `0.75rem` |
| Padding | `1.25rem 1.375rem` |
| Min-height | `10.5rem` (equal row height) |
| Shadow | `--josh-tint-shadow-card` (light) |
| Hover | one step hotter + shadow-hover |

**Preview list / “+N 更多”:** `gray-700`; more link `primary` on hover.

### 6. Reveal animation fix (behavior, no layout change)

Problem: hot cards start `opacity: 0` below fold; section looks empty.

**Fix:** In `josh-components.css`, add exception:

```css
.josh-page--tags.josh-reveal-ready .josh-tags-page__hot .josh-tag-card:not(.is-revealed) {
  opacity: 1;
  transform: none;
}
```

Hot cards still get staggered reveal on scroll if desired, but are **never hidden**. Alternatively set `is-revealed` on init in `initJoshArticleScrollReveal` for `.josh-page--tags .josh-tag-card` — prefer CSS exception to avoid JS churn.

---

## `/tag/*` detail page

### 1. Archive header

- `#tag` title: unchanged size.
- `.josh-blog-archive__count`: `color: var(--josh-color-primary)`; `font-weight: medium`.
- Back link: unchanged; hover already primary.

### 2. Article grid (`josh-blog-card`)

Keep white card + existing shadow in light mode. Add **tint accent** without layout change:

| Property | Value |
|----------|-------|
| `border-left` | `3px solid color-mix(primary 35%, transparent)` |
| Hover | `border-left-color: var(--josh-color-primary)`; existing title underline |
| Dark | accent via `border-left` on outlined card |

Grid gap / columns unchanged.

### 3. Empty state (`josh-empty-state`)

| Property | Value |
|----------|-------|
| Wrapper | `background: --josh-tint-step-2-bg` |
| Border-radius | `0.75rem` |
| Padding | `2.5rem 1.5rem` |
| Max-width | `28rem`; `margin-inline: auto` |
| Text | `gray-700` |
| Action button | `step-3-bg` background; primary outline on hover |

Hide duplicate back link visual weight: keep one back link in header only; empty-state action button remains. If two back links exist in DOM, style `.josh-empty-state .josh-archive-back { display: none }` on tag page only.

### 4. Related tags (`josh-archive-related`)

Related pills use **count-based tier** (reuse `joshTagIndexPillMarkup` tiers: hot ≥3, warm =2, more =1):

| Tier | Pill background via `--josh-pill-color` |
|------|----------------------------------------|
| hot | `var(--josh-color-primary)` |
| warm | `var(--josh-color-cloud-400)` |
| more | `var(--josh-color-cloud-300)` |

Override `.josh-pill__bg` opacity per tier on `.josh-page--tag` to match cloud pill steps (hot 0.92, warm 0.85, more 0.72 — already partially defined).

Section label: existing `josh-section-label`.

---

## Responsive

| Breakpoint | Adjustment |
|------------|------------|
| `< 48rem` | Stat grid 2×2; hot grid 1 col; cloud padding `1.25rem` |
| `≥ 40rem` | Hot grid 2 col (unchanged) |
| `≥ 56rem` | Hot grid 3 col (unchanged) |

No horizontal overflow; pill text wraps inside cloud.

---

## Files to touch

| File | Changes |
|------|---------|
| `josh-components.css` | Tint tokens; restyle `.josh-page--tags` / `.josh-page--tag` blocks; reveal exception; cloud item tiers; hot card nth-child; detail accents |
| `josh-interactions.js` | Only if CSS reveal exception insufficient (unlikely) |

**Do not modify:** `josh-home.js`, `josh-home.css`, `renderJoshTags` / `renderJoshTagPosts` markup unless empty-state duplicate link removal requires one line (prefer CSS-only).

---

## Verification

1. **Desktop light/dark** `1280×900`: `/tags` — all cloud tiers distinguishable; hot cards visible without scroll; stats numbers primary.
2. **Mobile** `375×900`: no overflow; pills readable.
3. **`/tag/RAG`**: 6 cards, left accent visible; count primary-colored.
4. **`/tag/LLM`** (or any 0-post tag): empty state in tinted box; single clear CTA.
5. **Theme toggle**: no flash; dark outlines readable.
6. Compare screenshots to `_probe-tags-now.png` baseline — structure identical, color hierarchy improved.

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Too many primary tints feel noisy | Cap step-5 to top cloud tier + top 2 hot cards only |
| Dark mode low contrast on step-1 | Minimum `outline` on coldest pills |
| nth-child rank wrong if sort changes | Hot section order already by `sortedTags`; document dependency |

---

## Summary

Tint Ladder keeps the current three-block `/tags` layout and detail archive grid, replacing opacity-based hierarchy with five primary/cloud background steps shared across stats, cloud pills, hot cards, and related pills. Hot cards are always visible; detail cards gain a subtle primary left accent. All work is CSS-scoped to `josh-page--tags` and `josh-page--tag`.
