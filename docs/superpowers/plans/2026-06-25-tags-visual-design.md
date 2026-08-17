# Tags Tint Ladder Visual Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Scheme C (Tint Ladder) visual tokens to `/tags` and `/tag/*` without changing DOM layout—colors, radii, shadows, reveal fix only.

**Architecture:** Add scoped CSS custom properties on `.josh-page--tags` and `.josh-page--tag`, then override stat cards, tag cloud pills, hot cards, detail blog cards, empty state, and related pills. One file (`josh-components.css`); no JS unless reveal CSS exception fails.

**Tech Stack:** Static SPA, vanilla CSS (`josh-components.css`), Playwright probes for visual verification (`npx serve . -l 3456`).

## Global Constraints

- DOM structure and content layout unchanged (stat grid + tag cloud + hot section remain).
- Scope: `/tags` index + `/tag/*` detail only.
- Do not modify `josh-home.js`, `josh-home.css`, `renderJoshTags`, `renderJoshTagPosts` markup (CSS-only; hide duplicate empty-state back link via CSS).
- Use existing palette: `--josh-color-primary`, `--josh-color-cloud-300/400`, `--josh-color-gray-700`, `--josh-color-text`.
- Tint Ladder: five background steps; remove opacity-based cloud pill hierarchy.
- Hot cards must be visible on first paint (no `opacity: 0` in hot section).

**Spec:** `docs/superpowers/specs/2026-06-25-tags-visual-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `josh-components.css` | All tint tokens + page-scoped overrides |
| `_verify-tags-tint.mjs` (create) | Playwright assertions for tint/reveal |

---

### Task 1: Tint Ladder tokens + dark mode

**Files:**
- Modify: `josh-components.css` (insert before `/* Tags index page — cloud + hot cards layout */` ~line 862)

**Interfaces:**
- Produces: CSS variables consumed by Tasks 2–4 on `.josh-page--tags, .josh-page--tag`

- [ ] **Step 1: Add token block**

```css
/* Tint Ladder — tags / tag detail pages */
.josh-page--tags,
.josh-page--tag {
  --josh-tint-step-5-bg: color-mix(in srgb, var(--josh-color-primary) 22%, white);
  --josh-tint-step-4-bg: color-mix(in srgb, var(--josh-color-primary) 14%, white);
  --josh-tint-step-3-bg: color-mix(in srgb, var(--josh-color-cloud-400) 38%, white);
  --josh-tint-step-2-bg: color-mix(in srgb, var(--josh-color-cloud-300) 62%, white);
  --josh-tint-step-1-bg: color-mix(in srgb, var(--josh-color-cloud-300) 38%, white);
  --josh-tint-step-5-fg: var(--josh-color-primary);
  --josh-tint-step-1-fg: var(--josh-color-gray-700);
  --josh-tint-radius-card: 0.75rem;
  --josh-tint-radius-pill: 999px;
  --josh-tint-shadow-card:
    rgba(155, 174, 187, 0.28) 0.3px 0.5px 0.7px 0,
    rgba(155, 174, 187, 0.28) 0.8px 1.6px 2px -0.8px,
    rgba(155, 174, 187, 0.29) 1.9px 3.8px 4.8px -1.6px,
    rgba(155, 174, 187, 0.3) 4.5px 9.1px 11.6px -2.4px;
}

html.josh-site.dark .josh-page--tags,
html.josh-site.dark .josh-page--tag,
html.josh-site[data-color-mode="dark"] .josh-page--tags,
html.josh-site[data-color-mode="dark"] .josh-page--tag {
  --josh-tint-step-5-bg: color-mix(in srgb, var(--josh-color-primary) 28%, transparent);
  --josh-tint-step-4-bg: color-mix(in srgb, var(--josh-color-primary) 18%, transparent);
  --josh-tint-step-3-bg: color-mix(in srgb, var(--josh-color-cloud-400) 22%, transparent);
  --josh-tint-step-2-bg: color-mix(in srgb, var(--josh-color-cloud-300) 18%, transparent);
  --josh-tint-step-1-bg: color-mix(in srgb, var(--josh-color-cloud-300) 10%, transparent);
  --josh-tint-step-5-fg: var(--josh-color-primary);
  --josh-tint-step-1-fg: var(--josh-color-gray-700);
}
```

- [ ] **Step 2: Verify tokens resolve**

Run dev server: `npx serve . -l 3456`  
Open `/tags`, DevTools → `.josh-page--tags` → confirm `--josh-tint-step-5-bg` is set.

---

### Task 2: `/tags` index — stats, cloud, pills

**Files:**
- Modify: `josh-components.css` — `.josh-stat-card`, `.josh-tag-cloud`, `.josh-tag-cloud__item*` (~lines 315–388 and `.josh-page--tags` block)

**Interfaces:**
- Consumes: Task 1 tint tokens

- [ ] **Step 1: Scope stat + cloud under `.josh-page--tags`**

```css
.josh-page--tags .josh-stat-card {
  padding: 0.875rem 1.125rem;
  border-radius: var(--josh-tint-radius-card);
  background: var(--josh-tint-step-2-bg);
  box-shadow: none;
}

.josh-page--tags .josh-stat-card__value {
  color: var(--josh-color-primary);
}

.josh-page--tags .josh-tag-cloud {
  padding: 1.5rem;
  border-radius: var(--josh-tint-radius-card);
  background: var(--josh-tint-step-1-bg);
}

.josh-page--tags .josh-tag-cloud__inner {
  justify-content: flex-start;
}

@media (max-width: 48rem) {
  .josh-page--tags .josh-tag-cloud {
    padding: 1.25rem;
  }
}
```

- [ ] **Step 2: Replace global cloud item opacity tiers with tint tiers (scoped)**

```css
.josh-page--tags .josh-tag-cloud__item {
  padding: 0.45rem 0.9rem;
  border-radius: var(--josh-tint-radius-pill);
  opacity: 1;
  background: var(--josh-tint-step-2-bg);
  color: var(--josh-color-text);
}

.josh-page--tags .josh-tag-cloud__item--1 { font-size: 0.875rem; background: var(--josh-tint-step-1-bg); color: var(--josh-tint-step-1-fg); }
.josh-page--tags .josh-tag-cloud__item--2 { font-size: 0.9375rem; background: var(--josh-tint-step-2-bg); }
.josh-page--tags .josh-tag-cloud__item--3 { font-size: 1rem; background: var(--josh-tint-step-3-bg); }
.josh-page--tags .josh-tag-cloud__item--4 { font-size: 1.125rem; background: var(--josh-tint-step-4-bg); }
.josh-page--tags .josh-tag-cloud__item--5 { font-size: 1.25rem; background: var(--josh-tint-step-5-bg); color: var(--josh-tint-step-5-fg); }

.josh-page--tags .josh-tag-cloud__item--4 .josh-tag-cloud__count,
.josh-page--tags .josh-tag-cloud__item--5 .josh-tag-cloud__count {
  padding: 0.1rem 0.4rem;
  border-radius: 999px;
  background: color-mix(in srgb, var(--josh-color-primary) 20%, transparent);
  color: var(--josh-color-primary);
}

.josh-page--tags .josh-tag-cloud__item:hover {
  filter: brightness(1.04);
  transform: translateY(-1px);
}

.josh-page--tags .josh-tag-cloud__item:focus-visible {
  outline: 2px solid var(--josh-color-primary);
  outline-offset: 2px;
}
```

- [ ] **Step 3: Dark outlines for coldest cloud shell + pills**

```css
html.josh-site.dark .josh-page--tags .josh-tag-cloud,
html.josh-site[data-color-mode="dark"] .josh-page--tags .josh-tag-cloud {
  outline: 1px solid color-mix(in srgb, var(--josh-color-cloud-300) 40%, var(--josh-color-gray-700));
}

html.josh-site.dark .josh-page--tags .josh-tag-cloud__item--1,
html.josh-site[data-color-mode="dark"] .josh-page--tags .josh-tag-cloud__item--1 {
  outline: 1px solid var(--josh-color-gray-700);
}
```

- [ ] **Step 4: Visual check**

Probe: cloud item `--5` background ≠ item `--1` background on `/tags` light mode.

---

### Task 3: `/tags` index — hot section + reveal fix

**Files:**
- Modify: `josh-components.css` — `.josh-page--tags .josh-tags-page__hot*` and reveal block (~2692)

- [ ] **Step 1: Hot title → section-label style**

```css
.josh-page--tags .josh-tags-page__hot-title {
  margin: 0 0 1rem;
  font-size: 0.8125rem;
  font-weight: var(--josh-font-weight-bold);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  line-height: 1.4;
  color: var(--josh-color-gray-700);
}
```

- [ ] **Step 2: Hot card nth-child tint ladder**

```css
.josh-page--tags .josh-tags-page__hot .josh-tag-card {
  min-height: 10.5rem;
  padding: 1.25rem 1.375rem;
  border-radius: var(--josh-tint-radius-card);
  box-shadow: var(--josh-tint-shadow-card);
  background: var(--josh-tint-step-3-bg);
}

.josh-page--tags .josh-tags-page__hot .josh-tag-card:nth-child(-n+2) {
  background: var(--josh-tint-step-5-bg);
}

.josh-page--tags .josh-tags-page__hot .josh-tag-card:nth-child(-n+2) .josh-tag-card__name {
  color: var(--josh-tint-step-5-fg);
  font-weight: var(--josh-font-weight-bold);
}

.josh-page--tags .josh-tags-page__hot .josh-tag-card:nth-child(n+3):nth-child(-n+4) {
  background: var(--josh-tint-step-4-bg);
}

.josh-page--tags .josh-tags-page__hot .josh-tag-card:nth-child(-n+4) .josh-tag-card__badge {
  background: color-mix(in srgb, var(--josh-color-primary) 24%, transparent);
  color: var(--josh-color-primary);
}

.josh-page--tags .josh-tags-page__hot .josh-tag-card:nth-child(n+5) .josh-tag-card__badge {
  background: color-mix(in srgb, var(--josh-color-cloud-400) 35%, transparent);
  color: var(--josh-color-gray-700);
}

.josh-page--tags .josh-tags-page__hot .josh-tag-card:hover {
  transform: translateY(-2px);
  filter: brightness(1.03);
}
```

- [ ] **Step 3: Reveal exception**

```css
.josh-page--tags.josh-reveal-ready .josh-tags-page__hot .josh-tag-card:not(.is-revealed) {
  opacity: 1;
  transform: none;
}
```

- [ ] **Step 4: Dark hot cards**

```css
html.josh-site.dark .josh-page--tags .josh-tags-page__hot .josh-tag-card,
html.josh-site[data-color-mode="dark"] .josh-page--tags .josh-tags-page__hot .josh-tag-card {
  box-shadow: none;
  outline: 1px solid var(--josh-color-gray-700);
}
```

- [ ] **Step 5: Assert hot cards visible without scroll**

```bash
node -e "const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1280,height:900}});await p.goto('http://localhost:3456/tags',{waitUntil:'networkidle'});const n=await p.evaluate(()=>[...document.querySelectorAll('.josh-tags-page__hot .josh-tag-card')].filter(c=>getComputedStyle(c).opacity==='1').length);console.log('visible',n,'/ 6');if(n<6)process.exit(1);await b.close();})();"
```

Expected: `visible 6 / 6`

---

### Task 4: `/tag/*` detail page

**Files:**
- Modify: `josh-components.css` — new `.josh-page--tag` block after tags block

- [ ] **Step 1: Header count primary**

```css
.josh-page--tag .josh-blog-archive__count {
  color: var(--josh-color-primary);
  font-weight: var(--josh-font-weight-medium);
}
```

- [ ] **Step 2: Blog card left accent**

```css
.josh-page--tag .josh-blog-card {
  border-left: 3px solid color-mix(in srgb, var(--josh-color-primary) 35%, transparent);
  transition: border-color 0.2s ease;
}

.josh-page--tag .josh-blog-card:hover {
  border-left-color: var(--josh-color-primary);
}
```

- [ ] **Step 3: Empty state tinted box + hide duplicate back**

```css
.josh-page--tag .josh-empty-state {
  max-width: 28rem;
  margin-inline: auto;
  padding: 2.5rem 1.5rem;
  border-radius: var(--josh-tint-radius-card);
  background: var(--josh-tint-step-2-bg);
}

.josh-page--tag .josh-empty-state .josh-archive-back {
  display: none;
}

.josh-page--tag .josh-empty-state__action {
  background: var(--josh-tint-step-3-bg);
}
```

- [ ] **Step 4: Related pills tier opacity (tag page scope)**

```css
.josh-page--tag .josh-pill--tag-hot .josh-pill__bg { opacity: 0.92; }
.josh-page--tag .josh-pill--tag-warm .josh-pill__bg { opacity: 0.85; }
.josh-page--tag .josh-pill--tag-more .josh-pill__bg { opacity: 0.72; }
```

- [ ] **Step 5: Verify `/tag/RAG` and empty tag**

```bash
node -e "const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();await p.goto('http://localhost:3456/tag/RAG',{waitUntil:'networkidle'});const c=await p.evaluate(()=>getComputedStyle(document.querySelector('.josh-blog-archive__count')).color);console.log('count',c);await b.close();})();"
```

---

### Task 5: Verification script + screenshots

**Files:**
- Create: `_verify-tags-tint.mjs`

- [ ] **Step 1: Write probe script** (checks tint steps, reveal, overflow, light/dark)

- [ ] **Step 2: Run full verification**

```bash
npx serve . -l 3456
node _verify-tags-tint.mjs
```

Expected: all assertions pass; save `_probe-tags-tint-light.png`, `_probe-tags-tint-dark.png`.

- [ ] **Step 3: Commit (if user requests)**

```bash
git add josh-components.css _verify-tags-tint.mjs docs/superpowers/
git commit -m "style: apply tint ladder visual system to tags pages"
```

---

## Spec coverage checklist

| Spec section | Task |
|--------------|------|
| Tint tokens light/dark | Task 1 |
| Stat grid | Task 2 |
| Tag cloud container + pills | Task 2 |
| Hot section title + cards | Task 3 |
| Reveal fix | Task 3 |
| Detail header count | Task 4 |
| Blog card accent | Task 4 |
| Empty state | Task 4 |
| Related pills | Task 4 |
| Responsive cloud padding | Task 2 |
| Verification | Task 5 |
