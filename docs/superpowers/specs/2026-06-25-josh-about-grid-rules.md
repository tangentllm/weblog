# Josh About Grid — 1:1 Fidelity Rules

**Date:** 2026-06-25  
**Reference:** [about-josh](https://www.joshwcomeau.com/about-josh/)  
**Goal:** Pixel-faithful card grid, interactions, and copy shell before any Tangentllm localization.

---

## 1) Card inventory (10 cards)

| Area | Role | Key interaction |
|------|------|-----------------|
| `map` | Origin + IP distance | Leaflet map, distance pill |
| `gay` | Pride flag | Click cycles flag stripes |
| `name` | Phonetic name | TTS “Comeau” |
| `tall` | Height joke | Toggle imperial / metric |
| `pkg` | OSS packages | Drum pads Q/W/E/R + sounds |
| `desk` | Voice coding desk photo | Static inset image |
| `side` | Generative machine | Toggle Power animation |
| `cat` | Cat allergy joke | Cosmetic pet click + sound |
| `conf` | Conference talks | Cover inset + talk list |
| `job` | Courses + stat | Count-up registrations |

HTML must set `--josh-about-area` to one of the names above. CSS `grid-template-areas` must use the **same names**.

---

## 2) Grid templates (canonical — from Josh `.w170x5fj`)

### Desktop — 8 columns, gap 16px

```css
grid-template-areas:
  "map  map  map  map  gay  gay  name name"
  "map  map  map  map  tall tall side side"
  "pkg  pkg  pkg  desk desk desk side side"
  "cat  cat  conf conf conf  job  job  job";
```

### Tablet — ≤51.875rem, 6 columns

```css
grid-template-areas:
  "map  map  map  map  gay  gay"
  "map  map  map  map  name name"
  "tall tall side side side side"
  "pkg  pkg  pkg  desk desk desk"
  "conf conf conf conf cat  cat"
  "job  job  job  job  job  job";
```

### Mobile — ≤39rem, 2 columns, gap 8px

```css
grid-template-areas:
  "map  map"
  "gay  name"
  "side side"
  "tall cat"
  "pkg  pkg"
  "desk desk"
  "conf conf"
  "job  job";
```

### Narrow — ≤24.5rem, 1 column

```css
grid-template-areas:
  "map"
  "gay"
  "name"
  "side"
  "tall"
  "cat"
  "pkg"
  "desk"
  "conf"
  "job";
```

---

## 3) Hard CSS Grid rules (why layouts break)

1. Every row in `grid-template-areas` must have the **same column count**.
2. Each named area must form a single **axis-aligned rectangle** (no L/T shapes).
3. If any rule is violated, the entire `grid-template-areas` declaration is **invalid**; the browser falls back to auto-placement → cards stack/overlap.
4. Do not use semicolons inside the quoted row list (only between rows in CSS syntax).

**Validation:** run `node _validate-about-grid-areas.mjs` after any grid edit.

---

## 4) Interaction binding checklist

| Selector | Card | Behavior |
|----------|------|----------|
| `#josh-about-flag-btn` | gay | Cycle `JOSH_ABOUT_PRIDE_FLAGS` |
| `#josh-about-name-btn` | name | `speechSynthesis` → “Comeau” |
| `#josh-about-arc-btn` | tall | Swap `6'2”` ↔ `188cm` |
| `.josh-about-drum` | pkg | Drum sound + key Q/W/E/R |
| `#josh-about-machine-btn` | side | Machine LED / gear animation |
| `#josh-about-cat-btn` | cat | Cosmetic click sound |
| `#josh-about-stat` | job | Count-up to `JOSH_ABOUT_COURSE_REGISTRATIONS` |
| `#josh-about-map-distance-inline` | map | Miles from `joshAboutFormatDistance` |

---

## 5) Files of record

| File | Responsibility |
|------|----------------|
| `josh-about.css` | Grid templates + card chrome |
| `josh-site.js` | Markup, constants, `initJoshAboutInteractions` |
| `_josh_about_css.css` | Scraped reference (do not load in prod) |

---

## 6) Non-goals (this spec)

- Chinese / Tangentllm copy (separate brand spec after 1:1 shell is verified)
- Hero cutout asset swap (keep Josh presenting cutout path for now)
