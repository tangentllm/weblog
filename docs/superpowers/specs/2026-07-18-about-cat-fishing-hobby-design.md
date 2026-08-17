# About Cat Card — Fishing Hobby Copy

**Date:** 2026-07-18  
**Route:** `/about` — cat card only (`josh-about-card--cat`)  
**Scope:** Replace cat-card body copy with fishing hobby persona  
**Approach:** Dual-line fixed copy (quiet + tactical); keep pet-the-cat interaction

---

## Goals

1. Surface a personal interest (钓鱼) that differentiates the brand from pure LLM résumé cards.
2. Blend quiet waiting (等漂) with tactical tinkering (看风/换饵/调漂 ≈ 调参).
3. Preserve Josh-style cat card structure and cosmetic pet interaction.

## Non-goals

- Changing other about cards, hero, grid layout, or CSS
- Replacing cat SVG / head image assets
- Adding click-to-cycle copy (rejected as heavier than needed)
- Swapping desk/conf images or adding fishing photos
- Git commit until explicitly requested

---

## Decisions (from brainstorm)

| Question | Choice |
|----------|--------|
| Brand gap to fill | **A** — interests / hobbies |
| Hobby | 钓鱼 |
| Card target | **A** — cat card only |
| Tone | **A + B** — quiet shore + tactical tuning |
| Copy pattern | Dual-line fixed (`p` + `p.small`) |

---

## Copy (approved)

**File:** `josh-site.js` → `joshAboutCatMarkup()`

| Slot | Before | After |
|------|--------|-------|
| Main (`p`) | 新 paper 出来，常会先复现一版——睡眠是后来的事。 | 周末常去岸边坐着——等漂的时候，脑子反而最干净。 |
| Sub (`p.small`) | 咖啡因是副产物，不是生产力神话。 | 看风向、换饵、调漂，跟调参差不多；上鱼是副产物。 |

**Tone notes:** Quiet first, tactical second. One light engineering echo (`调参` / `副产物`); no hard RAG/LLM jargon.

---

## Implementation

1. Edit the two paragraph strings inside `joshAboutCatMarkup()` only.
2. Leave ghost pet button, head image, main pet button, aria-labels, and CSS untouched.
3. Visual check `/about` cat card: layout unchanged; copy reads correctly in light/dark.

## Verification

- [ ] Cat card shows new main + small lines
- [ ] Pet-the-cat still works (sound/animation if enabled)
- [ ] No regression on neighboring cards or grid areas

---

## Out of scope follow-ups (optional later)

- desk / conf cards for scene photos or deeper life copy
- credo mix of life principles + engineering judgments
