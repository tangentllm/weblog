# About Cat Card Fishing Copy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the about-page cat card body copy with approved fishing-hobby lines (quiet + tactical).

**Architecture:** Single string swap inside existing `joshAboutCatMarkup()` in `josh-site.js`. No new components, CSS, assets, or interactions.

**Tech Stack:** Static SPA — `josh-site.js` markup helper; visual verify on `/about`.

**Spec:** `docs/superpowers/specs/2026-07-18-about-cat-fishing-hobby-design.md`

## Global Constraints

- Touch only the two `<p>` strings inside `joshAboutCatMarkup()`
- Keep pet-the-cat buttons, SVG, head image, aria-labels, and CSS unchanged
- Do not modify hero or other about cards
- Do not git commit unless the user explicitly asks

---

## File map

| File | Role |
|------|------|
| `josh-site.js` | `joshAboutCatMarkup()` — cat card HTML |
| `docs/superpowers/specs/2026-07-18-about-cat-fishing-hobby-design.md` | Approved copy (read-only reference) |

---

### Task 1: Swap cat card copy

**Files:**
- Modify: `josh-site.js` — `joshAboutCatMarkup()` (around lines 2621–2633)
- Test: manual visual check on `/about` (no automated test harness for this markup)

**Interfaces:**
- Consumes: none (hardcoded strings)
- Produces: same function signature `joshAboutCatMarkup()` → HTML string for cat card body

- [x] **Step 1: Confirm current strings**

Open `josh-site.js` and locate:

```javascript
<p>新 paper 出来，常会先复现一版——睡眠是后来的事。</p>
<p class="small">咖啡因是副产物，不是生产力神话。</p>
```

- [x] **Step 2: Apply approved copy**

Replace with:

```javascript
<p>周末常去岸边坐着——等漂的时候，脑子反而最干净。</p>
<p class="small">看风向、换饵、调漂，跟调参差不多；上鱼是副产物。</p>
```

Full function after edit:

```javascript
function joshAboutCatMarkup() {
  return `<button type="button" class="josh-about-cat-pet josh-about-cat-pet--ghost" tabindex="-1" aria-hidden="true">
    ${joshAboutCatTailSvg()}
  </button>
  <div class="josh-about-cat-body">
    <img class="josh-about-cat-head" src="https://www.joshwcomeau.com/images/star-cat-head.svg" alt="" width="150" height="150">
    <p>周末常去岸边坐着——等漂的时候，脑子反而最干净。</p>
    <p class="small">看风向、换饵、调漂，跟调参差不多；上鱼是副产物。</p>
  </div>
  <button type="button" class="josh-about-cat-pet" id="josh-about-cat-btn" aria-label="Illustration of a cat. Triggering this button pets the cat. This is a purely cosmetic effect.">
    ${joshAboutCatMainSvg()}
  </button>`;
}
```

Only the two paragraph texts change; leave all classes, ids, and SVG helpers as-is.

- [x] **Step 3: Verify in browser**

1. Open `/about` (local static server or existing preview).
2. Confirm cat card shows the new main + `.small` lines.
3. Click the cat pet control once — cosmetic pet effect still runs.
4. Spot-check neighboring cards (`name`, `side`) unchanged.

- [ ] **Step 4: Commit only if requested** — skipped; user asked not to commit

If the user asks to commit:

```bash
git add josh-site.js docs/superpowers/specs/2026-07-18-about-cat-fishing-hobby-design.md docs/superpowers/plans/2026-07-18-about-cat-fishing-hobby.md
git commit -m "$(cat <<'EOF'
Refine about cat card copy to fishing hobby persona.

EOF
)"
```

Otherwise leave working tree unstaged for the user.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Main fishing line | Task 1 Step 2 |
| Small tactical line | Task 1 Step 2 |
| Keep pet interaction | Task 1 Steps 2–3 (no markup/CSS change beyond text) |
| No other cards / hero | Global constraints |
| Manual verification | Task 1 Step 3 |
