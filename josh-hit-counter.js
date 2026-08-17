/* Josh W. Comeau — 7-segment hit counter for post tail stats */

const JOSH_7SEG_ON = {
  0: 'abcdef',
  1: 'bc',
  2: 'abged',
  3: 'abgcd',
  4: 'fgbc',
  5: 'afgcd',
  6: 'afgcde',
  7: 'abc',
  8: 'abcdefg',
  9: 'abfgcd',
};

const JOSH_7SEG_PATHS = {
  a: 'M5 3.5h20a2 2 0 0 1 2 2v2.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5.5a2 2 0 0 1 2-2z',
  b: 'M23.5 5h2.5a2 2 0 0 1 2 2v15a2 2 0 0 1-2 2H23.5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z',
  c: 'M23.5 26h2.5a2 2 0 0 1 2 2v15a2 2 0 0 1-2 2H23.5a2 2 0 0 1-2-2V28a2 2 0 0 1 2-2z',
  d: 'M5 42.5h20a2 2 0 0 1 2 2v2.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-2.5a2 2 0 0 1 2-2z',
  e: 'M4 26h2.5a2 2 0 0 1 2 2v15a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V28a2 2 0 0 1 2-2z',
  f: 'M4 5h2.5a2 2 0 0 1 2 2v15a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z',
  g: 'M5 23.5h20a2 2 0 0 1 2 2v2.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-2.5a2 2 0 0 1 2-2z',
};

function joshHitCounterDigits(value, digits = 6) {
  const n = Math.max(0, Math.floor(Number(value) || 0));
  return String(n).padStart(digits, '0').slice(-digits);
}

function joshHitCounterDigitSvg(char, index) {
  const on = new Set((JOSH_7SEG_ON[char] || '').split(''));
  const paths = Object.entries(JOSH_7SEG_PATHS).map(([key, d]) => {
    const active = on.has(key);
    return `<path class="josh-hit-counter__seg${active ? ' josh-hit-counter__seg--on' : ''}" data-seg="${key}" d="${d}"></path>`;
  }).join('');
  return `<svg class="josh-hit-counter__digit" viewBox="0 0 30 48" width="30" height="48" aria-hidden="true" data-digit-index="${index}">${paths}</svg>`;
}

function joshHitCounterInnerHtml(value, digits = 6) {
  return joshHitCounterDigits(value, digits)
    .split('')
    .map((ch, i) => joshHitCounterDigitSvg(ch, i))
    .join('');
}

function updateJoshHitCounter(el, value) {
  if (!el) return;
  const digits = Number(el.dataset.digits) || 6;
  const shown = joshHitCounterDigits(value, digits);
  el.innerHTML = joshHitCounterInnerHtml(shown, digits);
  el.dataset.value = shown;
  el.setAttribute('aria-label', `${Number(shown)} hits`);
}

function initJoshHitCounters(scope = document) {
  scope.querySelectorAll('.josh-hit-counter').forEach((el) => {
    const initial = el.dataset.value != null ? el.dataset.value : '0';
    updateJoshHitCounter(el, initial);
  });
}

window.joshHitCounterInnerHtml = joshHitCounterInnerHtml;
window.updateJoshHitCounter = updateJoshHitCounter;
window.initJoshHitCounters = initJoshHitCounters;
