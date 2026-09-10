/**
 * Classify magazine SVG diagram tone for dark-mode paper-card / invert.
 * Run: node scripts/test-html-magazine-svg-tone.mjs
 */
import assert from 'node:assert/strict';

export function joshParseCssColorToRgb(raw) {
  const s = String(raw || '').trim().toLowerCase();
  if (!s || s === 'none' || s === 'transparent' || s.startsWith('url(')) return null;
  if (s === 'white') return [255, 255, 255];
  if (s === 'black') return [0, 0, 0];
  let m = s.match(/^#([0-9a-f]{3})$/i);
  if (m) {
    const h = m[1];
    return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)];
  }
  m = s.match(/^#([0-9a-f]{6})$/i);
  if (m) {
    const h = m[1];
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  m = s.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i);
  if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
  return null;
}

export function joshRelLuminance(rgb) {
  const [r, g, b] = rgb.map((c) => {
    const x = c / 255;
    return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * @param {{ fill: string, area?: number }[]} samples
 * @returns {'light'|'dark'}
 */
export function joshClassifySvgToneFromSamples(samples) {
  const scored = [];
  for (const sample of samples || []) {
    const rgb = joshParseCssColorToRgb(sample.fill);
    if (!rgb) continue;
    const L = joshRelLuminance(rgb);
    const area = Number(sample.area) > 0 ? Number(sample.area) : 1;
    scored.push({ L, area });
  }
  if (!scored.length) return 'light';

  // Prefer large background-like fills
  scored.sort((a, b) => b.area - a.area);
  const top = scored.slice(0, Math.min(3, scored.length));
  const weighted = top.reduce((sum, s) => sum + s.L * s.area, 0)
    / top.reduce((sum, s) => sum + s.area, 0);

  if (weighted >= 0.55) return 'light';
  if (weighted <= 0.35) return 'dark';

  // Mid tones: use mean of all samples
  const mean = scored.reduce((sum, s) => sum + s.L, 0) / scored.length;
  return mean >= 0.45 ? 'light' : 'dark';
}

/**
 * Lightweight markup probe (no DOM) for unit tests / Node.
 */
export function joshClassifySvgToneFromMarkup(svgMarkup) {
  const samples = [];
  const re = /<(rect|path|circle|ellipse|polygon)\b([^>]*)>/gi;
  let m;
  while ((m = re.exec(String(svgMarkup || ''))) !== null) {
    const attrs = m[2];
    const fillMatch = attrs.match(/\bfill\s*=\s*["']([^"']+)["']/i)
      || attrs.match(/\bstyle\s*=\s*["'][^"']*fill\s*:\s*([^;"']+)/i);
    if (!fillMatch) continue;
    const fill = fillMatch[1].trim();
    const w = Number((attrs.match(/\bwidth\s*=\s*["']?([\d.]+)/i) || [])[1]) || 0;
    const h = Number((attrs.match(/\bheight\s*=\s*["']?([\d.]+)/i) || [])[1]) || 0;
    const area = w > 0 && h > 0 ? w * h : (m[1].toLowerCase() === 'rect' ? 100 : 1);
    samples.push({ fill, area });
  }
  return joshClassifySvgToneFromSamples(samples);
}

{
  assert.equal(
    joshClassifySvgToneFromMarkup('<svg><rect width="780" height="220" fill="#161b22"/><text fill="#8b949e"/></svg>'),
    'dark',
  );
  assert.equal(
    joshClassifySvgToneFromMarkup('<svg><rect width="760" height="340" fill="#fff" stroke="#e4e0d6"/><rect width="200" height="100" fill="#f4f8fb"/></svg>'),
    'light',
  );
  assert.equal(
    joshClassifySvgToneFromMarkup('<svg><rect width="100" height="100" fill="#fdfcf9"/></svg>'),
    'light',
  );
  assert.equal(joshClassifySvgToneFromSamples([]), 'light');
  assert.deepEqual(joshParseCssColorToRgb('#abc'), [170, 187, 204]);
}

console.log('ok: joshClassifySvgTone');
