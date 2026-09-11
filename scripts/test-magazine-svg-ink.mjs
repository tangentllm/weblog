/**
 * Smoke: light-paper SVG label ink annotation + CSS hooks.
 * Run: node scripts/test-magazine-svg-ink.mjs
 */
import assert from 'node:assert/strict';
import fs from 'fs';

function joshParseCssColorToRgb(raw) {
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
  return null;
}

function joshRelLuminance(rgb) {
  const [r, g, b] = rgb.map((c) => {
    const x = c / 255;
    return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function joshReadSvgFill(el) {
  const attr = el.getAttribute('fill');
  if (attr) return attr;
  const style = el.getAttribute('style') || '';
  const m = style.match(/(?:^|;)\s*fill\s*:\s*([^;]+)/i);
  return m ? m[1].trim() : '';
}

function joshAnnotateLightSvgInk(root) {
  root.querySelectorAll('.josh-magazine-fig[data-josh-svg-tone="light"]').forEach((fig) => {
    fig.querySelectorAll('text, tspan').forEach((el) => {
      const fill = joshReadSvgFill(el);
      if (!fill || /^currentcolor$/i.test(fill) || /^var\(/i.test(fill)) {
        el.setAttribute('data-josh-svg-ink', 'dark');
        return;
      }
      const rgb = joshParseCssColorToRgb(fill);
      if (!rgb) return;
      const L = joshRelLuminance(rgb);
      if (L >= 0.72) el.setAttribute('data-josh-svg-ink', 'light');
    });
  });
}

const src = fs.readFileSync('josh-site.js', 'utf8');
const css = fs.readFileSync('josh-prose.css', 'utf8');
assert.match(src, /function joshAnnotateLightSvgInk/);
assert.match(src, /data-josh-svg-ink/);
assert.match(css, /data-josh-svg-ink='dark'/);
assert.match(css, /\.case-study/);
const softIs = css.slice(css.indexOf('Soft surfaces'));
const softBlock = softIs.slice(softIs.indexOf(':is('), softIs.indexOf(') {'));
assert.doesNotMatch(softBlock, /\.svg-container/);
assert.match(softBlock, /\.case-study/);

class El {
  constructor(tag, attrs = {}, children = []) {
    this.tagName = tag.toUpperCase();
    this.attrs = { ...attrs };
    this.children = children;
  }
  getAttribute(k) { return this.attrs[k] ?? null; }
  setAttribute(k, v) { this.attrs[k] = v; }
  querySelectorAll(sel) {
    if (sel.includes('josh-magazine-fig')) {
      const figs = [];
      const walk = (n) => {
        if (n.attrs?.class?.includes('josh-magazine-fig') && n.attrs['data-josh-svg-tone'] === 'light') {
          figs.push(n);
        }
        for (const c of n.children || []) walk(c);
      };
      walk(this);
      return figs;
    }
    if (sel === 'text, tspan') {
      const texts = [];
      const walk = (n) => {
        if (n.tagName === 'TEXT' || n.tagName === 'TSPAN') texts.push(n);
        for (const c of n.children || []) walk(c);
      };
      walk(this);
      return texts;
    }
    return [];
  }
}

const a = new El('text', { id: 'a' });
const b = new El('text', { id: 'b', fill: '#fff' });
const c = new El('text', { id: 'c', fill: '#C0392B' });
const d = new El('text', { id: 'd', class: 'svg-text' });
const e = new El('text', { id: 'e', fill: '#2c3e50' });
const svg = new El('svg', {}, [a, b, c, d, e]);
const fig = new El('div', { class: 'josh-magazine-fig', 'data-josh-svg-tone': 'light' }, [svg]);
const root = new El('div', {}, [fig]);

joshAnnotateLightSvgInk(root);
assert.equal(a.getAttribute('data-josh-svg-ink'), 'dark');
assert.equal(b.getAttribute('data-josh-svg-ink'), 'light');
assert.equal(c.getAttribute('data-josh-svg-ink'), null);
assert.equal(d.getAttribute('data-josh-svg-ink'), 'dark');
assert.equal(e.getAttribute('data-josh-svg-ink'), null);
console.log('ok: joshAnnotateLightSvgInk');
