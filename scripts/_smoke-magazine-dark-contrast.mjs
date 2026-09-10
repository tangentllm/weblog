/**
 * Smoke: dark-mode contrast + magazine lightbox SVG size
 * Run: node scripts/_smoke-magazine-dark-contrast.mjs
 */
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFileSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.md': 'text/markdown; charset=utf-8',
};

const server = createServer((req, res) => {
  const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  const filePath = path.join(root, urlPath === '/' ? 'index.html' : urlPath);
  if (!filePath.startsWith(root) || !existsSync(filePath) || statSync(filePath).isDirectory()) {
    res.writeHead(404);
    res.end('missing');
    return;
  }
  res.writeHead(200, {
    'Content-Type': mime[path.extname(filePath)] || 'application/octet-stream',
    'Cache-Control': 'no-store',
  });
  res.end(readFileSync(filePath));
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const { port } = server.address();
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.addInitScript(() => {
  localStorage.setItem('josh-color-mode', 'dark');
  document.documentElement.classList.add('dark');
  document.documentElement.setAttribute('data-color-mode', 'dark');
});

function parseRgb(color) {
  const m = String(color || '').match(/[\d.]+/g);
  if (!m || m.length < 3) return null;
  const nums = m.slice(0, 3).map(Number);
  if (nums.every((n) => n <= 1)) return nums.map((n) => Math.round(n * 255));
  return nums.map((n) => Math.round(n));
}
function relL(rgb) {
  const [a, b, c] = rgb.map((v) => {
    const x = v / 255;
    return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * a + 0.7152 * b + 0.0722 * c;
}
function contrastRatio(fg, bg) {
  const L1 = relL(fg);
  const L2 = relL(bg);
  return (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
}

await page.goto(`http://127.0.0.1:${port}/?view=post&slug=rag-eval-03-regression-dataset`, {
  waitUntil: 'networkidle',
  timeout: 60000,
});
await page.waitForSelector('.josh-html-magazine');
await page.evaluate(() => {
  document.documentElement.classList.add('dark');
  document.documentElement.setAttribute('data-color-mode', 'dark');
});

const note = await page.evaluate(() => {
  const el = document.querySelector('.tech-note');
  const p = el?.querySelector('p');
  const lab = el?.querySelector('.tn-label');
  return {
    bg: el ? getComputedStyle(el).backgroundColor : null,
    pColor: p ? getComputedStyle(p).color : null,
    labColor: lab ? getComputedStyle(lab).color : null,
  };
});

const noteFg = parseRgb(note.pColor);
const noteBg = parseRgb(note.bg);
const noteContrast = noteFg && noteBg ? contrastRatio(noteFg, noteBg) : 0;
console.log(JSON.stringify({ note, noteContrast: Number(noteContrast.toFixed(2)) }));

await page.locator('.josh-magazine-fig').first().click();
await page.waitForSelector('.josh-magazine-lightbox');
const lb = await page.evaluate(() => {
  const svg = document.querySelector('.josh-magazine-lightbox__panel svg');
  const r = svg?.getBoundingClientRect();
  return {
    hasSvg: Boolean(svg),
    w: r?.width || 0,
    h: r?.height || 0,
  };
});
console.log(JSON.stringify({ lightbox: lb }));
await page.keyboard.press('Escape');

await page.goto(`http://127.0.0.1:${port}/?view=post&slug=rag-eval-04-eval-studio-release-gate`, {
  waitUntil: 'networkidle',
  timeout: 60000,
});
await page.waitForSelector('.josh-html-magazine');
await page.evaluate(() => {
  document.documentElement.classList.add('dark');
  document.documentElement.setAttribute('data-color-mode', 'dark');
});
const eval4 = await page.evaluate(() => {
  const pick = (sel) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const cs = getComputedStyle(el);
    let bgEl = el;
    let bg = cs.backgroundColor;
    while (bgEl && (!bg || /rgba?\(0,\s*0,\s*0,\s*0\)/.test(bg))) {
      bgEl = bgEl.parentElement;
      bg = bgEl ? getComputedStyle(bgEl).backgroundColor : '';
    }
    return { color: cs.color, bg };
  };
  return {
    h3no: pick('.h3-no'),
    loop: pick('.loop-final'),
    thead: pick('thead th'),
  };
});
const scores = {};
for (const [k, v] of Object.entries(eval4)) {
  const fg = parseRgb(v?.color);
  const bg = parseRgb(v?.bg);
  scores[k] = fg && bg ? Number(contrastRatio(fg, bg).toFixed(2)) : 0;
}
console.log(JSON.stringify({ scores }));

const fail = noteContrast < 4.5
  || lb.w < 200
  || lb.h < 100
  || scores.h3no < 3
  || scores.loop < 3
  || scores.thead < 3;

await browser.close();
server.close();
if (fail) {
  console.error('FAIL');
  process.exit(1);
}
console.log('PASS');
