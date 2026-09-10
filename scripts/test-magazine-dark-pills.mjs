/**
 * Batch A: dark-mode magazine pills/tags must not stay cream,
 * and semantic pill text must stay readable on soft surfaces.
 * Run: node scripts/test-magazine-dark-pills.mjs
 */
import assert from 'node:assert/strict';
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
  '.webp': 'image/webp',
  '.png': 'image/png',
  '.woff2': 'font/woff2',
};

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

function isCreamPaper(rgb) {
  if (!rgb) return false;
  const [r, g, b] = rgb;
  return r > 220 && g > 210 && b > 190 && r + g + b > 660;
}

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

async function openDark(slug) {
  const context = await browser.newContext();
  const page = await context.newPage({ viewport: { width: 1280, height: 900 } });
  await page.addInitScript(() => {
    localStorage.setItem('josh-color-mode', 'dark');
    document.documentElement.classList.add('dark');
    document.documentElement.setAttribute('data-color-mode', 'dark');
  });
  await page.goto(`http://127.0.0.1:${port}/?view=post&slug=${slug}`, {
    waitUntil: 'networkidle',
    timeout: 60000,
  });
  await page.evaluate(() => {
    document.documentElement.classList.add('dark');
    document.documentElement.setAttribute('data-color-mode', 'dark');
  });
  await page.waitForSelector('.josh-html-magazine');
  return { context, page };
}

const cases = [
  { slug: 'rag-eval-01-metrics-ragas', sel: '.layer-pill' },
  { slug: 'rag-eval-04-eval-studio-release-gate', sel: '.pill.neutral, .pill.pass' },
  { slug: 'enterprise-rag-01-why-need-rag', sel: '.tag.no, .tag.yes' },
];

const report = [];

for (const c of cases) {
  const { context, page } = await openDark(c.slug);
  const samples = await page.evaluate((sel) => {
    return [...document.querySelectorAll(sel)].slice(0, 12).map((el) => ({
      cls: String(el.className).slice(0, 48),
      text: el.textContent.trim().slice(0, 24),
      bg: getComputedStyle(el).backgroundColor,
      color: getComputedStyle(el).color,
    }));
  }, c.sel);
  assert.ok(samples.length > 0, `${c.slug}: expected matches for ${c.sel}`);

  for (const s of samples) {
    const bg = parseRgb(s.bg);
    const fg = parseRgb(s.color);
    const cream = isCreamPaper(bg);
    const ratio = fg && bg ? contrastRatio(fg, bg) : 0;
    report.push({ slug: c.slug, ...s, cream, ratio: Number(ratio.toFixed(2)) });
    assert.equal(cream, false, `${c.slug} ${s.cls} stayed cream: ${s.bg}`);
    assert.ok(ratio >= 3.5, `${c.slug} ${s.cls} contrast ${ratio.toFixed(2)} < 3.5 (${s.color} on ${s.bg})`);
  }
  await context.close();
}

await browser.close();
server.close();
console.log(JSON.stringify({ ok: true, checked: report.length, samples: report.slice(0, 8) }, null, 2));
