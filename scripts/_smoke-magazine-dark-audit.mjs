/**
 * Broad dark-mode audit for HTML magazine posts.
 * Checks: washed callouts, lightbox SVG fill loss, missing soft-class remaps.
 * Run: node scripts/_smoke-magazine-dark-audit.mjs
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

const slugs = [
  'rag-eval-01-metrics-ragas',
  'rag-eval-02-llm-as-judge',
  'rag-eval-03-regression-dataset',
  'rag-eval-04-eval-studio-release-gate',
  'enterprise-rag-01-why-need-rag',
  'enterprise-rag-04-chunking',
  'rag-master-01-simple-rag',
  'rag-master-08-reranking',
];

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

function parseRgb(color) {
  const m = String(color || '').match(/[\d.]+/g);
  if (!m || m.length < 3) return null;
  const nums = m.slice(0, 3).map(Number);
  if (nums.every((n) => n <= 1)) return nums.map((n) => Math.round(n * 255));
  return nums.map((n) => Math.round(n));
}
function lum(rgb) {
  return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
}
function contrast(fg, bg) {
  const L1 = lum(fg);
  const L2 = lum(bg);
  return (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
}

const report = [];
let failed = false;

for (const slug of slugs) {
  const context = await browser.newContext();
  const page = await context.newPage({ viewport: { width: 1400, height: 900 } });
  await page.addInitScript(() => {
    localStorage.setItem('josh-color-mode', 'dark');
    document.documentElement.classList.add('dark');
    document.documentElement.setAttribute('data-color-mode', 'dark');
  });

  await page.goto(`http://127.0.0.1:${port}/?view=post&slug=${slug}`, {
    waitUntil: 'networkidle',
    timeout: 60000,
  });

  const hasMag = await page.locator('.josh-html-magazine').count();
  if (!hasMag) {
    report.push({ slug, skip: 'no magazine' });
    await context.close();
    continue;
  }

  await page.evaluate(() => {
    document.documentElement.classList.add('dark');
    document.documentElement.setAttribute('data-color-mode', 'dark');
  });

  const pageAudit = await page.evaluate(() => {
    const softSel = [
      '.tnote', '.tech-note', '.callout', '.card', '.abstract', '.note',
      '.insight', '.warn', '.danger', '.lead-quote', '.pain', '.qa-block',
    ].join(',');

    const washed = [];
    document.querySelectorAll(softSel).forEach((el) => {
      const bg = getComputedStyle(el).backgroundColor;
      const bgImage = getComputedStyle(el).backgroundImage;
      const sample = el.querySelector('p, li, .tn-label, .co-label') || el;
      const fg = getComputedStyle(sample).color;
      washed.push({
        cls: String(el.className).slice(0, 40),
        bg,
        bgImage,
        fg,
        text: (sample.textContent || '').trim().slice(0, 36),
      });
    });

    const h3 = document.querySelector('.h3-no, .sec-num, .sec-no');
    const thead = document.querySelector('thead th');
    return {
      washed,
      h3: h3 ? { color: getComputedStyle(h3).color, text: h3.textContent.trim() } : null,
      thead: thead ? {
        color: getComputedStyle(thead).color,
        bg: getComputedStyle(thead).backgroundColor,
      } : null,
      figs: document.querySelectorAll('.josh-magazine-fig').length,
      tables: document.querySelectorAll('.tbl-wrap').length,
    };
  });

  const washedBad = [];
  for (const item of pageAudit.washed) {
    const bg = parseRgb(item.bg);
    const fg = parseRgb(item.fg);
    if (!bg || !fg) continue;
    const ratio = contrast(fg, bg);
    const hasGradient = item.bgImage && item.bgImage !== 'none';
    if (ratio < 3.5 || (hasGradient && lum(bg) > 0.55 && lum(fg) > 0.55)) {
      washedBad.push({
        cls: item.cls,
        contrast: Number(ratio.toFixed(2)),
        bgL: Number(lum(bg).toFixed(2)),
        fgL: Number(lum(fg).toFixed(2)),
        gradient: hasGradient,
        text: item.text,
      });
    }
  }

  // Lightbox: check up to 2 figs for black-fill regression
  const lightboxIssues = [];
  const figCount = Math.min(pageAudit.figs, 2);
  for (let i = 0; i < figCount; i += 1) {
    await page.locator('.josh-magazine-fig').nth(i).click();
    await page.waitForSelector('.josh-magazine-lightbox', { timeout: 5000 });
    const lb = await page.evaluate(() => {
      const svg = document.querySelector('.josh-magazine-lightbox__panel svg');
      if (!svg) return { missing: true };
      const rects = [...svg.querySelectorAll('rect')].slice(0, 8).map((r) => getComputedStyle(r).fill);
      const size = svg.getBoundingClientRect();
      const host = document.querySelector('[data-josh-lightbox-host]');
      return {
        missing: false,
        w: size.width,
        h: size.height,
        hasHost: Boolean(host),
        rects,
        blackRects: rects.filter((f) => f === 'rgb(0, 0, 0)' || f === 'rgb(0,0,0)').length,
      };
    });
    if (lb.missing || lb.w < 200 || lb.h < 56 || !lb.hasHost || lb.blackRects >= 3) {
      lightboxIssues.push({ index: i, ...lb });
    }
    await page.keyboard.press('Escape');
    await page.waitForTimeout(80);
  }

  const row = {
    slug,
    softBlocks: pageAudit.washed.length,
    washedBad: washedBad.length,
    washedSamples: washedBad.slice(0, 3),
    figs: pageAudit.figs,
    tables: pageAudit.tables,
    lightboxIssues,
    ok: washedBad.length === 0 && lightboxIssues.length === 0,
  };
  if (!row.ok) failed = true;
  report.push(row);
  console.log(JSON.stringify(row));
  await context.close();
}

await browser.close();
server.close();

const summary = {
  checked: report.filter((r) => !r.skip).length,
  failed: report.filter((r) => r.ok === false).length,
  passed: report.filter((r) => r.ok === true).length,
};
console.log(JSON.stringify({ summary }));
if (failed) process.exit(1);
