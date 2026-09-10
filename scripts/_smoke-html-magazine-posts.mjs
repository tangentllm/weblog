/**
 * Smoke: HTML magazine posts embed with theme bridge (layout kept, chrome stripped).
 * Run: node scripts/_smoke-html-magazine-posts.mjs
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
  const ext = path.extname(filePath);
  res.writeHead(200, {
    'Content-Type': mime[ext] || 'application/octet-stream',
    'Cache-Control': 'no-store',
  });
  res.end(readFileSync(filePath));
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const { port } = server.address();
const browser = await chromium.launch();

const slugs = [
  'enterprise-rag-01-why-need-rag',
  'rag-master-01-simple-rag',
  'rag-eval-01-metrics-ragas',
];

function relLuminance(rgb) {
  const [r, g, b] = rgb.map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function parseRgb(color) {
  const m = String(color || '').match(/\d+/g);
  if (!m || m.length < 3) return null;
  return m.slice(0, 3).map(Number);
}

const results = [];
let failed = false;

for (const mode of ['light', 'dark']) {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.addInitScript((colorMode) => {
    try {
      localStorage.setItem('josh-color-mode', colorMode);
      localStorage.setItem('color-mode', colorMode);
      localStorage.setItem('theme', colorMode);
    } catch {}
    document.documentElement.classList.toggle('dark', colorMode === 'dark');
    document.documentElement.setAttribute('data-color-mode', colorMode);
  }, mode);

  for (const slug of slugs) {
    await page.goto(`http://127.0.0.1:${port}/?view=post&slug=${slug}`, {
      waitUntil: 'networkidle',
      timeout: 60000,
    });
    await page.waitForSelector('.josh-html-magazine', { timeout: 30000 });
    await page.evaluate((colorMode) => {
      document.documentElement.classList.toggle('dark', colorMode === 'dark');
      document.documentElement.setAttribute('data-color-mode', colorMode);
    }, mode);
    await page.waitForTimeout(200);

    const result = await page.evaluate(() => {
      const mag = document.querySelector('.josh-html-magazine');
      const sample = mag?.querySelector('p') || mag?.querySelector('li, td, .abstract');
      const color = sample ? getComputedStyle(sample).color : '';
      let bgEl = sample;
      let bg = 'rgba(0,0,0,0)';
      while (bgEl) {
        bg = getComputedStyle(bgEl).backgroundColor;
        if (bg && !/^rgba?\(0,\s*0,\s*0,\s*0\)$/.test(bg) && bg !== 'transparent') break;
        bgEl = bgEl.parentElement;
      }
      if (!bgEl || /^rgba?\(0,\s*0,\s*0,\s*0\)$/.test(bg)) {
        bg = getComputedStyle(document.body).backgroundColor
          || getComputedStyle(document.documentElement).backgroundColor;
      }
      return {
        hasMagazine: Boolean(mag),
        hasMasthead: Boolean(document.querySelector('.josh-html-magazine .masthead, .josh-html-magazine .hero, .josh-html-magazine header')),
        hasInlineToc: Boolean(document.querySelector('.josh-html-magazine .toc, .josh-html-magazine .toc-inline')),
        hasSidebarToc: Boolean(document.querySelector('.josh-post-toc')),
        proseLen: document.querySelector('.josh-prose')?.textContent?.length || 0,
        color,
        bg,
        modeAttr: document.documentElement.getAttribute('data-color-mode'),
      };
    });

    const fg = parseRgb(result.color);
    const bg = parseRgb(result.bg);
    let contrast = 0;
    if (fg && bg) {
      const L1 = relLuminance(fg);
      const L2 = relLuminance(bg);
      const lighter = Math.max(L1, L2);
      const darker = Math.min(L1, L2);
      contrast = (lighter + 0.05) / (darker + 0.05);
    }
    const contrastOk = contrast >= 3;
    const ok = result.hasMagazine
      && !result.hasMasthead
      && !result.hasInlineToc
      && result.hasSidebarToc
      && result.proseLen > 800
      && contrastOk;

    if (!ok) failed = true;
    const row = { mode, slug, ok, contrast: Number(contrast.toFixed(2)), contrastOk, ...result };
    results.push(row);
    console.log(JSON.stringify(row));
  }
  await context.close();
}

await browser.close();
server.close();
if (failed) process.exit(1);
