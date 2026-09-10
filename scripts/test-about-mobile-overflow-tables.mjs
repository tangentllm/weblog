/**
 * Batch B: about mobile no page-level x-scroll; magazine tables wrapped.
 * Run: node scripts/test-about-mobile-overflow-tables.mjs
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

const server = createServer((req, res) => {
  const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  let filePath = path.join(root, urlPath === '/' ? 'index.html' : urlPath);
  // SPA fallback for /about, /categories, /tags, /post/:slug, etc.
  if (
    !filePath.startsWith(root)
    || !existsSync(filePath)
    || (existsSync(filePath) && statSync(filePath).isDirectory())
  ) {
    filePath = path.join(root, 'index.html');
  }
  if (!existsSync(filePath)) {
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

const aboutCtx = await browser.newContext({ viewport: { width: 375, height: 812 } });
const aboutPage = await aboutCtx.newPage();
await aboutPage.goto(`http://127.0.0.1:${port}/about`, { waitUntil: 'networkidle', timeout: 60000 });
await aboutPage.waitForSelector('.josh-about-intro__cutout-img', { timeout: 15000 });
await aboutPage.waitForFunction(() => {
  const img = document.querySelector('.josh-about-intro__cutout-img');
  return img && img.complete && img.naturalWidth > 0;
}, { timeout: 15000 });
const about = await aboutPage.evaluate(() => ({
  bodySW: document.body.scrollWidth,
  docSW: document.documentElement.scrollWidth,
  client: document.documentElement.clientWidth,
  cutoutRight: Math.round(document.querySelector('.josh-about-intro__cutout')?.getBoundingClientRect().right || 0),
}));
assert.ok(
  about.docSW <= about.client + 2,
  `about page still scrolls horizontally: scrollWidth=${about.docSW} client=${about.client} cutoutRight=${about.cutoutRight}`,
);
await aboutCtx.close();

const postCtx = await browser.newContext({ viewport: { width: 375, height: 812 } });
const postPage = await postCtx.newPage();
await postPage.goto(`http://127.0.0.1:${port}/?view=post&slug=rag-eval-01-metrics-ragas`, {
  waitUntil: 'networkidle',
  timeout: 60000,
});
await postPage.waitForSelector('.josh-html-magazine');
const tables = await postPage.evaluate(() => {
  const all = [...document.querySelectorAll('.josh-html-magazine table')];
  return {
    count: all.length,
    unwrapped: all.filter((t) => !t.closest('.tbl-wrap')).length,
    pageOverflow: Math.max(document.body.scrollWidth, document.documentElement.scrollWidth)
      > document.documentElement.clientWidth + 2,
    wrapsHaveOverflow: [...document.querySelectorAll('.josh-html-magazine .tbl-wrap')].slice(0, 3).map((w) =>
      getComputedStyle(w).overflowX),
  };
});
assert.ok(tables.count > 0, 'expected magazine tables');
assert.equal(tables.unwrapped, 0, `unwrapped tables: ${tables.unwrapped}`);
assert.equal(tables.pageOverflow, false, 'magazine post page overflows x');
assert.ok(tables.wrapsHaveOverflow.every((v) => v === 'auto' || v === 'scroll'),
  `tbl-wrap overflow-x expected auto, got ${tables.wrapsHaveOverflow}`);
await postCtx.close();

await browser.close();
server.close();
console.log(JSON.stringify({ ok: true, about, tables }, null, 2));
