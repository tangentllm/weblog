/**
 * Hot-tag previews prefer unique titles; line-clamp is 2.
 * Run: node scripts/test-categories-tags-polish.mjs
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
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

await page.goto(`http://127.0.0.1:${port}/tags`, { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForSelector('.josh-tag-card--hot, .josh-tags-page__hot');
const tags = await page.evaluate(() => {
  const titles = [...document.querySelectorAll('.josh-tag-card--hot .josh-tag-card__preview-item')]
    .map((el) => el.textContent.trim())
    .filter(Boolean);
  const counts = new Map();
  for (const t of titles) counts.set(t, (counts.get(t) || 0) + 1);
  const dupes = [...counts.entries()].filter(([, n]) => n > 1);
  const clamp = getComputedStyle(document.querySelector('.josh-tag-card__preview-item')).webkitLineClamp;
  return { titleCount: titles.length, dupes, clamp };
});
assert.equal(tags.dupes.length, 0, `duplicate hot-tag previews: ${JSON.stringify(tags.dupes)}`);
assert.equal(String(tags.clamp), '2', `expected line-clamp 2, got ${tags.clamp}`);

await browser.close();
server.close();
console.log(JSON.stringify({ ok: true, tags }, null, 2));
