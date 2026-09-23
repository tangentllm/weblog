/**
 * Dark-mode contrast audit for newly published Multi-Agent + Agent Memory posts.
 * Run: node scripts/_smoke-new-posts-dark-audit.mjs
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
  'production-agent-11-agent-state-memory',
  'multi-agent-system-architecture',
  'multi-agent-communication-state',
  'multi-agent-reliability-engineering',
  'production-multi-agent',
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

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const { port } = server.address();
const browser = await chromium.launch();
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
    timeout: 90000,
  });

  const hasMag = await page.locator('.josh-html-magazine').count();
  if (!hasMag) {
    console.log(JSON.stringify({ slug, skip: 'no magazine' }));
    failed = true;
    await context.close();
    continue;
  }

  const audit = await page.evaluate(() => {
    const softSel = [
      '.card', '.callout', '.abstract', '.toc', '.stat', '.pull', '.lead-card',
      '.co-warn', '.co-info', '.co-good', '.case', '.blist li', 'figure', '.qa',
      '.mini', '.next-post', 'blockquote', '.tbl-wrap', '.hero', '.masthead',
    ].join(',');

    const samples = [];
    document.querySelectorAll(softSel).forEach((el) => {
      const bg = getComputedStyle(el).backgroundColor;
      const sample = el.querySelector('p, li, h2, h3, h4, .ttl, .lab, .t, span') || el;
      const fg = getComputedStyle(sample).color;
      samples.push({
        cls: String(el.className).slice(0, 48),
        bg,
        fg,
        text: (sample.textContent || '').trim().slice(0, 40),
      });
    });

    const bodyP = document.querySelector('.josh-html-magazine p');
    const hero = document.querySelector('.josh-html-magazine .hero, .josh-html-magazine .masthead, .josh-html-magazine header.hero');
    const h1 = document.querySelector('.josh-html-magazine h1, .josh-html-magazine h1.title');
    const thead = document.querySelector('.josh-html-magazine thead th');
    const inlineCode = document.querySelector('.josh-html-magazine p code, .josh-html-magazine li code');

    return {
      samples,
      bodyP: bodyP ? { fg: getComputedStyle(bodyP).color, bg: getComputedStyle(bodyP).backgroundColor, text: bodyP.textContent.trim().slice(0, 40) } : null,
      hero: hero ? { fg: getComputedStyle(hero).color, bg: getComputedStyle(hero).backgroundColor, bgImage: getComputedStyle(hero).backgroundImage } : null,
      h1: h1 ? { fg: getComputedStyle(h1).color, text: h1.textContent.trim().slice(0, 40) } : null,
      thead: thead ? { fg: getComputedStyle(thead).color, bg: getComputedStyle(thead).backgroundColor } : null,
      inlineCode: inlineCode ? { fg: getComputedStyle(inlineCode).color, bg: getComputedStyle(inlineCode).backgroundColor } : null,
    };
  });

  const bad = [];
  for (const item of audit.samples) {
    const bg = parseRgb(item.bg);
    const fg = parseRgb(item.fg);
    if (!bg || !fg) continue;
    const ratio = contrast(fg, bg);
    if (ratio < 3.5) {
      bad.push({ cls: item.cls, contrast: Number(ratio.toFixed(2)), text: item.text });
    }
  }

  const checks = [];
  if (audit.bodyP) {
    const fg = parseRgb(audit.bodyP.fg);
    if (fg && lum(fg) < 0.45) checks.push({ type: 'bodyP-too-dark', fg: audit.bodyP.fg });
  }
  if (audit.h1) {
    const fg = parseRgb(audit.h1.fg);
    if (fg && lum(fg) < 0.45) checks.push({ type: 'h1-too-dark', fg: audit.h1.fg, text: audit.h1.text });
  }
  if (audit.hero) {
    const bg = parseRgb(audit.hero.bg);
    const fg = parseRgb(audit.hero.fg);
    if (bg && fg && lum(bg) > 0.7 && lum(fg) > 0.7) {
      checks.push({ type: 'hero-light-on-light', bg: audit.hero.bg, fg: audit.hero.fg });
    }
  }
  if (audit.thead) {
    const bg = parseRgb(audit.thead.bg);
    const fg = parseRgb(audit.thead.fg);
    if (bg && fg && contrast(fg, bg) < 3.5) {
      checks.push({ type: 'thead-low-contrast', bg: audit.thead.bg, fg: audit.thead.fg });
    }
  }

  const row = {
    slug,
    washedBad: bad.length,
    washedSamples: bad.slice(0, 5),
    checks,
    ok: bad.length === 0 && checks.length === 0,
  };
  if (!row.ok) failed = true;
  console.log(JSON.stringify(row));
  await context.close();
}

await browser.close();
server.close();
if (failed) process.exit(1);
