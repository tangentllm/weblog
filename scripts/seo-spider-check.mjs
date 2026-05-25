/**
 * 抓取线上 HTML，检查爬虫可见性（无 JS 执行）。
 * 运行: node scripts/seo-spider-check.mjs
 */
const ORIGIN = 'https://tangentllm.github.io';
const BASE = '/weblog';

import { readFileSync, existsSync } from 'node:fs';

const URLS = [
  `${ORIGIN}${BASE}/`,
  `${ORIGIN}${BASE}/#/post/attention-from-scratch`,
  `${ORIGIN}${BASE}/post/attention-from-scratch/`,
];

const LOCAL_PRERENDER = 'post/attention-from-scratch/index.html';

function extractSignals(html) {
  const title = html.match(/<title[^>]*>([^<]*)<\/title>/i)?.[1]?.trim() || '';
  const desc = html.match(/<meta[^>]+name=["']description["'][^>]+content=["']([^"']*)["']/i)?.[1]
    || html.match(/<meta[^>]+content=["']([^"']*)["'][^>]+name=["']description["']/i)?.[1]
    || '';
  const canonical = html.match(/<link[^>]+rel=["']canonical["'][^>]+href=["']([^"']*)["']/i)?.[1]
    || html.match(/<link[^>]+href=["']([^"']*)["'][^>]+rel=["']canonical["']/i)?.[1]
    || '';
  const hasArticleBody = /<article[\s>]/i.test(html) || /<main[\s>]/i.test(html);
  const hasPostHeading = /attention|多头注意力/i.test(html);
  const jsonLdCount = (html.match(/application\/ld\+json/gi) || []).length;
  return { title, desc: desc.slice(0, 80), canonical, hasArticleBody, hasPostHeading, jsonLdCount };
}

async function checkUrl(url) {
  const r = await fetch(url, {
    headers: { 'User-Agent': 'TangentllmSEOCheck/1.0' },
    redirect: 'follow',
  });
  const html = await r.text();
  const signals = extractSignals(html);
  return { url, status: r.status, finalUrl: r.url, ...signals, htmlBytes: html.length };
}

async function main() {
  console.log('=== SEO 爬虫可见性自检（纯 HTML，不执行 JS）===\n');
  const results = [];
  for (const url of URLS) {
    try {
      results.push(await checkUrl(url));
    } catch (e) {
      results.push({ url, error: String(e.message || e) });
    }
  }
  console.table(results);

  const postHash = results.find((r) => r.url.includes('#/post/'));
  const postPath = results.find((r) => r.url.includes('/post/attention') && !r.url.includes('#'));

  if (existsSync(LOCAL_PRERENDER)) {
    const html = readFileSync(LOCAL_PRERENDER, 'utf-8');
    const local = { url: `file://${LOCAL_PRERENDER}`, status: 'local', ...extractSignals(html), htmlBytes: html.length };
    console.log('\n=== 本地预渲染页 ===\n');
    console.table([local]);
  }

  console.log('\n--- 结论 ---');
  if (postHash && !postHash.hasPostHeading) {
    console.log('Hash 文章 URL：首屏 HTML 通常不含文章标题（依赖 JS）→ 收录弱。');
  }
  if (postPath && postPath.hasPostHeading) {
    console.log('History/预渲染路径：首屏含文章信号 → 利于收录。');
  } else if (postPath) {
    console.log('History 路径已请求；若仍无正文，需确认 prerender 是否已部署。');
  }
  console.log('\n手动补充: Google Rich Results Test、百度搜索 site:tangentllm.github.io');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
