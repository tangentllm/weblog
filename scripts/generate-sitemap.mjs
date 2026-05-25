/**
 * 从 manifest 生成无 hash 的 sitemap.xml
 * 运行: node scripts/generate-sitemap.mjs
 */
import { readFileSync, writeFileSync } from 'node:fs';

const ORIGIN = 'https://tangentllm.github.io';
const BASE = '/weblog';

function loc(path = '') {
  const p = path ? `${BASE}/${path.replace(/^\//, '')}` : `${BASE}/`;
  return `${ORIGIN}${p.endsWith('/') ? p : p + '/'}`;
}

function readManifest(file) {
  return JSON.parse(readFileSync(file, 'utf-8'));
}

function urlEntry(path, lastmod, priority, changefreq = 'monthly') {
  return `  <url>
    <loc>${loc(path)}</loc>
    <lastmod>${lastmod}</lastmod>
    <changefreq>${changefreq}</changefreq>
    <priority>${priority}</priority>
  </url>`;
}

function main() {
  const posts = readManifest('content/posts/manifest.json');
  const projects = readManifest('content/projects/manifest.json');
  const latestPostDate = posts.reduce((max, p) => (p.date > max ? p.date : max), '2026-01-01');
  const staticPages = [
    { path: '', lastmod: latestPostDate, priority: '1.0', changefreq: 'daily' },
    { path: 'categories', lastmod: latestPostDate, priority: '0.8', changefreq: 'weekly' },
    { path: 'tags', lastmod: latestPostDate, priority: '0.8', changefreq: 'weekly' },
    { path: 'projects', lastmod: latestPostDate, priority: '0.8', changefreq: 'weekly' },
    { path: 'about', lastmod: latestPostDate, priority: '0.6', changefreq: 'monthly' },
  ];

  const entries = [
    ...staticPages.map((p) => urlEntry(p.path, p.lastmod, p.priority, p.changefreq)),
    ...posts.map((p) => urlEntry(`post/${p.slug}`, p.date, '0.9')),
    ...projects.map((p) => urlEntry(`project/${p.slug}`, p.period?.slice(0, 10) || latestPostDate, '0.7')),
  ];

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml"
        xmlns:mobile="http://www.google.com/schemas/sitemap-mobile/1.0"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">

${entries.join('\n\n')}

</urlset>
`;

  writeFileSync('sitemap.xml', xml, 'utf-8');
  console.log(`sitemap.xml generated: ${entries.length} URLs`);
}

main();
