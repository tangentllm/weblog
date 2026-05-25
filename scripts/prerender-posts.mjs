/**
 * 为每篇文章生成可爬取的静态页：post/{slug}/index.html
 * 运行: node scripts/prerender-posts.mjs
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';

const ORIGIN = 'https://tangentllm.github.io';
const BASE = '/weblog';

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function stripFrontmatter(text) {
  return text.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, '').replace(/^\uFEFF/, '');
}

function markdownToPlainHtml(md) {
  let html = escapeHtml(stripFrontmatter(md));
  html = html
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
  return `<p>${html}</p>`;
}

function buildPage({ title, description, slug, date, bodyHtml, isHtmlArticle }) {
  const canonical = `${ORIGIN}${BASE}/post/${slug}/`;
  const appUrl = `${ORIGIN}${BASE}/?view=post&slug=${encodeURIComponent(slug)}`;
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'BlogPosting',
    headline: title,
    description,
    datePublished: date,
    inLanguage: 'zh-CN',
    author: { '@type': 'Person', name: 'tangentllm' },
    mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
    url: canonical,
  };

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(title)} - Tangentllm Notes</title>
  <meta name="description" content="${escapeHtml(description)}">
  <link rel="canonical" href="${canonical}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="${escapeHtml(title)}">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:url" content="${canonical}">
  <meta property="og:locale" content="zh_CN">
  <meta name="robots" content="index, follow">
  <script type="application/ld+json">${JSON.stringify(jsonLd)}</script>
  <style>
    body{font-family:system-ui,sans-serif;max-width:48rem;margin:2rem auto;padding:0 1rem;line-height:1.7;color:#222}
    a{color:#00a870}
    .meta{color:#666;font-size:.9rem}
    article{margin-top:1.5rem}
    code{background:#f4f4f4;padding:.1em .35em;border-radius:4px}
  </style>
</head>
<body>
  <p class="meta"><a href="${ORIGIN}${BASE}/">Tangentllm Notes</a> · ${escapeHtml(date)}</p>
  <h1>${escapeHtml(title)}</h1>
  <p>${escapeHtml(description)}</p>
  <p><a href="${appUrl}">在交互式阅读器中打开全文 →</a></p>
  <article>${isHtmlArticle ? '<p>本文为 HTML 长文，请使用上方链接进入完整版。</p>' : bodyHtml}</article>
  <script>location.replace(${JSON.stringify(appUrl)});</script>
</body>
</html>`;
}

function main() {
  const manifest = JSON.parse(readFileSync('content/posts/manifest.json', 'utf-8'));
  let count = 0;
  for (const post of manifest) {
    const dir = path.join('post', post.slug);
    mkdirSync(dir, { recursive: true });
    let bodyHtml = '';
    const isHtmlArticle = post.format === 'html' && post.htmlFile;
    if (!isHtmlArticle && post.file) {
      const mdPath = post.file.replace(/^\.\//, '');
      try {
        const md = readFileSync(mdPath, 'utf-8');
        const plain = stripFrontmatter(md);
        bodyHtml = markdownToPlainHtml(plain.slice(0, 12000));
      } catch {
        bodyHtml = '<p></p>';
      }
    }
    const html = buildPage({
      title: post.title,
      description: post.excerpt || post.title,
      slug: post.slug,
      date: post.date,
      bodyHtml,
      isHtmlArticle,
    });
    writeFileSync(path.join(dir, 'index.html'), html, 'utf-8');
    count += 1;
  }
  console.log(`Prerendered ${count} posts under post/{slug}/index.html`);
}

main();
