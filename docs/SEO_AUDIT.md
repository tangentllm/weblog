# SEO 自检结果

运行：`node scripts/seo-spider-check.mjs`

## 改造前（2026-05-25 线上）

| URL | 状态 | 爬虫可见文章标题 |
|-----|------|------------------|
| `/weblog/` | 200 | 仅首页默认 title |
| `/weblog/#/post/attention-from-scratch` | 200（实为首页） | 否 |
| `/weblog/post/attention-from-scratch/` | 404 | 否 |

结论：Hash 路由下爬虫几乎只能索引首页；路径 URL 需 `404.html` SPA 回退 + 预渲染静态页。

## 改造后（本仓库已实施，部署后生效）

- SPA 交互路由：`/weblog/?view=post&slug={slug}`（`404.html` 回退）
- 爬虫/分享 URL：`/weblog/post/{slug}/`（`post/{slug}/index.html` 预渲染）
- `sitemap.xml`：22 条无 hash URL（`node scripts/generate-sitemap.mjs`）
- 每篇文章页含 `BlogPosting` JSON-LD（预渲染 + SPA `updateMetaTags`）

本地复检预渲染页：

```bash
node scripts/seo-spider-check.mjs
```

## 站长验证

在 [`index.html`](../index.html) 的 `siteMeta.siteVerification` 填入各平台验证码后重新部署，再提交 sitemap：

- [百度搜索资源平台](https://ziyuan.baidu.com/)
- [Google Search Console](https://search.google.com/search-console)
- [Bing Webmaster](https://www.bing.com/webmasters)
- 360 / 搜狗站长（可选）

详见 [WEBMASTER_SETUP.md](./WEBMASTER_SETUP.md)。
