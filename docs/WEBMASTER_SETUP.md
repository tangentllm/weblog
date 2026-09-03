# 站长平台配置指南

## 1. 填写验证码

编辑 [`index.html`](../index.html) 中 `siteMeta.siteVerification`：

```javascript
siteVerification: {
  baidu: 'codeva-xxxxxxxx',      // 百度搜索资源平台
  qihoo360: 'xxxxxxxx',          // 360 站长
  sogou: 'xxxxxxxx',             // 搜狗站长
},
```

留空则不会写入对应 meta 标签。部署后访问首页，用「查看源代码」确认 meta 已更新。

## 2. 提交 Sitemap

部署后 sitemap 地址：

`https://tangentllm.github.io/weblog/sitemap.xml`

在各站长平台「链接提交 / Sitemap」中添加上述 URL。

## 3. 构建时自动生成

CI 与本地构建顺序：

```bash
node scripts/generate-manifest.mjs
node scripts/generate-project-manifest.mjs
node scripts/generate-sitemap.mjs
node scripts/prerender-posts.mjs
# 404.html 由 deploy 工作流从 index.html 复制
```

## 4. Umami 分析

国家、来源、设备看 [Umami Cloud](https://cloud.umami.is)。Website ID 写在 `index.html` 的 `siteMeta.umamiWebsiteId`。域名填 `tangentllm.github.io`（不要带 `https://` 或 `/weblog`）。

部署后访问线上站点，再在 Umami 实时面板确认有数据。本地预览不会上报。

## 5. 验证收录

- Google：[Rich Results Test](https://search.google.com/test/rich-results) 测试 `https://tangentllm.github.io/weblog/post/attention-from-scratch/`
- 百度：搜索 `site:tangentllm.github.io`
