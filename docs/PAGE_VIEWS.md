# 浏览量说明

## 展示数字 vs 真实 PV

- **页面显示**：`真实 PV + 稳定偏移（100~5000）`，偏移由 `slug` 哈希决定，同一篇文章不变。
- **Supabase `page_views.count`**：全站真实 PV（未注水）。

## 查看真实数据

```bash
node scripts/verify-page-views.mjs
```

或在 Supabase SQL Editor：

```sql
SELECT slug, kind, count, updated_at FROM public.page_views ORDER BY count DESC;
```

## 计数规则

- 进入文章/项目详情页时 +1（同浏览器会话内同一 slug 只计一次）。
- 列表页不增加 PV。
- 爬虫、禁用 JS、RPC 失败不计入远端；失败时回退到本机 `localStorage`。

## Supabase 保活

免费项目约 7 天无数据库活动会被 Supabase 自动暂停。仓库通过 GitHub Actions 每日调用只读 RPC `get_page_view`（slug `_keepalive`，**不增加浏览量**）：

- Workflow：`.github/workflows/supabase-keepalive.yml`（每天 03:00 UTC）
- 本地手动：`node scripts/supabase-keepalive.mjs`
- Actions 页可手动 **Run workflow** 立即保活

合并到 `main` 后，在 GitHub → Actions → **Supabase keepalive** 确认已启用 scheduled runs。

## Umami（国家 / 来源 / 设备）

文章真实 PV 仍看 Supabase。国家分布、来源、设备走 [Umami Cloud](https://cloud.umami.is)（Hobby 免费档，数据保留 6 个月）。

- Website ID 写在 `index.html` 的 `siteMeta.umamiWebsiteId`
- 仅在 `tangentllm.github.io` 上报（本地预览不计）
- 脚本自动监听 History API，SPA 换页会记 pageview

部署后打开**线上站点**（不要用 localhost），再在 Umami 看 **Views**（访问次数），不要只看 Visitors（独立访客）。首页一次打开就会打很多条 Supabase `get_page_view`，所以后台请求数会涨得很快，和 Umami 的 1 次浏览不是一回事。
