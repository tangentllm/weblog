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
