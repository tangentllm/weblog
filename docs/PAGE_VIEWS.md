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
