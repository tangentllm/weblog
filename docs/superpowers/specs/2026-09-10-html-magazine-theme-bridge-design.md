# HTML 杂志稿 → Josh 主题桥接设计

**日期：** 2026-09-10  
**状态：** 已批准并实现  
**取代方向：** 不再以「杂志 HTML → Josh Markdown」为主路径（见 `2026-09-10-rag-eval-md-migration-design.md`，该方案对用户意图不适用）

## 问题

大量博文是独立排好的杂志 HTML（版面、配色、插图、表格、callout）。嵌入 Josh 壳后出现：

- 奶油纸 / 衬线 / 硬编码近黑字与 Josh 明暗主题冲突
- 或过度剥样式后版式塌掉、「看起来很差」

用户目标：**保留原稿版式与图文排布**，颜色跟站点主题走；不要迁成 Markdown。

## 决策摘要

| 项 | 选择 |
|----|------|
| 颜色策略 | **B**：布局/图文全留，表面与正文色跟明暗主题 |
| 页头壳 | **A**：去掉 masthead / cover / hero / 文内 TOC / next-up；保留正文块 |
| 落地做法 | **方案 1**：CSS scope + token 映射 + 暗色硬编码补丁 |
| Accent | **保留**各篇 `--accent*`；只适配 soft 底，不强制成 Josh `#4242fa` |
| RAG Eval 四篇 | **还原**为 `format: html`，不以 MD 为渲染源 |

## 架构

HTML 博文继续：`format: html` + `htmlFile` → `extractHtmlArticleProse`。

流水线：

1. 收集原稿 `<style>`，scope 到 `.josh-prose .josh-html-magazine`
2. 剥离 chrome：`header, nav, footer, .masthead, .hero, .cover, .next-box` 及 `.toc*`
3. 从 chrome 中 salvage 正文 keeper（如 `.abstract`、`.subtitle`、`.series-index`）再删壳
4. 去掉与站点标题重复的文内 H1
5. 注入主题 isolation：把杂志 token 映射到 Josh token；外层背景透明
6. 包一层 `<div class="josh-html-magazine">…</div>`
7. 暗色补丁放在 `josh-prose.css`（硬编码奶油/近黑/浅表条纹等）
8. 代码块继续走现有 Shiki 路径；magazine 内代码不强制 `color: inherit`

**不做：** iframe；按系列手写整套适配 CSS；批量 MD 迁移；改首页。

## 颜色映射

### 强制跟主题（表面 / 正文）

在 `.josh-html-magazine` 上覆盖（示例，实现时可扩展同义名）：

| 杂志 token | Josh 映射 |
|------------|-----------|
| `--paper` / `--paper-*` | 透明 / `background` 系 mix |
| `--ink` / `--ink-soft` / `--ink-mute` | `--josh-color-text` / gray-700 / gray-500 |
| `--card` | cloud-300 与 background 的 mix |
| `--line` / `--line-soft` / `--rule` | text 色低透明度 mix |
| `--code-bg` / `--code-ink` | Josh code / syntax token |

容器（`.page` / `.wrap` / `.paper` / `.container` / `main`）：`max-width:100%`，左右 padding 清零，背景透明，避免双栏宽。

### 保留系列个性（accent）

- `--accent`、`--accent-2`、`--accent-color` 等 **保持原稿色值**
- `--accent-soft`、`--warn-soft`、`--ok-soft` 等软底：暗色改为 `color-mix` 深表面，避免奶油高亮块
- 语义色（warn / ok / bad）保留色相，只改 soft 表面可读性

### 硬编码兜底（暗色）

对未走变量的字面量（如 `#fff`、`#fbfaf7`、`tbody tr:nth-child(even)` 浅绿条、近黑 `color`）：

- 在 `josh-prose.css` 用 scoped 规则覆盖表面/文字
- **避免**对整棵子树无差别 `color: inherit !important`（会抹掉 accent / 状态色 / Shiki）
- SVG：优先原稿；对比崩了再对 magazine 内 `svg` 做 `invert` 类兜底（可开关）

## 范围与文件

**改：**

- `josh-site.js` — extract / scope / token 注入 / chrome 选择器
- `josh-prose.css` — 暗色补丁收敛（可读优先，少误伤 accent）
- `josh-shiki.js` — 仅必要时微调 magazine 代码识别（已有则保持）
- RAG Eval 四篇 frontmatter + HTML 归位 + `manifest.json`
- smoke：断言存在 `.josh-html-magazine` 且关键对比可读（替代「无 magazine」的 MD smoke）

**还原 RAG Eval：**

1. `_archive/*.html` → `content/posts/`
2. 各 `rag-eval-0N-*.md` 恢复 `format: html` + `htmlFile`
3. 正文以 HTML 为准；已生成的 MD 长文 / `content/assets/posts/rag-eval/**` 可保留但不参与渲染
4. `node scripts/generate-manifest.mjs`

**不改：** 各系列 HTML 正文内容；`josh-home.*`；其它框架迁移。

## 验收

抽测 slug：`enterprise-rag-01-why-need-rag`、`rag-master-01-simple-rag`、`rag-eval-01-metrics-ragas`（还原后）。

| 检查项 | 期望 |
|--------|------|
| DOM | 有 `.josh-html-magazine`；无文内 `.toc` / `.masthead` |
| 壳 | Josh 标题 / 侧栏目录 / 上下篇正常 |
| 浅色 | 版式接近原稿；无双标题 |
| 暗色 | 无奶油纸孤岛；正文对比可读；callout / 表 / 图可读；accent 仍可辨 |
| 代码 | Shiki 块不被 magazine 文字色覆盖 |

## 风险

- 各稿 CSS 变量命名不一（`--accent` vs `--accent-color`）→ token 表需覆盖常见别名，漏网靠硬编码补丁
- 部分稿几乎全硬编码色 → 暗色补丁可能要迭代 1–2 轮抽测
- 还原 RAG Eval 后，未提交的 MD 迁移改动需理清：以 HTML 路径为准，避免 manifest 仍指向纯 MD

## 后续（非本轮）

若个别系列暗色仍差，再考虑该系列薄适配层；仍不作为默认批量 MD 化。
