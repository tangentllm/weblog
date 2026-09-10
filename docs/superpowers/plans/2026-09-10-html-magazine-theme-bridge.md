# HTML Magazine Theme Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Josh 壳内保真渲染杂志 HTML：剥页头壳、保留版式图文，表面/正文色跟明暗主题，accent 保留。

**Architecture:** `extractHtmlArticleProse` scope 原稿 CSS 到 `.josh-html-magazine`，注入 token 映射；`josh-prose.css` 暗色硬编码补丁避免抹掉 accent；还原 RAG Eval 四篇为 `format: html`。

**Tech Stack:** 静态 SPA（`josh-site.js` / `josh-prose.css` / Playwright smoke）、Node assert 单测。

**Spec:** `docs/superpowers/specs/2026-09-10-html-magazine-theme-bridge-design.md`

## Global Constraints

- 不改各篇 HTML 正文内容；不改 `josh-home.*`
- 不强制 `--accent` → Josh `#4242fa`
- 禁止对 magazine 子树无差别 `color: inherit !important`
- 用户未要求时不 git commit

---

### Task 1: Token 注入契约单测 + 实现

**Files:**
- Modify: `scripts/test-scope-html-article-css.mjs`（或新建 `scripts/test-html-magazine-theme-tokens.mjs`）
- Modify: `josh-site.js` — `extractHtmlArticleProse` isolationCss

**Interfaces:**
- Produces: isolation 块含 `--paper/--ink/--card/--line/--code-*` → Josh token；**不含**把 `--accent` 强制成 primary

- [ ] **Step 1:** 单测：isolation 字符串映射表面 token，且不覆盖 `--accent:`
- [ ] **Step 2:** 跑测确认当前失败或缺口
- [ ] **Step 3:** 补全 `isolationCss`（含常见别名如 `--bg`、`--text` 若需要）
- [ ] **Step 4:** 单测通过

### Task 2: 收敛暗色补丁（少误伤 accent）

**Files:**
- Modify: `josh-prose.css` — `.josh-html-magazine` dark bridge

- [ ] **Step 1:** 去掉/收窄整棵 `* { color: inherit !important }`；改为表面/正文选择器 + soft 底
- [ ] **Step 2:** soft 底（`.callout`/`.card`/table zebra 等）用 cloud mix；accent 链接/标签保留原稿色或 primary 仅作链接兜底
- [ ] **Step 3:** SVG invert 保留为可选兜底

### Task 3: 还原 RAG Eval 四篇为 HTML

**Files:**
- Move: `_archive/*.html` → `content/posts/`
- Modify: `rag-eval-01`…`04` `.md` frontmatter → `format: html` + `htmlFile`
- Run: `node scripts/generate-manifest.mjs`
- Modify: `scripts/_smoke-eval-md-posts.mjs` → 改为断言 **有** magazine（或改名为 `_smoke-html-magazine-posts.mjs`）

映射：

| slug | htmlFile |
|------|----------|
| rag-eval-01-metrics-ragas | `./content/posts/《RAG Evaluation：从“检索到了”到“回答正确”，Ragas 到底在评估什么？》.html` |
| rag-eval-02-llm-as-judge | `./content/posts/《RAG Evaluation 工程：从指标、Judge 到 Eval Studio 与生产质量门禁》.html` |
| rag-eval-03-regression-dataset | `./content/posts/《RAG Regression：如何建立一套真正有价值的 Evaluation Dataset？》.html` |
| rag-eval-04-eval-studio-release-gate | `./content/posts/《Eval Studio：把 RAG Evaluation 接入 CICD，建立真正的上线质量门禁》.html` |

- [ ] **Step 1:** 移回 HTML，改 frontmatter，重生 manifest
- [ ] **Step 2:** smoke：有 `.josh-html-magazine`、无 `.masthead`/文内 `.toc`、侧栏目录存在

### Task 4: 抽测验收

- [ ] 浏览器/Playwright：`enterprise-rag-01`、`rag-master-01`、`rag-eval-01` 浅色+暗色
- [ ] 确认无奶油纸孤岛、正文可读、accent 可辨、Shiki 不被盖掉
