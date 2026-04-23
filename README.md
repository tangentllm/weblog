# Model Atlas | 大模型学习博客

一个单文件实现的技术博客站点，面向大模型学习笔记沉淀与公开分享。

## 特性

- 单个 `index.html`，内联 CSS/JS，支持 hash 路由无刷新切页
- 深色优先设计，支持亮暗主题切换并记忆用户选择
- 文章内容来自 `content/posts/*.md`，新增博文无需改 `index.html`
- Markdown 渲染 + 代码高亮 + KaTeX 数学公式
- 文章页支持 TOC 高亮、代码复制、图片放大、阅读进度条
- 分类/标签/作品集/关于页完整可用
- GitHub Pages 自动部署
- 文章支持 Markdown 一键提交流程（跳转 GitHub Issue / PR）

## 本地预览

请使用静态服务器启动（不要直接双击 `index.html`），例如：

```bash
python -m http.server 8000
```

然后访问 `http://localhost:8000`。

## 一键部署到 GitHub Pages

1. 将项目推送到 GitHub 仓库 `main` 分支。
2. 在仓库设置中进入 `Settings -> Pages`。
3. `Build and deployment` 选择 `GitHub Actions`。
4. 首次 push 后会自动触发 `.github/workflows/deploy.yml` 完成部署。

## 自动化写作流（无需改 HTML）

### 1) 新增文章

在 `content/posts/` 下新建一个 `.md` 文件，并写入 frontmatter：

```md
---
title: 你的文章标题
slug: your-post-slug
date: 2026-04-22
readTime: 12 分钟
category: 工程实践
tags: vLLM, Quantization
cover: https://picsum.photos/seed/your-post-slug/1000/500
excerpt: 一句话摘要
---

## 正文开始
```

### 2) 提交并推送

push 到 `main` 后，GitHub Actions 会自动执行：

- `node scripts/generate-manifest.mjs`
- 生成 `content/posts/manifest.json`
- 发布到 GitHub Pages

### 3) 自动生效

站点会优先读取 `content/posts/manifest.json` 和对应的 Markdown 文件，文章会自动出现在首页、分类、标签、搜索和详情页。

## 作品 Markdown 自动发布

在 `content/projects/` 下新建项目 `.md` 文件，frontmatter 示例：

```md
---
title: 你的项目名
slug: your-project-slug
subtitle: 一句话定位
status: 已上线
period: 2026 Q2
tags: RAG, Agent
summary: 项目摘要
cover: https://picsum.photos/seed/your-project-slug/1200/700
github: https://github.com/your/repo
demo: https://your-demo.com
docs: https://your-docs.com
---

## 详细分析
这里写架构、实现细节、效果对比、公式和代码块。
```

部署时会自动执行：

- `node scripts/generate-project-manifest.mjs`

并生成 `content/projects/manifest.json`，作品列表和详情会自动更新。

说明：作品详情页当前仅保留 `GitHub / 在线 Demo / 技术文档` 展示按钮，
不展示一键提交 Issue/PR 按钮。

## Markdown 一键提交 GitHub

文章详情页底部有两个按钮：

- `一键提交 Markdown 到 GitHub Issue`
- `一键发起 GitHub PR 草稿`

使用前请先修改 `index.html` 里的仓库地址：

```js
repoBase: "https://github.com/your-name/your-repo"
```

按钮会把当前文章的标题和 Markdown 正文自动拼接到 GitHub 新建页面 URL 中，实现快速提交。

## 后续替换为 CMS 的建议

- 先保留 `posts` 结构不变
- 将数据源替换成 API 或静态 JSON
- 保持 `renderByRoute()` 和页面渲染函数不变，迁移成本最低
