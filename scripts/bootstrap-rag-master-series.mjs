import { promises as fs } from "node:fs";
import path from "node:path";

const postsDir = path.resolve("content/posts");

const SERIES = "从零到 RAG 大师";
const CATEGORY = "RAG 与检索";
const COVER = "./content/assets/posts/covers/rag.svg";
const TAGS = "RAG, 检索增强生成, 教程系列";

const ARTICLES = [
  {
    order: 0,
    html: "0_前言.html",
    slug: "rag-master-preface",
    title: "从零到 RAG 大师 · 前言：检索增强生成技术全景导读",
    subtitle: "21 篇系列教程全景地图与学习路线",
    readTime: "20 分钟",
    excerpt:
      "从 Naive RAG 到 Agentic RAG，系统梳理检索增强生成的技术全景、演进脉络与 21 篇系列教程学习路线。",
  },
  {
    order: 1,
    html: "1_基础rag_blog.html",
    slug: "rag-master-01-simple-rag",
    title: "从零到 RAG 大师（一）：Simple RAG 入门指南",
    subtitle: "最基础的检索增强生成管道",
    readTime: "18 分钟",
    excerpt: "从零搭建 Simple RAG：文档加载、切块、向量化、检索与生成，理解 RAG 的核心工作流。",
  },
  {
    order: 2,
    html: "2_语义分块_blog.html",
    slug: "rag-master-02-semantic-chunking",
    title: "从零到 RAG 大师（二）：语义分块",
    subtitle: "让检索更「懂」文本",
    readTime: "16 分钟",
    excerpt: "告别固定长度切分：用语义边界切块，提升检索精度与上下文连贯性。",
  },
  {
    order: 3,
    html: "3_切块大小选择_blog.html",
    slug: "rag-master-03-chunk-size",
    title: "从零到 RAG 大师（三）：切块大小选择",
    subtitle: "检索精度与上下文完整性的平衡",
    readTime: "15 分钟",
    excerpt: "切块大小如何影响召回与生成质量？实验对比与工程选型建议。",
  },
  {
    order: 4,
    html: "4_上下文增强检索_blog.html",
    slug: "rag-master-04-contextual-retrieval",
    title: "从零到 RAG 大师（四）：上下文增强检索 RAG",
    subtitle: "为每个 chunk 注入文档级上下文",
    readTime: "17 分钟",
    excerpt: "Anthropic 式 Contextual Retrieval：切块时附加上下文，显著降低检索失败率。",
  },
  {
    order: 5,
    html: "5_上下文片段标题提取_blog.html",
    slug: "rag-master-05-cch",
    title: "从零到 RAG 大师（五）：上下文片段标题提取 (CCH)",
    subtitle: "为每个片段生成描述性标题",
    readTime: "14 分钟",
    excerpt: "Contextual Chunk Headers：用 LLM 为 chunk 生成标题，增强可检索性与可解释性。",
  },
  {
    order: 6,
    html: "6_文档增强rag_blog.html",
    slug: "rag-master-06-doc-enhancement",
    title: "从零到 RAG 大师（六）：文档增强 RAG",
    subtitle: "通过问题生成进行文档增强",
    readTime: "18 分钟",
    excerpt: "用 LLM 为文档生成假设性问题，扩展索引维度，提升多样 query 的召回率。",
  },
  {
    order: 7,
    html: "7_查询转换_blog.html",
    slug: "rag-master-07-query-transformation",
    title: "从零到 RAG 大师（七）：查询转换 Query Transformation",
    subtitle: "Multi-Query、HyDE 与查询分解",
    readTime: "16 分钟",
    excerpt: "查询改写、多查询扩展与分解策略，弥合用户问题与文档表述之间的语义鸿沟。",
  },
  {
    order: 8,
    html: "8_重排序_blog.html",
    slug: "rag-master-08-reranking",
    title: "从零到 RAG 大师（八）：重排序（Reranking）",
    subtitle: "Cross-Encoder 精排提升 Top-K 质量",
    readTime: "15 分钟",
    excerpt: "Bi-Encoder 召回 + Cross-Encoder 精排：Reranker 原理、选型与生产部署要点。",
  },
  {
    order: 9,
    html: "9_相关段落提取_blog.html",
    slug: "rag-master-09-rse",
    title: "从零到 RAG 大师（九）：相关段落提取 (RSE)",
    subtitle: "从长文档中定位最相关段落",
    readTime: "14 分钟",
    excerpt: "Relevant Segment Extraction：在检索到的长文档中精确定位与 query 最相关的段落。",
  },
  {
    order: 10,
    html: "10_上下文压缩_blog.html",
    slug: "rag-master-10-contextual-compression",
    title: "从零到 RAG 大师（十）：上下文压缩 Contextual Compression",
    subtitle: "在有限窗口内塞入更多有效信息",
    readTime: "15 分钟",
    excerpt: "用 LLM 压缩检索结果，去除冗余，在 context window 内最大化有效信息量。",
  },
  {
    order: 11,
    html: "11_反馈回路_blog.html",
    slug: "rag-master-11-feedback-loop",
    title: "从零到 RAG 大师（十一）：反馈回路机制",
    subtitle: "让 RAG 越用越聪明",
    readTime: "16 分钟",
    excerpt: "用户反馈与检索质量闭环：如何收集信号并持续优化索引与检索策略。",
  },
  {
    order: 12,
    html: "12_自适应检索_blog.html",
    slug: "rag-master-12-adaptive-retrieval-intent",
    title: "从零到 RAG 大师（十二）：自适应检索——意图识别",
    subtitle: "按查询意图动态选择检索策略",
    readTime: "17 分钟",
    excerpt: "识别用户查询意图，在向量检索、关键词检索与混合策略间自适应切换。",
  },
  {
    order: 13,
    html: "13_自适应rag_blog.html",
    slug: "rag-master-13-self-rag",
    title: "从零到 RAG 大师（十三）：自适应 RAG（Self-RAG）",
    subtitle: "检索、生成与自我批判的一体化",
    readTime: "18 分钟",
    excerpt: "Self-RAG：模型自主决定何时检索、如何检索，并对生成结果进行自我批判与修正。",
  },
  {
    order: 14,
    html: "14_命题分块_blog.html",
    slug: "rag-master-14-proposition-chunking",
    title: "从零到 RAG 大师（十四）：命题分块",
    subtitle: "让每个原子事实独立可检索",
    readTime: "16 分钟",
    excerpt: "Proposition-based Chunking：将文档分解为独立可验证的原子命题，提升细粒度检索精度。",
  },
  {
    order: 15,
    html: "15_多模态rag_blog.html",
    slug: "rag-master-15-multimodal-rag",
    title: "从零到 RAG 大师（十五）：多模态 RAG 与图像描述",
    subtitle: "图文混合知识库的检索增强",
    readTime: "19 分钟",
    excerpt: "多模态 RAG 架构：图像描述生成、跨模态嵌入与图文联合检索的工程实践。",
  },
  {
    order: 16,
    html: "16_融合检索_blog.html",
    slug: "rag-master-16-fusion-retrieval",
    title: "从零到 RAG 大师（十六）：融合检索（Fusion Retrieval）",
    subtitle: "多路召回与 RRF 排名融合",
    readTime: "16 分钟",
    excerpt: "稠密 + 稀疏 + 多查询多路召回，用 RRF 等算法融合排名，兼顾语义与关键词匹配。",
  },
  {
    order: 17,
    html: "17_图rag_blog.html",
    slug: "rag-master-17-graph-rag",
    title: "从零到 RAG 大师（十七）：图增强型 RAG（Graph RAG）",
    subtitle: "知识图谱与社区摘要增强检索",
    readTime: "20 分钟",
    excerpt: "Graph RAG：从文档构建知识图谱，用社区摘要与图遍历增强多跳推理能力。",
  },
  {
    order: 18,
    html: "18_分层检索rag_blog.html",
    slug: "rag-master-18-hierarchical-rag",
    title: "从零到 RAG 大师（十八）：分层索引检索（Hierarchical RAG）",
    subtitle: "父子层级索引与两阶段检索",
    readTime: "17 分钟",
    excerpt: "Hierarchical RAG：构建父子层级索引，先粗筛文档再精确定位 chunk，兼顾效率与精度。",
  },
  {
    order: 19,
    html: "19_假设文档嵌入rag_blog.html",
    slug: "rag-master-19-hyde",
    title: "从零到 RAG 大师（十九）：假设文档嵌入 HyDE",
    subtitle: "用「假答案」弥合语义鸿沟",
    readTime: "15 分钟",
    excerpt: "Hypothetical Document Embeddings：用 LLM 生成假设答案再检索，提升 query-document 语义对齐。",
  },
  {
    order: 20,
    html: "20_动态纠正rag_blog.html",
    slug: "rag-master-20-crag",
    title: "从零到 RAG 大师（二十）：动态纠正 RAG（CRAG）",
    subtitle: "检索质量评估与动态纠正",
    readTime: "17 分钟",
    excerpt: "Corrective RAG：评估检索质量，低质量时触发网络搜索或知识库重写，动态纠正信息源。",
  },
  {
    order: 21,
    html: "21_强化学习增强rag_blog.html",
    slug: "rag-master-21-rl-rag",
    title: "从零到 RAG 大师（二十一）：强化学习增强 RAG",
    subtitle: "让 Agent 自主学习最优检索策略",
    readTime: "20 分钟",
    excerpt: "用强化学习训练 RAG Agent 自主决策检索时机、策略与工具调用，全系列收官篇。",
  },
];

function seriesDate(order) {
  const base = new Date("2025-01-01T12:00:00Z");
  base.setUTCDate(base.getUTCDate() + order);
  return base.toISOString().slice(0, 10);
}

function mdStub(article) {
  const mdName = `${article.slug}.md`;
  const htmlFile = `./content/posts/${article.html}`;
  return {
    mdName,
    content: `---
title: ${article.title}
slug: ${article.slug}
date: ${seriesDate(article.order)}
readTime: ${article.readTime}
category: ${CATEGORY}
tags: ${TAGS}
cover: ${COVER}
format: html
htmlFile: ${htmlFile}
series: ${SERIES}
seriesOrder: ${article.order}
subtitle: ${article.subtitle}
excerpt: ${article.excerpt}
---
`,
  };
}

async function main() {
  for (const article of ARTICLES) {
    const htmlPath = path.join(postsDir, article.html);
    try {
      await fs.access(htmlPath);
    } catch {
      console.warn(`Skip missing HTML: ${article.html}`);
      continue;
    }
    const { mdName, content } = mdStub(article);
    const mdPath = path.join(postsDir, mdName);
    await fs.writeFile(mdPath, content, "utf-8");
    console.log(`Wrote ${mdName}`);
  }
  console.log(`Done: ${ARTICLES.length} stubs`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
