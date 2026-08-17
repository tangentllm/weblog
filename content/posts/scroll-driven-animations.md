---
title: 滚动驱动动画：深入理解 `animation-timeline` API
slug: scroll-driven-animations
date: 2026-04-28
readTime: 22 分钟
category: 基础原理
tags: CSS, Animation, Scroll, animation-timeline
cover: ./content/assets/posts/covers/scroll-driven-animations.svg
excerpt: 用原生 CSS 把关键帧动画绑定到滚动距离而非时间——从 view() 与 scroll() 入门，到 animation-range、联动时间线与 fill-mode 踩坑。
subtitle: 探索宏大的新 `animation-timeline` API
---

# 滚动驱动动画：深入理解 `animation-timeline` API

给网站加一点「滚动时动起来」的个性，是最划算的动效投入之一。过去这通常要手写 JavaScript 监听 `scroll`，而 **Animation Timeline** 系列 API 让浏览器原生支持：**把 `@keyframes` 的进度轴从「时间」换成「滚动距离」**。

如果你已经会用 CSS 关键帧动画，那你已经掌握了这套 API 80% 的语法——剩下的主要是理解 `animation-timeline`、`animation-range` 和几种时间线类型。

> **适用读者**
> 本文假设你已熟悉 CSS 基础与 `@keyframes` 关键帧动画。若对关键帧还不熟，建议先补一篇关键帧交互指南，再回来看滚动驱动部分。

> **浏览器支持**
> 现在就可以在生产环境做**渐进增强**级别的滚动动效。`animation-timeline` 在 caniuse 上约 **85%** 覆盖率，主流 Chromium 与 Safari 已支持；Firefox 实现完整但默认仍受 flag 控制，Nightly 已默认开启。有 polyfill 可用，但对 `timeline-scope` 等高级特性支持不完整——复杂联动时间线请做好降级。

---

## 核心概念：把时间轴换成滚动轴

CSS 关键帧动画的本质，是在两个样式状态之间平滑插值：

```css
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}

.elem {
  animation: fadeIn 1000ms;
}
```

上面这段代码在 **1000ms** 内从透明淡入。Animation Timeline API 的核心问题是：

**如果把 0%→100% 映射到元素穿过视口的进度，而不是时钟，会发生什么？**

```css
.elem {
  animation: fadeIn;
  animation-timeline: view();
}
```

`animation-timeline: view()` 表示：滚动条向下移动时，就像在「 scrub 」这段关键帧——元素在视口中的位置决定当前帧。

```playground
{
  "template": "vanilla",
  "files": {
    "/index.html": "<p>👇 在右侧结果区上下滚动 👇</p><div class=\"elem\"></div>",
    "/styles.css": "body { margin: 0; padding: 1.25rem; font-family: system-ui, sans-serif; background: #fff; color: #111; min-height: 220vh; }\np { margin: 0 0 1rem; }\n.elem { width: 100px; height: 100px; background: goldenrod; margin-top: 40vh; animation: fadeIn; animation-timeline: view(); }\n@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }",
    "/index.js": ""
  },
  "options": { "editorHeight": 360, "editorWidthPercentage": 50 }
}
```

> **注意：你的浏览器不支持 `animation-timeline`**
> 上方 Playground 依赖 `animation-timeline: view()`。若结果区方块没有随滚动淡入，说明当前浏览器尚未支持该属性。Firefox 可在 `about:config` 中启用 `layout.css.scroll-driven-animations.enabled`；也可查阅 [MDN：animation-timeline](https://developer.mozilla.org/en-US/docs/Web/CSS/animation-timeline)。

传统思路里，关键帧百分比默认绑定在 **duration** 上；而 `@keyframes` 本身并没有规定百分比必须代表时间——**任何能从 0% 走到 100% 的输入** 都可以作为驱动源。`view()` 只是把输入换成了元素在视口中的几何进度。

---

## 缓动曲线仍然有效

滚动驱动并不等于线性。`animation` 简写里的 easing 依然生效：

```css
.box {
  --super-ease-out: cubic-bezier(0.15, 0.75, 0.35, 1);
  animation: spin var(--super-ease-out);
  animation-timeline: view();
}
```

你仍然在用同一套 `@keyframes`，只是时间线换了来源。

> **考虑动效敏感**
> 网页动效对部分用户会引发眩晕等不适。应通过 `@media (prefers-reduced-motion: no-preference)` 包裹滚动动画，为系统开启「减少动态效果」的用户提供静态兜底。多元素视差同时运动尤其需要这道闸门。

---

## 动画范围：`animation-range`

默认情况下，`view()` 从元素**刚进入**视口顶边就开始计量，直到**完全离开**底边才结束。有时你希望动画在元素**完全进入视口后**才开始——这时用 `animation-range`：

```css
.elem {
  animation: fadeIn;
  animation-timeline: view();
  animation-range: cover;   /* 默认：从进入到离开 */
}

.elem--contain {
  animation-range: contain; /* 仅在元素完全处于视口内时计量 */
}
```

`cover` 与 `contain` 的区别，决定了动画是否会在用户「只看到元素一角」时就已经跑完大半。

> **可定制**
> `animation-range` 还支持 entry / exit 百分比与 `cover 20% cover 80%` 这类精细区间——适合「滑入到 60% 就停住」的编排。把它当成滚动版的 in/out 点即可。

---

## 填充模式踩坑

时间驱动动画里，`animation-fill-mode` 控制动画结束后是否保持最终帧。滚动驱动下行为更容易让人困惑：

> **踩坑：fill modes**
> 若关键帧在 100% 定义了 `opacity: 1`，但滚动进度还没到达终点，元素可能仍保持初始样式。检查 `animation-fill-mode: both` 是否必要，并确认 `animation-range` 的终点是否与你期望的「完全可见」时刻一致——两者不一致时，会出现「滚完了但样式弹回」的错觉。

---

## 滚动进度时间线：`scroll()`

`view()` 跟踪**元素自身**在视口中的位置。`scroll()` 则跟踪**某个滚动容器**的滚动偏移：

```css
.scroller {
  overflow: auto;
  scroll-timeline: --section;
}

.parallax-layer {
  animation: parallaxShift linear;
  animation-timeline: --section;
}
```

适合在固定高度面板内做内部视差，而不必监听 `window` scroll。

---

## 联动时间线：`view-timeline` 与 `timeline-scope`

有时你想让**子元素**订阅**另一个元素**的视口进度，而不是自己的：

```css
.content {
  view-timeline: --tracked-elem;
}

.badge {
  animation: reveal linear;
  animation-timeline: --tracked-elem;
}
```

`.content` 创建命名时间线 `--tracked-elem`，同树内的 `.badge` 可以「跟着那块内容的滚动进度」做动画。

> **注意：你的浏览器不支持 `timeline-scope`**
> 跨子树共享命名时间线时，常配合 `timeline-scope` 扩大可见范围。该属性覆盖率低于 `view()`，部署前请在目标浏览器实测。

> **工作原理**
> 在目标元素上写 `view-timeline: --name` 会创建一条 **view progress timeline**。其他元素通过 `animation-timeline: --name` 订阅同一条进度——语义上类似「滚动广播」。

> **踩坑提醒**
> `--name` **不是全局 CSS 变量**。它只能被**声明它的元素及其后代**引用。想把时间线「借」给兄弟节点，需要 `timeline-scope: --name` 显式提升作用域——这是联动时间线最容易写错的地方。

---

## 触发动画 vs 驱动动画

> **滚动触发动画**
> `animation-timeline` 驱动的是**连续 scrub**（滚动多少，动画进度多少）。若你只需要「滚到某处播放一次」的离散触发，应使用 **Scroll-driven Animations** 规范中的 `animation-trigger` 或继续用 Intersection Observer / JS。两者语义不同：前者是「跟手」，后者是「点火」。

---

## 小结

| 概念 | 作用 |
|------|------|
| `animation-timeline: view()` | 用元素在视口中的进度驱动关键帧 |
| `animation-timeline: scroll()` | 用滚动容器偏移驱动 |
| `view-timeline` + 命名 | 创建可订阅的视口进度时间线 |
| `animation-range` | 限定进度计量的起止区间 |
| `prefers-reduced-motion` | 无障碍降级 |

滚动驱动动画不是替代 JS 的全部滚动特效，而是把**大量跟手式 cosmetic 动效**收回到 CSS，减轻主线程滚动监听负担。从 `view()` + 简单 `fadeIn` 开始，在真机滚动测试通过后再叠 `animation-range` 与联动时间线——与 Josh Comeau 原文的建议一致：**不必等 100% 覆盖才做增强，但务必为不支持的浏览器准备静态兜底。**
