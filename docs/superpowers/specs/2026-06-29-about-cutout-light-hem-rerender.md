# About Cutout v2 — 浅下摆重渲 Prompt（仅资产，不改页面）

**Date:** 2026-06-29  
**原因:** 深灰一体衣 + Josh 暗云 `#262b31` 在腰～腿交界带同族融色；后处理 / CSS 已回滚。  
**解法:** 保留瓷感公仔身份，**云带高度（人物 42%～58%）必须是浅色衣料**，对齐 [Josh About](https://www.joshwcomeau.com/about-josh/)「亮部叠暗云」的层次逻辑。

---

## 设计变更（相对 v1 深灰一体）

| 区域 | v1（问题） | v2（定稿） |
|------|------------|------------|
| 胸～腹上（<40% 身高） | 深灰 `#3a3836` + Sticker | **不变** — 深灰釉 + Sticker |
| 腹～腰～大腿上（40%～58%） | 深灰延续 | **浅暖灰白釉** `#d8d0c8`～`#ece4dc`，明显亮于云 |
| 大腿下（>58%） | 深裤 | **中暖灰** `#a89890`～`#b8a8a0`，仍亮于 `#262b31` |
| 皮肤/脸/手 | 暖瓷感 | **不变** |
| 姿态 | 双手抬起讲解 | **不变** |

**关键:** 不是整件换白 T，而是 **上深下浅分区釉绘**（潮玩喷漆分色），胸口 Sticker 区保持品牌识别。

---

## 技术规格（与现站一致）

- 竖构图半身，腰～大腿中部，左侧留空给文案
- 导出透明 WebP：`782×1466`（2×）→ 显示 `391×733`
- 文件覆盖（抠图后）：
  - `content/assets/about-cutout-ceramic-dark.webp`
  - `content/assets/about-cutout-ceramic-light.webp`
- **不要**再跑 `_export-about-cutout-webp.mjs` 的衣物调色 / 描边 / halo（已证明越改越差）

---

## 中文 Prompt（复制用）

### 角色 + 服装分色（暗/亮共用）

```
3D 瓷感潮玩盲盒公仔，4～5 头身，成年东亚男性工程师自画像，30 岁。
块面雕塑短发，细框矩形眼镜，大眼睛露齿微笑，双手抬起掌心朝前讲解，左手黑色智能手表。

材质：半光泽陶瓷树脂皮肤 + 公仔一体彩绘釉衣（不是真实布料）。

【服装分色 — 重要】
- 胸腹上部（腰带以上）：oversize T 恤，深灰暖调陶瓷釉 #3a3836～#4a4744；
  胸前 Sticker 珐琅彩绘：粉星、黄笑脸、绿闪电、橙行星、白箭头、青心，白描边。
- 下摆与腰际（腰带到大腿根）：同一件 T 的下半段，改为浅暖灰白陶瓷釉 #d8d0c8～#ece4dc，
  明显比深灰胸区亮两个阶，像 Josh 站点 hero 里浅色衣料叠在暗色云上的效果。
- 裤区（大腿）：中暖灰釉 #a89890～#b8a8a0，比背景云更亮、更暖，不要深蓝灰。

腰际可有 subtle 瓷釉高光 rim（1～2px 暖白），不要描边线、不要外发光、不要 halo。

高品质 C4D/Octane 产品渲染，竖构图半身，腰到大腿中部，人物右侧构图、左侧留白。
```

### 暗色版

```
【角色 + 服装分色】

背景纯黑 #0d0f12（后期抠透明）。
主光左上方冷侧光，皮肤暖色；浅灰白下摆区在暗背景下仍清晰可读，亮度明显高于 #262b31 云层。
```

### 亮色版

```
【角色 + 服装分色】

背景 #f5f3ef（后期抠透明）。
warm 顶光 + 轻 rim，肤色略亮；浅下摆区为 #e8e0d8～#f2ebe4 暖白釉。
同一张脸、同一姿势、同一胸区深灰 Sticker，仅光照与肤色调暖。
```

### 负向

```
灰模、豆眼、写实照片、Pixar 真人皮、Q 版大头、低模、模糊、水印、
真实棉布褶皱、整件深色一体衣、腰际以下仍深灰、裤腿接近黑色、
外描边、halo、外发光、镜头光晕、云雾、脚底云、电路板、肥胖
```

---

## 英文 Prompt（备用）

```
3D ceramic vinyl-toy figurine, 4-5 head tall, East Asian male engineer age 30, stylized self-portrait.
Sculpted blocky hair, thin rectangular glasses, big friendly eyes, open smile, both hands raised palms forward presenting pose, black smartwatch on left wrist.

Semi-gloss ceramic skin, ONE-PIECE painted enamel outfit (not real fabric).

COLOR BLOCKING (critical):
- Upper chest and torso: dark warm gray ceramic glaze #3a3836-#4a4744, sticker decals with white enamel borders (pink star, yellow smiley, green bolt, orange planet, white arrow, cyan heart).
- Lower shirt hem and waist band (40-58% of figure height): light warm gray-white glaze #d8d0c8-#ece4dc, clearly TWO STEPS brighter than chest — like a light shirt over dark clouds on a hero banner.
- Thighs: medium warm gray glaze #a89890-#b8a8a0, warmer and brighter than #262b31 blue-gray.

Subtle warm ceramic specular on waist edge only. No outline stroke, no glow halo, no ground fog.

Vertical half-body framing, waist to mid-thigh, subject on the right third, empty space on left for typography.
Octane product render, clean silhouette for cutout.

Negative: photo-real, pixar skin, all-dark shirt, dark pants matching clouds, outline, halo, fog at feet, blurry, watermark
```

---

## 验收（导出 WebP 后目测）

在 `/about` 暗色主题、1440px 宽度：

1. 云带高度（人物腰际）看到的是 **浅衣料**，不是深灰裤腰  
2. 与 SVG 云 `#262b31` 并排时，腰区 **一眼可分**（暖 + 亮，不靠描边）  
3. 胸区 Sticker 深灰身份保留  
4. 无 halo / 描边 / 滤镜感  

---

## 接入（仅换文件）

1. 外部工具出图 → 抠透明 → `782×1466` WebP  
2. 覆盖 `content/assets/about-cutout-ceramic-dark.webp` 与 `-light.webp`  
3. 硬刷新 `/about` — **无需改** `josh-site.js` / `josh-about.css`
