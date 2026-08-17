# About Hero — 3D 瓷感潮玩 Cutout 替换方案



**Date:** 2026-06-26  

**Route:** `/about` Hero 右侧 cutout  

**Status:** 已接入代码（概念 PNG 占位，待抠透明 WebP 替换）



---



## 目标



用 **3D 瓷感潮玩公仔风** 自画像，替换 Josh 真人 cutout（`about-presenting-cutout.webp`），保留现有槽位与布局，支持 **明/暗双资产** 随站点主题切换。



## 定稿风格（用户确认 2026-06-26）



- **方案：** R3 Sticker 散贴 + **深灰** 一体彩绘公仔

- **暗版概念：** `assets/about-cutout-ceramic-r3-sticker-deep-gray-base.png`

- **亮版概念：** `assets/about-cutout-ceramic-r3-sticker-deep-gray-light.png`



## 风格锚点



- **比例：** 4～5 头身，潮玩/盲盒公仔，非 Q 版豆眼、非 Pixar 真人皮

- **体型：** 造型饱满 = 立体层次丰富，**不是** 变胖

- **材质：** 半光泽瓷感皮肤（SSS）+ 块面雕塑发 + **衣服与身体同材质釉绘** + 硬塑料手表

- **表情：** 大眼神、露齿笑、讲解/展示态

- **姿态：** 双手抬起、掌心朝前

- **背景：** 生成时用纯色底，定稿 **透明 WebP** + 底部云由页面 SVG 承担



## 角色设定



| 项 | 规格 |

|----|------|

| 身份 | 用户 stylized 自画像，30 岁东亚男性工程师 |

| 眼镜 | 细框矩形，公仔配件感 |

| 发型 | 短发，束状雕塑发片，半光泽 |

| 服装 | **公仔一体彩绘**——宽松 oversize T 雕塑造型，**深灰暖调釉底** `#3a3836`～`#4a4744`；胸前 **Sticker 散贴**（粉星、黄笑脸、绿闪电、橙行星、白箭头、青心，白边珐琅漆），**不是真实布料** |

| 配件 | 黑色哑光智能手表 |



## 双资产



| 文件 | 光/底 | 用途 |

|------|-------|------|

| `content/assets/about-cutout-ceramic-dark.webp` | 纯黑底生成 → 透明；左上方冷侧光，暖肤 | dark mode |

| `content/assets/about-cutout-ceramic-light.webp` | `#f5f3ef` 底生成 → 透明；warm 顶光 + rim，肤略亮 | light mode |



**导出：** 源图约 780×1466 → **391×733** WebP，透明底，人物高约 85%，腰～大腿中部。



## 产出流程



1. 以定稿概念图为参考，外部工具精修（可选 img2img / 局部重绘）

2. 抠透明底、裁切 391×733、导出 WebP 至 `content/assets/`

3. 1440 / 390 目测 About hero

4. 代码接入双 `<img>` 主题切换

5. 归档 `about-presenting-cutout.webp`



## 代码接入（导出 WebP 后）



- `joshAboutCutoutMarkup()`：双 `<img>`，`--dark` / `--light` class

- CSS：复用 map mascot / job mascot 主题切换模式

- **不改** cutout 定位、动画、云层 DOM



## 非目标



- 不改 Hero 文案、10 卡 grid

- 不做 Three.js 实时 3D



---



## 中文生成 Prompt（定稿）



### 角色设定表（暗/亮共用）



```

3D 瓷感潮玩盲盒公仔，4～5 头身，成年东亚男性工程师自画像，30 岁。

造型饱满丰富：立体瓷感、块面发、光影圆润，正常体型不要变胖。

半光泽陶瓷树脂皮肤，次表面散射；块面雕塑短发；细框矩形眼镜。

大眼睛带高光，露齿友好微笑；双手抬起掌心朝前讲解姿态；左手黑色智能手表。



服装必须是公仔一体彩绘：oversize 宽松 T 恤为同材质雕塑造型 + 珐琅釉色喷漆，绝对不是真实棉布。

T 恤底色：深灰暖调陶瓷釉 #3a3836～#4a4744，哑光偏 satin。

胸前 Sticker 散贴图案（珐琅彩绘）：粉色五角星、黄色笑脸、绿色闪电、橙色带环行星、白色弯曲箭头、青色小心，均有白色描边。

图案布局 curated 不杂乱，年轻街头潮牌感，不要像素网格，不要电路板。



高品质 C4D/Octane 产品渲染。竖构图半身，腰到大腿中部，左侧留空给文字。

```



### 暗色版



```

【角色设定表】

背景纯黑 #0d0f12，底部仅少量柔和白色云形剪影（后期可裁掉改透明）。

主光左上方冷色侧光，皮肤暖色与深灰衣对比；半光泽瓷感高光在额头、鼻尖、发束。

```



### 亮色版



```

【角色设定表】

背景浅暖灰 #f5f3ef（后期抠透明）。

主光 warm 顶光 + 轻 rim，肤色略亮，陶感仍 semi-gloss。

同一张脸、同一姿势、同一深灰 Sticker 衣，仅光照与肤色调暖。

```



### 负向



```

灰模、单色陶、豆眼、写实照片、Pixar 真人皮、Q 版大头、低模、模糊、水印、

真实布料、棉质褶皱、衬衫、像素风、电路板、工程师符号、肥胖、大肚子、变胖

```



---



## 概念图索引



| 阶段 | 文件 |

|------|------|

| 头像参考 | `assets/c__Users_81917_..._image-a6d7616a....png` |

| **定稿·暗** | `assets/about-cutout-ceramic-r3-sticker-deep-gray-base.png` |

| **定稿·亮** | `assets/about-cutout-ceramic-r3-sticker-deep-gray-light.png` |

| 备选暖灰 | `assets/about-cutout-ceramic-r3-sticker-warm-gray-base.png` |


