/* Josh W. Comeau homepage — hero & home grid (shared shell in josh-site.js) */

const JOSH_INITIAL_ARTICLE_COUNT = 12;

const JOSH_POPULAR_SLUGS = [
  'claude-code-best-practices',
  'transformer-in-depth',
  'attention-from-scratch',
  'rag-hybrid-retrieval-strategy',
  'mcp-kw-guide',
  'embedding-finetune-domain-rag',
  'rag-production-refactor',
  'llm-sft-note',
  'tokenization-guide',
  'everything-claude-code-zh-guide',
];

const JOSH_CLOUD_PATH_500 =
  'M2467 198C2478.93 198 2508.5 148.5 2692.3 167C2855.77 183.454 2890 275.92 2940.45 271C2978.5 267.29 3025.5 66.1073 3208.04 55.5002C3364.5 46.408 3407.37 123 3419.5 123.5C3431.63 124 3448.89 83.0002 3564.32 83.0002C3728 83.0002 3767.67 198.501 3779.08 198C3790.5 197.5 3808 45.0002 4044.68 45.0002C4238.5 45.0002 4245.32 120.5 4256.5 116.5C4267.69 112.5 4277 13.5002 4417.9 13.5002C4567 13.5002 4590.74 115.5 4608.5 116.5C4626.26 117.5 4640.5 13.5007 4795 13.5004C4946 13.5002 4954.43 76.5003 4970.51 76.5003C4986.6 76.5003 4983 8.5 5077 8.5C5147.13 8.5 5148.62 62.7657 5148.14 74.3437C5148.08 75.8075 5148 77.2344 5148 78.6994V360V361.5C5148 383.592 5130.09 401.5 5108 401.5H9C-13.0914 401.5 -31 383.592 -31 361.5V133.5V76.0021V76.0002C-31 75.9604 -30.9925 -7.80104e-05 24 0C103.747 0.000113126 132.617 67.9717 143.069 117.186C148.413 142.347 172.927 161.481 197.99 155.7L478.5 91C598.5 64.5 646 110.5 659 110.5C672 110.5 714 31 856 33.5C998 36 996.5 76 1008.5 73.5C1020.5 71 1014.28 28.0329 1174.5 31C1309.5 33.5 1298.5 110.5 1327.5 110.5C1366.31 110.5 1378.25 109.457 1388 110.5C1406.69 112.5 1429.5 27 1615 27C1743.74 27 1771.09 161.183 1855.16 167C1930.28 172.198 1914.5 85 2032.05 90.0002C2108.93 93.2702 2132.33 148 2146.16 148C2160 148 2184 81.6655 2318.08 102.5C2440.5 121.524 2455.07 198 2467 198Z';

const JOSH_CLOUD_PATH_300 =
  'M2617 234C2496.99 229.765 2429.72 276.108 2400.53 303.732C2388.43 315.177 2372.83 323.5 2356.18 323.5H2135.62C2111.05 323.5 2089.95 305.704 2082.79 282.198C2061.56 212.504 2001.53 78.3592 1852.75 71.0003C1691 63 1645 185 1622 186.5C1599 188 1587 88.5 1368.5 88.5C1211 88.5 1180 157.5 1158.4 161.5C1136.8 165.501 1074.33 111 931 129.5C787.671 148 789.676 214.5 770 214C750.324 213.5 736.5 129.5 535.029 142.5C416.863 150.125 382.163 211.07 373.669 260.166C368.141 292.123 343.421 323.5 310.99 323.5H280.024C249.079 323.5 225.052 295.503 224.331 264.567C222.732 195.98 200.305 92 79 92C17.4738 92 3.47982 128.37 0.653094 139.38C0.122368 141.447 0 143.571 0 145.705V398C0 412.36 11.6404 424 25.9998 424H5100C5127.61 424 5150 401.615 5150 374V365V181.851C5150 149.381 5119.54 125.514 5087.89 132.773C5054.67 140.392 5019.02 148.008 5011.31 147.5C4996.11 146.501 4966.41 99.9071 4859.43 95.5003C4731 90.2096 4684 213.5 4663 213.5H4531.84C4513.48 213.5 4496.63 203.435 4485.66 188.715C4451.8 143.286 4365.08 52.9127 4220.67 71.0003C4061 91.0002 4023.5 150.5 4006.5 150.5C3989.5 150.5 3925.6 96.5092 3797.5 100.5C3637 105.5 3599 235.5 3589 231.5C3563.12 221.148 3430.32 192.596 3405.38 180.145C3382.96 168.954 3354.61 161.5 3318.87 161.5C3175.43 161.5 3129.73 224 3116.87 224C3104 224 3073.62 179.5 2953.5 179.5C2782 179.5 2771.92 286 2756 284.5C2740.08 283 2721.1 237.674 2617 234Z';

const JOSH_CLOUD_PATH_HORIZON =
  'M2641.4 401.5C2613.31 399.999 2525.75 198 2121.01 198C1862 198 1840 264.5 1806.88 259.5C1773.77 254.499 1723.34 129.991 1562.17 136C1401 142.009 1366.58 313.5 1339 321C1311.42 328.5 1279 226.5 1034.79 234.5C802.99 242.093 724.297 318.5 697 313C669.703 307.5 681 75.9996 430.496 32.4996C214.304 -5.042 99.7464 183.937 60.6394 266.475C51.4353 285.9 27.9703 295.392 8.5729 286.129C-15.3473 274.705 -43 292.144 -43 318.652V429.5C-43 443.859 -31.3592 455.5 -16.9999 455.5H5103C5127.3 455.5 5147 435.8 5147 411.5V232.89C5147 226.643 5146.46 220.404 5144.55 214.457C5136.92 190.729 5108.7 128.5 5022.5 128.5C4881 128.5 4935 253.704 4838.83 249C4808.16 247.499 4757.27 55.5004 4535 59C4312.73 62.4996 4283.98 270.5 4250.5 268.5C4217.02 266.5 4197 199 4037.27 189.5C3834.76 177.455 3790.86 285 3753.5 279C3716.14 273 3652.96 98.8238 3377.5 153.5C3156.46 197.374 3191.5 387.48 3139.82 376.5C3118.64 371.999 3078.5 339 2948.03 339C2894.2 339 2890.37 330.676 2837.19 339C2708.5 359.141 2669.5 403 2641.4 401.5Z';

/* Article hero wave (s1trgvaz / w1cg7uq0 on joshwcomeau.com post pages) */
const JOSH_POST_CLOUD_PATH_500 =
  'M2970.83 162.311c-80.18 35.471-73.69 108.447-47.7 124.208 26 15.762 93.47 40.054 93.47 40.054h1305s57.5 19.5 57.5-51-94.97-66.296-94.97-66.296 40.27-77.869-52.89-120.793c-93.16-42.924-146.42 13.009-146.42 13.009s-5.54-62.255-95.87-73.305c-90.33-11.05-137.76 48.38-137.76 48.38s-4.64-52.293-92.06-70.127c-87.42-17.834-162.6 40.045-162.6 40.045S3565.05-8.443 3454.48 1.38c-110.56 9.822-137.31 62.814-137.31 62.814s-51.15-37.036-129.29-7.003c-78.15 30.034-77.58 104.275-77.58 104.275s-59.3-34.626-139.47.845zM330 57.505c-98 9-132 69.5-132 69.5l-88 225.5h2094.5v-.5s-11-77-135.5-89-147 17-159.5 16.5-46-34-131-43.5-154.5 29.5-165 27-32-50-134.5-58-142 20-152 17.5-20-41-100.5-65-162.5 5.5-174 0-5.5-59-110.5-100-166.5 11-181 6.5-40.5-40.5-151.5-40-134.5 53-151.5 54-30-29.5-128-20.5z';

const JOSH_POST_CLOUD_PATH_BG =
  'M2741.5 299.5c10-1.428 20.5-71 203.5-91.5s216 49 226.5 49 56.5-74 240-49 189 86 199 84.5 49-63.5 226-71 207 63 216 63 71.5-55 243-27.5 181.5 74 195.5 73.5-8.5-102 199-139 247.5 30.5 262 30 85.5-103.5 245-58.5 0 194 0 194H-92s-72-149.5 0-177.5S1 208 12 208s42.5-98.5 250.5-100.5S511 206.5 518 208s97-68 269-34.5 194 134 203 138 88.5-24 256.5-12 186 47.5 192.5 47.5 50.5-66 204.5-65.5 193 51 202 52 68.5-27.5 248-22 215 35.5 226.5 35.5c11.5.001 38-67.142 204-79 166-11.857 207.5 32.929 217.5 31.5z';

function joshCloudSvgMarkup() {
  return `<svg class="josh-sky__cloud-svg" width="5120" height="456" viewBox="0 0 5120 456" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path class="josh-sky__cloud-path--500" d="${JOSH_CLOUD_PATH_500}"></path>
    <path class="josh-sky__cloud-path--300" d="${JOSH_CLOUD_PATH_300}"></path>
  </svg>`;
}

function joshSkyHorizonCloudMarkup() {
  return `<svg class="josh-sky__cloud-svg" width="5120" height="456" viewBox="0 0 5120 456" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path class="josh-sky__cloud-path--horizon" d="${JOSH_CLOUD_PATH_HORIZON}"></path>
  </svg>`;
}

function joshPostCloudSvgMarkup() {
  return `<svg class="josh-post-hero__cloud-svg" width="5120" height="357" viewBox="0 0 5120 357" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path class="josh-post-hero__cloud-path--500" fill="var(--josh-color-cloud-500)" d="${JOSH_POST_CLOUD_PATH_500}"></path>
    <path class="josh-post-hero__cloud-path--bg" fill="var(--josh-color-background)" d="${JOSH_POST_CLOUD_PATH_BG}"></path>
  </svg>`;
}

function joshPostHeroWaveMarkup() {
  return `<div class="josh-post-hero__wave" aria-hidden="true">${joshPostCloudSvgMarkup()}</div>`;
}

/* About page hero wave (bw0i50u / m126aln0 / s1q6y8ki on joshwcomeau.com/about-josh/) */
const JOSH_ABOUT_WAVE_PATH_BACK =
  'M824.001 403L875.501 422H1999V63.1446C1977.5 77.3383 1961.18 93.6986 1958 93.5001C1950 93.0001 1946 25.5001 1846 11.5001C1746 -2.49994 1705.5 70.0001 1698.5 69.5001C1691.5 69.0001 1678 -0.499932 1559 0.500068C1440 1.50007 1439.5 117 1432.5 119.5C1425.5 122 1363.5 79.0001 1292 113.5C1220.5 148 1244.5 212 1237.5 216C1230.5 220 1191 172 1106.5 198.5C1022 225 1035 289 1024 291C1013 293 967.001 231.5 875.501 278C784.001 324.5 824.001 403 824.001 403Z';

const JOSH_ABOUT_WAVE_PATH_FRONT =
  'M925.5 340.5C1010.5 364.432 1005.5 408.5 1005.5 408.5L1065.5 451.5H0V281.079C15.5746 272.042 39.5753 264.56 70.5013 260.5C200.001 243.5 219.003 293.5 227.501 292C236 290.5 257.001 219.5 387.501 233C518.001 246.5 530.002 317.5 540.501 319C551 320.5 601.502 263 695.001 283.5C788.5 304 788.502 352.5 796.501 354.5C804.5 356.5 840.501 316.568 925.5 340.5Z';

const JOSH_ABOUT_WAVE_PATH_WHITE =
  'M1347 413C1365.5 413 1395.5 375 1538.5 375C1681.5 375 1717 404 1729 403.5C1741 403 1752.5 323.5 1914.5 322.5C1947.52 322.296 1975.45 324.357 1999 327.767V453.5H0V393.684C29.7494 380.632 86.3331 368.933 191.5 375C399.5 387 447.5 444 457 444C466.5 444 488 390.5 676 375C864 359.5 931.5 413 940 413C948.5 413 990 340.5 1155.5 346.5C1321 352.5 1328.5 413 1347 413Z';

const JOSH_ABOUT_WAVE_PATH_HORIZON =
  'M2262 93C2122.5 82.5987 2116 21.5 2096.5 21.5C2077 21.5 2070.5 77.5238 1920.5 93C1794.5 106 1786 62 1771.5 63.5C1757 65 1687 155.5 1580 142C1473 128.5 1446.5 90 1435 93C1423.5 96 1448.03 199.005 1340 214C1181.5 236 1155.5 142 1144 142C1132.5 142 1105.5 269 946.5 236C787.5 203 799 115 784 114.5C769 114 732.5 162 544 158C382 154.562 352.5 81 341 84.5C329.5 88 358 269 168 326C-22 383 -75.5 180 -75.5 180V0.5H5189.5L5193.5 46C5193.5 46 5200 94 5069.5 100.5C4939 107 4923.5 21.5 4906.5 21.5C4889.5 21.5 4870 35 4835 93.5C4800 152 4765.5 169.5 4643.5 173.5C4521.5 177.5 4436.5 69 4425.5 76.5C4414.5 84 4413.5 212 4235 222C4056.5 232 4045.5 92 4033.5 89C4021.5 86 3968.5 169.5 3823.5 172.5C3678.5 175.5 3573.5 104 3562.5 106.5C3551.5 109 3553.5 167.5 3396 201C3238.5 234.5 3171.5 168.5 3161 172.5C3150.5 176.5 3164 273 3076.5 294.5C2975.99 319.197 2935 228 2920 225.5C2905 223 2862 276.955 2749 245C2671.4 223.057 2672.5 149 2660.5 151.5C2648.5 154 2622.5 181.04 2548.5 158C2425 119.548 2427.5 53.5 2412 51C2396.5 48.5 2376 101.5 2262 93Z';

function joshAboutHeroWaveMarkup() {
  return `<svg class="josh-about-sky__wave-dark" viewBox="0 0 1999 454" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path class="josh-about-sky__wave-path--back" d="${JOSH_ABOUT_WAVE_PATH_BACK}"></path>
    <path class="josh-about-sky__wave-path--front" d="${JOSH_ABOUT_WAVE_PATH_FRONT}"></path>
  </svg>`;
}

function joshAboutBodyWaveMarkup() {
  return `<svg class="josh-about-sky__wave-white" viewBox="0 0 1999 454" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path class="josh-about-sky__wave-path--white" fill="var(--josh-color-background)" d="${JOSH_ABOUT_WAVE_PATH_WHITE}"></path>
  </svg>`;
}

function joshAboutEndBandWaveMarkup() {
  return `<div class="josh-about-end-band__wave-wrap" aria-hidden="true">
    <svg class="josh-about-end-band__wave" viewBox="0 0 1999 454" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
      <path class="josh-about-sky__wave-path--back" d="${JOSH_ABOUT_WAVE_PATH_BACK}"></path>
      <path class="josh-about-sky__wave-path--front" d="${JOSH_ABOUT_WAVE_PATH_FRONT}"></path>
    </svg>
  </div>`;
}

function joshPopularMarkup(allPosts) {
  const bySlug = new Map(allPosts.map((post) => [post.slug, post]));
  const popular = JOSH_POPULAR_SLUGS.map((slug) => bySlug.get(slug)).filter(Boolean);
  return popular.map((post) => `
    <li>
      <a class="josh-popular__link" href="${Routes.post(post.slug)}">
        <svg class="josh-popular__arrow" width="1.25rem" height="1.25rem" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="5" y1="12" x2="18" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
        <span class="josh-popular__label">${post.title}</span>
      </a>
    </li>
  `).join('');
}

function buildJoshHomeStickyMarkup(homeHref) {
  return `<div class="josh-sky__sticky" id="josh-sky-sticky" data-is-over-threshold="false">
    <div class="josh-sky__blur" aria-hidden="true"></div>
    <div class="josh-sky__header-wrap josh-container">
      <header class="josh-header">
        ${joshLogoMarkup(homeHref)}
        <nav class="josh-nav" aria-label="主导航">
          <ul class="josh-nav__list">
            ${joshNavLinks(Routes.home())}
          </ul>
        </nav>
        <div class="josh-header__actions">
          ${joshHeaderUtilityActionsMarkup()}
        </div>
        <button type="button" class="josh-mobile-toggle" id="josh-mobile-toggle" aria-expanded="false" aria-controls="josh-mobile-menu" aria-label="打开菜单">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>
      </header>
    </div>
    <div class="josh-mobile-menu" id="josh-mobile-menu" hidden>
      ${joshMobileNavLinks(Routes.home())}
    </div>
  </div>`;
}

function buildJoshHomeHeroMarkup() {
  return `<div class="josh-hero">
    <div class="josh-sky" id="josh-sky">
      <div class="josh-sky__clouds-back">
        ${joshCloudSvgMarkup()}
      </div>
      <div class="josh-sky__mascot-ground">
        <div class="josh-sky__clouds-horizon" aria-hidden="true">
          <svg class="josh-sky__cloud-svg" width="5120" height="456" viewBox="0 0 5120 456" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
            <path class="josh-sky__cloud-path--horizon" d="${JOSH_CLOUD_PATH_HORIZON}"></path>
          </svg>
        </div>
        <div class="josh-sky__hero-arcs" aria-hidden="true">
          <div class="josh-sky__hero-arcs-wrap">
            <canvas class="josh-sky__hero-arcs-canvas" width="1000" height="520"></canvas>
          </div>
        </div>
        <div class="josh-sky__mascot-lane">
          ${joshMascotMarkup()}
        </div>
      </div>
    </div>
  </div>`;
}

function buildJoshHomeHTML() {
  const homeHref = Routes.home();
  const sorted = [...posts].sort((a, b) => new Date(b.date) - new Date(a.date));
  const visible = sorted.slice(0, JOSH_INITIAL_ARTICLE_COUNT);
  const hidden = sorted.slice(JOSH_INITIAL_ARTICLE_COUNT);
  const catsWithPosts = joshCategoriesWithPosts();

  return `<div class="josh-page">
    <div class="josh-home-shell">
      ${buildJoshHomeStickyMarkup(homeHref)}
      ${buildJoshHomeHeroMarkup()}
      <main class="josh-main josh-container">
      <div class="josh-blocker" id="josh-blocker" data-is-stuck="false" aria-hidden="true"></div>
      <section class="josh-articles" aria-labelledby="josh-articles-heading">
        <h2 class="josh-section-label" id="josh-articles-heading">文章与教程</h2>
        <div class="josh-articles__list" id="josh-articles-list">
          ${visible.map((p) => joshArticleMarkup(p, false)).join('')}
          ${hidden.map((p) => joshArticleMarkup(p, true)).join('')}
        </div>
        ${hidden.length > 0 ? `
        <div class="josh-show-more-wrap">
          <button type="button" class="josh-show-more" id="josh-show-more">
            <span class="josh-show-more__icon" aria-hidden="true">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="m19 12-7 7-7-7"/></svg>
            </span>
            <span class="josh-show-more__label">显示更多</span>
          </button>
        </div>` : ''}
      </section>
      <section class="josh-categories" aria-labelledby="josh-categories-heading">
        <h2 class="josh-section-label" id="josh-categories-heading">按分类浏览</h2>
        <div class="josh-pills">
          ${catsWithPosts.map(joshHomeCategoryPillMarkup).join('')}
        </div>
      </section>
      <section class="josh-popular" aria-labelledby="josh-popular-heading">
        <h2 class="josh-section-label" id="josh-popular-heading">热门内容</h2>
        <ol class="josh-popular__list">
          ${joshPopularMarkup(posts)}
        </ol>
      </section>
      </main>
    </div>
    ${buildJoshFooterMarkup()}
  </div>`;
}

function renderJoshHome(app) {
  syncJoshSiteClass(true);
  updateMetaTags({
    title: 'Tangentllm Notes',
    description: '专注于大模型技术的学习笔记、实践经验与项目分享。涵盖 Transformer、RAG、Agent、微调等核心技术。',
    keywords: '大模型,LLM,Transformer,RAG,Agent,微调,SFT,LoRA,Prompt Engineering',
    url: absolutePageUrl(),
    type: 'website',
  });
  app.innerHTML = buildJoshHomeHTML();
  queueMicrotask(() => initJoshHomeInteractions(app));
}

let joshHomeCleanup = null;

const JOSH_NAV_SCROLL_RAMP_PX = 80;

const JOSH_HERO_ARCS_COLORS = {
  light: [
    [340, 100, 50], [340, 100, 50], [310, 100, 40], [310, 100, 40],
    [270, 100, 40], [270, 100, 40], [240, 100, 30], [240, 100, 30],
    [230, 100, 20], [230, 100, 20],
  ],
  dark: [
    [350, 100, 55], [350, 100, 55], [50, 100, 50], [50, 100, 50],
    [150, 100, 50], [150, 100, 50], [240, 100, 70], [240, 100, 70],
    [270, 100, 80], [270, 100, 80],
  ],
};

const JOSH_HERO_ARCS_BASE = {
  width: 1000,
  height: 520,
};

/** Josh ArtProvider RESET_VALUES default (chunk 21768), biased slightly coarser. */
const JOSH_HERO_ARCS_RESET_CONFIG = {
  lineWidth: -0.35,
  lineLength: 0.45,
  density: 55,
  numOfRows: 10,
  linecap: 'round',
  shape: 'line',
  springiness: 75,
};

/** Josh drawArt — render caps; random configs are clamped below these ceilings. */
const JOSH_HERO_ARCS_STROKE_MAX = {
  line: 12,
  circle: 14,
  dot: 10,
  tick: 12,
  cross: 11,
  diamond: 11,
};
/** Floor so random draws never land on hairline grains. */
const JOSH_HERO_ARCS_STROKE_MIN = 5;
const JOSH_HERO_ARCS_SEGMENT_MIN = 10;
const JOSH_HERO_ARCS_SEGMENT_MAX = 28;
/** Max rendered grain size — prevents thick strokes from blobbing together. */
const JOSH_HERO_ARCS_GRAIN_STROKE_CEILING = 11;
const JOSH_HERO_ARCS_GRAIN_SEGMENT_CEILING = 24;
/** Minimum clear gap (px) between grain edges along the arc. */
const JOSH_HERO_ARCS_GRAIN_GAP_MIN = 4;
const JOSH_HERO_ARCS_DENSITY_SPACING_MIN = 115;
const JOSH_HERO_ARCS_DENSITY_SPACING_MAX = 450;
const JOSH_HERO_ARCS_MOUSE_BUFFER = {
  line: 100,
  circle: 220,
  dot: 150,
  tick: 110,
  cross: 130,
  diamond: 130,
};

/** Josh randomizeValues() weighted pools — not a fixed style list. */
const JOSH_HERO_ARCS_MIN_ROWS = 8;
const JOSH_HERO_ARCS_MAX_ROWS = 9;
const JOSH_HERO_ARCS_ROW_PITCH_BASE = 100 * 0.1575;
/** Keep upper flank below sticky nav / search icons (canvas logical px). */
const JOSH_HERO_ARCS_TOP_CLIP_Y = 520 * 0.22;
const JOSH_HERO_ARCS_RANDOM_DENSITY = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
const JOSH_HERO_ARCS_RANDOM_NUM_ROWS = [8, 8, 9, 9, 9, 9, 9, 9];
const JOSH_HERO_ARCS_RANDOM_LINECAP = ['round', 'square'];
const JOSH_HERO_ARCS_RANDOM_SHAPE = ['line', 'circle', 'dot', 'tick', 'cross', 'diamond'];

function joshHeroArcsPickRandom(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function joshHeroArcsRandomFloat(min, max) {
  return min + Math.random() * (max - min);
}

function joshHeroArcsNormalizeShape(shape) {
  if (shape && Object.prototype.hasOwnProperty.call(JOSH_HERO_ARCS_STROKE_MAX, shape)) {
    return shape;
  }
  return 'line';
}

function joshHeroArcsGrainStrokeWidth(config) {
  const shape = joshHeroArcsNormalizeShape(config.shape);
  return joshHeroArcsMapRange(
    config.lineWidth,
    -1,
    1,
    JOSH_HERO_ARCS_STROKE_MAX[shape],
    JOSH_HERO_ARCS_STROKE_MIN,
  );
}

function joshHeroArcsGrainSegmentLength(config) {
  return joshHeroArcsMapRange(
    config.lineLength,
    -1,
    1,
    JOSH_HERO_ARCS_SEGMENT_MIN,
    JOSH_HERO_ARCS_SEGMENT_MAX,
  );
}

/** Along-arc footprint used to keep center-to-center spacing above grain size + gap. */
function joshHeroArcsGrainArcExtent(config) {
  const shape = joshHeroArcsNormalizeShape(config.shape);
  const strokeWidth = joshHeroArcsGrainStrokeWidth(config);
  const segmentLength = joshHeroArcsGrainSegmentLength(config);

  if (shape === 'dot' || shape === 'diamond') {
    return strokeWidth * 1.1;
  }

  if (shape === 'tick') {
    return Math.max(strokeWidth, segmentLength * 0.42);
  }

  if (shape === 'cross') {
    return Math.max(strokeWidth, segmentLength * 0.62);
  }

  return Math.max(strokeWidth, segmentLength * 0.55);
}

function joshHeroArcsSpacingForDensity(density) {
  return joshHeroArcsMapRange(
    density,
    0,
    100,
    JOSH_HERO_ARCS_DENSITY_SPACING_MAX,
    JOSH_HERO_ARCS_DENSITY_SPACING_MIN,
  );
}

function joshHeroArcsMinSpacingForConfig(config) {
  return (joshHeroArcsGrainArcExtent(config) + JOSH_HERO_ARCS_GRAIN_GAP_MIN) * (2 * Math.PI);
}

function joshHeroArcsResolveSpacing(config, density) {
  return Math.max(joshHeroArcsSpacingForDensity(density), joshHeroArcsMinSpacingForConfig(config));
}

/** Nudge random config toward thinner/shorter grains when over ceiling. */
function joshHeroArcsEnforceGrainThresholds(config) {
  let lineWidth = config.lineWidth;
  let lineLength = config.lineLength;

  while (joshHeroArcsGrainStrokeWidth({ ...config, lineWidth }) > JOSH_HERO_ARCS_GRAIN_STROKE_CEILING && lineWidth < 1) {
    lineWidth = joshHeroArcsClamp(lineWidth + 0.05, -1, 1);
  }

  while (joshHeroArcsGrainSegmentLength({ ...config, lineLength }) > JOSH_HERO_ARCS_GRAIN_SEGMENT_CEILING && lineLength > -1) {
    lineLength = joshHeroArcsClamp(lineLength - 0.05, -1, 1);
  }

  return { ...config, lineWidth, lineLength };
}

/** Lower density when random draw would pack grains tighter than the gap rule. */
function joshHeroArcsEnforceGrainSpacing(config) {
  let density = config.density;

  while (joshHeroArcsSpacingForDensity(density) < joshHeroArcsMinSpacingForConfig(config) && density > 0) {
    density -= 5;
  }

  return { ...config, density };
}

/** Mirrors Josh ArtProvider randomizeValues() in chunk 21768, biased toward coarser grains. */
function joshHeroArcsRandomizeConfig() {
  return joshHeroArcsNormalizeConfig(
    joshHeroArcsEnforceGrainThresholds({
      ...JOSH_HERO_ARCS_BASE,
      // lineWidth: -1 = thick, 1 = thin — keep most draws on the thick half.
      lineWidth: joshHeroArcsRandomFloat(-1, 0.25),
      // lineLength: -1 = short, 1 = long — avoid the hairline-short end.
      lineLength: joshHeroArcsRandomFloat(-0.2, 1),
      density: joshHeroArcsPickRandom(JOSH_HERO_ARCS_RANDOM_DENSITY),
      numOfRows: joshHeroArcsPickRandom(JOSH_HERO_ARCS_RANDOM_NUM_ROWS),
      linecap: joshHeroArcsPickRandom(JOSH_HERO_ARCS_RANDOM_LINECAP),
      shape: joshHeroArcsPickRandom(JOSH_HERO_ARCS_RANDOM_SHAPE),
      springiness: joshHeroArcsRandomFloat(0, 100),
    }),
  );
}

function joshHeroArcsNormalizeConfig(config) {
  const rows = Math.round(config.numOfRows || JOSH_HERO_ARCS_MIN_ROWS);
  return joshHeroArcsEnforceGrainSpacing(joshHeroArcsEnforceGrainThresholds({
    ...config,
    numOfRows: joshHeroArcsClamp(rows, JOSH_HERO_ARCS_MIN_ROWS, JOSH_HERO_ARCS_MAX_ROWS),
  }));
}

function joshHeroArcsConfigForRefresh() {
  return joshHeroArcsRandomizeConfig();
}

function joshHeroArcsMapRange(value, inMin, inMax, outMin, outMax) {
  if (inMax === inMin) return outMin;
  return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
}

function joshHeroArcsClamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function joshHeroArcsLerp(a, b, t) {
  return a + (b - a) * t;
}

function joshHeroArcsCreateSpring(initial, tension = 170, friction = 26) {
  let value = initial;
  let velocity = 0;
  let target = initial;
  return {
    setTarget(next) {
      target = next;
    },
    step(dt = 1 / 60) {
      const accel = -tension * (value - target) - friction * velocity;
      velocity += accel * dt;
      value += velocity * dt;
      return value;
    },
    get() {
      return value;
    },
  };
}

function joshHeroArcsStrokeColor(rowIndex, numRows, lightModeRatio) {
  const palette = JOSH_HERO_ARCS_COLORS;
  const colorIndex = Math.floor(joshHeroArcsMapRange(rowIndex, 0, numRows, 0, numRows === 8 ? 8 : 10));
  const hue = Math.round(
    joshHeroArcsMapRange(lightModeRatio, 1, 0, palette.light[colorIndex][0] + 360, palette.dark[colorIndex][0] + 360),
  ) - 360;
  const sat = joshHeroArcsMapRange(lightModeRatio, 1, 0, palette.light[colorIndex][1], palette.dark[colorIndex][1]);
  const light = joshHeroArcsMapRange(lightModeRatio, 1, 0, palette.light[colorIndex][2], palette.dark[colorIndex][2]);
  return `hsl(${hue}deg ${sat}% ${light}%)`;
}

function joshHeroArcsBlendAngles(arcAngle, mouseAngle, mouseRatio) {
  const arcPct = joshHeroArcsMapRange(arcAngle, -Math.PI, Math.PI, 0, 100);
  const mousePct = joshHeroArcsMapRange(mouseAngle, -Math.PI, Math.PI, 0, 100);
  return joshHeroArcsMapRange(arcPct * (1 - mouseRatio) + mousePct * mouseRatio, 0, 100, -Math.PI, Math.PI);
}

function joshHeroArcsIsLightMode() {
  const html = document.documentElement;
  return !html.classList.contains('dark') && html.getAttribute('data-color-mode') !== 'dark';
}

function joshHeroArcsDrawGrain(ctx, grain) {
  const {
    shape,
    x,
    y,
    arcAngle,
    adjustedMouseAngle,
    mousePosition,
    mouseEnabledRatio,
    strokeWidth,
    segmentLength,
    centerX,
    centerY,
  } = grain;
  const grainShape = joshHeroArcsNormalizeShape(shape);
  const radialAngle = Math.atan2(centerY - y, centerX - x);

  if (grainShape === 'circle') {
    /* Josh 21768 chunk: circle = short line segments, elastic near mouse (300px) */
    const halfLen = segmentLength * 0.5;
    const dist = Math.hypot(mousePosition.x - x, mousePosition.y - y);
    const towardMouseX = x - mousePosition.x;
    const towardMouseY = y - mousePosition.y;
    const scale = halfLen / dist;
    const offsetX = towardMouseX * scale;
    const offsetY = towardMouseY * scale;
    let fromX;
    let fromY;
    let toX;
    let toY;

    if (dist <= 300) {
      const influence = joshHeroArcsMapRange(dist, 0, 300, 1, 0);
      fromX = x - offsetX * influence;
      fromY = y - offsetY * influence;
      toX = x + offsetX * influence;
      toY = y + offsetY * influence;
    } else {
      const tangentX = 0.01 * Math.cos(arcAngle);
      const tangentY = 0.01 * Math.sin(arcAngle);
      fromX = x - tangentX;
      fromY = y - tangentY;
      toX = x + tangentX;
      toY = y + tangentY;
    }

    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();
    return;
  }

  if (grainShape === 'dot') {
    const radius = Math.max(1, strokeWidth * 0.45);
    const dist = Math.hypot(mousePosition.x - x, mousePosition.y - y);
    let drawX = x;
    let drawY = y;

    if (dist <= 200) {
      const pull = joshHeroArcsMapRange(dist, 0, 200, 0.35, 0) * mouseEnabledRatio;
      drawX += (mousePosition.x - x) * pull * 0.15;
      drawY += (mousePosition.y - y) * pull * 0.15;
    }

    ctx.beginPath();
    ctx.arc(drawX, drawY, radius, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  if (grainShape === 'tick') {
    const blended = joshHeroArcsBlendAngles(radialAngle, adjustedMouseAngle, mouseEnabledRatio * 0.65);
    const halfLen = segmentLength * 0.4;
    const offsetX = halfLen * Math.cos(blended);
    const offsetY = halfLen * Math.sin(blended);
    ctx.beginPath();
    ctx.moveTo(x - offsetX, y - offsetY);
    ctx.lineTo(x + offsetX, y + offsetY);
    ctx.stroke();
    return;
  }

  if (grainShape === 'cross') {
    const tangentAngle = joshHeroArcsBlendAngles(arcAngle, adjustedMouseAngle, mouseEnabledRatio);
    const arm = segmentLength * 0.3;
    const tcos = Math.cos(tangentAngle) * arm;
    const tsin = Math.sin(tangentAngle) * arm;
    const rcos = Math.cos(radialAngle) * arm;
    const rsin = Math.sin(radialAngle) * arm;

    ctx.beginPath();
    ctx.moveTo(x - tcos, y - tsin);
    ctx.lineTo(x + tcos, y + tsin);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - rcos, y - rsin);
    ctx.lineTo(x + rcos, y + rsin);
    ctx.stroke();
    return;
  }

  if (grainShape === 'diamond') {
    const size = Math.max(2, strokeWidth * 0.95);
    const tangentAngle = joshHeroArcsBlendAngles(arcAngle, adjustedMouseAngle, mouseEnabledRatio * 0.5);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(tangentAngle + Math.PI * 0.25);
    ctx.fillRect(-size * 0.5, -size * 0.5, size, size);
    ctx.restore();
    return;
  }

  const blended = joshHeroArcsBlendAngles(arcAngle, adjustedMouseAngle, mouseEnabledRatio);
  const halfLen = segmentLength * 0.5;
  const offsetX = halfLen * Math.cos(blended);
  const offsetY = halfLen * Math.sin(blended);
  ctx.beginPath();
  ctx.moveTo(x - offsetX, y - offsetY);
  ctx.lineTo(x + offsetX, y + offsetY);
  ctx.stroke();
}

function joshHeroArcsDrawFrame(ctx, canvasBox, options) {
  const {
    mountAt,
    mouseEnabledRatio,
    lightModeRatio,
    enableInteractiveFeatures,
    mousePosition,
    lineWidth,
    lineLength,
    density,
    numOfRows,
    linecap,
    shape,
  } = options;

  const elapsed = Date.now() - (mountAt + 200);
  const phase = elapsed < 2000 && enableInteractiveFeatures ? 'enter' : 'default';
  ctx.clearRect(0, 0, canvasBox.width, canvasBox.height);

  const grainConfig = { lineWidth, lineLength, shape };
  const spacing = joshHeroArcsResolveSpacing(grainConfig, density);
  const rowScale = joshHeroArcsMapRange(numOfRows, 3, 10, 1.2, 1.325);
  const centerX = canvasBox.width * 0.5;
  const centerY = canvasBox.height * rowScale;
  const baseRadius = canvasBox.width * 0.4;
  const strokeWidth = joshHeroArcsGrainStrokeWidth({ lineWidth, shape });
  const segmentLength = joshHeroArcsGrainSegmentLength({ lineLength });

  ctx.save();
  ctx.beginPath();
  ctx.rect(0, JOSH_HERO_ARCS_TOP_CLIP_Y, canvasBox.width, canvasBox.height - JOSH_HERO_ARCS_TOP_CLIP_Y);
  ctx.clip();

  ctx.lineWidth = strokeWidth;
  ctx.lineCap = linecap;

  for (let row = 0; row < numOfRows; row += 1) {
    const radius = baseRadius + row * JOSH_HERO_ARCS_ROW_PITCH_BASE;
    const angleStep = spacing / (2 * Math.PI * radius);
    const startAngle = Math.PI;
    const endAngle = Math.PI * 2;

    ctx.strokeStyle = joshHeroArcsStrokeColor(row, numOfRows, lightModeRatio);
    ctx.fillStyle = ctx.strokeStyle;

    for (let angle = startAngle; angle < endAngle; angle += angleStep) {
      if (phase === 'enter') {
        const reveal = joshHeroArcsMapRange(elapsed, 0, 2000, 0, 1);
        const angleProgress = joshHeroArcsMapRange(angle, startAngle, endAngle, 0, 1);
        if (angleProgress > reveal) continue;
      }

      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      const deltaX = mousePosition.x - x;
      const mouseAngle = Math.atan2(mousePosition.y - y, deltaX);
      let arcAngle = angle + Math.PI * 0.5;

      arcAngle = joshHeroArcsClamp(arcAngle, -Math.PI, Math.PI);
      let adjustedMouseAngle = deltaX > 0 ? mouseAngle - Math.PI : mouseAngle;
      adjustedMouseAngle = joshHeroArcsClamp(adjustedMouseAngle - Math.PI, -Math.PI, Math.PI);

      joshHeroArcsDrawGrain(ctx, {
        shape,
        x,
        y,
        arcAngle,
        adjustedMouseAngle,
        mousePosition,
        mouseEnabledRatio,
        strokeWidth,
        segmentLength,
        centerX,
        centerY,
      });
    }
  }

  ctx.restore();
}

function initJoshHeroArcsCanvas(scope) {
  const root = scope?.querySelector('.josh-sky__hero-arcs');
  const canvas = root?.querySelector('.josh-sky__hero-arcs-canvas');
  if (!canvas) return () => {};

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const mountAt = Date.now();
  const config = joshHeroArcsConfigForRefresh();
  const logicalWidth = config.width;
  const logicalHeight = config.height;

  // Josh homepage canvas uses 1:1 backing store (1000×520), not devicePixelRatio scaling.
  canvas.width = logicalWidth;
  canvas.height = logicalHeight;

  const ctx = canvas.getContext('2d');
  if (!ctx) return () => {};

  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const canvasBox = { width: logicalWidth, height: logicalHeight };
  let rafId = null;
  let targetMouseX = -1000;
  let targetMouseY = -1000;
  const mouseFriction = joshHeroArcsMapRange(config.springiness, 0, 100, 50, 12) || 20;
  const mouseSpringX = joshHeroArcsCreateSpring(-1000, 300, mouseFriction);
  const mouseSpringY = joshHeroArcsCreateSpring(-1000, 300, mouseFriction);
  const mouseEnabledSpring = joshHeroArcsCreateSpring(0);
  const lightModeSpring = joshHeroArcsCreateSpring(joshHeroArcsIsLightMode() ? 1 : 0);
  const mouseBuffer = JOSH_HERO_ARCS_MOUSE_BUFFER[joshHeroArcsNormalizeShape(config.shape)] ?? 120;
  let targetMouseEnabledRatio = 0;
  let targetLightModeRatio = joshHeroArcsIsLightMode() ? 1 : 0;
  lightModeSpring.setTarget(targetLightModeRatio);

  const isAnimationSettled = () => {
    const elapsed = Date.now() - (mountAt + 200);
    const pastEnter = elapsed >= 2000 || reducedMotion;
    return pastEnter
      && Math.abs(mouseEnabledSpring.get() - targetMouseEnabledRatio) < 0.002
      && Math.abs(mouseSpringX.get() - targetMouseX) < 0.5
      && Math.abs(mouseSpringY.get() - targetMouseY) < 0.5
      && Math.abs(lightModeSpring.get() - targetLightModeRatio) < 0.002;
  };

  const draw = () => {
    mouseSpringX.setTarget(targetMouseX);
    mouseSpringY.setTarget(targetMouseY);
    mouseEnabledSpring.setTarget(targetMouseEnabledRatio);
    lightModeSpring.setTarget(targetLightModeRatio);

    const mouseX = mouseSpringX.step();
    const mouseY = mouseSpringY.step();
    const mouseEnabledRatio = mouseEnabledSpring.step();
    const lightModeRatio = lightModeSpring.step();

    joshHeroArcsDrawFrame(ctx, canvasBox, {
      mountAt,
      mouseEnabledRatio,
      lightModeRatio,
      enableInteractiveFeatures: !reducedMotion,
      mousePosition: { x: mouseX, y: mouseY },
      lineWidth: config.lineWidth,
      lineLength: config.lineLength,
      density: config.density,
      numOfRows: config.numOfRows,
      linecap: config.linecap,
      shape: config.shape,
    });
  };

  const scheduleDraw = () => {
    if (rafId !== null) return;
    rafId = requestAnimationFrame(() => {
      rafId = null;
      draw();
      if (!reducedMotion && !isAnimationSettled()) scheduleDraw();
    });
  };

  const setPointer = (clientX, clientY) => {
    const rect = canvas.getBoundingClientRect();
    targetMouseX = clientX - rect.left;
    targetMouseY = clientY - rect.top;
  };

  const pointerInside = (clientX, clientY) => {
    const rect = canvas.getBoundingClientRect();
    return (
      clientX >= rect.left - mouseBuffer
      && clientX <= rect.right + mouseBuffer
      && clientY >= rect.top - mouseBuffer
      && clientY <= rect.bottom + mouseBuffer
    );
  };

  const onPointerMove = (event) => {
    if (!pointerInside(event.clientX, event.clientY)) {
      targetMouseEnabledRatio = 0;
      scheduleDraw();
      return;
    }
    targetMouseEnabledRatio = 1;
    setPointer(event.clientX, event.clientY);
    scheduleDraw();
  };

  const onPointerLeave = () => {
    targetMouseEnabledRatio = 0;
    scheduleDraw();
  };

  const syncTheme = () => {
    targetLightModeRatio = joshHeroArcsIsLightMode() ? 1 : 0;
    lightModeSpring.setTarget(targetLightModeRatio);
    scheduleDraw();
  };

  const themeObserver = new MutationObserver(syncTheme);
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class', 'data-color-mode'],
  });

  canvas.addEventListener('pointermove', onPointerMove, { passive: true });
  canvas.addEventListener('pointerleave', onPointerLeave, { passive: true });
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('resize', scheduleDraw, { passive: true });

  draw();
  if (!reducedMotion) scheduleDraw();

  return () => {
    if (rafId !== null) cancelAnimationFrame(rafId);
    themeObserver.disconnect();
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('resize', scheduleDraw);
  };
}

function joshHomeNavScrollMetrics(sticky, blocker) {
  if (!sticky || !blocker) {
    const y = window.scrollY;
    if (y >= 520) return { progress: 1, over: true };
    if (y <= 440) return { progress: 0, over: false };
    const progress = (y - 440) / JOSH_NAV_SCROLL_RAMP_PX;
    return { progress, over: progress >= 1 };
  }

  const headerWrap = sticky.querySelector('.josh-sky__header-wrap');
  const headerBottom = headerWrap
    ? headerWrap.getBoundingClientRect().bottom
    : sticky.getBoundingClientRect().bottom;
  const blockerTop = blocker.getBoundingClientRect().top;
  const stickyBottom = sticky.getBoundingClientRect().bottom;
  const gap = blockerTop - headerBottom;

  if (stickyBottom <= 0 && blockerTop <= 1) {
    return { progress: 1, over: true };
  }

  if (gap >= JOSH_NAV_SCROLL_RAMP_PX) {
    return { progress: 0, over: false };
  }

  if (gap <= 0) {
    return { progress: 1, over: true };
  }

  const progress = 1 - gap / JOSH_NAV_SCROLL_RAMP_PX;
  return { progress, over: progress >= 1 };
}

function initJoshHomeInteractions(app) {
  if (typeof joshHomeCleanup === 'function') {
    joshHomeCleanup();
    joshHomeCleanup = null;
  }

  const sticky = app.querySelector('#josh-sky-sticky');
  const sky = app.querySelector('#josh-sky');
  const blocker = app.querySelector('#josh-blocker');
  const showMoreBtn = app.querySelector('#josh-show-more');

  let navScrollRaf = null;

  const syncNavScrollState = () => {
    let { progress, over } = joshHomeNavScrollMetrics(sticky, blocker);

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      progress = over ? 1 : 0;
    }

    const progressValue = String(progress);

    if (sky) {
      sky.style.setProperty('--josh-nav-scroll-progress', progressValue);
    }
    if (sticky) {
      sticky.classList.toggle('is-scrolled', over);
      sticky.setAttribute('data-is-over-threshold', String(over));
    }
    if (blocker) {
      blocker.classList.toggle('is-stuck', over);
      blocker.setAttribute('data-is-stuck', String(over));
    }
  };

  const scheduleNavScrollState = () => {
    if (navScrollRaf !== null) return;
    navScrollRaf = requestAnimationFrame(() => {
      navScrollRaf = null;
      syncNavScrollState();
    });
  };

  window.addEventListener('scroll', scheduleNavScrollState, { passive: true });
  window.addEventListener('resize', syncNavScrollState, { passive: true });
  syncNavScrollState();

  if (showMoreBtn) {
    showMoreBtn.addEventListener('click', () => {
      const revealed = [...app.querySelectorAll('.josh-article.is-hidden')];
      revealed.forEach((el) => {
        el.classList.remove('is-hidden');
        el.classList.add('is-revealed');
      });
      if (typeof joshPlaySound === 'function') joshPlaySound('click');
      showMoreBtn.closest('.josh-show-more-wrap')?.remove();
    });
  }

  initJoshSiteInteractions(app);
  const cleanupHeroArcs = initJoshHeroArcsCanvas(app);

  joshHomeCleanup = () => {
    cleanupHeroArcs();
    if (navScrollRaf !== null) {
      cancelAnimationFrame(navScrollRaf);
      navScrollRaf = null;
    }
    window.removeEventListener('scroll', scheduleNavScrollState);
    window.removeEventListener('resize', syncNavScrollState);
    if (typeof joshSiteCleanup === 'function') joshSiteCleanup();
  };
}
