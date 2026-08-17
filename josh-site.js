/* Josh W. Comeau site shell — shared across all pages */

const JOSH_MASCOT_LIGHT = 'https://www.joshwcomeau.com/images/josh/josh-happy-light.png';
const JOSH_MASCOT_DARK = 'https://www.joshwcomeau.com/images/josh/josh-happy-dark.png';
const JOSH_MASCOT_HEAD_VERY_LIGHT = 'https://www.joshwcomeau.com/images/josh/head-very-happy-light.png';
const JOSH_MASCOT_HEAD_VERY_DARK = 'https://www.joshwcomeau.com/images/josh/head-very-happy-dark.png';

const JOSH_FOOTER_MASCOT_OFFSETS = { hidden: -226, visible: -82 };
const JOSH_FOOTER_MASCOT_SPRING = { tension: 160, friction: 32 };
let joshFooterMascotCleanup = null;

function joshMascotMarkup() {
  return `<div class="josh-sky__mascot-spring">
    <div class="josh-sky__mascot-figure">
      <div class="josh-sky__mascot-picture josh-sky__mascot-picture--dark">
        <img class="josh-sky__mascot-dark" src="${JOSH_MASCOT_DARK}" alt="" width="138" height="232" loading="eager" decoding="async">
      </div>
      <div class="josh-sky__mascot-picture josh-sky__mascot-picture--light" data-use-global-color-mode="true">
        <img class="josh-sky__mascot-light" src="${JOSH_MASCOT_LIGHT}" alt="" width="138" height="232" loading="eager" decoding="async">
      </div>
    </div>
  </div>`;
}

function joshFooterMascotBodyPicture(src) {
  return `<picture><img class="josh-footer__mascot-body" src="${src}" alt="" loading="lazy" decoding="async"></picture>`;
}

function joshFooterMascotHeadPicture(mood, mode) {
  if (mood === 'happy') return '';
  const src = mode === 'dark' ? JOSH_MASCOT_HEAD_VERY_DARK : JOSH_MASCOT_HEAD_VERY_LIGHT;
  return `<picture><img class="josh-footer__mascot-head" src="${src}" alt="" loading="lazy" decoding="async"></picture>`;
}

function joshFooterMascotMarkup() {
  return `<div class="josh-footer__mascot-layer" id="josh-footer-mascot-layer" data-mood="happy">
    <div class="josh-footer__mascot-figure">
      <div class="josh-footer__mascot-picture josh-footer__mascot-picture--dark">
        ${joshFooterMascotBodyPicture(JOSH_MASCOT_DARK)}
      </div>
      <div class="josh-footer__mascot-picture josh-footer__mascot-picture--light" data-use-global-color-mode="true" style="--opacity: 1;">
        ${joshFooterMascotBodyPicture(JOSH_MASCOT_LIGHT)}
      </div>
    </div>
  </div>`;
}

function syncJoshFooterMascotMood(layer) {
  if (!layer) return;
  const mood = layer.dataset.mood || 'happy';
  layer.querySelectorAll('.josh-footer__mascot-picture--dark, .josh-footer__mascot-picture--light').forEach((pic) => {
    const mode = pic.classList.contains('josh-footer__mascot-picture--dark') ? 'dark' : 'light';
    const headPicture = pic.querySelector('picture:has(.josh-footer__mascot-head)');
    const shouldHaveHead = mood !== 'happy';
    if (shouldHaveHead && !headPicture) {
      pic.insertAdjacentHTML('beforeend', joshFooterMascotHeadPicture(mood, mode));
    } else if (!shouldHaveHead && headPicture) {
      headPicture.remove();
    }
  });
}

function syncJoshFooterMascotLightOpacity() {
  const light = document.querySelector('#josh-footer-mascot-layer .josh-footer__mascot-picture--light');
  if (!light) return;
  const isDark = document.documentElement.classList.contains('dark');
  light.style.setProperty('--opacity', isDark ? '0' : '1');
}

function joshFooterMascotMood(offset) {
  return offset < -100 ? 'happy' : 'very-happy';
}

function joshFooterMascotTargetFromFooterRect(footerRect) {
  const { hidden, visible } = JOSH_FOOTER_MASCOT_OFFSETS;
  const viewH = window.innerHeight || 900;
  const rampStart = viewH * 0.74;
  const rampEnd = viewH * 0.41;
  const span = rampStart - rampEnd;
  if (span <= 0) return hidden;
  const progress = Math.max(0, Math.min(1, 1 - (footerRect.top - rampEnd) / span));
  return hidden + progress * (visible - hidden);
}

function joshSpringFollowFooterMascot(layer, getTargetOffset, config = JOSH_FOOTER_MASCOT_SPRING) {
  const readOffset = () => {
    const raw = getComputedStyle(layer).getPropertyValue('--josh-footer-mascot-offset').trim();
    const parsed = parseFloat(raw);
    return Number.isFinite(parsed) ? parsed : JOSH_FOOTER_MASCOT_OFFSETS.hidden;
  };

  let value = readOffset();
  let velocity = 0;
  let running = true;
  let rafId = null;

  const apply = (nextOffset) => {
    layer.style.setProperty('--josh-footer-mascot-offset', `${nextOffset}px`);
    const mood = joshFooterMascotMood(nextOffset);
    if (layer.dataset.mood !== mood) {
      layer.dataset.mood = mood;
      syncJoshFooterMascotMood(layer);
    }
  };

  const step = () => {
    if (!running) return;
    const targetOffset = getTargetOffset();
    const displacement = targetOffset - value;
    const acceleration = config.tension * displacement - config.friction * velocity;
    velocity += acceleration * (1 / 60);
    value += velocity * (1 / 60);
    apply(value);
    rafId = requestAnimationFrame(step);
  };

  rafId = requestAnimationFrame(step);

  return () => {
    running = false;
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  };
}

function joshSpringAnimateOffset(layer, targetOffset, config = JOSH_FOOTER_MASCOT_SPRING) {
  const readOffset = () => {
    const raw = getComputedStyle(layer).getPropertyValue('--josh-footer-mascot-offset').trim();
    const parsed = parseFloat(raw);
    return Number.isFinite(parsed) ? parsed : JOSH_FOOTER_MASCOT_OFFSETS.hidden;
  };

  let value = readOffset();
  let velocity = 0;
  let rafId = null;
  const precision = 0.25;

  const apply = (nextOffset) => {
    layer.style.setProperty('--josh-footer-mascot-offset', `${nextOffset}px`);
    const mood = joshFooterMascotMood(nextOffset);
    if (layer.dataset.mood !== mood) {
      layer.dataset.mood = mood;
      syncJoshFooterMascotMood(layer);
    }
  };

  const step = () => {
    const displacement = targetOffset - value;
    const acceleration = config.tension * displacement - config.friction * velocity;
    velocity += acceleration * (1 / 60);
    value += velocity * (1 / 60);
    apply(value);

    if (Math.abs(displacement) > precision || Math.abs(velocity) > precision) {
      rafId = requestAnimationFrame(step);
      return;
    }

    value = targetOffset;
    velocity = 0;
    apply(targetOffset);
    rafId = null;
  };

  if (rafId !== null) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(step);

  return () => {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  };
}

function initJoshFooterMascotAnimation(app) {
  if (typeof joshFooterMascotCleanup === 'function') {
    joshFooterMascotCleanup();
    joshFooterMascotCleanup = null;
  }

  const root = app || document.getElementById('app') || document;
  const footer = root.querySelector('.josh-footer');
  const mascotLayer = root.querySelector('#josh-footer-mascot-layer');
  if (!footer || !mascotLayer) return () => {};

  mascotLayer.style.setProperty('--josh-footer-mascot-offset', `${JOSH_FOOTER_MASCOT_OFFSETS.hidden}px`);
  mascotLayer.dataset.mood = 'happy';
  syncJoshFooterMascotMood(mascotLayer);
  syncJoshFooterMascotLightOpacity();

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const applyOffset = (targetOffset) => {
    mascotLayer.style.setProperty('--josh-footer-mascot-offset', `${targetOffset}px`);
    const mood = joshFooterMascotMood(targetOffset);
    if (mascotLayer.dataset.mood !== mood) {
      mascotLayer.dataset.mood = mood;
      syncJoshFooterMascotMood(mascotLayer);
    }
  };

  const readTarget = () => joshFooterMascotTargetFromFooterRect(footer.getBoundingClientRect());

  if (reducedMotion) {
    const sync = () => applyOffset(readTarget());
    window.addEventListener('scroll', sync, { passive: true });
    window.addEventListener('resize', sync, { passive: true });
    sync();
    joshFooterMascotCleanup = () => {
      window.removeEventListener('scroll', sync);
      window.removeEventListener('resize', sync);
    };
    return joshFooterMascotCleanup;
  }

  const stopSpring = joshSpringFollowFooterMascot(mascotLayer, readTarget);
  const syncOnScroll = () => {
    syncJoshFooterMascotLightOpacity();
  };
  window.addEventListener('scroll', syncOnScroll, { passive: true });
  window.addEventListener('resize', syncOnScroll, { passive: true });
  joshFooterMascotCleanup = () => {
    stopSpring();
    window.removeEventListener('scroll', syncOnScroll);
    window.removeEventListener('resize', syncOnScroll);
  };
  return joshFooterMascotCleanup;
}

const JOSH_CATEGORY_COLORS = {
  '基础原理': '#b8ddf2',
  '模型架构': '#c8daf8',
  '微调与对齐': '#dcc8f8',
  'RAG 与检索': '#b8ecd8',
  '智能体': '#c8f0e0',
  '评测与质量': '#e8d4f8',
  '多模态': '#ffe0b8',
  '论文解读': '#f4c0dc',
  '随想与思考': '#ccd8ff',
};

function syncJoshSiteClass(isHome) {
  const html = document.documentElement;
  html.classList.add('josh-site');
  html.classList.toggle('josh-home-page', isHome);
  syncJoshColorMode();
}

function syncJoshColorMode() {
  const html = document.documentElement;
  if (!html.classList.contains('josh-site')) {
    html.removeAttribute('data-color-mode');
    return;
  }
  html.setAttribute(
    'data-color-mode',
    html.classList.contains('dark') ? 'dark' : 'light',
  );
  syncJoshFooterMascotLightOpacity();
}

function joshLockThemeScroll(y) {
  window.scrollTo({ top: y, left: 0, behavior: 'instant' });
}

function joshFinishThemeSwap() {
  const html = document.documentElement;
  const y = Number(html.dataset.joshThemeScrollY ?? window.scrollY);
  requestAnimationFrame(() => {
    joshLockThemeScroll(y);
    requestAnimationFrame(() => {
      joshLockThemeScroll(y);
      html.classList.remove('josh-theme-swapping');
      delete html.dataset.joshThemeScrollY;
    });
  });
}

function joshLogoMarkup(homeHref) {
  return `<a class="josh-logo" href="${homeHref}" aria-label="Tangentllm Notes 首页">
    <span class="josh-logo__name">Tangent</span>
    <span class="josh-logo__w-wrap" aria-hidden="true">
      <svg class="josh-logo__w-main" viewBox="0 0 14 12" aria-hidden="true"><path d="M1.84 4.19C2.5 5.36 3.23 6.56 3.8 7.77C4.05 8.3 4.36 9.36 4.81 9.75C5.73 10.54 6.3 10.27 6.77 9.06C7.13 8.13 7.2 7.09 7.55 6.13C7.91 5.11 9.24 6.7 9.78 7.07C10.32 7.42 13.57 9.53 12.89 7.6C12.67 6.99 12.61 6.34 12.48 5.71C12.35 5.08 12.11 4.28 12.08 3.65C12.04 2.88 11.72 2.13 11.72 1.32" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>
      <svg class="josh-logo__w-tail" viewBox="0 0 9 10" aria-hidden="true"><path d="M1 9C2.24 7.12 3.87 5.19 4.18 2.6C4.22 2.22 4.11 0.73 4.11 1.04C4.11 1.49 4.42 2.03 4.56 2.41C4.99 3.56 5.31 4.69 5.93 5.7C6.43 6.54 7.08 7.96 8 7.96" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>
      <span class="josh-logo__w-letter">l</span>
    </span>
    <span class="josh-logo__name">lm</span>
  </a>`;
}

let joshThemeToggleUid = 0;

function joshThemeToggleMarkup() {
  const uid = `jtt-${++joshThemeToggleUid}`;
  const sunDots = [
    [22, 12, 280], [19.0711, 19.0711, 200], [12, 22, 120], [4.9289, 19.0711, 40],
    [2, 12, 0], [4.9289, 4.9289, 80], [12, 2, 160], [19.0711, 4.9289, 240],
  ].map(([cx, cy, delay]) => (
    `<circle class="josh-theme-toggle__orbit-dot" cx="${cx}" cy="${cy}" r="1.5" style="--enter-delay:${delay}ms"/>`
  )).join('');
  return `<button type="button" class="josh-icon-btn josh-theme-toggle" aria-label="切换主题">
    <svg class="josh-theme-toggle__glyph" width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
      <mask id="${uid}-sun-dot"><rect x="-10" y="-10" width="44" height="44" fill="#FFF"/><circle r="6" cx="12" cy="12" fill="#000"/></mask>
      <mask id="${uid}-moon-cut"><rect x="0" y="0" width="24" height="24" fill="#FFF"/><circle cx="12" cy="-4" r="8" fill="#000"/></mask>
      <mask id="${uid}-moon-cres"><rect x="0" y="0" width="24" height="24" fill="#000"/><circle r="7" cx="12" cy="12" fill="#FFF"/></mask>
      <g class="josh-theme-toggle__sun" mask="url(#${uid}-sun-dot)">${sunDots}</g>
      <g class="josh-theme-toggle__moon-cut" mask="url(#${uid}-moon-cut)"><circle cx="12" cy="12" stroke="currentColor" fill="none" r="6"/></g>
      <g class="josh-theme-toggle__moon-cres" mask="url(#${uid}-moon-cres)"><circle cx="12" cy="-4" r="8" stroke="currentColor" fill="none"/></g>
    </svg>
  </button>`;
}

function joshSoundToggleMarkup() {
  return `<button type="button" class="josh-icon-btn josh-sound-toggle" aria-label="禁用音效" aria-pressed="true" title="音效">
    <svg class="josh-sound-toggle__icon josh-sound-toggle__on" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
    <svg class="josh-sound-toggle__icon josh-sound-toggle__off" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>
  </button>`;
}

function joshNavLinks(activeHref) {
  const links = [
    { href: Routes.categories(), label: '分类' },
    { href: Routes.tags(), label: '标签' },
    { href: Routes.projects(), label: '作品' },
    { href: Routes.about(), label: '关于' },
  ];
  const normalize = (h) => (h || '').replace(/\/$/, '');
  const active = normalize(activeHref);
  return links.map((link) => {
    const isActive = normalize(link.href) === active;
    return `<li class="josh-nav__item"><a class="josh-nav__link${isActive ? ' is-active' : ''}" href="${link.href}"${isActive ? ' aria-current="page"' : ''}>${link.label}</a></li>`;
  }).join('');
}

function joshMobileNavLinks(activeHref) {
  const links = [
    { href: Routes.home(), label: '首页' },
    { href: Routes.categories(), label: '分类' },
    { href: Routes.tags(), label: '标签' },
    { href: Routes.projects(), label: '作品' },
    { href: Routes.about(), label: '关于' },
  ];
  const normalize = (h) => (h || '').replace(/\/$/, '');
  const active = normalize(activeHref);
  return links.map((link) => {
    const isActive = normalize(link.href) === active;
    return `<a class="josh-mobile-menu__link${isActive ? ' is-active' : ''}" href="${link.href}"${isActive ? ' aria-current="page"' : ''}>${link.label}</a>`;
  }).join('');
}

function joshGithubLinkMarkup() {
  return `<a class="josh-icon-btn" href="https://github.com/tangentllm/weblog" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/></svg>
  </a>`;
}

function joshHeaderUtilityActionsMarkup() {
  return `<button type="button" class="josh-icon-btn" onclick="openSearch()" aria-label="搜索">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        </button>
        ${joshSoundToggleMarkup()}
        ${joshThemeToggleMarkup()}
        ${joshGithubLinkMarkup()}`;
}

function buildJoshInnerHeaderMarkup(activeHref, options = {}) {
  const homeHref = Routes.home();
  const aboutSkyEmbed = Boolean(options.aboutSkyEmbed);
  const aboutBodySticky = Boolean(options.aboutBodySticky);
  const postBackdrop = options.postNavLayers && !aboutSkyEmbed && !aboutBodySticky ? `
    <div class="josh-post-header__sky-blur" aria-hidden="true"></div>
    <div class="josh-post-header__white-bg" aria-hidden="true"></div>
    <div class="josh-post-header__white-blur" aria-hidden="true"></div>
  ` : '';
  const outerClass = aboutSkyEmbed
    ? 'josh-about-sky__nav josh-inner-header'
    : aboutBodySticky
      ? 'josh-about-body__nav josh-inner-header'
      : 'josh-inner-header';
  const headerId = aboutBodySticky ? 'josh-about-body-header' : 'josh-inner-header';
  const mobileToggleId = aboutBodySticky ? 'josh-about-body-mobile-toggle' : 'josh-mobile-toggle';
  const mobileMenuId = aboutBodySticky ? 'josh-about-body-mobile-menu' : 'josh-mobile-menu';
  return `<div class="${outerClass}" id="${headerId}">
    ${postBackdrop}
    <div class="josh-sky__header-wrap josh-container">
      <header class="josh-header">
        ${joshLogoMarkup(homeHref)}
        <nav class="josh-nav" aria-label="主导航">
          <ul class="josh-nav__list">
            ${joshNavLinks(activeHref)}
          </ul>
        </nav>
        <div class="josh-header__actions">
          ${joshHeaderUtilityActionsMarkup()}
        </div>
        <button type="button" class="josh-mobile-toggle" id="${mobileToggleId}" aria-expanded="false" aria-controls="${mobileMenuId}" aria-label="打开菜单">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>
      </header>
    </div>
    <div class="josh-mobile-menu" id="${mobileMenuId}" hidden>
      ${joshMobileNavLinks(activeHref)}
    </div>
  </div>`;
}

function joshCategoriesWithPosts() {
  return categories.filter((c) => c.count > 0 || posts.some((p) => p.category === c.name));
}

function joshFooterWaveAccentMarkup() {
  return `<svg class="josh-footer__wave-accent" viewBox="0 0 1557 213" width="1590" height="213" preserveAspectRatio="none" aria-hidden="true"><path d="M1331 165.5C1320.5 162.5 1311 112 1221.5 92.5C1132 73 1104.5 114 1093.5 112C1082.5 110 1052 6 903.5 1C771 -3.5 732.5 90 722.5 92.5C712.5 95 647.5 54 535.5 74C423.5 94 421.5 165.5 411.5 165.5C401.5 165.5 359 58.5 203.5 49.5C48 40.5 0.5 136 0.5 209.5C0.5 283 62.5 309.5 62.5 309.5H1556.5V232.5C1556.5 215.5 1542.5 159 1464 145.5C1368.81 129.1 1341.5 168.5 1331 165.5Z"></path></svg>`;
}

function joshFooterWaveMainMarkup() {
  return `<div class="josh-footer__wave-main-wrap" aria-hidden="true">
    <svg class="josh-footer__wave-main" width="320rem" height="15.625rem" viewBox="0 0 5120 337" preserveAspectRatio="none" aria-hidden="true"><path d="M2262 93C2122.5 82.6 2116 21.5 2096.5 21.5C2077 21.5 2070.5 77.5 1920.5 93C1794.5 106 1786 62 1771.5 63.5C1757 65 1687 155.5 1580 142C1473 128.5 1446.5 90 1435 93C1423.5 96 1448 199 1340 214C1181.5 236 1155.5 142 1144 142C1132.5 142 1105.5 269 946.5 236C787.5 203 799 115 784 114.5C769 114 732.5 162 544 158C382 154.6 352.5 81 341 84.5C329.5 88 358 269 168 326C-22 383 -75.5 180 -75.5 180V0.5H5189.5L5193.5 46C5193.5 46 5200 94 5069.5 100.5C4939 107 4923.5 21.5 4906.5 21.5C4889.5 21.5 4870 35 4835 93.5C4800 152 4765.5 169.5 4643.5 173.5C4521.5 177.5 4436.5 69 4425.5 76.5C4414.5 84 4413.5 212 4235 222C4056.5 232 4045.5 92 4033.5 89C4021.5 86 3968.5 169.5 3823.5 172.5C3678.5 175.5 3573.5 104 3562.5 106.5C3551.5 109 3553.5 167.5 3396 201C3238.5 234.5 3171.5 168.5 3161 172.5C3150.5 176.5 3164 273 3076.5 294.5C2976 319.2 2935 228 2920 225.5C2905 223 2862 277 2749 245C2671.4 223.1 2672.5 149 2660.5 151.5C2648.5 154 2622.5 181 2548.5 158C2425 119.5 2427.5 53.5 2412 51C2396.5 48.5 2376 101.5 2262 93Z" fill="var(--josh-color-background)"></path></svg>
  </div>`;
}

/* Josh search sheet cloud edge (s1n3w95p / s132djyi on joshwcomeau.com) */
const JOSH_SEARCH_WAVE_PATH =
  'M1825 113C1837.5 112 1874.5 145 2058.5 145.5C2242.5 146 2273 119.5 2288.5 119C2304 118.5 2312 140 2519.5 129.5C2727 119 2739.5 81.5 2748.5 82.5C2757.5 83.5 2821.5 105 2980.5 105C3139.5 105 3198 65 3211.5 61C3225 57 3230 91 3447.5 83.5C3665 76 3663 27.5 3675 29.5C3687 31.5 3710 76 3904 76C4098 76 4088 22 4098 19C4102.46 17.6626 4104.43 23.0822 4112.91 29.5V0.5H0.416992V28C24.7581 17.693 38.7983 8.78176 45.5013 9.5C59.5002 11 60.0014 72 226.501 72.5C393.001 73 432.003 19.5 440.502 18C449 16.5 474.502 77.5 669.002 85C863.502 92.5 889.003 56 902.502 56C916 56 937.002 92 1132.5 102.5C1328 113 1353 85 1363.5 85C1374 85 1391 124.5 1596 131.5C1801 138.5 1812.5 114 1825 113Z';

function joshSearchWaveMarkup() {
  return `<div class="josh-search__wave" aria-hidden="true">
    <svg class="josh-search__wave-svg" viewBox="0 0 4113 146" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none" aria-hidden="true">
      <path fill="var(--josh-color-background)" d="${JOSH_SEARCH_WAVE_PATH}"></path>
    </svg>
  </div>`;
}

function initJoshSearchWave() {
  const sheet = document.querySelector('.josh-search__sheet');
  if (!sheet || sheet.querySelector('.josh-search__wave')) return;
  sheet.insertAdjacentHTML('beforeend', joshSearchWaveMarkup());
}

function buildJoshFooterMarkup(options = {}) {
  const aboutPage = Boolean(options.aboutPage);
  const homeHref = Routes.home();
  const catsWithPosts = joshCategoriesWithPosts();
  const footerSkyMarkup = joshFooterMascotMarkup();
  const footerWavesMarkup = `${joshFooterWaveAccentMarkup()}${joshFooterWaveMainMarkup()}`;

  return `<footer class="josh-footer${aboutPage ? ' josh-footer--about' : ''}" role="contentinfo">
    <div class="josh-footer__sky" aria-hidden="true">
      ${footerSkyMarkup}
    </div>
    ${footerWavesMarkup}
    <div class="josh-footer__body josh-container">
      <div class="josh-footer__grid">
        <div class="josh-footer__intro">
          ${joshLogoMarkup(homeHref)}
          <div class="josh-footer__intro-spacer" aria-hidden="true"></div>
        </div>
        <div class="josh-footer__grid-spacer" aria-hidden="true"></div>
        <div class="josh-footer__legal">
          <span>© 2024–${new Date().getFullYear()} Tangentllm Notes</span>
          <ul>
            <li><a href="${Routes.about()}">关于本站</a></li>
          </ul>
        </div>
        <div class="josh-footer__links">
          <div class="josh-footer__col-cats">
            <h2 class="josh-footer__col-title">按分类浏览</h2>
            <ul class="josh-footer__cat-grid">
              ${catsWithPosts.map((c) => `<li><a href="${Routes.category(c.name)}">${c.name}</a></li>`).join('')}
            </ul>
          </div>
          <div class="josh-footer__col-courses">
            <h2 class="josh-footer__col-title">站点导航</h2>
            <ul class="josh-footer__stack">
              <li><a href="${Routes.home()}">全部文章</a></li>
              <li><a href="${Routes.projects()}">作品展示</a></li>
              <li><a href="${Routes.about()}">关于作者</a></li>
            </ul>
          </div>
          <div class="josh-footer__col-misc">
            <h2 class="josh-footer__col-title">General</h2>
            <ul class="josh-footer__stack">
              <li><a href="${Routes.about()}">关于本站</a></li>
              <li><a href="${Routes.home()}">返回首页</a></li>
              <li><a href="https://github.com/tangentllm/weblog" target="_blank" rel="noopener noreferrer">GitHub<span class="josh-footer__ext" aria-hidden="true"> ↗</span></a></li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </footer>`;
}

function buildJoshPageShell(mainHtml, activeHref, options = {}) {
  const mainClass = options.mainClass || 'josh-inner-main josh-container';
  const header = options.omitHeader ? '' : buildJoshInnerHeaderMarkup(activeHref);
  return `<div class="josh-page${options.pageClass || ''}">
    ${header}
    <div class="${mainClass}">
      ${mainHtml}
    </div>
    ${buildJoshFooterMarkup(options.footerOptions || {})}
  </div>`;
}

const JOSH_POST_NAV_SCROLL_RAMP_PX = 80;

function joshNormalizeHeadingText(text) {
  return String(text || '')
    .replace(/<[^>]+>/g, '')
    .replace(/[：:]/g, ':')
    .replace(/\s+/g, '')
    .toLowerCase();
}

function extractHtmlArticleProse(html, title) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');
  doc.querySelectorAll('script, style, link[rel="stylesheet"]').forEach((el) => el.remove());

  const container = doc.querySelector('.container, .wrap') || doc.body;
  container.querySelectorAll('header, nav, footer').forEach((el) => el.remove());
  container.querySelectorAll('.toc').forEach((el) => el.remove());
  container.querySelectorAll('.callout[style]').forEach((el) => el.removeAttribute('style'));
  container.querySelectorAll('pre[style], pre code[style]').forEach((el) => el.removeAttribute('style'));
  container.querySelectorAll('pre span[style]').forEach((el) => el.removeAttribute('style'));

  return stripDuplicatePostH1(container.innerHTML.trim(), title);
}

function stripDuplicatePostH1(html, title) {
  const target = joshNormalizeHeadingText(title);
  if (!target) return html;
  return html.replace(/^\s*<h1\b[^>]*>([\s\S]*?)<\/h1>\s*/i, (match, inner) => {
    const heading = joshNormalizeHeadingText(inner);
    if (!heading) return match;
    if (heading === target || heading.startsWith(target.slice(0, 24)) || target.startsWith(heading.slice(0, 24))) {
      return '';
    }
    return match;
  });
}

function processPostProseHeadings(proseHtml) {
  const headings = [];
  let index = 0;
  const contentWithIds = proseHtml.replace(
    /<h([23])\b([^>]*)>([\s\S]*?)<\/h\1>/gi,
    (match, level, attrs, content) => {
      const existingId = attrs.match(/\bid\s*=\s*["']([^"']+)["']/i);
      const id = existingId ? existingId[1] : `heading-${index}`;
      headings.push({
        id,
        tag: `h${level}`,
        text: content.replace(/<[^>]+>/g, '').trim(),
      });
      index += 1;
      if (existingId) return match;
      const attrStr = attrs.trim() ? ` ${attrs.trim()}` : '';
      return `<h${level}${attrStr} id="${id}">${content}</h${level}>`;
    }
  );
  return { headings, contentWithIds };
}

function joshShouldShowPostCover(post) {
  const cover = String(post?.cover || '').trim();
  if (!cover) return false;
  if (/picsum\.photos/i.test(cover)) return false;
  if (cover.startsWith('data:') && decodeURIComponent(cover).includes('Inline fallback cover')) {
    return false;
  }
  // OG / 社交用 SVG 封面不在正文展示（对齐 joshwcomeau.com 文章页）
  if (/\.svg([?#]|$)/i.test(cover) || /\/covers\//i.test(cover)) return false;
  return true;
}

function joshShouldShowProjectCover(project) {
  return joshShouldShowPostCover(project);
}

function joshProjectCoverSrc(project) {
  const cover = String(project?.cover || '').trim();
  if (!cover) return '';
  return typeof resolveAssetUrl === 'function' ? resolveAssetUrl(cover) : cover;
}

function joshProjectUpdatedLabel(project) {
  return String(project?.updated || project?.period || '').trim();
}

function joshProjectReadTime(project) {
  if (project?.readTime) return project.readTime;
  const md = String(project?.rawMarkdown || '').trim();
  if (md && typeof estimateReadTimeFromMarkdown === 'function') {
    return estimateReadTimeFromMarkdown(
      typeof stripFrontmatter === 'function' ? stripFrontmatter(md) : md,
    );
  }
  return '';
}

function joshProjectStatusPillMarkup(project) {
  const status = String(project?.status || '').trim();
  if (!status) return '';
  return `<span class="josh-status-pill">${status}</span>`;
}

function joshPostTailStatsMarkup(slug, options = {}) {
  const updated = String(options.updated || '').trim();
  const updatedHtml = updated
    ? `<p class="josh-post-tail-stats__updated">最后更新于 <strong>${updated}</strong></p>`
    : '';
  const hitsInner = typeof joshHitCounterInnerHtml === 'function'
    ? joshHitCounterInnerHtml(0)
    : '000000';
  return `<aside class="josh-post-tail-stats" aria-label="阅读统计">
    ${updatedHtml}
    <div class="josh-post-tail-stats__row">
      <div class="josh-post-tail-stats__hits">
        <span class="josh-post-tail-stats__hits-label"># of hits</span>
        <div class="josh-hit-counter josh-post-tail-stats__hits-value" id="view-count-tail-${slug}" data-value="0" data-digits="6" aria-live="polite" aria-label="0 hits">${hitsInner}</div>
      </div>
    </div>
  </aside>`;
}

function joshRelatedPosts(entry, limit = 3) {
  const slug = String(entry?.slug || '').trim();
  if (!slug || !Array.isArray(posts) || !posts.length) return [];

  const tags = Array.isArray(entry.tags) ? entry.tags : [];
  const category = String(entry.category || '').trim();

  const scored = posts
    .filter((item) => item.slug !== slug)
    .map((item) => {
      let score = 0;
      if (category && item.category === category) score += 3;
      const sharedTags = (item.tags || []).filter((tag) => tags.includes(tag));
      score += sharedTags.length * 2;
      return { item, score };
    })
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return String(b.item.date || '').localeCompare(String(a.item.date || ''));
    });

  const pool = scored.some(({ score }) => score > 0)
    ? scored.filter(({ score }) => score > 0)
    : scored;

  const picked = [];
  pool.forEach(({ item }) => {
    if (picked.length >= limit) return;
    picked.push(item);
  });

  if (picked.length < limit) {
    posts.forEach((item) => {
      if (picked.length >= limit) return;
      if (item.slug === slug) return;
      if (picked.some((p) => p.slug === item.slug)) return;
      picked.push(item);
    });
  }

  return picked.slice(0, limit);
}

function joshSeriesSiblingPosts(post) {
  const series = String(post?.series || '').trim();
  if (!series || !Array.isArray(posts) || !posts.length) {
    return { prev: null, next: null };
  }
  const siblings = posts
    .filter((item) => item.series === series && item.seriesOrder != null && !Number.isNaN(item.seriesOrder))
    .sort((a, b) => a.seriesOrder - b.seriesOrder);
  const idx = siblings.indexOf(post);
  if (idx < 0) {
    return { prev: null, next: null };
  }
  return {
    prev: idx > 0 ? siblings[idx - 1] : null,
    next: idx < siblings.length - 1 ? siblings[idx + 1] : null,
  };
}

function joshRelatedPostsMarkup(entry) {
  const related = joshRelatedPosts(entry);
  if (!related.length) return '';

  return `<section class="josh-related" aria-label="相关文章">
    <h2 class="josh-related__title">相关文章</h2>
    <div class="josh-related__grid">
      ${related.map((item) => `
        <a class="josh-related-card" href="${Routes.post(item.slug)}">
          <p class="josh-related-card__title">${item.title}</p>
          <p class="josh-related-card__excerpt">${item.excerpt || joshArticleSubtitle(item)}</p>
        </a>
      `).join('')}
    </div>
  </section>`;
}

function joshPostUpdatedLabel(entry) {
  const raw = String(entry?.updated || entry?.date || entry?.period || '').trim();
  if (!raw) return '';
  if (/^\d{4}-\d{2}-\d{2}/.test(raw) && typeof formatDate === 'function') {
    return formatDate(raw);
  }
  return raw;
}

function joshPostNavItemMarkup(item, { label, next = false } = {}) {
  const nextClass = next ? ' josh-post-nav__item--next' : '';
  return `<a class="josh-post-nav__item${nextClass}" href="${item.href}">
    <p class="josh-post-nav__label">${label}</p>
    <p class="josh-post-nav__title">${item.title}</p>
  </a>`;
}

function joshPostNavMarkup(prevItem, nextItem, options = {}) {
  const prevLabel = options.prevLabel || '上一篇';
  const nextLabel = options.nextLabel || '下一篇';
  const ariaLabel = options.ariaLabel || '文章导航';
  if (!prevItem && !nextItem) return '';

  if (prevItem && nextItem) {
    return `<nav class="josh-post-nav" aria-label="${ariaLabel}">
      ${joshPostNavItemMarkup(prevItem, { label: prevLabel })}
      ${joshPostNavItemMarkup(nextItem, { label: nextLabel, next: true })}
    </nav>`;
  }

  const soloClass = nextItem ? ' josh-post-nav--solo-next' : ' josh-post-nav--solo-prev';
  const soloItem = nextItem || prevItem;
  const soloLabel = nextItem ? nextLabel : prevLabel;
  return `<nav class="josh-post-nav josh-post-nav--solo${soloClass}" aria-label="${ariaLabel}">
    ${joshPostNavItemMarkup(soloItem, { label: soloLabel, next: Boolean(nextItem) })}
  </nav>`;
}

const JOSH_HEART_SHAPE_PATH =
  'M13.2537 0.0255029C23.4033 0.0255029 25.0273 10.5191 25.0273 10.5191C25.0273 10.5191 26.6512 -0.60088 37.6129 0.0255029C44.3441 0.410148 48.7484 6.32169 48.9804 12.1981C49.7924 32.7656 28.7678 41.5 25.0273 41.5C21.2868 41.5 -0.549833 32.3459 1.07416 12.1981C1.54782 6.32169 6.29929 0.0255029 13.2537 0.0255029Z';

function joshHeartUid(slug) {
  return String(slug || 'post').replace(/[^a-z0-9]/gi, '').slice(0, 12) || 'post';
}

function joshHeartMarkup(slug, options = {}) {
  const uid = `${joshHeartUid(slug)}${options.instance || ''}`;
  return `<div class="josh-post-toc__like">
    <div class="josh-heart" data-slug="${slug}">
      <button type="button" class="josh-heart__btn" aria-label="为这篇内容点赞">
        <svg class="josh-heart__svg" width="48" height="42" viewBox="0 0 50 42" fill="none" aria-hidden="true">
          <defs>
            <linearGradient id="josh-heart-active-${uid}" x1="25" y1="42" x2="26.38" y2="0.05" gradientUnits="userSpaceOnUse">
              <stop stop-color="hsl(353deg, 100%, 52%)"></stop>
              <stop offset="1" stop-color="hsl(313deg, 100%, 52%)"></stop>
            </linearGradient>
            <linearGradient id="josh-heart-inactive-${uid}" x1="15" y1="41" x2="42" y2="-1.5" gradientUnits="userSpaceOnUse">
              <stop stop-color="var(--josh-heart-lower)" stop-opacity="0.8"></stop>
              <stop offset="1" stop-color="var(--josh-heart-upper)" stop-opacity="0.8"></stop>
            </linearGradient>
          </defs>
          <path class="josh-heart__fill josh-heart__fill--idle" d="${JOSH_HEART_SHAPE_PATH}" fill="url(#josh-heart-inactive-${uid})"></path>
          <path class="josh-heart__fill josh-heart__fill--liked" d="${JOSH_HEART_SHAPE_PATH}" fill="url(#josh-heart-active-${uid})"></path>
          <g class="josh-heart__face">
            <circle cx="18.2" cy="17.5" r="2.1" fill="var(--josh-heart-face)"></circle>
            <circle cx="31.8" cy="17.5" r="2.1" fill="var(--josh-heart-face)"></circle>
            <path d="M18.5 25.5 Q25 29.5 31.5 25.5" stroke="var(--josh-heart-face)" stroke-width="1.75" stroke-linecap="round" fill="none"></path>
          </g>
        </svg>
      </button>
      <p class="josh-heart__count" aria-live="polite"></p>
    </div>
  </div>`;
}

const JOSH_CALLOUT_TAB_PATH_INFO =
  'M54 0V0.716804C54 25.9434 35.0653 47.1517 10 50L0 57V0H54Z';
const JOSH_CALLOUT_TAB_ACCENT_INFO =
  'M56.9961 4.15364C57.0809 2.49896 55.8083 1.08879 54.1536 1.00394C52.499 0.919082 51.0888 2.19168 51.0039 3.84636L56.9961 4.15364ZM9.09704 51.7557L8.49716 48.8163L9.09704 51.7557ZM6 69V59.2227H0V69H6ZM9.69692 54.6951L14.3373 53.7481L13.1375 47.8693L8.49716 48.8163L9.69692 54.6951ZM14.3373 53.7481C38.202 48.8777 55.7486 28.4783 56.9961 4.15364L51.0039 3.84636C49.8967 25.4384 34.3213 43.5461 13.1375 47.8693L14.3373 53.7481ZM6 59.2227C6 57.0268 7.54537 55.1342 9.69692 54.6951L8.49716 48.8163C3.55195 49.8255 0 54.1756 0 59.2227H6Z';
const JOSH_CALLOUT_TAB_PATH_WARNING =
  'M54.3545 15.0483L46 0.5H0V64L5.42113 56.1695C7.66253 52.9319 11.3497 51 15.2874 51H33.542C51.9925 51 63.5426 31.0483 54.3545 15.0483Z';
const JOSH_CALLOUT_TAB_ACCENT_WARNING =
  'M52.1101 4.52096C51.2932 3.07946 49.4625 2.57308 48.021 3.38993C46.5795 4.20678 46.0731 6.03754 46.8899 7.47904L52.1101 4.52096ZM15 54H37.2466V48H15V54ZM0 63V69.5H6V63H0ZM46.8899 7.47904L53.777 19.6328L58.9972 16.6747L52.1101 4.52096L46.8899 7.47904ZM37.2466 54C56.4023 54 68.4412 33.3405 58.9972 16.6747L53.777 19.6328C60.9545 32.2988 51.8049 48 37.2466 48V54ZM15 48C6.71573 48 0 54.7157 0 63H6C6 58.0294 10.0294 54 15 54V48Z';

function joshCalloutTabMarkup(variant) {
  const isWarning = variant === 'warning';
  const viewBox = isWarning ? '0 0 75 70' : '0 0 57 69';
  const width = isWarning ? '37.5' : '28.5';
  const height = isWarning ? '35' : '34.5';
  const bgPath = isWarning ? JOSH_CALLOUT_TAB_PATH_WARNING : JOSH_CALLOUT_TAB_PATH_INFO;
  const accentPath = isWarning ? JOSH_CALLOUT_TAB_ACCENT_WARNING : JOSH_CALLOUT_TAB_ACCENT_INFO;
  return `<div class="josh-callout__rail" aria-hidden="true">
    <div class="josh-callout__tab">
      <svg class="josh-callout__tab-svg" xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${viewBox}" fill="none" preserveAspectRatio="none" aria-hidden="true">
        <path class="josh-callout__tab-bg" d="${bgPath}"></path>
        <path class="josh-callout__tab-accent" d="${accentPath}"></path>
      </svg>
    </div>
    <span class="josh-callout__stripe"></span>
  </div>`;
}

function joshCalloutIconMarkup(variant) {
  if (variant === 'success') {
    return `<div class="josh-callout__icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.801 10A10 10 0 1 1 17 3.335"></path><path d="m9 11 3 3L22 4"></path></svg></div>`;
  }
  if (variant === 'warning') {
    return `<div class="josh-callout__icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"></path><path d="M12 9v4"></path><path d="M12 17h.01"></path></svg></div>`;
  }
  return `<div class="josh-callout__icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg></div>`;
}

const JOSH_CALLOUT_TITLE_RE =
  /^(key takeaways|适用读者|适用场景|前置|阅读前提|阅读前|阅读基础|intended audience|browser support|what.?s the browser|浏览器支持|浏览器|不支持|延伸阅读|注意|警告|踩坑|踩坑提醒|关键提示|gotcha|说明|准确说明|重要区分|重要澄清|直接回答|第一原则|一句话|痛点|关键设计|关键|澄清|综合建议|总结|提醒|基础|前提|考虑|可定制|工作原理|动效敏感|滚动触发|scroll[- ]triggered|considering motion|motion sensitiv|fill mode)/i;

function joshNormalizeCalloutTitle(title) {
  return (title || '').trim().replace(/[:：]\s*$/, '');
}

function joshIsCalloutTitle(title) {
  const raw = (title || '').trim();
  if (!raw || raw.length > 80) return false;
  if (/^[""「『"']/.test(raw)) return false;
  const normalized = joshNormalizeCalloutTitle(raw);
  if (JOSH_CALLOUT_TITLE_RE.test(normalized)) return true;
  if (/[:：]$/.test(raw) && raw.length <= 36) return true;
  return false;
}

function preprocessJoshMarkdown(markdown) {
  return String(markdown || '').replace(
    /^::: callout(?:\s+(info|warning|success|blue|coral|teal))?\s*\n([\s\S]*?)\n:::/gm,
    (_, variant, body) => {
      const v = String(variant || 'info').toLowerCase();
      const legacyClass = ['blue', 'coral', 'teal'].includes(v) ? ` ${v}` : '';
      return `<div class="callout${legacyClass}">\n${body.trim()}\n</div>`;
    },
  );
}

function joshParseJoshMarkdown(markdown) {
  const normalized = preprocessJoshMarkdown(markdown);
  if (typeof marked === 'undefined') return normalized;
  return marked.parse(normalized);
}

function joshInferCalloutVariant(titleText, className = '') {
  const text = `${joshNormalizeCalloutTitle(titleText)} ${className}`.toLowerCase();
  if (/warning|warn|注意|警告|踩坑|gotcha|caution|alert|motion sensitiv|browser support|浏览器|不支持|关键提示|动效敏感|fill mode/.test(text) || /\bcoral\b/.test(className)) {
    return 'warning';
  }
  if (/success|takeaway|key takeaway|要点|总结|key point|credit/.test(text) || /\bteal\b/.test(className)) {
    return 'success';
  }
  if (/\bblue\b/.test(className)) return 'info';
  return 'info';
}

function joshBuildCalloutMarkup(variant, title, bodyHtml) {
  const titleHtml = title
    ? `<strong class="josh-callout__title">${title}</strong>`
    : '';
  return `${joshCalloutTabMarkup(variant)}${joshCalloutIconMarkup(variant)}${titleHtml}<div class="josh-callout__body">${bodyHtml}</div>`;
}

function joshUpgradeCalloutElement(el, variant, title, bodyHtml) {
  const aside = document.createElement('aside');
  aside.className = `josh-callout josh-callout--${variant}`;
  aside.setAttribute('data-josh-enhanced', 'true');
  aside.innerHTML = joshBuildCalloutMarkup(variant, title, bodyHtml);
  el.replaceWith(aside);
  return aside;
}

function joshExtractCalloutTitle(root) {
  const strong = root.querySelector(':scope > strong, :scope > p > strong, :scope > p > b, :scope > b');
  if (!strong) return { title: '', bodyHtml: root.innerHTML };

  const title = strong.textContent.trim();
  const hostP = strong.closest('p');
  strong.remove();

  if (hostP && hostP.parentElement === root && !hostP.textContent.trim()) {
    hostP.remove();
  }

  return { title, bodyHtml: root.innerHTML.trim() };
}

function enhanceJoshProseCallouts(scope = document) {
  const proseRoots = scope.querySelectorAll('.josh-prose');
  proseRoots.forEach((prose) => {
    prose.querySelectorAll('.callout:not([data-josh-enhanced])').forEach((el) => {
      const variant = joshInferCalloutVariant('', el.className);
      const { title, bodyHtml } = joshExtractCalloutTitle(el);
      joshUpgradeCalloutElement(el, variant, title || '提示', bodyHtml);
    });

    prose.querySelectorAll('blockquote:not([data-josh-enhanced]):not(.quote):not(.josh-quote)').forEach((el) => {
      const { title, bodyHtml } = joshExtractCalloutTitle(el);
      if (!title || !bodyHtml || !joshIsCalloutTitle(title)) return;
      const variant = joshInferCalloutVariant(title, '');
      joshUpgradeCalloutElement(el, variant, joshNormalizeCalloutTitle(title), bodyHtml);
    });

    prose.querySelectorAll('.josh-callout:not([data-josh-enhanced])').forEach((el) => {
      let variant = 'info';
      if (el.classList.contains('josh-callout--warning')) variant = 'warning';
      else if (el.classList.contains('josh-callout--success')) variant = 'success';
      else if (el.classList.contains('josh-callout--info')) variant = 'info';
      else variant = joshInferCalloutVariant(el.querySelector('.josh-callout__title, strong')?.textContent || '', el.className);

      const existingTitle = el.querySelector('.josh-callout__title, strong');
      const existingBody = el.querySelector('.josh-callout__body');
      const title = existingTitle?.textContent?.trim() || '';
      const bodyHtml = existingBody?.innerHTML?.trim() || el.innerHTML.trim();
      el.setAttribute('data-josh-enhanced', 'true');
      el.className = `josh-callout josh-callout--${variant}`;
      el.innerHTML = joshBuildCalloutMarkup(variant, title, bodyHtml);
    });
  });
}

function joshReadMoreArrowMarkup() {
  return `<span class="josh-read-more__arrow" aria-hidden="true">
    <svg class="josh-read-more__arrow-svg" width="36" height="12" viewBox="0 0 36 12" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path class="josh-read-more__arrow-main" d="M0.75 6H11.25 M6 0.75L11.25 6L6 11.25"></path>
      <path class="josh-read-more__arrow-tail" d="M15 10L19.5 5.5L15 1"></path>
      <path class="josh-read-more__arrow-tail josh-read-more__arrow-tail--66" d="M23 10L27.5 5.5L23 1"></path>
      <path class="josh-read-more__arrow-tail josh-read-more__arrow-tail--35" d="M31 10L35.5 5.5L31 1"></path>
    </svg>
  </span>`;
}

function joshArticleSubtitle(post) {
  if (post.subtitle) return post.subtitle;
  if (post.excerpt) {
    const hook = post.excerpt.split(/[。！？.!?]/)[0].trim();
    if (hook && hook.length <= 80) return hook;
  }
  return post.category || '';
}

function joshPostHeroTitleMarkup(title, subtitle, options = {}) {
  const includeSubtitle = options.includeSubtitle !== false;
  const sub = includeSubtitle ? String(subtitle || '').trim() : '';
  const subHtml = sub ? `<span class="josh-post-title__sub">${sub}</span>` : '';
  return `<h1 class="josh-post-title"><span class="josh-post-title__main">${title}</span>${subHtml}</h1>`;
}

function joshProjectMetaMarkup(project) {
  const primaryTag = project.tags?.[0];
  const categoryHtml = primaryTag
    ? `<a class="josh-post-meta__link" href="${Routes.tag(primaryTag)}">${primaryTag}</a>`
    : `<a class="josh-post-meta__link" href="${Routes.projects()}">作品</a>`;
  const readTime = joshProjectReadTime(project);
  const published = String(project.period || '').trim();
  const readTimeHtml = readTime
    ? `<span class="josh-post-meta__sep" aria-hidden="true">·</span><span>${readTime}</span>`
    : '';
  return `<div class="josh-post-meta" role="contentinfo" aria-label="项目元信息">
    <span>收录于</span>
    ${categoryHtml}
    <span class="josh-post-meta__sep" aria-hidden="true">·</span>
    <span>发布于</span>
    <span>${published}</span>
    ${readTimeHtml}
  </div>`;
}

function joshProjectProseLinksMarkup(project) {
  const links = project.links || {};
  const items = [];
  if (!joshIsPlaceholderLink(links.demo)) {
    items.push(`<a href="${links.demo}" target="_blank" rel="noopener noreferrer">${joshProjectDemoLabel(links.demo)}</a>`);
  }
  if (links.github && links.github !== '#') {
    items.push(`<a href="${links.github}" target="_blank" rel="noopener noreferrer">GitHub</a>`);
  }
  if (!joshIsPlaceholderLink(links.docs)) {
    items.push(`<a href="${links.docs}" target="_blank" rel="noopener noreferrer">技术文档</a>`);
  }
  if (!items.length) return '';
  return `<p class="josh-project-prose-links">${items.join(' · ')}</p>`;
}

function joshArticleMarkup(post, hidden) {
  const subtitle = joshArticleSubtitle(post);
  return `<article class="josh-article${hidden ? ' is-hidden' : ''}" data-slug="${post.slug}">
    <a class="josh-article__title" href="${Routes.post(post.slug)}">${post.title}</a>
    <p class="josh-article__subtitle">${subtitle}</p>
    <p class="josh-article__desc">${post.excerpt}</p>
    <a class="josh-read-more" href="${Routes.post(post.slug)}">
      <span>阅读全文</span>
      ${joshReadMoreArrowMarkup()}
    </a>
  </article>`;
}

function joshCategoryPillMarkup(cat, options = {}) {
  const { uniformColor = false } = options;
  const colorStyle = uniformColor
    ? ''
    : ` style="--josh-pill-color: ${JOSH_CATEGORY_COLORS[cat.name] || '#ddeef8'}"`;
  return `<a class="josh-pill" href="${Routes.category(cat.name)}"${colorStyle}>
    <span class="josh-pill__bg" aria-hidden="true"></span>
    ${cat.name}
  </a>`;
}

function joshHomeCategoryPillMarkup(cat) {
  return joshCategoryPillMarkup(cat, { uniformColor: true });
}

function joshCategoryPostCount(cat) {
  if (posts.length) {
    return posts.filter((p) => p.category === cat.name).length;
  }
  if (typeof cat.count === 'number') return cat.count;
  return 0;
}

const JOSH_CATEGORY_ICON_SVG = {
  brain: '<path d="M12 18V5"/><path d="M15 13a4.17 4.17 0 0 1-3-4 4.17 4.17 0 0 1-3 4"/><path d="M17.598 6.5A3 3 0 1 0 12 5a3 3 0 1 0-5.598 1.5"/><path d="M17.997 5.125a4 4 0 0 1 2.526 5.77"/><path d="M18 18a4 4 0 0 0 2-7.464"/><path d="M19.967 17.483A4 4 0 1 1 12 18a4 4 0 1 1-7.967-.517"/><path d="M6 18a4 4 0 0 1-2-7.464"/><path d="M6.003 5.125a4 4 0 0 0-2.526 5.77"/>',
  boxes: '<path d="M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z"/><path d="M7 16.5 4.74 2.85"/><path d="M7 16.5 5-3"/><path d="M7 16.5v5.17"/><path d="M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z"/><path d="M17 16.5-5-3"/><path d="M17 16.5 4.74-2.85"/><path d="M17 16.5v5.17"/><path d="M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z"/><path d="M12 8 7.26 5.15"/><path d="m12 8 4.74-2.85"/><path d="M12 13.5V8"/>',
  'sliders-horizontal': '<path d="M10 5H3"/><path d="M21 12H3"/><path d="M16 19H3"/><path d="M14 5v14"/><path d="M18 12v7"/>',
  wrench: '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76Z"/>',
  'scan-eye': '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="1"/><path d="M18.944 12.944a8 8 0 0 0-8-8 8 8 0 0 0-8 8 8 8 0 0 0 8 8 8 8 0 0 0 8-8Z"/>',
  'file-text': '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
  lightbulb: '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  bot: '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
  'chart-bar': '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M7 16h8"/><path d="M7 11h12"/><path d="M7 6h3"/>',
};

function joshCategoryIconMarkup(iconName) {
  const paths = JOSH_CATEGORY_ICON_SVG[iconName] || JOSH_CATEGORY_ICON_SVG.boxes;
  return `<span class="josh-category-card__icon-wrap" aria-hidden="true">
    <svg class="josh-category-card__icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${paths}</svg>
  </span>`;
}

function joshRainbowHue(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) {
    hash = ((hash << 5) - hash + label.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % 360;
}

function joshCategoryCardMarkup(cat) {
  const hue = joshRainbowHue(cat.name);
  const count = joshCategoryPostCount(cat);
  const desc = String(cat.desc || '').trim();
  const descHtml = desc ? `<p class="josh-category-card__desc">${desc}</p>` : '';
  const iconHtml = cat.icon ? joshCategoryIconMarkup(cat.icon) : '';
  return `<a class="josh-category-card" href="${Routes.category(cat.name)}" style="--josh-tag-hue: ${hue}">
    ${iconHtml}
    <h2 class="josh-category-card__title">${cat.name}</h2>
    ${descHtml}
    <span class="josh-category-card__count">${count} 篇文章</span>
  </a>`;
}

function joshCategoryCardGridMarkup(cats) {
  if (!cats.length) {
    return '<p class="josh-archive-pills__empty">暂无分类</p>';
  }
  return `<div class="josh-category-grid" role="list">
    ${cats.map(joshCategoryCardMarkup).join('')}
  </div>`;
}

function joshTagsStatsMarkup(stats) {
  const { tagCount, postCount, hottestCount, avgCount } = stats;
  const items = [
    [tagCount, '标签总数', 350],
    [postCount, '文章总数', 45],
    [hottestCount, '最热标签', 145],
    [avgCount, '平均文章数', 230],
  ];
  return `<div class="josh-stat-grid" role="list">
    ${items.map(([value, label, hue]) => `
    <div class="josh-stat-card" role="listitem" style="--josh-tag-hue: ${hue}">
      <p class="josh-stat-card__value">${value}</p>
      <p class="josh-stat-card__label">${label}</p>
    </div>`).join('')}
  </div>`;
}

function joshTagCloudItemMarkup(tag, count, maxCount, minCount) {
  const sizeClass = joshTagCloudSizeClass(count, maxCount, minCount);
  const hue = joshRainbowHue(tag);
  return `<a class="josh-tag-cloud__item ${sizeClass}" href="${Routes.tag(tag)}" style="--josh-tag-hue: ${hue}">
    <span>#${tag}</span>
    <span class="josh-tag-cloud__count">${count}</span>
  </a>`;
}

function joshTagCloudMarkup(sortedTags) {
  if (!sortedTags.length) {
    return '<p class="josh-archive-pills__empty">暂无标签</p>';
  }
  const counts = sortedTags.map(([, list]) => list.length);
  const maxCount = Math.max(...counts);
  const minCount = Math.min(...counts);
  const items = sortedTags
    .map(([tag, list]) => joshTagCloudItemMarkup(tag, list.length, maxCount, minCount))
    .join('');
  return `<div class="josh-tag-cloud josh-tags-page__cloud" role="navigation" aria-label="全部标签">
    <div class="josh-tag-cloud__inner">${items}</div>
  </div>`;
}

function joshHotTagCardMarkup(tag, tagPosts) {
  const count = tagPosts.length;
  const sorted = [...tagPosts].sort((a, b) => new Date(b.date) - new Date(a.date));
  const previewItems = sorted.slice(0, 2).map((post) => {
    const title = String(post.title || '').trim();
    return title ? `<li class="josh-tag-card__preview-item">${title}</li>` : '';
  }).join('');
  const previewsHtml = previewItems
    ? `<ul class="josh-tag-card__previews">${previewItems}</ul>`
    : '';
  const moreHtml = count > 2
    ? `<p class="josh-tag-card__more-link">更多</p>`
    : '';
  return `<a class="josh-tag-card josh-tag-card--hot" href="${Routes.tag(tag)}">
    <div class="josh-tag-card__head">
      <span class="josh-tag-card__name">#${tag}</span>
      <span class="josh-tag-card__badge">${count}</span>
    </div>
    ${previewsHtml}
    ${moreHtml}
  </a>`;
}

function joshTagsHotSectionMarkup(sortedTags, limit = 6) {
  const hotTags = sortedTags.slice(0, limit);
  if (!hotTags.length) return '';
  const cards = hotTags.map(([tag, tagPosts]) => joshHotTagCardMarkup(tag, tagPosts)).join('');
  return `<section class="josh-tags-page__hot" aria-labelledby="josh-tags-hot-heading">
    <h2 class="josh-tags-page__hot-title" id="josh-tags-hot-heading">热门标签</h2>
    <div class="josh-tag-card-grid" role="list">${cards}</div>
  </section>`;
}

function joshTagIndexPillMarkup(tag, count, tier = 'more') {
  const tierColors = {
    hot: 'var(--josh-color-primary)',
    warm: 'var(--josh-color-cloud-400)',
    more: 'var(--josh-color-cloud-300)',
  };
  const pillColor = tierColors[tier] || tierColors.more;
  const countHtml = tier === 'more' && count === 1
    ? ''
    : `<span class="josh-pill__count">${count}</span>`;
  return `<a class="josh-pill josh-pill--tag-index josh-pill--tag-${tier}" href="${Routes.tag(tag)}" style="--josh-pill-color: ${pillColor}">
    <span class="josh-pill__bg" aria-hidden="true"></span>
    <span>#${tag}</span>
    ${countHtml}
  </a>`;
}

function joshCollectRelatedTags(postList, excludeTags = []) {
  const exclude = new Set(excludeTags);
  const counts = new Map();
  postList.forEach((post) => {
    (post.tags || []).forEach((tag) => {
      if (exclude.has(tag)) return;
      counts.set(tag, (counts.get(tag) || 0) + 1);
    });
  });
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

function joshArchiveRelatedTagsMarkup(postList, options = {}) {
  const excludeTags = options.excludeTags || [];
  const limit = options.limit ?? 10;
  const heading = options.heading || '相关标签';
  const related = joshCollectRelatedTags(postList, excludeTags).slice(0, limit);
  if (!related.length) return '';
  return `<section class="josh-archive-related" aria-labelledby="josh-archive-related-tags-heading">
    <h2 class="josh-section-label" id="josh-archive-related-tags-heading">${heading}</h2>
    <div class="josh-pills josh-archive-pills josh-archive-pills--tags" role="list">
      ${related.map(([tag, count]) => {
        const tier = count >= 3 ? 'hot' : count === 2 ? 'warm' : 'more';
        return joshTagIndexPillMarkup(tag, count, tier);
      }).join('')}
    </div>
  </section>`;
}

function joshArchiveTagGroupMarkup(id, label, tagEntries) {
  if (!tagEntries.length) return '';
  return `<section class="josh-archive-tag-group" aria-labelledby="josh-tag-group-${id}">
    <h2 class="josh-section-label" id="josh-tag-group-${id}">${label}</h2>
    <div class="josh-pills josh-archive-pills josh-archive-pills--tags" role="list">
      ${tagEntries.map(([tag, tagPosts]) => joshTagIndexPillMarkup(tag, tagPosts.length, id)).join('')}
    </div>
  </section>`;
}

function joshArchiveTagGroupsMarkup(sortedTags) {
  if (!sortedTags.length) {
    return '<p class="josh-archive-pills__empty">暂无标签</p>';
  }
  const hot = sortedTags.filter(([, list]) => list.length >= 3);
  const warm = sortedTags.filter(([, list]) => list.length === 2);
  const more = sortedTags.filter(([, list]) => list.length === 1);
  const groups = [
    joshArchiveTagGroupMarkup('hot', '热门标签', hot),
    joshArchiveTagGroupMarkup('warm', '常用标签', warm),
    joshArchiveTagGroupMarkup('more', '更多标签', more),
  ].filter(Boolean);
  if (!groups.length) {
    return `<div class="josh-pills josh-archive-pills josh-archive-pills--tags" role="list">
      ${sortedTags.map(([tag, tagPosts]) => joshTagIndexPillMarkup(tag, tagPosts.length)).join('')}
    </div>`;
  }
  return `<div class="josh-archive-tag-groups">${groups.join('')}</div>`;
}

function joshBlogArchiveHeaderMarkup(title, countText) {
  return `<header class="josh-blog-archive__header">
    <h1 class="josh-blog-archive__title">${title}</h1>
    <span class="josh-blog-archive__count">${countText}</span>
  </header>`;
}

function joshBlogArchiveDetailHeaderMarkup(title, countText, backHref, backLabel) {
  return `<header class="josh-blog-archive__header josh-blog-archive__header--detail">
    <div class="josh-blog-archive__header-start">
      ${joshArchiveBackLinkMarkup(backHref, backLabel, 'josh-archive-back--header')}
    </div>
    <h1 class="josh-blog-archive__title">${title}</h1>
    <span class="josh-blog-archive__count">${countText}</span>
  </header>`;
}

function joshArchiveBackLinkMarkup(href, label, className = '', options = {}) {
  const extraClass = className ? ` ${className}` : '';
  const showArrow = options.showArrow !== false;
  const arrowHtml = showArrow ? '<span aria-hidden="true">←</span>\n    ' : '';
  return `<a class="josh-back-link josh-archive-back${extraClass}" href="${href}">
    ${arrowHtml}<span>${label}</span>
  </a>`;
}

function joshArchiveRecentPostsMarkup(limit = 2) {
  const recent = [...posts]
    .sort((a, b) => new Date(b.date) - new Date(a.date))
    .slice(0, limit);
  if (!recent.length) return '';
  return `<section class="josh-archive-preview" aria-labelledby="josh-archive-preview-heading">
    <h2 class="josh-section-label" id="josh-archive-preview-heading">最新文章</h2>
    ${joshPostBlogArchiveGridMarkup(recent, { gridClass: 'josh-blog-grid josh-blog-grid--preview' })}
  </section>`;
}

function joshPostTagMarkup(tag) {
  return `<a class="josh-pill" href="${Routes.tag(tag)}" style="--josh-pill-color: var(--josh-color-cloud-300)">
    <span class="josh-pill__bg" aria-hidden="true"></span>
    ${tag}
  </a>`;
}

function joshPostTailTagMarkup(tag) {
  return joshTagRainbowPillMarkup(tag);
}

function joshTagRainbowPillMarkup(tag) {
  const hue = joshRainbowHue(tag);
  return `<a class="josh-tag-rainbow-pill" href="${Routes.tag(tag)}" style="--josh-tag-hue: ${hue}">${tag}</a>`;
}

function joshPageIntroMarkup({ label, title, description, backHref, backLabel }) {
  const backHtml = backHref ? `
    <a class="josh-back-link" href="${backHref}">
      <span aria-hidden="true">←</span>
      <span>${backLabel || '返回'}</span>
    </a>
  ` : '';
  const descHtml = description ? `<p class="josh-page-desc">${description}</p>` : '';
  return `
    <header class="josh-page-intro">
      ${backHtml}
      <p class="josh-section-label">${label}</p>
      <h1 class="josh-page-title">${title}</h1>
      ${descHtml}
    </header>
  `;
}

function joshArticleMetaMarkup(post) {
  return `<p class="josh-article__meta">
    <time datetime="${post.date}">${formatDate(post.date)}</time>
    <span class="josh-post-meta__dot" aria-hidden="true"></span>
    <span>${post.readTime}</span>
    <span class="josh-post-meta__dot" aria-hidden="true"></span>
    <span id="view-list-post-${post.slug}">…</span>
  </p>`;
}

function joshArticleListItemMarkup(post, options = {}) {
  const subtitle = options.subtitle
    ?? (post.tags && post.tags[0] ? post.tags[0] : post.category);
  const tagsHtml = options.showTags && post.tags?.length
    ? `<div class="josh-list-tags">${post.tags.map((t) => joshPostTagMarkup(t)).join('')}</div>`
    : '';
  return `<article class="josh-article" data-slug="${post.slug}">
    ${joshArticleMetaMarkup(post)}
    <a class="josh-article__title" href="${Routes.post(post.slug)}">${post.title}</a>
    <p class="josh-article__subtitle">${subtitle}</p>
    <p class="josh-article__desc">${post.excerpt}</p>
    ${tagsHtml}
    <a class="josh-read-more" href="${Routes.post(post.slug)}">
      <span>阅读全文</span>
      ${joshReadMoreArrowMarkup()}
    </a>
  </article>`;
}

function joshArticleListMarkup(postList, options = {}) {
  if (!postList.length) {
    return `<div class="josh-empty-state"><p>${options.emptyText || '暂无内容'}</p></div>`;
  }
  return `<div class="josh-article-list">
    ${postList.map((post) => joshArticleListItemMarkup(post, options)).join('')}
  </div>`;
}

function joshTagCloudSizeClass(count, maxCount, minCount) {
  if (maxCount === minCount) return 'josh-tag-cloud__item--3';
  const ratio = (count - minCount) / (maxCount - minCount);
  if (ratio >= 0.8) return 'josh-tag-cloud__item--5';
  if (ratio >= 0.6) return 'josh-tag-cloud__item--4';
  if (ratio >= 0.4) return 'josh-tag-cloud__item--3';
  if (ratio >= 0.2) return 'josh-tag-cloud__item--2';
  return 'josh-tag-cloud__item--1';
}

function joshMountListPage(app, mainHtml, activeHref) {
  app.innerHTML = buildJoshPageShell(mainHtml, activeHref);
  queueMicrotask(() => initJoshSiteInteractions(app));
}

function buildJoshSkyListPageShell(heroHtml, mainHtml, activeHref, options = {}) {
  const mainClass = options.mainClass || 'josh-inner-main josh-container';
  const pageClass = ` josh-page--sky-list${options.pageClass || ''}`;
  return `<div class="josh-page${pageClass}">
    ${buildJoshInnerHeaderMarkup(activeHref, { postNavLayers: true })}
    ${heroHtml}
    <div class="${mainClass}">
      ${mainHtml}
    </div>
    ${buildJoshFooterMarkup()}
  </div>`;
}

function joshMountSkyListPage(app, heroHtml, mainHtml, activeHref, options = {}) {
  app.innerHTML = buildJoshSkyListPageShell(heroHtml, mainHtml, activeHref, options);
  queueMicrotask(() => initJoshSiteInteractions(app));
}

function joshMountBlogArchivePage(app, mainHtml, activeHref, options = {}) {
  const navHtml = buildJoshInnerHeaderMarkup(activeHref);
  const archiveHtml = `
    <div class="josh-blog-archive-page">
      <div class="josh-blog-archive-sticky">
        <div class="josh-blog-archive-sticky__blur" aria-hidden="true"></div>
        <div class="josh-blog-archive-sticky__content">
          ${navHtml}
        </div>
      </div>
      <div class="josh-blog-archive-column">
        <div class="josh-blog-archive josh-container">
          ${mainHtml}
        </div>
      </div>
    </div>`;
  app.innerHTML = buildJoshPageShell(archiveHtml, activeHref, {
    pageClass: ` josh-page--blog-archive${options.pageClass || ''}`,
    mainClass: 'josh-blog-archive-main',
    omitHeader: true,
  });
  queueMicrotask(() => initJoshSiteInteractions(app));
}

function joshSkyPageHeroCloudsMarkup() {
  if (typeof joshCloudSvgMarkup !== 'function') return '';
  const horizonHtml = typeof joshSkyHorizonCloudMarkup === 'function'
    ? `<div class="josh-page-hero__clouds-horizon" aria-hidden="true">${joshSkyHorizonCloudMarkup()}</div>`
    : '';
  return `<div class="josh-page-hero__clouds-back" aria-hidden="true">${joshCloudSvgMarkup()}</div>${horizonHtml}`;
}

function joshPageHeroMascotMarkup() {
  if (typeof joshMascotMarkup !== 'function') return '';
  return `<div class="josh-page-hero__decor" aria-hidden="true">
    <div class="josh-page-hero__mascot-ground">
      <div class="josh-page-hero__mascot-lane">${joshMascotMarkup()}</div>
    </div>
  </div>`;
}

function joshSkyPageHeroMarkup({
  label,
  title,
  description,
  backHref,
  backLabel,
  profileHtml,
  innerClass,
  showMascot = true,
}) {
  const backHtml = backHref ? `
    <a class="josh-back-link" href="${backHref}">
      <span aria-hidden="true">←</span>
      <span>${backLabel || '返回'}</span>
    </a>
  ` : '';
  const descHtml = description ? `<p class="josh-page-desc josh-page-hero__desc">${description}</p>` : '';
  const headingHtml = profileHtml || `
    <h1 class="josh-page-title">${title}</h1>
    ${descHtml}
  `;
  const innerCls = innerClass || 'josh-container';
  const decorHtml = showMascot ? joshPageHeroMascotMarkup() : '';
  return `<div class="josh-page-hero">
    <div class="josh-page-hero__band">
      ${joshSkyPageHeroCloudsMarkup()}
      ${decorHtml}
      <div class="josh-page-hero__inner ${innerCls}">
        <header class="josh-page-hero__header">
          ${backHtml}
          <p class="josh-section-label">${label}</p>
          ${headingHtml}
        </header>
      </div>
    </div>
  </div>`;
}

function joshSkyListNavScrollMetrics(innerHeader, pageHero) {
  const headerWrap = innerHeader.querySelector('.josh-sky__header-wrap');
  const headerBottom = headerWrap
    ? headerWrap.getBoundingClientRect().bottom
    : innerHeader.getBoundingClientRect().bottom;
  const heroRect = pageHero.getBoundingClientRect();
  const gap = heroRect.bottom - headerBottom;
  const ramp = JOSH_POST_NAV_SCROLL_RAMP_PX;
  let whiteProgress = 0;
  if (gap <= 0) whiteProgress = 1;
  else if (gap < ramp) whiteProgress = 1 - gap / ramp;

  const heroHeight = Math.max(pageHero.offsetHeight, 1);
  let skyProgress = 1;
  if (heroRect.top < 0) {
    skyProgress = Math.max(0, Math.min(1, 1 + heroRect.top / (heroHeight * 0.45)));
  }

  const over = whiteProgress >= 0.98;
  return { whiteProgress, skyProgress, over };
}

function joshPostNavScrollMetrics(innerHeader, postHero, postBlocker) {
  const headerWrap = innerHeader.querySelector('.josh-sky__header-wrap');
  const headerBottom = headerWrap
    ? headerWrap.getBoundingClientRect().bottom
    : innerHeader.getBoundingClientRect().bottom;
  const ramp = JOSH_POST_NAV_SCROLL_RAMP_PX;

  let whiteProgress = 0;
  let skyProgress = 1;

  if (postHero) {
    const heroRect = postHero.getBoundingClientRect();
    const gap = heroRect.bottom - headerBottom;
    if (gap <= 0) whiteProgress = 1;
    else if (gap < ramp) whiteProgress = 1 - gap / ramp;

    const heroHeight = Math.max(postHero.offsetHeight, 1);
    if (heroRect.top < 0) {
      skyProgress = Math.max(0, Math.min(1, 1 + heroRect.top / (heroHeight * 0.45)));
    }
  }

  if (postBlocker && whiteProgress < 1) {
    const blockerGap = postBlocker.getBoundingClientRect().top - headerBottom;
    let blockerProgress = 0;
    if (blockerGap <= 0) blockerProgress = 1;
    else if (blockerGap < ramp) blockerProgress = 1 - blockerGap / ramp;
    whiteProgress = Math.max(whiteProgress, blockerProgress);
  }

  const over = whiteProgress >= 0.98;
  return { whiteProgress, skyProgress, over };
}

function joshAboutBodyNavScrollMetrics(stickyWrap) {
  const ramp = JOSH_POST_NAV_SCROLL_RAMP_PX;
  const stickyTop = stickyWrap.getBoundingClientRect().top;
  let progress = 0;
  if (stickyTop <= 0) progress = 1;
  else if (stickyTop < ramp) progress = 1 - stickyTop / ramp;
  const over = progress >= 0.98;
  return { progress, over };
}

function joshIsPlaceholderLink(url) {
  if (!url || url === '#' || String(url).trim() === '') return true;
  try {
    const host = new URL(url, location.origin).hostname;
    return host === 'example.com';
  } catch {
    return true;
  }
}

function joshProjectDemoLabel(demoUrl) {
  const url = String(demoUrl || '').toLowerCase();
  if (url.includes('github.io/weblog') || url.includes('/weblog')) return '查看博客';
  return '在线 Demo';
}

function joshProjectLinkButtonsMarkup(project) {
  const links = project.links || {};
  const buttons = [];
  if (!joshIsPlaceholderLink(links.demo)) {
    buttons.push(`<a class="josh-btn josh-btn--primary" href="${links.demo}" target="_blank" rel="noopener noreferrer">${joshProjectDemoLabel(links.demo)}</a>`);
  }
  if (links.github && links.github !== '#') {
    buttons.push(`<a class="josh-btn" href="${links.github}" target="_blank" rel="noopener noreferrer">
      <svg viewBox="0 0 24 24" aria-hidden="true" class="fill-current"><path d="M12 .5A12 12 0 0 0 8.2 23.9c.6.1.8-.3.8-.6v-2.3c-3.3.7-4-1.6-4-1.6-.6-1.5-1.4-1.9-1.4-1.9-1.2-.8.1-.8.1-.8 1.3.1 2 1.4 2 1.4 1.2 2 3.1 1.4 3.9 1.1.1-.9.5-1.4.8-1.8-2.6-.3-5.3-1.3-5.3-5.7 0-1.3.5-2.3 1.3-3.1-.1-.3-.6-1.5.1-3.1 0 0 1.1-.3 3.4 1.2 1-.3 2-.4 3-.4s2 .1 3 .4c2.3-1.6 3.4-1.2 3.4-1.2.7 1.6.2 2.8.1 3.1.8.8 1.3 1.8 1.3 3.1 0 4.4-2.7 5.4-5.3 5.7.5.4.9 1.2.9 2.4v3.5c0 .3.2.7.8.6A12 12 0 0 0 12 .5Z"/></svg>
      GitHub
    </a>`);
  }
  if (!joshIsPlaceholderLink(links.docs)) {
    buttons.push(`<a class="josh-btn" href="${links.docs}" target="_blank" rel="noopener noreferrer">技术文档</a>`);
  }
  return buttons.length ? `<div class="josh-project-actions">${buttons.join('')}</div>` : '';
}

function joshProjectFilterQuery({ status, tag }) {
  const params = new URLSearchParams();
  if (status && status !== '全部') params.set('status', status);
  if (tag && tag !== '全部') params.set('tag', tag);
  const query = params.toString();
  return query ? `${query}` : '';
}

function joshProjectStatusFilterMarkup(statusOptions, activeStatus, activeTag) {
  return `<div class="josh-pills josh-project-filters" role="group" aria-label="按状态筛选">
    ${statusOptions.map((status) => {
      const isActive = status === activeStatus;
      const href = Routes.projects(joshProjectFilterQuery({ status, tag: activeTag }));
      return `<a class="josh-pill josh-project-filter${isActive ? ' is-active' : ''}" data-filter-kind="status" href="${href}"${isActive ? ' aria-current="true"' : ''}>
        <span class="josh-pill__bg" aria-hidden="true"></span>
        ${status}
      </a>`;
    }).join('')}
  </div>`;
}

function joshProjectTagFilterMarkup(tagOptions, activeStatus, activeTag) {
  if (tagOptions.length <= 1) return '';
  return `<div class="josh-pills josh-project-filters josh-project-filters--tags" role="group" aria-label="按标签筛选">
    ${tagOptions.map((tag) => {
      const isActive = tag === activeTag;
      const href = Routes.projects(joshProjectFilterQuery({ status: activeStatus, tag }));
      return `<a class="josh-pill josh-project-filter${isActive ? ' is-active' : ''}" data-filter-kind="tag" href="${href}"${isActive ? ' aria-current="true"' : ''}>
        <span class="josh-pill__bg" aria-hidden="true"></span>
        ${tag}
      </a>`;
    }).join('')}
  </div>`;
}

function joshProjectArchiveSubtitle(project) {
  const subtitle = String(project.subtitle || '').trim();
  const title = String(project.title || '').trim();
  if (!subtitle || subtitle === title) return '';
  return subtitle;
}

function joshProjectArchiveExcerpt(project) {
  const excerpt = String(project.excerpt || project.summary || '').trim();
  return excerpt;
}

function joshBlogArchiveCardMarkup(project) {
  const href = Routes.project(project.slug);
  const subtitle = joshProjectArchiveSubtitle(project);
  const subtitleHtml = subtitle
    ? `<p class="josh-blog-card__subtitle">${subtitle}</p>`
    : '';
  const excerpt = joshProjectArchiveExcerpt(project);
  return `<div class="josh-blog-card" data-slug="${project.slug}">
    <article>
      <a class="josh-blog-card__title" href="${href}">${project.title}</a>
      ${subtitleHtml}
      <p class="josh-blog-card__desc">${excerpt}</p>
      <a class="josh-read-more" href="${href}" aria-label="阅读全文：${project.title}">
        <span>阅读全文</span>
        ${joshReadMoreArrowMarkup()}
      </a>
    </article>
  </div>`;
}

function joshBlogArchiveGridMarkup(projects, options = {}) {
  if (!projects.length) {
    const backHtml = options.emptyBackHref
      ? joshArchiveBackLinkMarkup(
        options.emptyBackHref,
        options.emptyBackLabel || '返回',
        options.emptyBackClass || '',
        { showArrow: options.emptyBackShowArrow },
      )
      : '';
    return `<div class="josh-empty-state">
      <p class="josh-empty-state__text">${options.emptyText || '暂无作品'}</p>
      ${backHtml}
    </div>`;
  }
  const useSparse = options.sparse !== false && projects.length === 1;
  const sparseClass = useSparse ? ' josh-blog-grid--sparse' : '';
  return `<div class="josh-blog-grid${sparseClass}" role="list">
    ${projects.map((project) => joshBlogArchiveCardMarkup(project)).join('')}
  </div>`;
}

function joshPostBlogArchiveCardDesc(post, options = {}) {
  const excerpt = String(post.excerpt || '').trim();
  if (!options.richDesc) return excerpt;

  const subtitle = String(post.subtitle || '').trim();
  if (!subtitle) return excerpt;

  const normalize = (value) => value.replace(/\s+/g, '');
  const excerptNorm = normalize(excerpt);
  const subtitleNorm = normalize(subtitle);
  if (excerptNorm && subtitleNorm && excerptNorm.includes(subtitleNorm.slice(0, Math.min(12, subtitleNorm.length)))) {
    return excerpt;
  }

  const joiner = excerpt && !/[。！？.!?]$/.test(excerpt) ? '。' : '';
  const combined = excerpt ? `${excerpt}${joiner}${subtitle}` : subtitle;
  const maxLen = 320;
  if (combined.length <= maxLen) return combined;

  const clipped = combined.slice(0, maxLen);
  const lastStop = Math.max(clipped.lastIndexOf('。'), clipped.lastIndexOf('！'), clipped.lastIndexOf('？'));
  if (lastStop > maxLen * 0.55) return clipped.slice(0, lastStop + 1);
  return `${clipped.trim()}…`;
}

function joshBlogCardMetaMarkup(post, options = {}) {
  const parts = [];
  if (options.showCategory && post.category) {
    parts.push(`<a class="josh-blog-card__meta-link" href="${Routes.category(post.category)}">${post.category}</a>`);
  }
  if (post.date) {
    parts.push(`<time datetime="${post.date}">${formatDate(post.date)}</time>`);
  }
  if (post.readTime) {
    parts.push(`<span>${post.readTime}</span>`);
  }
  if (!parts.length) return '';

  const inner = parts.map((part, index) => {
    if (index === 0) return part;
    return `<span class="josh-blog-card__meta-sep" aria-hidden="true">·</span>${part}`;
  }).join('');

  return `<p class="josh-blog-card__meta">${inner}</p>`;
}

function joshBlogCardTagsMarkup(post, options = {}) {
  const omit = new Set(options.omitTags || []);
  const tags = (post.tags || []).filter((tag) => !omit.has(tag)).slice(0, options.maxTags || 3);
  if (!tags.length) return '';
  return `<div class="josh-blog-card__tags">${tags.map((tag) => joshTagRainbowPillMarkup(tag)).join('')}</div>`;
}

function joshPostBlogArchiveCardMarkup(post, options = {}) {
  const href = Routes.post(post.slug);
  const showSubtitle = options.showSubtitle !== false;
  const subtitle = showSubtitle ? joshArticleSubtitle(post) : '';
  const subtitleHtml = subtitle
    ? `<p class="josh-blog-card__subtitle">${subtitle}</p>`
    : '';
  const excerpt = joshPostBlogArchiveCardDesc(post, { richDesc: Boolean(options.showMeta) });
  const metaHtml = options.showMeta
    ? joshBlogCardMetaMarkup(post, { showCategory: options.showCategory })
    : '';
  const tagsHtml = options.showTags
    ? joshBlogCardTagsMarkup(post, { omitTags: options.omitTags, maxTags: options.maxTags })
    : '';
  const footerHtml = tagsHtml
    ? `<div class="josh-blog-card__footer">${tagsHtml}
      <a class="josh-read-more" href="${href}" aria-label="阅读全文：${post.title}">
        <span>阅读全文</span>
        ${joshReadMoreArrowMarkup()}
      </a>
    </div>`
    : `<a class="josh-read-more" href="${href}" aria-label="阅读全文：${post.title}">
      <span>阅读全文</span>
      ${joshReadMoreArrowMarkup()}
    </a>`;
  return `<div class="josh-blog-card" data-slug="${post.slug}">
    <article>
      <a class="josh-blog-card__title" href="${href}">${post.title}</a>
      ${metaHtml}
      ${subtitleHtml}
      <p class="josh-blog-card__desc">${excerpt}</p>
      ${footerHtml}
    </article>
  </div>`;
}

function joshPostBlogArchiveGridMarkup(postList, options = {}) {
  if (!postList.length) {
    const backHtml = options.emptyBackHref
      ? joshArchiveBackLinkMarkup(
        options.emptyBackHref,
        options.emptyBackLabel || '返回',
        options.emptyBackClass || '',
        { showArrow: options.emptyBackShowArrow },
      )
      : '';
    return `<div class="josh-empty-state">
      <p class="josh-empty-state__text">${options.emptyText || '暂无文章'}</p>
      ${backHtml}
    </div>`;
  }
  const useSparse = options.sparse !== false && postList.length === 1;
  const sparseClass = useSparse ? ' josh-blog-grid--sparse' : '';
  const gridClass = `${options.gridClass || 'josh-blog-grid'}${sparseClass}`;
  return `<div class="${gridClass}" role="list">
    ${postList.map((post) => joshPostBlogArchiveCardMarkup(post, options)).join('')}
  </div>`;
}

function joshPostTocMarkup(headings, options = {}) {
  if (!headings.length) return '';
  const heartHtml = options.heartSlug ? joshHeartMarkup(options.heartSlug, { instance: 'toc' }) : '';
  const tocLabel = options.tocLabel || '目录';
  const tocAria = options.tocAria || '文章目录';
  return `<div class="josh-post-toc" id="toc-sidebar">
    <nav class="josh-post-toc__nav" aria-label="${tocAria}">
      <h2 class="josh-post-toc__label">${tocLabel}</h2>
      ${headings.map((h) => `<a href="#${h.id}" class="josh-toc-link${h.tag === 'h3' ? ' josh-toc-link--sub' : ''}" data-target="${h.id}">${h.text}</a>`).join('')}
    </nav>
    ${heartHtml}
  </div>`;
}

function joshProjectProseStartsWithHeading(html) {
  return /^<h[1-6][\s>]/i.test(String(html).trim());
}

function joshProjectProseLeadMarkup(project) {
  const lead = String(project.excerpt || project.summary || '').trim();
  if (!lead) return '';
  return `<p>${lead}</p>`;
}

function joshProjectProseMarkup(project) {
  const parts = [];
  const hasMdContent = Boolean(project.content && String(project.content).replace(/<[^>]+>/g, '').trim().length > 40);

  if (hasMdContent) {
    const content = String(project.content).trim();
    if (joshProjectProseStartsWithHeading(content)) {
      const lead = joshProjectProseLeadMarkup(project);
      if (lead) return `${lead}\n${content}`;
    }
    return `<h2 id="introduction">概述</h2>\n${content}`;
  }

  const lead = joshProjectProseLeadMarkup(project);
  if (lead) parts.push(lead);

  if (project.features?.length) {
    parts.push('<h2 id="features">核心功能</h2>');
    parts.push(`<ul>${project.features.map((item) => `<li>${item}</li>`).join('')}</ul>`);
  }

  if (project.architecture?.length) {
    parts.push('<h2 id="architecture">架构设计</h2>');
    parts.push(`<ul>${project.architecture.map((item) => `<li>${item}</li>`).join('')}</ul>`);
  }

  if (project.metrics?.length) {
    parts.push('<h2 id="metrics">关键指标</h2>');
    parts.push(`<ul>${project.metrics.map((item) => `<li>${item}</li>`).join('')}</ul>`);
  }

  if (project.screenshots?.length) {
    parts.push('<h2 id="screenshots">效果图</h2>');
    parts.push(`<div class="josh-screenshot-grid">${project.screenshots.map((image) => `<img src="${image}" alt="${project.title} 截图" loading="lazy">`).join('')}</div>`);
  }

  return parts.join('\n');
}

function bindJoshProjectFilters(app) {
  app.querySelectorAll('.josh-project-filter').forEach((link) => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = new URL(link.href, window.location.origin);
      history.pushState(null, '', `${target.pathname}${target.search}`);
      router();
    });
  });
}

const JOSH_ABOUT_PRIDE_FLAG_GRADIENTS = {
  rainbow: 'linear-gradient(to bottom, var(--rainbow-black, hsl(0deg 0% 18%)) 0%, var(--rainbow-black, hsl(0deg 0% 18%)) 12.5%, hsl(30deg 60% 30%) 12.5%, hsl(30deg 60% 30%) 25%, hsl(0deg 90% 55%) 25%, hsl(0deg 90% 55%) 37.5%, hsl(30deg 95% 65%) 37.5%, hsl(30deg 95% 65%) 50%, hsl(55deg 90% 65%) 50%, hsl(55deg 90% 65%) 62.5%, hsl(100deg 65% 45%) 62.5%, hsl(100deg 65% 45%) 75%, hsl(220deg 80% 55%) 75%, hsl(220deg 80% 55%) 87.5%, hsl(265deg 80% 50%) 87.5%, hsl(265deg 80% 50%) 100%)',
  trans: 'linear-gradient(rgb(113, 200, 244) 0%, rgb(113, 200, 244) 20%, rgb(249, 184, 195) 20%, rgb(249, 184, 195) 40%, rgb(255, 255, 255) 40%, rgb(255, 255, 255) 60%, rgb(249, 184, 195) 60%, rgb(249, 184, 195) 80%, rgb(113, 200, 244) 80%, rgb(113, 200, 244) 100%)',
  pan: 'linear-gradient(rgb(255, 26, 136) 0%, rgb(255, 26, 136) 33.3333%, rgb(255, 213, 0) 33.3333%, rgb(255, 213, 0) 66.6667%, rgb(26, 179, 255) 66.6667%, rgb(26, 179, 255) 100%)',
};

const JOSH_ABOUT_CREDOS = [
  {
    label: '阅读：从 Transformer 论文到科幻小说，杂食。',
    text: '📚 阅读杂食：从 Transformer 论文到科幻小说。',
    subtitle: '最近在看的东西，总比手头的活有趣一点。',
    gradient: JOSH_ABOUT_PRIDE_FLAG_GRADIENTS.rainbow,
    columns: 10,
  },
  {
    label: '户外：徒步和钓鱼，是给大脑清缓存的方式。',
    text: '🎣 徒步和钓鱼，是给大脑清缓存的方式。',
    subtitle: '看风向、换饵、调漂——跟调参差不多；上鱼是副产物。',
    gradient: JOSH_ABOUT_PRIDE_FLAG_GRADIENTS.trans,
    columns: 10,
  },
  {
    label: '交互设计：痴迷把抽象原理画成能动的图。',
    text: '🎨 痴迷把抽象原理画成能动的图。',
    subtitle: '如果一张图不能让人「哦——」，那就重画。',
    gradient: JOSH_ABOUT_PRIDE_FLAG_GRADIENTS.pan,
    columns: 10,
  },
  {
    label: '咖啡：手冲爱好者，偏浅烘的酸。',
    text: '☕ 手冲爱好者，偏浅烘的酸。',
    subtitle: '写代码前磨一杯豆子，比任何仪式感都管用。',
    gradient: JOSH_ABOUT_PRIDE_FLAG_GRADIENTS.rainbow,
    columns: 10,
  },
];

const JOSH_ABOUT_DESK_RADAR = [
  { label: '检索系统', score: 0.88 },
  { label: '模型训练', score: 0.72 },
  { label: '性能工程', score: 0.82 },
  { label: '工程实践', score: 0.78 },
];

const JOSH_ABOUT_DESK_TAGS = ['Python', 'PyTorch', 'FastAPI', 'Docker', 'Redis', 'PostgreSQL'];

const JOSH_ABOUT_DESK_CREDOS = [
  {
    text: '演示环境 1.9 秒，老板点头；灰度一周，P95 到了 4.2 秒，工单来了。',
    subtitle: '先画延迟瀑布，别在 Embedding 上空转。',
  },
  {
    text: '相关性上去，延迟未必好看。',
    subtitle: '瓶颈在生成和 Rerank，不在更大的 Embedding。',
  },
  {
    text: '没 Trace，优化容易打在错觉上。',
    subtitle: '先分阶段量 latency，再动 Top-K 和缓存。',
  },
  {
    text: '默认初始化，有时会悄悄使坏。',
    subtitle: '十几行 Embedding，也能让训练前期失稳。',
  },
  {
    text: '纯向量检索，会漏掉字面匹配。',
    subtitle: 'SKU、错误码、专有名词，往往要 BM25 补位。',
  },
];

const JOSH_ABOUT_PRIDE_FLAG_COLUMN_DELAYS_MS = [-1000, -900, -800, -700, -600, -500, -400, -300, -200, -100];
const JOSH_ABOUT_PRIDE_FLAG_COLUMN_SINS_DEG = [0, 45, 90, 135, 180, 225, 270, 315, 360, 405];

const JOSH_ABOUT_SIDE_LINKS = [
  { slug: 'transformer-in-depth', label: '《Transformer 原理》' },
  { slug: 'attention-from-scratch', label: '《Attention 从零实现》' },
];

const JOSH_ABOUT_JOB_CATEGORY_LINKS = [
  { label: '生产 RAG 与检索链路', category: 'RAG 与检索' },
  { label: '智能体与工具调用', category: '智能体' },
  { label: '原理与手写实现', category: '基础原理' },
  { label: '微调、对齐与训练', category: '微调与对齐' },
];

const JOSH_ABOUT_POST_HOOKS = {
  'rag-production-performance-optimization': {
    hook: '相关性拉到 0.86，P95 却从 1.8s 涨到 4.2s',
    takeaway: '先画延迟瀑布，别在 Embedding 上空转',
  },
  'claude-code-best-practices': {
    hook: 'Agent 看起来很强，翻车往往更早',
    takeaway: '先把 CLAUDE.md 和 Plan 写稳，再谈编排',
  },
  'transformer-in-depth': {
    hook: '注意力不是背完公式就结束',
    takeaway: 'QKV、位置编码、训练策略要连成一条线',
  },
  'attention-from-scratch': {
    hook: '手写 Attention，坑在 mask 和 shape',
    takeaway: '实现细节比推导公式更耗时间',
  },
  'embedding-from-scratch': {
    hook: 'Embedding 像查表，直到 loss 前期开始抖',
    takeaway: '初始化尺度会和位置编码打架',
  },
  'embedding-finetune-domain-rag': {
    hook: '通用 bge 搜不对领域里的黑话',
    takeaway: '用 Recall@K 决定要不要微调，别凭感觉',
  },
  'rag-hybrid-retrieval-strategy': {
    hook: '纯向量会漏 SKU、漏错误码',
    takeaway: 'BM25 和向量并行，RRF 融合是常态',
  },
  'rag-production-refactor': {
    hook: '上线第二周，embedding 账单吃掉四成预算',
    takeaway: '中文术语召回飘，往往要先换 embedding 栈',
  },
  'multi-turn-context-management-three-approaches': {
    hook: 'context 窗口有限，对话一长就失忆',
    takeaway: '滑动、摘要、检索注入，三种取舍不同',
  },
  'ppo-proximal-policy-optimization-pytorch': {
    hook: 'RLHF 听起来远，PPO 的 Clip 离落地很近',
    takeaway: '先搞清四模型分工，再写训练循环',
  },
};

const JOSH_ABOUT_CONF_COVER = 'https://www.joshwcomeau.com/images/josh-grabby-hands.jpg';

const JOSH_ABOUT_TOPIC_FALLBACK = ['RAG', 'Transformer', 'RLHF', 'Agent'];

const JOSH_ABOUT_DRUM_PADS = [
  { key: 'q', label: 'kick', kind: 'kick' },
  { key: 'w', label: 'hihat', kind: 'hihat' },
  { key: 'e', label: 'snare', kind: 'snare' },
  { key: 'r', label: 'cowbell', kind: 'cowbell' },
];

const JOSH_ABOUT_AUTHOR_CITY = '深圳';
const JOSH_ABOUT_CUTOUT_ASSET_V = '2026062919';
const JOSH_ABOUT_CUTOUT_DARK_PATH = 'content/assets/about-cutout-ceramic-dark.webp';
const JOSH_ABOUT_CUTOUT_LIGHT_PATH = 'content/assets/about-cutout-ceramic-light.webp';

function joshAboutArcUnits() {
  return [
    {
      value: '10+',
      width: '2.75rem',
      prefix: '',
      suffix: '年，都在跟交付打交道。',
      detail: '性能、架构、上线后的告警——先问「会不会在用户那边爆」。',
    },
    {
      value: '2015',
      width: '3.25rem',
      prefix: '',
      suffix: '年入行，从客户端开始。',
      detail: 'UI、稳定性、版本节奏，在那套环境里磨出来的习惯。',
    },
    {
      value: '2024',
      width: '3.25rem',
      prefix: '',
      suffix: '年转向 LLM 应用。',
      detail: 'Demo 能跑不算数；第一次被灰度里的 P95 曲线教育。',
    },
  ];
}

function joshAboutPostHook(post) {
  const curated = JOSH_ABOUT_POST_HOOKS[post.slug];
  if (curated) return curated;
  const excerpt = String(post.excerpt || '').trim();
  if (!excerpt) {
    return { hook: post.title, takeaway: post.category || '阅读全文' };
  }
  const dot = excerpt.indexOf('。');
  const hook = dot > 0 ? excerpt.slice(0, dot + 1) : excerpt.slice(0, 52);
  const rest = dot > 0 ? excerpt.slice(dot + 1).trim() : '';
  const takeaway = rest ? rest.slice(0, 48) : post.title;
  return { hook, takeaway };
}

function joshAboutParseReadMinutes(readTime) {
  if (!readTime) return 0;
  const match = String(readTime).match(/(\d+)/);
  return match ? Number(match[1]) : 0;
}

function joshAboutTotalReadMinutes(postList) {
  return postList.reduce((sum, post) => sum + joshAboutParseReadMinutes(post.readTime), 0);
}

function joshAboutCategoryPostCount(categoryName, postList) {
  return postList.filter((post) => post.category === categoryName).length;
}

function joshAboutFormatPostMonth(dateStr) {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  if (Number.isNaN(date.getTime())) return '';
  const month = String(date.getMonth() + 1).padStart(2, '0');
  return `${date.getFullYear()}-${month}`;
}

function joshAboutHotPostSlugs(limit = 3) {
  const postList = typeof posts !== 'undefined' ? posts : [];
  const bySlug = new Map(postList.map((post) => [post.slug, post]));

  if (typeof JOSH_POPULAR_SLUGS !== 'undefined' && JOSH_POPULAR_SLUGS.length) {
    const fromPopular = JOSH_POPULAR_SLUGS
      .map((slug) => bySlug.get(slug))
      .filter(Boolean)
      .slice(0, limit);
    if (fromPopular.length) return fromPopular.map((post) => post.slug);
  }

  if (typeof readLocalViewCount === 'function') {
    const ranked = [...postList]
      .sort((a, b) => readLocalViewCount(b.slug, 'post') - readLocalViewCount(a.slug, 'post'))
      .slice(0, limit);
    if (ranked.some((post) => readLocalViewCount(post.slug, 'post') > 0)) {
      return ranked.map((post) => post.slug);
    }
  }

  return [...postList]
    .sort((a, b) => new Date(b.date) - new Date(a.date))
    .slice(0, limit)
    .map((post) => post.slug);
}

function joshAboutPostStats() {
  const postList = typeof posts !== 'undefined' ? posts : [];
  const categoryList = typeof categories !== 'undefined' ? categories : [];
  const tagSet = new Set();
  postList.forEach((post) => {
    (post.tags || []).forEach((tag) => tagSet.add(tag));
  });
  const latestPost = postList.length
    ? [...postList].sort((a, b) => new Date(b.date) - new Date(a.date))[0]
    : null;
  return {
    postCount: postList.length,
    categoryCount: categoryList.length,
    tagCount: tagSet.size,
    latestPost,
  };
}

function joshAboutTopTags(postList, limit = 4) {
  const counts = new Map();
  postList.forEach((post) => {
    (post.tags || []).forEach((tag) => {
      counts.set(tag, (counts.get(tag) || 0) + 1);
    });
  });
  const ranked = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'zh-CN'))
    .map(([name]) => name);
  const names = [...ranked];
  JOSH_ABOUT_TOPIC_FALLBACK.forEach((tag) => {
    if (names.length >= limit) return;
    if (!names.includes(tag)) names.push(tag);
  });
  return names.slice(0, limit).map((name, index) => ({
    name,
    count: counts.get(name) || 0,
    key: JOSH_ABOUT_DRUM_PADS[index].key,
    kind: JOSH_ABOUT_DRUM_PADS[index].kind,
    label: JOSH_ABOUT_DRUM_PADS[index].label,
  }));
}

function joshAboutCutoutAsset(path) {
  const base = typeof resolveAssetUrl === 'function' ? resolveAssetUrl(path) : path;
  const joiner = base.includes('?') ? '&' : '?';
  return `${base}${joiner}v=${JOSH_ABOUT_CUTOUT_ASSET_V}`;
}

function joshAboutCutoutMarkup() {
  const darkSrc = joshAboutCutoutAsset(JOSH_ABOUT_CUTOUT_DARK_PATH);
  const lightSrc = joshAboutCutoutAsset(JOSH_ABOUT_CUTOUT_LIGHT_PATH);
  const imgAttrs = 'alt="" width="391" height="758" loading="eager" decoding="async" fetchpriority="high" draggable="false"';
  return `<div class="josh-about-intro__cutout" aria-hidden="true">
    <img class="josh-about-intro__cutout-img josh-about-intro__cutout-img--dark" src="${darkSrc}" srcset="${darkSrc} 2x" ${imgAttrs}>
    <img class="josh-about-intro__cutout-img josh-about-intro__cutout-img--light" src="${lightSrc}" srcset="${lightSrc} 2x" ${imgAttrs}>
  </div>`;
}

function joshAboutHeroMarkup({ title, paragraphs, activeHref }) {
  const headerHtml = buildJoshInnerHeaderMarkup(activeHref, { aboutSkyEmbed: true });
  const waveHtml = typeof joshAboutHeroWaveMarkup === 'function'
    ? joshAboutHeroWaveMarkup()
    : '';
  const bodyHtml = paragraphs.map((p) => `<p>${p}</p>`).join('');
  return `<div class="josh-about-sky" id="josh-about-sky">
    <div class="josh-about-sky__band">
      <div class="josh-about-sky__veil" aria-hidden="true"></div>
      ${headerHtml}
      ${waveHtml}
      <div class="josh-about-intro">
        ${joshAboutCutoutMarkup()}
        <div class="josh-about-intro__content">
          <h1 class="josh-about-intro__title">${title}</h1>
          ${bodyHtml}
        </div>
      </div>
    </div>
  </div>`;
}

const JOSH_ABOUT_AUTHOR_GEO = { lat: 22.5431, lon: 114.0579, city: '深圳' };
const JOSH_ABOUT_LEAFLET_VERSION = '1.9.4';
// Leaflet sets SVG stroke attributes directly — use hex/rgb literals, not hsl()/oklch() tokens.
const JOSH_ABOUT_MAP_ROUTE_COLORS = {
  light: '#4242fa',
  dark: '#ff3366',
};
const JOSH_ABOUT_MAP_TILE_LAYERS = {
  primary: {
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    options: {
      subdomains: 'abc',
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> 贡献者',
      maxZoom: 19,
      keepBuffer: 8,
      updateWhenZooming: false,
    },
  },
  fallback: {
    url: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    options: {
      subdomains: 'abcd',
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> 贡献者 · &copy; <a href="https://carto.com/attributions">CARTO</a>',
      maxZoom: 19,
      keepBuffer: 8,
      updateWhenZooming: false,
    },
  },
};

let joshAboutLeafletPromise = null;
let joshAboutLeafletState = null;

function joshAboutEnsureLeaflet() {
  if (window.L) return Promise.resolve(window.L);
  if (joshAboutLeafletPromise) return joshAboutLeafletPromise;
  joshAboutLeafletPromise = new Promise((resolve, reject) => {
    if (!document.querySelector('link[data-josh-leaflet-css]')) {
      const css = document.createElement('link');
      css.rel = 'stylesheet';
      css.href = `https://cdn.jsdelivr.net/npm/leaflet@${JOSH_ABOUT_LEAFLET_VERSION}/dist/leaflet.css`;
      css.setAttribute('data-josh-leaflet-css', '');
      document.head.appendChild(css);
    }
    const script = document.createElement('script');
    script.src = `https://cdn.jsdelivr.net/npm/leaflet@${JOSH_ABOUT_LEAFLET_VERSION}/dist/leaflet.js`;
    script.onload = () => resolve(window.L);
    script.onerror = () => reject(new Error('leaflet load failed'));
    document.head.appendChild(script);
  });
  return joshAboutLeafletPromise;
}

function joshAboutVisitorPlaceLabel(data) {
  if (!data) return '你的位置';
  const city = (data.city || '').trim();
  const region = (data.region || '').trim();
  const country = (data.country || '').trim();
  if (city) return city;
  if (region) return region;
  if (country) return country;
  return '你的位置';
}

async function joshAboutPlaceLabelZh(lat, lon, fallbackData) {
  try {
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timeoutId = controller ? window.setTimeout(() => controller.abort(), 4500) : null;
    const url = `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${encodeURIComponent(lat)}&longitude=${encodeURIComponent(lon)}&localityLanguage=zh`;
    const response = await fetch(url, controller ? { signal: controller.signal } : undefined);
    if (timeoutId) window.clearTimeout(timeoutId);
    if (!response.ok) throw new Error('reverse geocode unavailable');
    const data = await response.json();
    const place = (data.city || data.locality || data.principalSubdivision || data.countryName || '').trim();
    if (place) return place;
  } catch {
    /* fallback below */
  }
  return joshAboutVisitorPlaceLabel(fallbackData);
}

function joshAboutMapMascotMarkerHtml() {
  return `<span class="josh-about-map__mascot-marker" aria-hidden="true">
    <img class="josh-about-map__mascot josh-about-map__mascot--dark" src="${JOSH_MASCOT_DARK}" alt="" width="40" height="68" loading="lazy" decoding="async">
    <img class="josh-about-map__mascot josh-about-map__mascot--light" src="${JOSH_MASCOT_LIGHT}" alt="" width="40" height="68" loading="lazy" decoding="async">
  </span>`;
}

function joshAboutMapAuthorIcon(L) {
  return L.divIcon({
    className: 'josh-about-map__leaflet-icon josh-about-map__leaflet-icon--author',
    html: joshAboutMapMascotMarkerHtml(),
    iconSize: [40, 67],
    iconAnchor: [20, 34],
    popupAnchor: [0, -30],
  });
}

function joshAboutMapVisitorIcon(L) {
  return L.divIcon({
    className: 'josh-about-map__leaflet-icon josh-about-map__leaflet-icon--visitor',
    html: '<span class="josh-about-map__pin-marker" aria-hidden="true"></span>',
    iconSize: [16, 16],
    iconAnchor: [8, 8],
    popupAnchor: [0, -8],
  });
}

function joshAboutMapRouteColor() {
  const html = document.documentElement;
  const isDark = html.classList.contains('dark') || html.getAttribute('data-color-mode') === 'dark';
  return isDark ? JOSH_ABOUT_MAP_ROUTE_COLORS.dark : JOSH_ABOUT_MAP_ROUTE_COLORS.light;
}

function joshAboutMapRouteStyle() {
  const html = document.documentElement;
  const isDark = html.classList.contains('dark') || html.getAttribute('data-color-mode') === 'dark';
  if (isDark) {
    return {
      color: joshAboutMapRouteColor(),
      weight: 3.25,
      opacity: 0.96,
      dashArray: '7 6',
      lineCap: 'round',
      lineJoin: 'round',
    };
  }
  return {
    color: joshAboutMapRouteColor(),
    weight: 2.5,
    opacity: 0.85,
    dashArray: '8 7',
    lineCap: 'round',
    lineJoin: 'round',
  };
}

function joshAboutSyncRouteStyleWithTheme() {
  if (!joshAboutLeafletState?.routeLine) return;
  joshAboutLeafletState.routeLine.setStyle(joshAboutMapRouteStyle());
}

function joshAboutMapWidgetMarkup() {
  return `<div class="josh-about-map">
    <div class="josh-about-map__canvas" id="josh-about-map-canvas">
      <div id="josh-about-map-leaflet" class="josh-about-map__leaflet" role="region" aria-label="我与访客位置的地图"></div>
      <div class="josh-about-map__hud" aria-hidden="true" hidden>
        <div class="josh-about-map__legend">
          <span class="josh-about-map__legend-item josh-about-map__legend-item--author"><i></i>${JOSH_ABOUT_AUTHOR_GEO.city} · 我</span>
          <span class="josh-about-map__legend-item josh-about-map__legend-item--visitor"><i></i><span id="josh-about-map-visitor-place">Locating…</span></span>
        </div>
        <div class="josh-about-map__distance-pill" id="josh-about-map-distance-pill" hidden>…</div>
      </div>
    </div>
  </div>`;
}

function joshAboutAttachMapTiles(map, L) {
  const primary = L.tileLayer(
    JOSH_ABOUT_MAP_TILE_LAYERS.primary.url,
    JOSH_ABOUT_MAP_TILE_LAYERS.primary.options,
  );
  const fallback = L.tileLayer(
    JOSH_ABOUT_MAP_TILE_LAYERS.fallback.url,
    JOSH_ABOUT_MAP_TILE_LAYERS.fallback.options,
  );
  let errorCount = 0;
  let switched = false;
  const onTileError = () => {
    errorCount += 1;
    if (!switched && errorCount >= 4) {
      switched = true;
      map.removeLayer(primary);
      fallback.addTo(map);
      if (joshAboutLeafletState) {
        joshAboutLeafletState.tileLayer = fallback;
      }
    }
  };
  primary.on('tileerror', onTileError);
  primary.addTo(map);
  return primary;
}

function joshAboutScheduleMapResize(map) {
  const refresh = () => {
    if (!map || !map.getContainer?.()?.isConnected) return;
    map.invalidateSize({ pan: false });
  };
  queueMicrotask(refresh);
  const t1 = window.setTimeout(refresh, 120);
  const t2 = window.setTimeout(refresh, 600);
  return () => {
    window.clearTimeout(t1);
    window.clearTimeout(t2);
  };
}

async function joshAboutInitLeafletMap(app) {
  const container = app.querySelector('#josh-about-map-leaflet');
  if (!container || joshAboutLeafletState?.map) return joshAboutLeafletState;
  const L = await joshAboutEnsureLeaflet();
  const map = L.map(container, {
    zoomControl: false,
    attributionControl: true,
    scrollWheelZoom: false,
    dragging: true,
    worldCopyJump: true,
    fadeAnimation: false,
  });
  const tileLayer = joshAboutAttachMapTiles(map, L);
  const { lat, lon } = JOSH_ABOUT_AUTHOR_GEO;
  const authorMarker = L.marker([lat, lon], {
    icon: joshAboutMapAuthorIcon(L),
    title: `我在${JOSH_ABOUT_AUTHOR_GEO.city}`,
    zIndexOffset: 1000,
  }).addTo(map);
  map.setView([lat, lon], 4);
  const cancelResize = joshAboutScheduleMapResize(map);
  joshAboutLeafletState = {
    map,
    L,
    tileLayer,
    authorMarker,
    visitorMarker: null,
    routeLine: null,
    cancelResize,
  };
  const html = document.documentElement;
  const observer = new MutationObserver(() => joshAboutSyncRouteStyleWithTheme());
  observer.observe(html, { attributes: true, attributeFilter: ['class', 'data-color-mode'] });
  joshAboutLeafletState.routeStyleObserver = observer;
  return joshAboutLeafletState;
}

function joshAboutDestroyLeafletMap() {
  if (joshAboutLeafletState?.cancelResize) joshAboutLeafletState.cancelResize();
  if (joshAboutLeafletState?.routeStyleObserver) joshAboutLeafletState.routeStyleObserver.disconnect();
  if (joshAboutLeafletState?.map) {
    joshAboutLeafletState.map.remove();
  }
  joshAboutLeafletState = null;
}

function joshAboutUpdateMapRoute(visitorLon, visitorLat, visitorMeta = {}, visitorPlaceLabel) {
  const state = joshAboutLeafletState;
  if (!state?.map || !window.L) return;
  const L = window.L;
  const { lat: authorLat, lon: authorLon } = JOSH_ABOUT_AUTHOR_GEO;
  const authorLatLng = L.latLng(authorLat, authorLon);
  const visitorLatLng = L.latLng(visitorLat, visitorLon);
  const place = visitorPlaceLabel || joshAboutVisitorPlaceLabel(visitorMeta);
  const visitorPlace = document.getElementById('josh-about-map-visitor-place');
  const mapCanvas = document.getElementById('josh-about-map-canvas') || document.querySelector('.josh-about-map__canvas');

  if (state.visitorMarker) state.map.removeLayer(state.visitorMarker);
  if (state.routeLine) state.map.removeLayer(state.routeLine);

  state.visitorMarker = L.marker(visitorLatLng, {
    icon: joshAboutMapVisitorIcon(L),
    title: `You are in ${place}`,
    zIndexOffset: 900,
  }).addTo(state.map);

  state.routeLine = L.polyline([authorLatLng, visitorLatLng], joshAboutMapRouteStyle()).addTo(state.map);

  const bounds = L.latLngBounds([authorLatLng, visitorLatLng]).pad(0.22);
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Leaflet resize 会触发瓦片重绘；只在容器尺寸尚未就绪时才触发，避免与初始化重排抢占导致“闪一下”。
  const size = state.map.getSize?.() || { x: 1, y: 1 };
  if (!size.x || !size.y) state.map.invalidateSize({ pan: false });
  state.map.stop();
  if (prefersReduced) {
    state.map.fitBounds(bounds, { animate: false });
  } else {
    // 平滑优先：避免 flyToBounds 的缩放飞行，改为“静态定缩放 + 动态平移”。
    const targetZoom = state.map.getBoundsZoom(bounds);
    const currentZoom = state.map.getZoom();
    if (Number.isFinite(targetZoom) && Math.abs(targetZoom - currentZoom) > 0.01) {
      state.map.setZoom(targetZoom, { animate: false });
    }
    state.map.panTo(bounds.getCenter(), {
      animate: true,
      duration: 0.85,
      easeLinearity: 0.2,
      noMoveStart: true,
    });
  }

  if (visitorPlace) visitorPlace.textContent = `${place} · you`;
  if (mapCanvas) mapCanvas.classList.add('is-connected');
  const routeHint = document.getElementById('josh-about-map-route-hint');
  if (routeHint) routeHint.hidden = false;
}

function joshAboutFlagColumnsMarkup(flag) {
  const columnCount = flag?.columns || JOSH_ABOUT_PRIDE_FLAG_COLUMN_DELAYS_MS.length;
  const gradient = flag?.gradient || JOSH_ABOUT_PRIDE_FLAG_GRADIENTS.rainbow;
  return Array.from({ length: columnCount }, (_, index) => {
    const delay = JOSH_ABOUT_PRIDE_FLAG_COLUMN_DELAYS_MS[index] ?? -100 * (columnCount - index);
    const sin = JOSH_ABOUT_PRIDE_FLAG_COLUMN_SINS_DEG[index] ?? (index * 45);
    return `<div class="josh-about-flag__column" style="--billow:4%;--sin:${sin}deg;animation-duration:600ms;animation-delay:${delay}ms;background-image:${gradient}"></div>`;
  }).join('');
}

function joshAboutSwapText(el, nextText, duration = 220) {
  if (!el) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    el.textContent = nextText;
    return;
  }
  el.classList.add('is-leaving');
  window.setTimeout(() => {
    el.textContent = nextText;
    el.classList.remove('is-leaving');
    el.classList.add('is-entering');
    void el.offsetWidth;
    el.classList.remove('is-entering');
  }, duration);
}

function joshAboutGayMarkup(flagIndex = 0) {
  const credo = JOSH_ABOUT_CREDOS[flagIndex];
  return `<button type="button" class="josh-about-flag-btn" id="josh-about-flag-btn" aria-label="${credo.label}">
    <span class="josh-about-flag" id="josh-about-flag-strip" aria-hidden="true">
      ${joshAboutFlagColumnsMarkup(credo)}
    </span>
  </button>
  <p id="josh-about-flag-text">${credo.text}</p>
  <p class="small" id="josh-about-flag-sub">${credo.subtitle}</p>`;
}

function joshAboutSpawnWaves(fader, hue) {
  const knob = fader.querySelector('.josh-about-fader__knob');
  if (!knob) return;
  const col = fader.querySelector('.josh-about-fader__col');
  if (!col) return;
  const wrap = document.createElement('div');
  wrap.className = 'josh-about-fader__waves';
  const knobRect = knob.getBoundingClientRect();
  const colRect = col.getBoundingClientRect();
  wrap.style.top = (knobRect.top - colRect.top + knobRect.height / 2) + 'px';
  for (let i = 0; i < 3; i++) {
    const ring = document.createElement('span');
    ring.className = 'josh-about-wave-ring';
    ring.style.setProperty('--ring-hue', hue);
    wrap.appendChild(ring);
  }
  col.appendChild(wrap);
  window.setTimeout(() => wrap.remove(), 700);
}

function joshAboutSpawnParticles(fader, hue) {
  const knob = fader.querySelector('.josh-about-fader__knob');
  if (!knob) return;
  const col = fader.querySelector('.josh-about-fader__col');
  if (!col) return;
  const wrap = document.createElement('div');
  wrap.className = 'josh-about-fader__particles';
  const knobRect = knob.getBoundingClientRect();
  const colRect = col.getBoundingClientRect();
  wrap.style.top = (knobRect.top - colRect.top + knobRect.height / 2) + 'px';
  const count = 7;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('span');
    p.className = 'josh-about-particle';
    const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.6;
    const dist = 14 + Math.random() * 18;
    p.style.setProperty('--ptc-x', (Math.cos(angle) * dist).toFixed(1) + 'px');
    p.style.setProperty('--ptc-y', (Math.sin(angle) * dist - 8).toFixed(1) + 'px');
    p.style.setProperty('--ptc-hue', hue);
    p.style.animationDelay = (Math.random() * 0.08).toFixed(2) + 's';
    wrap.appendChild(p);
  }
  col.appendChild(wrap);
  window.setTimeout(() => wrap.remove(), 600);
}

const JOSH_ABOUT_FADER_HUES = { kick: 195, hihat: 150, snare: 35, cowbell: 330 };

function joshAboutDrumsMarkup() {
  const postList = typeof posts !== 'undefined' ? posts : [];
  const topics = joshAboutTopTags(postList, 4);
  const maxCount = Math.max(...topics.map((t) => t.count), 1);
  const ticks = [0.25, 0.5, 0.75].map((p) => `<span class="josh-about-fader__tick" style="bottom:${p * 100}%"></span>`).join('');
  return `<div class="josh-about-faders" role="group" aria-label="博客主题调音台">
    ${topics.map((topic) => {
      const level = 0.3 + (topic.count / maxCount) * 0.6;
      const hue = JOSH_ABOUT_FADER_HUES[topic.kind] || 195;
      return `
      <div class="josh-about-fader" data-drum="${topic.kind}" data-drum-key="${topic.key}" style="--fader-hue:${hue};--fader-level:${level.toFixed(3)}">
        <div class="josh-about-fader__col">
          <div class="josh-about-fader__track">
            <div class="josh-about-fader__ticks">${ticks}</div>
            <div class="josh-about-fader__fill"></div>
            <div class="josh-about-fader__knob"></div>
          </div>
          <button type="button" class="josh-about-fader__btn" aria-label="播放 ${topic.name} 主题音效。也可以按 ${topic.key.toUpperCase()} 键。"></button>
        </div>
        <span class="josh-about-fader__label">${topic.name}</span>
        <span class="josh-about-fader__key">${topic.key.toUpperCase()}</span>
      </div>`;
    }).join('')}
  </div>`;
}

function joshAboutTallFigureMarkup() {
  return `<svg class="josh-about-tall-figure" width="50" height="100" viewBox="0 0 50 100" xmlns="http://www.w3.org/2000/svg" fill="none" aria-hidden="true">
    <g stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <circle class="josh-about-tall-figure__head" cx="25" cy="35" r="15"/>
      <line x1="25" y1="50" x2="25" y2="80"/>
      <line x1="25" y1="80" x2="15" y2="100"/>
      <line x1="25" y1="80" x2="35" y2="100"/>
      <line x1="25" y1="55" x2="15" y2="70"/>
      <line x1="25" y1="55" x2="35" y2="70"/>
    </g>
  </svg>`;
}

function joshAboutDeskRadarMarkup() {
  const cx = 120, cy = 105, R = 70;
  const axes = JOSH_ABOUT_DESK_RADAR;
  const n = axes.length;

  function pt(i, r) {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  }
  function poly(r) {
    return Array.from({ length: n }, (_, i) => pt(i, r).map(v => Math.round(v * 10) / 10).join(',')).join(' ');
  }
  const dataPts = axes.map((d, i) => pt(i, R * d.score).map(v => Math.round(v * 10) / 10).join(',')).join(' ');
  const labelOff = 18;
  const labels = axes.map((d, i) => {
    const [lx, ly] = pt(i, R + labelOff);
    const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
    const anchor = Math.abs(Math.cos(a)) < 0.01 ? 'middle' : Math.cos(a) > 0 ? 'start' : 'end';
    const dy = Math.sin(a) < -0.5 ? '-0.4em' : Math.sin(a) > 0.5 ? '1em' : '0.35em';
    return `<text x="${Math.round(lx * 10) / 10}" y="${Math.round(ly * 10) / 10}" text-anchor="${anchor}" dy="${dy}">${d.label}</text>`;
  }).join('');

  return `<div class="josh-about-desk-viz">
    <svg class="josh-about-desk-radar" viewBox="0 0 240 210" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <polygon points="${poly(R)}" class="josh-radar-grid"/>
      <polygon points="${poly(R * 0.66)}" class="josh-radar-grid"/>
      <polygon points="${poly(R * 0.33)}" class="josh-radar-grid"/>
      ${axes.map((_, i) => { const [ax, ay] = pt(i, R); return `<line x1="${cx}" y1="${cy}" x2="${Math.round(ax * 10) / 10}" y2="${Math.round(ay * 10) / 10}" class="josh-radar-axis"/>`; }).join('')}
      <polygon points="${dataPts}" class="josh-radar-data"/>
      ${axes.map((d, i) => { const [dx, dy] = pt(i, R * d.score); return `<circle cx="${Math.round(dx * 10) / 10}" cy="${Math.round(dy * 10) / 10}" r="3.5" class="josh-radar-dot"/>`; }).join('')}
      <g class="josh-radar-labels">${labels}</g>
    </svg>
    <div class="josh-about-desk-tags">
      ${JOSH_ABOUT_DESK_TAGS.map((tag, i) => `<span class="josh-about-desk-tag" style="--tag-i:${i}">${tag}</span>`).join('')}
    </div>
  </div>`;
}

function joshAboutDeskMarkup() {
  return joshAboutDeskRadarMarkup();
}

function joshAboutConfCoverMarkup() {
  return `<span class="josh-about-card__inset josh-about-conf-cover">
    <img src="${JOSH_ABOUT_CONF_COVER}" alt="Josh presenting at CSS Day 2024" loading="lazy" width="500" height="318">
  </span>`;
}

function joshAboutSetCatMood(catBtn, mood) {
  if (!catBtn) return;
  if (!mood || mood === 'normal') {
    catBtn.removeAttribute('data-mood');
  } else {
    catBtn.dataset.mood = mood;
  }
}

function joshAboutCreateCatPetController(catBtn, bodyGroup) {
  const state = {
    angle: 0,
    velocity: 0,
    target: 0,
    tension: 170,
    friction: 25,
    clickCount: 0,
    resetTimer: null,
    raf: null,
    last: 0,
    mood: 'normal',
  };

  const apply = () => {
    bodyGroup.style.transform = `rotate(${state.angle}deg)`;
  };

  const setMood = (mood) => {
    state.mood = mood;
    joshAboutSetCatMood(catBtn, mood);
  };

  const tick = (now) => {
    if (!state.last) state.last = now;
    const dt = Math.min((now - state.last) / 1000, 0.064);
    state.last = now;
    const steps = Math.max(1, Math.ceil(dt / (1 / 60)));
    const subDt = dt / steps;
    for (let i = 0; i < steps; i += 1) {
      const accel = (-state.tension * (state.angle - state.target)) - (state.friction * state.velocity);
      state.velocity += accel * subDt;
      state.angle += state.velocity * subDt;
    }
    apply();

    if (Math.abs(state.angle - state.target) > 0.01 || Math.abs(state.velocity) > 0.05) {
      state.raf = requestAnimationFrame(tick);
    } else {
      state.angle = state.target;
      state.velocity = 0;
      apply();
      state.raf = null;
      state.last = 0;
    }
  };

  const ensureRaf = () => {
    if (!state.raf) {
      state.last = 0;
      state.raf = requestAnimationFrame(tick);
    }
  };

  const pet = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (state.mood !== 'normal') return;

    state.clickCount += 1;
    const isMad = state.clickCount % 8 === 0;

    if (typeof joshPlaySound === 'function') {
      joshPlaySound(isMad ? 'cat-on' : 'cat-press');
    }

    if (state.resetTimer) {
      clearTimeout(state.resetTimer);
      state.resetTimer = null;
    }

    setMood(isMad ? 'mad' : 'happy');

    if (isMad) {
      state.target = -5;
      state.tension = 350;
      state.friction = 10;
      state.resetTimer = window.setTimeout(() => {
        setMood('normal');
        state.target = 0;
        state.tension = 170;
        state.friction = 25;
        state.resetTimer = null;
        ensureRaf();
      }, 450);
    } else {
      state.target = 6;
      state.tension = 170;
      state.friction = 10;
      state.resetTimer = window.setTimeout(() => {
        setMood('normal');
        state.target = 0;
        state.tension = 170;
        state.friction = 25;
        state.resetTimer = null;
        ensureRaf();
      }, 220);
    }

    ensureRaf();
  };

  const cleanup = () => {
    if (state.resetTimer) {
      clearTimeout(state.resetTimer);
      state.resetTimer = null;
    }
    if (state.raf) {
      cancelAnimationFrame(state.raf);
      state.raf = null;
    }
    state.angle = 0;
    state.velocity = 0;
    state.target = 0;
    setMood('normal');
    apply();
  };

  apply();
  return { pet, cleanup };
}

function joshAboutCatTailSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" fill="none" width="180" viewBox="0 0 399 472" aria-hidden="true" class="josh-about-cat-svg"><mask id="josh-about-cat-tail-mask" style="mask-type:alpha" width="314" height="320" x="168" y="141" maskUnits="userSpaceOnUse"><path fill="#C4C4C4" d="M168.196 379.181H420.076V622.857H168.196z" transform="rotate(-70.611 168.196 379.181)"></path></mask><g mask="url(#josh-about-cat-tail-mask)"><path fill="#fff" stroke="var(--josh-color-outline, #222222)" stroke-width="10" d="M456.06 447.462c-68.217-6.823-127.014-26.426-174.079-69.961-39.169-36.231-53.406-51.886-39.7-99.789 15.825-55.31 33.898-63.644 94.143-68.039 65.215-4.757 125.106 18.51 182.786 58.349 20.481 14.146 31.776 23.601 44.597 41.786 19.539 27.718 14.051 53.196 5.806 77.601-8.329 24.653-19.684 48.135-53.127 56.859-21.349 5.569-35.93 5.645-60.426 3.194z"></path></g></svg>`;
}

function joshAboutCatMainSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" fill="none" width="180" viewBox="0 0 399 472" aria-hidden="true" class="josh-about-cat-svg"><mask id="josh-about-cat-main-mask" style="mask-type:alpha" width="314" height="320" x="168" y="141" maskUnits="userSpaceOnUse"><path fill="#C4C4C4" d="M168.196 379.181H420.076V622.857H168.196z" transform="rotate(-70.611 168.196 379.181)"></path></mask><path fill="#fff" d="M350.844 376.47c-4.779 29.526-96.148 5.616-96.148 5.616l-16-33 6-41.5S252.824 329.583 281 344c35 0 72.133 18.331 69.844 32.47z"></path><path stroke="var(--josh-color-outline, #222222)" stroke-linecap="round" stroke-linejoin="round" stroke-width="10" d="M296.06 347.518s59.563-.574 54.784 28.952c-4.779 29.526-62.617 15.22-62.617 15.22s-39.03-10.104-46.531-27.104c-7.5-17-4-50-4-50"></path><path fill="#fff" d="M375.738 277.062c-6.754 29.138-63.499 10.973-63.499 10.973l10.787-43.544s59.466 3.433 52.712 32.571z"></path><path stroke="var(--josh-color-outline, #222222)" stroke-linecap="round" stroke-linejoin="round" stroke-width="10" d="M312.239 288.035s56.745 18.165 63.499-10.973c6.754-29.138-52.712-32.571-52.712-32.571"></path><g class="josh-about-cat-body-group"><path fill="#fff" stroke="var(--josh-color-outline, #222222)" stroke-linecap="round" stroke-linejoin="round" stroke-width="12" d="M274.468 303.833c-60.778 43.563-129.657 65.408-178.119 18.305-26.47-25.729-42.082-49.209-38.523-84.759 0 0-37.906-37.801-46.532-69.205-2.506-9.126-4.576-13.051-4.23-23.862.178-5.547.863-9.34 4.594-13.45 3.29-3.626 3.88-4.176 7.277-6.287 27.27-16.945 85.464 10.596 85.464 10.596 12.343-13.177 48.735-34.45 64.037-41.187a117.579 117.579 0 016.462-2.618s7.051-45.275 23.845-67.732c3.92-5.243 7.63-9.46 13.442-12.473 4.667-2.42 6.793-3.702 12.039-3.36 7.976.52 18.023 5.12 27.152 14.328 21.538 21.729 38.142 74.419 38.142 74.419 33.625 14.091 46.474 37.476 58.623 69.682 22.716 60.217-21.362 100.11-73.673 137.603z"></path><path fill="#FFE62A" d="M121.285 226.052l2.934 18.675s.195 1.469-.02 4.328c-.295 3.932-4.711 7.243-4.711 7.243l-18.487 14.723s-5.796 5.247-4.917 9.529c.946 4.603 11.435 5.835 11.435 5.835l23.879-.812s1.811-.574 4.525 2.05c2.714 2.623 3.176 19.065 6.392 29.473 1.495 4.839 3.745 10.436 8.427 10.947 5.173.565 10.48-11.912 10.48-11.912l7.247-17.988s2.917-6.874 4.457-8.164c2.179-1.825 9.616-.581 9.616-.581l26.047 2.163s7.636-.971 9.233-5.197c1.685-4.46-5.449-12.258-5.449-12.258s-7.962-7.281-14.755-12.41l-.153-.116c-2.508-1.893-7.71-5.82-7.984-7.235-.28-1.444 6.518-19.16 6.518-19.16 1.423-4.965 6.914-15.509 3.452-18.862-2.496-2.417-7.195-1.474-12.77.377-5.575 1.852-18.327 9.235-18.327 9.235s-6.735 2.985-9.502 2.662c-2.476-.289-5.152-2.187-5.152-2.187l-15.996-10.56s-8.083-4.952-12.413-2.536c-4.563 2.545-4.006 12.738-4.006 12.738z"></path><path fill="#FBCECA" d="M37.232 175.569c9.985 16.071 29.58 38.132 29.58 38.132s5.462-22.101 9.784-32.569c3.9-9.444 15.327-23.651 15.327-23.651s-38.205-19.234-54.514-9.1c0 0-10.162 11.118-.177 27.188zM245.347 44.637c8.883 16.705 19.002 51.084 19.002 51.084s-19.89-5.545-31.096-7.073c-10.111-1.378-30.042.504-30.042.504s4.152-48.922 21.071-57.916c0 0 12.181-3.304 21.065 13.401z"></path><g class="josh-about-cat-face josh-about-cat-face--normal"><circle cx="155.264" cy="260.614" r="18" fill="var(--josh-color-outline, #222222)" transform="rotate(-31.855 155.264 260.614)"></circle><circle cx="270.781" cy="188.838" r="18" fill="var(--josh-color-outline, #222222)" transform="rotate(-31.855 270.781 188.838)"></circle><path stroke="#fff" stroke-linecap="round" stroke-width="6" d="M146.036 262.816s-.397-1.904.201-4.05c.528-1.895 1.407-2.836 1.407-2.836"></path><path stroke="#fff" stroke-linecap="round" stroke-width="6" d="M261.553 191.04s-.397-1.904.201-4.049c.529-1.896 1.408-2.837 1.408-2.837"></path><path stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width="5" d="M217.525 266.078s11.053.41 13.475-4.841c2.146-4.654-3.2-12.728-3.2-12.728s6.088 7.999 11.269 7.715c5.202-.286 9.034-9.146 9.034-9.146"></path></g><g class="josh-about-cat-face josh-about-cat-face--happy"><path stroke="var(--josh-color-outline, #222222)" stroke-linecap="round" stroke-width="10" d="M261.103 201.517s-5.373-17.544 3.012-23.054c8.385-5.509 22.359 7.292 22.359 7.292M145.58 273.284s-5.895-17.201 3.012-23.054c8.908-5.853 22.36 7.292 22.36 7.292"></path><path fill="#222" stroke="#222" d="M241.341 270.302c-8.392 6.39-14.988-4.228-14.988-4.228l5.489-11.062 12.349-.019s5.541 8.919-2.85 15.309z"></path><path fill="#FF9BA7" d="M241.16 258.641c2.332 3.753.885 6.514-1.93 8.263-2.815 1.749-5.931 1.823-8.263-1.93-2.331-3.753-1.94-8.213.875-9.962 2.815-1.749 6.987-.124 9.318 3.629z"></path></g><g class="josh-about-cat-face josh-about-cat-face--mad"><path fill="var(--josh-color-outline, #222222)" stroke="#FFE62A" stroke-linecap="round" stroke-width="4" d="M133.292 233.912a6 6 0 10-.613 11.984l43.772 2.239a6 6 0 00.613-11.984l-43.772-2.239z"></path><path fill="var(--josh-color-outline, #222222)" stroke="#fff" stroke-linecap="round" stroke-width="4" d="M240.665 198.667a6 6 0 0011.312 4.005l14.628-41.316a6 6 0 00-11.312-4.005l-14.628 41.316z"></path><path stroke="var(--josh-color-outline, #222222)" stroke-linecap="round" stroke-width="6" d="M229.53 239.92c-7.904.659-17.202 5.019-15.148 11.299 1.61 4.924 9.162 6.158 12.348 6.461.691.066 1.171.782.975 1.448-.924 3.136-2.431 10.696 4.223 11.415 8.417.911 10.515-9.837 10.515-9.837"></path></g><path fill="#DEDEDE" d="M143.161 160.731c-10.609-7.581-23.023-31.06-23.023-31.06s3.91-3.612 6.679-5.539c2.427-1.688 6.51-3.844 6.51-3.844s20.42 18.604 23.37 32.45c.953 4.474 1.849 8.219-.752 10.81-3.256 3.244-8.316.376-12.784-2.817zM169.264 144.092c-10.337-8.398-20.028-33.143-20.028-33.143s4.438-2.596 7.376-4.253c2.575-1.452 7.234-3.56 7.234-3.56s18.321 19.668 20.017 34.1c.534 4.543 1.397 8.197-1.436 10.532-3.548 2.922-8.9-.213-13.163-3.676z"></path></g></svg>`;
}

function joshAboutCatMarkup() {
  return `<button type="button" class="josh-about-cat-pet josh-about-cat-pet--ghost" tabindex="-1" aria-hidden="true">
    ${joshAboutCatTailSvg()}
  </button>
  <div class="josh-about-cat-body">
    <img class="josh-about-cat-head" src="https://www.joshwcomeau.com/images/star-cat-head.svg" alt="" width="150" height="150">
    <p>十年客户端，两年 LLM。路还长。</p>
    <p class="small">2014 入行 → 2024 转型 → 现在，还在写代码、画交互图。</p>
  </div>
  <button type="button" class="josh-about-cat-pet" id="josh-about-cat-btn" aria-label="Illustration of a cat. Triggering this button pets the cat. This is a purely cosmetic effect.">
    ${joshAboutCatMainSvg()}
  </button>`;
}

function joshAboutHotPostsMarkup(limit = 3) {
  const postList = typeof posts !== 'undefined' ? posts : [];
  const items = joshAboutHotPostSlugs(limit)
    .map((slug) => postList.find((post) => post.slug === slug))
    .filter(Boolean);
  if (!items.length) return '';
  return `<ul class="josh-about-talks">
    ${items.map((post) => `<li>
      <a href="${Routes.post(post.slug)}">
        ${joshAboutTalkArrowMarkup()}
        <span>${post.title}</span>
      </a>
    </li>`).join('')}
  </ul>`;
}

function joshAboutPicksMarkup(slugs) {
  const postList = typeof posts !== 'undefined' ? posts : [];
  const items = slugs
    .map((slug) => postList.find((post) => post.slug === slug))
    .filter(Boolean);
  if (!items.length) return '';
  return `<ul class="josh-about-picks">
    ${items.map((post) => {
      const href = Routes.post(post.slug);
      const cover = post.cover && typeof resolveAssetUrl === 'function'
        ? resolveAssetUrl(post.cover)
        : '';
      const coverHtml = cover
        ? `<span class="josh-about-pick-cover"><img src="${cover}" alt="" loading="lazy" width="40" height="40"></span>`
        : '';
      return `<li>
        <a class="josh-about-pick" href="${href}">
          ${coverHtml}
          <span class="josh-about-pick__title">${post.title}</span>
        </a>
      </li>`;
    }).join('')}
  </ul>`;
}

function joshAboutSideCopyMarkup() {
  const postList = typeof posts !== 'undefined' ? posts : [];
  const links = JOSH_ABOUT_SIDE_LINKS
    .map((item) => {
      const post = postList.find((entry) => entry.slug === item.slug);
      if (!post) return null;
      return `<a href="${Routes.post(post.slug)}">${item.label}</a>`;
    })
    .filter(Boolean);
  if (links.length >= 2) {
    return `<div class="josh-about-side-text-default">
      <p>有些原理，用交互图比段落好懂——我更喜欢把东西画「动」。</p>
      <p>例如 ${links[0]} 和 ${links[1]}。</p>
    </div>
    <div class="josh-about-side-text-active" data-scene="sigmoid">
      <p>Sigmoid 把任意实数压到 0~1 之间——门控信号的基础。</p>
      <p>→ 去看看 ${links[0]}</p>
    </div>
    <div class="josh-about-side-text-active" data-scene="attention">
      <p>Q·Kᵀ 算出每对 token 的相关性，softmax 归一化成注意力权重。</p>
      <p>→ 去看看 ${links[1]}</p>
    </div>
    <div class="josh-about-side-text-active" data-scene="aggregate">
      <p>用权重对 V 加权求和——Attention 的输出就是这么来的。</p>
      <p>→ 去看看 ${links[0]}</p>
    </div>`;
  }
  return `<div class="josh-about-side-text-default">
    <p>有些原理，用交互图比段落好懂——我更喜欢把东西画「动」。</p>
    <p>Side project 和博客，都是这个路子。</p>
  </div>
  <div class="josh-about-side-text-active" data-scene="sigmoid">
    <p>Sigmoid 把任意实数压到 0~1——门控信号的基础。</p>
  </div>
  <div class="josh-about-side-text-active" data-scene="attention">
    <p>Q·Kᵀ → softmax → 注意力权重，这就是 Attention 的核心。</p>
  </div>
  <div class="josh-about-side-text-active" data-scene="aggregate">
    <p>用权重对 V 加权求和——Attention 的输出就是这么来的。</p>
  </div>`;
}

function joshAboutPkgCopyMarkup() {
  return `<p>写得最多的是这几类问题——点一下，听个响。</p>
  <p class="small">Q / W / E / R</p>`;
}

function joshAboutMapCopyMarkup() {
  return `<p class="small">我在这里写代码、记笔记。不表演成长，只留以后还能翻出来核对的东西。<span id="josh-about-map-route-hint" hidden> 路线已连上。</span></p>`;
}

function joshAboutJobCategoriesMarkup() {
  return `<ul class="josh-about-talks">
    ${JOSH_ABOUT_JOB_CATEGORY_LINKS.map(({ label, category }) => `<li><a href="${Routes.category(category)}">${joshAboutTalkArrowMarkup()}<span>${label}</span></a></li>`).join('')}
  </ul>`;
}

function joshAboutArcMarkup() {
  const unit = joshAboutArcUnits()[0];
  return `${joshAboutTallFigureMarkup()}
  <div class="josh-about-arc-copy" id="josh-about-arc-copy">
    <p><strong class="josh-about-focus-line"><span id="josh-about-arc-prefix">${unit.prefix}</span><button type="button" class="josh-about-focus-btn" id="josh-about-arc-btn" aria-label="切换职业里程碑：工程年限、入行年份、转型年份。" style="width:${unit.width}"><span class="josh-about-focus-btn__value josh-about-swap-text" id="josh-about-arc-value">${unit.value}</span></button><span id="josh-about-arc-suffix">${unit.suffix}</span></strong></p>
  </div>`;
}

function joshAboutTalkArrowMarkup() {
  return `<svg class="josh-about-talks__arrow" xmlns="http://www.w3.org/2000/svg" width="1.25rem" height="1.25rem" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="5" y1="12" x2="18" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>`;
}

function joshAboutMachineMatrixLedColor(index) {
  if (index < 6) return '#1AD9FF';
  if (index < 12) return '#32FF98';
  return '#FFEB33';
}

function joshAboutMachineMatrixMarkup() {
  const circles = Array.from({ length: 16 }, (_, index) => {
    const col = Math.floor(index / 2);
    const cx = (6 * col) + 5;
    const cy = index % 2 === 0 ? 5 : 11;
    const fill = joshAboutMachineMatrixLedColor(index);
    return `<circle class="josh-about-machine-hit__led" cx="${cx}" cy="${cy}" r="2" fill="${fill}"/>`;
  }).join('');
  return `<svg width="52" height="16" viewBox="0 0 52 16" fill="none" aria-hidden="true"><rect width="52" height="16" rx="4" fill="#2B2B2B"/>${circles}</svg>`;
}

function joshAboutMachineMiniMatrixMarkup() {
  const circles = Array.from({ length: 8 }, (_, index) => {
    const col = Math.floor(index / 2);
    const cx = (6 * col) + 5;
    const cy = index % 2 === 0 ? 5 : 11;
    const fill = index < 4 ? '#1AD9FF' : '#32FF98';
    return `<circle class="josh-about-machine-hit__led" cx="${cx}" cy="${cy}" r="2" fill="${fill}"/>`;
  }).join('');
  return `<svg class="josh-about-machine-hit__mini-matrix" width="32" height="16" viewBox="0 0 32 16" fill="none" aria-hidden="true"><rect width="32" height="16" rx="4" fill="#2B2B2B"/>${circles}</svg>`;
}

function joshAboutMachineRainbowMarkup() {
  const strokes = [
    'hsl(190deg 100% 55%)',
    'hsl(150deg 100% 60%)',
    'hsl(54deg 100% 60%)',
    'hsl(25deg 100% 55%)',
    'hsl(5deg 100% 55%)',
  ];
  return `<svg class="josh-about-machine-hit__rainbow-svg" width="32" height="34" viewBox="0 0 32 34" fill="none" aria-hidden="true">${strokes.map((stroke, index) => `<polyline class="josh-about-machine-hit__rainbow-line" data-line="${index}" points="0,3.4 28,3.4" stroke="${stroke}" stroke-width="2.5" stroke-linecap="round"/>`).join('')}</svg>`;
}

function joshAboutMachineRand(min, max) {
  return min + Math.random() * (max - min);
}

function joshAboutMachineRandInt(min, max) {
  return Math.floor(joshAboutMachineRand(min, max + 1));
}

function joshAboutMachineMap(value, inMin, inMax, outMin, outMax) {
  return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
}

function joshAboutMachineLerp(from, to, amount) {
  return from + (to - from) * amount;
}

function joshAboutMachineSpring(tension, friction) {
  let value = 0;
  let velocity = 0;
  let target = 0;
  return {
    set(next) { target = next; },
    get() { return value; },
    reset(next = 0) {
      value = next;
      velocity = 0;
      target = next;
    },
    snap(next) {
      value = next;
      velocity = 0;
      target = next;
    },
    step(dt) {
      const accel = -tension * (value - target) - friction * velocity;
      velocity += accel * dt;
      value += velocity * dt;
      return value;
    },
  };
}

function joshAboutMachineWavePoints(width, height, timeSeconds) {
  const innerWidth = width - 16;
  const innerHeight = height - 16;
  const samples = innerWidth / 2;
  const cycles = 3 * Math.PI;
  const points = [];
  for (let index = 0; index < samples; index += 1) {
    const ratio = index / (samples - 1);
    const x = 8 + ratio * innerWidth;
    const y = 8 + innerHeight / 2 + Math.sin(joshAboutMachineMap(ratio + timeSeconds, 0, 1, 0, cycles)) * (innerHeight / 2);
    points.push(`${x},${y.toFixed(2)}`);
  }
  return points.join(' ');
}

function joshAboutMachineRandomArcPoints(width, height) {
  return {
    start: [joshAboutMachineRandInt(8, 16), joshAboutMachineRandInt(8, height - 8)],
    control: [joshAboutMachineRandInt(8, width - 8), joshAboutMachineRandInt(-(0.3 * height), height)],
    end: [joshAboutMachineRandInt(width - 16, width - 8), joshAboutMachineRandInt(8, height - 8)],
  };
}

function joshAboutMachineArcPath(points) {
  const { start, control, end } = points;
  return `M ${start[0].toFixed(2)} ${start[1].toFixed(2)} Q ${control[0].toFixed(2)} ${control[1].toFixed(2)} ${end[0].toFixed(2)} ${end[1].toFixed(2)}`;
}

function joshAboutMachineRainbowLinePoints(value, width, height, lineIndex, numLines) {
  const omegaRatio = value / 100;
  const samplesPerRow = Math.ceil(width / 4);
  const rowHeight = height / numLines;
  const centerY = ((lineIndex + 1) / numLines) * height - rowHeight / 2;
  const points = [];
  for (let sampleIndex = 0; sampleIndex < samplesPerRow; sampleIndex += 1) {
    const baseX = 4 * sampleIndex;
    const angle = joshAboutMachineMap(sampleIndex, 0, samplesPerRow - 1, 0, Math.PI * 2) + (Math.PI / 2);
    const radius = joshAboutMachineLerp(height - centerY, centerY, omegaRatio) * 0.7;
    const polarX = Math.cos(angle) * radius;
    const polarY = Math.sin(angle) * radius;
    const x = joshAboutMachineLerp(polarX + width / 2, baseX, omegaRatio);
    const y = joshAboutMachineLerp(polarY + height / 2, centerY, omegaRatio);
    points.push(`${x},${y}`);
  }
  return points.join(' ');
}

let joshAboutMachineAnimRaf = null;
let joshAboutMachineAnimState = null;

function joshAboutStopMachineAnimations(machineEl) {
  if (joshAboutMachineAnimRaf) {
    cancelAnimationFrame(joshAboutMachineAnimRaf);
    joshAboutMachineAnimRaf = null;
  }
  if (joshAboutMachineAnimState) {
    joshAboutMachineAnimState.timers.forEach((timerId) => window.clearTimeout(timerId));
    joshAboutMachineAnimState = null;
  }
  if (!machineEl) return;
  machineEl.querySelectorAll('.josh-about-machine-hit__led').forEach((led) => {
    led.style.opacity = '';
  });
  machineEl.querySelectorAll('.josh-about-machine-hit__slider-knob').forEach((knob) => {
    knob.style.transform = '';
  });
  const rainbowInner = machineEl.querySelector('.josh-about-machine-hit__rainbow-inner');
  if (rainbowInner) rainbowInner.style.transform = '';
}

function joshAboutStartMachineAnimations(machineEl) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const polyline = machineEl.querySelector('.josh-about-machine-hit__wave');
  const arcPath = machineEl.querySelector('.josh-about-machine-hit__arc path');
  const knobs = [...machineEl.querySelectorAll('.josh-about-machine-hit__slider-knob')];
  const mainMatrixRows = [
    [...machineEl.querySelectorAll('.josh-about-machine-hit__matrix > svg:first-of-type .josh-about-machine-hit__led')],
    [...machineEl.querySelectorAll('.josh-about-machine-hit__matrix > svg:last-of-type .josh-about-machine-hit__led')],
  ];
  const miniLeds = [...machineEl.querySelectorAll('.josh-about-machine-hit__mini-matrix .josh-about-machine-hit__led')];
  const rainbowLines = [...machineEl.querySelectorAll('.josh-about-machine-hit__rainbow-line')];
  const rainbowInner = machineEl.querySelector('.josh-about-machine-hit__rainbow-inner');

  const matrixSprings = mainMatrixRows.map(() => joshAboutMachineSpring(40, 6));
  const miniSpring = joshAboutMachineSpring(40, 6);
  const rainbowSpring = joshAboutMachineSpring(40, 18);
  const sliderSprings = knobs.map(() => joshAboutMachineSpring(170, 26));
  const arcSprings = {
    startX: joshAboutMachineSpring(10, 5),
    startY: joshAboutMachineSpring(10, 5),
    controlX: joshAboutMachineSpring(10, 5),
    controlY: joshAboutMachineSpring(10, 5),
    endX: joshAboutMachineSpring(10, 5),
    endY: joshAboutMachineSpring(10, 5),
  };

  const arcCurrent = joshAboutMachineRandomArcPoints(64, 38);
  const arcTarget = {
    start: [...arcCurrent.start],
    control: [...arcCurrent.control],
    end: [...arcCurrent.end],
  };
  arcSprings.startX.reset(arcCurrent.start[0]);
  arcSprings.startY.reset(arcCurrent.start[1]);
  arcSprings.controlX.reset(arcCurrent.control[0]);
  arcSprings.controlY.reset(arcCurrent.control[1]);
  arcSprings.endX.reset(arcCurrent.end[0]);
  arcSprings.endY.reset(arcCurrent.end[1]);
  const sliderTargets = knobs.map(() => joshAboutMachineRandInt(0, 100));
  sliderSprings.forEach((spring, index) => {
    if (reducedMotion) spring.snap(sliderTargets[index]);
    else spring.reset(0);
  });

  const state = {
    machineEl,
    timers: [],
    waveStart: performance.now(),
    rainbowExpanded: true,
    schedule(fn, delay) {
      const timerId = window.setTimeout(fn, delay);
      state.timers.push(timerId);
      return timerId;
    },
  };
  joshAboutMachineAnimState = state;

  const scheduleMatrixRow = (spring) => {
    const tickMatrix = () => {
      spring.set(joshAboutMachineRandInt(1, 16));
      state.schedule(tickMatrix, joshAboutMachineRand(1500, 2200));
    };
    tickMatrix();
  };
  const scheduleMini = () => {
    miniSpring.set(joshAboutMachineRandInt(1, 8));
    state.schedule(scheduleMini, joshAboutMachineRand(1500, 2200));
  };
  const scheduleArc = () => {
    const next = joshAboutMachineRandomArcPoints(64, 38);
    arcTarget.start = [...next.start];
    arcTarget.control = [...next.control];
    arcTarget.end = [...next.end];
    if (reducedMotion) {
      arcCurrent.start = [...next.start];
      arcCurrent.control = [...next.control];
      arcCurrent.end = [...next.end];
      arcSprings.startX.snap(next.start[0]);
      arcSprings.startY.snap(next.start[1]);
      arcSprings.controlX.snap(next.control[0]);
      arcSprings.controlY.snap(next.control[1]);
      arcSprings.endX.snap(next.end[0]);
      arcSprings.endY.snap(next.end[1]);
    }
    state.schedule(scheduleArc, joshAboutMachineRand(250, 1500));
  };
  const scheduleSliders = () => {
    knobs.forEach((_, index) => {
      sliderTargets[index] = joshAboutMachineRandInt(0, 100);
      if (reducedMotion) sliderSprings[index].snap(sliderTargets[index]);
      else sliderSprings[index].set(sliderTargets[index]);
    });
    const delayMin = reducedMotion ? 2000 : 800;
    const delayMax = reducedMotion ? 4000 : 1600;
    state.schedule(scheduleSliders, joshAboutMachineRand(delayMin, delayMax));
  };
  const scheduleRainbow = () => {
    state.rainbowExpanded = !state.rainbowExpanded;
    rainbowSpring.set(state.rainbowExpanded ? 100 : 0);
    if (rainbowInner) {
      rainbowInner.style.transform = state.rainbowExpanded ? 'translateX(0)' : 'translateX(1px)';
    }
    state.schedule(scheduleRainbow, 1600);
  };

  matrixSprings.forEach((spring) => spring.set(joshAboutMachineRandInt(1, 16)));
  miniSpring.set(joshAboutMachineRandInt(1, 8));
  rainbowSpring.set(100);
  matrixSprings.forEach((spring) => scheduleMatrixRow(spring));
  scheduleMini();
  scheduleArc();
  state.schedule(scheduleSliders, joshAboutMachineRand(reducedMotion ? 200 : 200, reducedMotion ? 600 : 600));
  state.schedule(scheduleRainbow, 1600);

  let lastFrame = performance.now();

  const tick = (now) => {
    if (!machineEl.classList.contains('is-on')) return;
    const dt = Math.min(0.05, (now - lastFrame) / 1000);
    lastFrame = now;

    if (polyline) {
      const waveTime = (now - state.waveStart) / 1000;
      polyline.setAttribute('points', joshAboutMachineWavePoints(48, 34, waveTime));
    }

    if (arcPath) {
      arcSprings.startX.set(arcTarget.start[0]);
      arcSprings.startY.set(arcTarget.start[1]);
      arcSprings.controlX.set(arcTarget.control[0]);
      arcSprings.controlY.set(arcTarget.control[1]);
      arcSprings.endX.set(arcTarget.end[0]);
      arcSprings.endY.set(arcTarget.end[1]);
      arcCurrent.start[0] = arcSprings.startX.step(dt);
      arcCurrent.start[1] = arcSprings.startY.step(dt);
      arcCurrent.control[0] = arcSprings.controlX.step(dt);
      arcCurrent.control[1] = arcSprings.controlY.step(dt);
      arcCurrent.end[0] = arcSprings.endX.step(dt);
      arcCurrent.end[1] = arcSprings.endY.step(dt);
      arcPath.setAttribute('d', joshAboutMachineArcPath(arcCurrent));
    }

    knobs.forEach((knob, index) => {
      const value = reducedMotion
        ? sliderTargets[index]
        : sliderSprings[index].step(dt);
      const y = joshAboutMachineMap(value, 0, 100, 4, -22);
      knob.style.transform = `translateY(${y.toFixed(2)}px)`;
    });

    matrixSprings.forEach((spring, rowIndex) => {
      const matrixValue = spring.step(dt);
      mainMatrixRows[rowIndex].forEach((led, index) => {
        led.style.opacity = matrixValue > index ? '1' : '0';
      });
    });

    const miniValue = miniSpring.step(dt);
    miniLeds.forEach((led, index) => {
      led.style.opacity = miniValue > index ? '1' : '0';
    });

    const rainbowValue = rainbowSpring.step(dt);
    rainbowLines.forEach((line, index) => {
      line.setAttribute('points', joshAboutMachineRainbowLinePoints(rainbowValue, 32, 34, index, 5));
    });

    joshAboutMachineAnimRaf = requestAnimationFrame(tick);
  };

  joshAboutMachineAnimRaf = requestAnimationFrame(tick);
}

function joshAboutSetMachinePower(machineEl, isOn) {
  if (!machineEl) return;
  joshAboutStopMachineAnimations(machineEl);
  machineEl.classList.toggle('is-on', isOn);
  const vizEl = machineEl.closest('.josh-about-card--side')?.querySelector('#josh-about-viz');
  if (isOn && vizEl) joshAboutVizStart(vizEl);
  if (!isOn || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  joshAboutStartMachineAnimations(machineEl);
}

/* ── Viz panel: Transformer visualizations ── */
let joshAboutVizRaf = null;
let joshAboutVizTimers = [];

function joshAboutVizStop() {
  if (joshAboutVizRaf) { cancelAnimationFrame(joshAboutVizRaf); joshAboutVizRaf = null; }
  joshAboutVizTimers.forEach(clearTimeout);
  joshAboutVizTimers = [];
}

function joshAboutVizStart(vizEl) {
  joshAboutVizStop();
  const scenes = ['sigmoid', 'attention', 'aggregate'];
  let idx = 0;

  const showScene = () => {
    scenes.forEach((s, i) => {
      const el = vizEl.querySelector(`.josh-about-viz__scene--${s}`);
      if (el) el.classList.toggle('is-visible', i === idx);
    });
    const sideCard = vizEl.closest('.josh-about-card--side');
    if (sideCard) {
      sideCard.setAttribute('data-viz-scene', scenes[idx]);
    }
    if (scenes[idx] === 'sigmoid') joshAboutVizRenderSigmoid(vizEl);
    else if (scenes[idx] === 'attention') joshAboutVizRenderAttention(vizEl);
    else joshAboutVizRenderAggregate(vizEl);
  };

  showScene();
  const advance = () => {
    idx = (idx + 1) % scenes.length;
    showScene();
    const tid = setTimeout(advance, 5000);
    joshAboutVizTimers.push(tid);
  };
  const tid = setTimeout(advance, 5000);
  joshAboutVizTimers.push(tid);
}

function joshAboutVizRenderSigmoid(vizEl) {
  const curve = vizEl.querySelector('.josh-about-viz__sigmoid-curve');
  if (!curve) return;
  const totalPoints = 120;
  const xMin = -6, xMax = 6;
  const svgXMin = 30, svgXMax = 300;
  const svgYMin = 130, svgYMax = 20;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const buildPoints = (count) => {
    const pts = [];
    for (let i = 0; i <= count; i++) {
      const ratio = i / totalPoints;
      const x = xMin + ratio * (xMax - xMin);
      const y = 1 / (1 + Math.exp(-x));
      const sx = svgXMin + ratio * (svgXMax - svgXMin);
      const sy = svgYMin - y * (svgYMin - svgYMax);
      pts.push(`${sx.toFixed(1)},${sy.toFixed(1)}`);
    }
    return pts.join(' ');
  };

  if (reducedMotion) {
    curve.setAttribute('points', buildPoints(totalPoints));
    return;
  }

  let drawn = 0;
  const step = () => {
    if (drawn >= totalPoints) return;
    drawn = Math.min(drawn + 3, totalPoints);
    curve.setAttribute('points', buildPoints(drawn));
    joshAboutVizRaf = requestAnimationFrame(step);
  };
  curve.setAttribute('points', '');
  step();
}

function joshAboutVizRenderAttention(vizEl) {
  const heatmapG = vizEl.querySelector('.josh-about-viz__heatmap');
  if (!heatmapG) return;
  heatmapG.innerHTML = '';

  const size = 4;
  const cellW = 60, cellH = 28, gap = 3;
  const offsetX = (320 - size * (cellW + gap) + gap) / 2;
  const offsetY = (160 - size * (cellH + gap) + gap) / 2;
  const scores = [];
  for (let r = 0; r < size; r++) {
    scores[r] = [];
    for (let c = 0; c < size; c++) {
      scores[r][c] = Math.random();
    }
  }
  const rowSums = scores.map(row => row.reduce((a, b) => a + b, 0));
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const labels = ['Q1', 'Q2', 'Q3', 'Q4'];
  labels.forEach((lbl, i) => {
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', String(offsetX - 8));
    text.setAttribute('y', String(offsetY + i * (cellH + gap) + cellH / 2 + 3));
    text.setAttribute('fill', '#888');
    text.setAttribute('font-size', '9');
    text.setAttribute('font-family', 'monospace');
    text.setAttribute('text-anchor', 'end');
    text.textContent = lbl;
    heatmapG.appendChild(text);
  });
  ['K1', 'K2', 'K3', 'K4'].forEach((lbl, i) => {
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', String(offsetX + i * (cellW + gap) + cellW / 2));
    text.setAttribute('y', String(offsetY - 6));
    text.setAttribute('fill', '#888');
    text.setAttribute('font-size', '9');
    text.setAttribute('font-family', 'monospace');
    text.setAttribute('text-anchor', 'middle');
    text.textContent = lbl;
    heatmapG.appendChild(text);
  });

  if (reducedMotion) {
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const val = scores[r][c] / rowSums[r];
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', String(offsetX + c * (cellW + gap)));
        rect.setAttribute('y', String(offsetY + r * (cellH + gap)));
        rect.setAttribute('width', String(cellW));
        rect.setAttribute('height', String(cellH));
        rect.setAttribute('rx', '4');
        const intensity = Math.round(val * 255);
        rect.setAttribute('fill', `rgb(${intensity}, ${Math.round(intensity * 0.15)}, ${intensity})`);
        heatmapG.appendChild(rect);
      }
    }
    return;
  }

  let cellIdx = 0;
  const totalCells = size * size;

  const drawCell = () => {
    if (cellIdx >= totalCells) return;
    const r = Math.floor(cellIdx / size);
    const c = cellIdx % size;
    const val = scores[r][c] / rowSums[r];
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', offsetX + c * (cellW + gap));
    rect.setAttribute('y', offsetY + r * (cellH + gap));
    rect.setAttribute('width', cellW);
    rect.setAttribute('height', cellH);
    rect.setAttribute('rx', '4');
    const intensity = Math.round(val * 255);
    rect.setAttribute('fill', `rgb(${intensity}, ${Math.round(intensity * 0.15)}, ${intensity})`);
    rect.setAttribute('opacity', '0');
    rect.style.transition = 'opacity 0.25s ease';
    heatmapG.appendChild(rect);
    requestAnimationFrame(() => { rect.setAttribute('opacity', '1'); });
    cellIdx++;
    const tid = setTimeout(drawCell, 80);
    joshAboutVizTimers.push(tid);
  };

  drawCell();
}

function joshAboutVizRenderAggregate(vizEl) {
  const vGroup = vizEl.querySelector('.josh-about-viz__v-vectors');
  const linesG = vizEl.querySelector('.josh-about-viz__attn-lines');
  const outGroup = vizEl.querySelector('.josh-about-viz__output-vector');
  if (!vGroup || !linesG || !outGroup) return;
  vGroup.innerHTML = '';
  linesG.innerHTML = '';
  outGroup.innerHTML = '';

  const numV = 4;
  const barW = 40, barGap = 16, barMaxH = 70;
  const offsetX = (320 - (numV * barW + (numV - 1) * barGap)) / 2;
  const offsetY = 40;
  const vHeights = Array.from({ length: numV }, () => 25 + Math.random() * 45);
  const weights = [0.45, 0.1, 0.35, 0.1];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const outH = weights.reduce((sum, w, i) => sum + w * vHeights[i], 0);

  vHeights.forEach((h, i) => {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', String(offsetX + i * (barW + barGap)));
    rect.setAttribute('y', String(offsetY + barMaxH - h));
    rect.setAttribute('width', String(barW));
    rect.setAttribute('height', String(h));
    rect.setAttribute('rx', '4');
    rect.setAttribute('fill', '#1AD9FF');
    vGroup.appendChild(rect);

    const wt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    wt.setAttribute('x', String(offsetX + i * (barW + barGap) + barW / 2));
    wt.setAttribute('y', String(offsetY + barMaxH + 16));
    wt.setAttribute('fill', '#FF27FF');
    wt.setAttribute('font-size', '10');
    wt.setAttribute('font-family', 'monospace');
    wt.setAttribute('text-anchor', 'middle');
    wt.textContent = `×${weights[i].toFixed(2)}`;
    vGroup.appendChild(wt);
  });

  const outX = offsetX + (numV * barW + (numV - 1) * barGap) / 2 - 24;
  const outRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  outRect.setAttribute('x', String(outX));
  outRect.setAttribute('y', String(offsetY + barMaxH - outH));
  outRect.setAttribute('width', '48');
  outRect.setAttribute('height', String(outH));
  outRect.setAttribute('rx', '4');
  outRect.setAttribute('fill', '#32FF98');
  outGroup.appendChild(outRect);

  const outLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  outLabel.setAttribute('x', String(outX + 24));
  outLabel.setAttribute('y', String(offsetY + barMaxH + 16));
  outLabel.setAttribute('fill', '#32FF98');
  outLabel.setAttribute('font-size', '10');
  outLabel.setAttribute('font-family', 'monospace');
  outLabel.setAttribute('text-anchor', 'middle');
  outLabel.textContent = 'Σ output';
  outGroup.appendChild(outLabel);

  if (reducedMotion) return;

  vGroup.querySelectorAll('rect').forEach((rect, i) => {
    rect.setAttribute('opacity', '0');
    rect.style.transition = `opacity 0.3s ease ${i * 0.1}s`;
    requestAnimationFrame(() => rect.setAttribute('opacity', '1'));
  });
  vGroup.querySelectorAll('text').forEach((t, i) => {
    t.setAttribute('opacity', '0');
    t.style.transition = `opacity 0.3s ease ${0.4 + i * 0.1}s`;
    requestAnimationFrame(() => t.setAttribute('opacity', '1'));
  });

  outRect.setAttribute('opacity', '0');
  outRect.style.transition = 'opacity 0.5s ease 0.9s';
  requestAnimationFrame(() => outRect.setAttribute('opacity', '1'));
  outLabel.setAttribute('opacity', '0');
  outLabel.style.transition = 'opacity 0.4s ease 1.1s';
  requestAnimationFrame(() => outLabel.setAttribute('opacity', '1'));
}

function joshAboutMachineChassisSvg() {
  return `<svg width="145" height="177" viewBox="0 0 145 177" fill="none" aria-hidden="true" class="josh-about-machine-chassis__svg"><mask id="josh-about-machine-mask" maskUnits="userSpaceOnUse" x="0" y="0" width="145" height="177"><path d="M12 4.99999L4 13L0 17V21V171V177H6H122H128L132 173L140 165C140 165 142.13 162.531 143 160.5C143.923 158.346 144 155 144 155V6.99999C144 6.99999 144.28 3.27999 143 2C141.72 0.72001 138 0.999992 138 0.999992H22C22 0.999992 18.1537 1.07698 16 1.99999C13.9695 2.87021 12 4.99999 12 4.99999Z" fill="#FFFFFF"></path></mask><g mask="url(#josh-about-machine-mask)"><path d="M16 1L0 17H128L144 1H16Z" fill="#ABABAB"></path><path d="M144 1L128 17V177L144 161V1Z" fill="#959595"></path><rect y="17" width="128" height="160" fill="#C4C4C4"></rect></g></svg>`;
}

function joshAboutMachinePowerSvg() {
  const uid = 'josh-about-power';
  return `<svg class="josh-about-machine-hit__power" width="30" height="30" viewBox="0 0 40 40" fill="none" aria-hidden="true"><mask id="${uid}-mask" mask-type="alpha" maskUnits="userSpaceOnUse" x="0" y="0" width="40" height="40"><circle cx="20" cy="20" r="15" fill="#000000"></circle></mask><g><circle cx="20" cy="20" r="20" fill="#2B2B2B"></circle><path d="M40 20C40 31.0457 31.0457 40 20 40C8.95431 40 0 31.0457 0 20C0 8.95431 8.95431 0 20 0C31.0457 0 40 8.95431 40 20ZM4.81942 20C4.81942 28.384 11.616 35.1806 20 35.1806C28.384 35.1806 35.1806 28.384 35.1806 20C35.1806 11.616 28.384 4.81942 20 4.81942C11.616 4.81942 4.81942 11.616 4.81942 20Z" fill="url(#${uid}-outer-ring)"></path><g filter="url(#${uid}-blur)"><path d="M36.5 20C36.5 24.3761 34.7616 28.5729 31.6673 31.6673C28.5729 34.7616 24.3761 36.5 20 36.5C15.6239 36.5 11.4271 34.7616 8.33274 31.6673C5.23839 28.5729 3.5 24.3761 3.5 20" stroke="url(#${uid}-underside-glow)" stroke-opacity="0.5" stroke-linecap="round"></path></g><g filter="url(#${uid}-highlight-edge)"><path d="M8.21985 5.39015C11.4832 2.56555 15.6528 1.00752 19.9687 1.00003C24.2847 0.992532 28.4597 2.53607 31.7328 5.34932" stroke="url(#${uid}-top-glow)" stroke-opacity="0.5" stroke-width="2" stroke-linecap="round" style="mix-blend-mode:luminosity"></path></g><g mask="url(#${uid}-mask)"><g class="josh-about-machine-hit__power-inner"><circle cx="20" cy="20" r="15" fill="hsl(350deg 100% 50%)"></circle><circle cx="20" cy="20" r="13" fill="url(#${uid}-3d)" class="josh-about-machine-hit__power-highlight"></circle></g></g></g><defs><filter id="${uid}-blur" x="2" y="18.5" width="36" height="19.5" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB"><feFlood flood-opacity="0" result="BackgroundImageFix"></feFlood><feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape"></feBlend><feGaussianBlur stdDeviation="0.5" result="effect1_foregroundBlur"></feGaussianBlur></filter><filter id="${uid}-highlight-edge" x="5.21982" y="-2" width="29.513" height="10.3902" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB"><feFlood flood-opacity="0" result="BackgroundImageFix"></feFlood><feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape"></feBlend><feGaussianBlur stdDeviation="1" result="effect1_foregroundBlur"></feGaussianBlur></filter><radialGradient id="${uid}-outer-ring" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(20 20) rotate(90) scale(20)"><stop offset="0.773481" stop-color="hsl(348deg 90% 25%)"></stop><stop offset="1" stop-color="hsl(350deg 100% 50%)"></stop></radialGradient><linearGradient id="${uid}-underside-glow" x1="40.7187" y1="20" x2="-1.25" y2="20" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="hsl(350deg 100% 50%)" stop-opacity="0"></stop><stop offset="0.490056" stop-color="white"></stop><stop offset="1" stop-color="#FF0000" stop-opacity="0"></stop></linearGradient><linearGradient id="${uid}-top-glow" x1="1" y1="0" x2="39.5" y2="0" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="white" stop-opacity="0.25"></stop><stop offset="0.508287" stop-color="white"></stop><stop offset="1" stop-color="white" stop-opacity="0.44"></stop></linearGradient><linearGradient id="${uid}-3d" x1="20" y1="8" x2="20" y2="32" gradientUnits="userSpaceOnUse"><stop stop-opacity="0.33"></stop><stop offset="1" stop-color="white" stop-opacity="0.21"></stop></linearGradient></defs></svg>`;
}

function joshAboutMachineMarkup() {
  return `<div class="josh-about-machine" id="josh-about-machine">
    <div class="josh-about-machine-stage">
      <div class="josh-about-machine-chassis" aria-hidden="true">
        ${joshAboutMachineChassisSvg()}
      </div>
      <button type="button" class="josh-about-machine-hit" id="josh-about-machine-btn" aria-pressed="false" aria-label="Illustration of a small machine. Triggering this button powers on the machine. This is a purely cosmetic effect.">
        <span class="josh-about-machine-hit__rows" aria-hidden="true">
          <span class="josh-about-machine-hit__row">
            <svg class="josh-about-machine-hit__screen" width="48" height="34" viewBox="0 0 48 34" fill="none" aria-hidden="true" style="backface-visibility:hidden">
              <rect width="48" height="34" rx="4" fill="#2B2B2B"/>
              <polyline class="josh-about-machine-hit__wave" points="8,20 10,15 12,11 14,9 16,10 18,13 20,18 22,23 24,26 26,26 28,23 30,18 32,13 34,9 36,8 38,10" stroke="#32FF98" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <span class="josh-about-machine-hit__gap" aria-hidden="true"></span>
            <span class="josh-about-machine-hit__matrix">
              ${joshAboutMachineMatrixMarkup()}
              ${joshAboutMachineMatrixMarkup()}
            </span>
          </span>
          <span class="josh-about-machine-hit__gap" aria-hidden="true"></span>
          <span class="josh-about-machine-hit__row">
            <svg class="josh-about-machine-hit__arc" width="64" height="38" viewBox="0 0 64 38" fill="none" aria-hidden="true">
              <rect width="64" height="38" rx="4" fill="#2B2B2B"/>
              <path d="M10.788 21.89 Q18.474 3.84 52.922 21.47" stroke="#FF27FF" stroke-width="4" stroke-linecap="round"/>
            </svg>
            <span class="josh-about-machine-hit__gap" aria-hidden="true"></span>
            <span class="josh-about-machine-hit__rainbow-wrap">
              <span class="josh-about-machine-hit__rainbow-inner">
                ${joshAboutMachineRainbowMarkup()}
              </span>
            </span>
          </span>
          <span class="josh-about-machine-hit__gap" aria-hidden="true"></span>
          <span class="josh-about-machine-hit__row">
            <span class="josh-about-machine-hit__sliders">
              <svg width="20" height="52" viewBox="0 0 20 52" fill="none" aria-hidden="true">
                <rect x="2" width="16" height="52" rx="4" fill="#2B2B2B"/>
                <line x1="6" y1="5" x2="14" y2="5" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="11" x2="14" y2="11" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="17" x2="14" y2="17" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="23" x2="14" y2="23" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="29" x2="14" y2="29" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="35" x2="14" y2="35" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="41" x2="14" y2="41" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="47" x2="14" y2="47" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <rect class="josh-about-machine-hit__slider-knob" y="44" width="20" height="8" rx="4" fill="#FF27FF"/>
              </svg>
              <svg width="20" height="52" viewBox="0 0 20 52" fill="none" aria-hidden="true">
                <rect x="2" width="16" height="52" rx="4" fill="#2B2B2B"/>
                <line x1="6" y1="5" x2="14" y2="5" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="11" x2="14" y2="11" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="17" x2="14" y2="17" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="23" x2="14" y2="23" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="29" x2="14" y2="29" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="35" x2="14" y2="35" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="41" x2="14" y2="41" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <line x1="6" y1="47" x2="14" y2="47" stroke="#fff" stroke-opacity="0.51" stroke-width="2"/>
                <rect class="josh-about-machine-hit__slider-knob josh-about-machine-hit__slider-knob--delay" y="44" width="20" height="8" rx="4" fill="#FF27FF"/>
              </svg>
            </span>
            <span class="josh-about-machine-hit__gap" aria-hidden="true"></span>
            <span class="josh-about-machine-hit__gears-wrap">
              <svg class="josh-about-machine-hit__gears" width="30" height="28" viewBox="0 0 30 28" fill="none" aria-hidden="true">
                <circle cx="15" cy="17" r="11" stroke="#1AD9FF" stroke-width="2"/>
                <mask id="josh-about-machine-gear-mask" maskUnits="userSpaceOnUse" x="6" y="8" width="18" height="18">
                  <circle cx="15" cy="17" r="8.8" fill="#C4C4C4"/>
                </mask>
                <g mask="url(#josh-about-machine-gear-mask)" stroke="#32FF98" stroke-width="2" stroke-linejoin="round">
                  <g class="josh-about-machine-hit__gear-shapes">
                    <path d="M8.03333 17.0001L4 12.0501V9.30015L6.30476 8.20015H9.7619L10.9143 10.9501L13.219 10.4001L14.3714 13.1501L12.0667 13.7001L10.3381 16.4501L13.7952 17.5501L16.1 19.7501L13.7952 22.5001L10.9143 24.7001L12.0667 20.8501L10.3381 18.1001L8.03333 17.0001Z"/>
                    <path d="M19.7038 18.1001L20.5 15.9001L22.7 14.8001L24.9 15.3501H27.1L28.75 18.6501L29.85 21.9501L27.65 24.7001L24.9 24.1501L23.25 19.7501L21.6 20.3001L19.7038 18.1001Z"/>
                    <path d="M20.5 13.1501L21.6 10.9501L20.5 7.65015L23.25 8.20015L30.4 10.4001L26 13.7001L23.8 12.6001L20.5 13.1501Z"/>
                  </g>
                </g>
              </svg>
              ${joshAboutMachineMiniMatrixMarkup()}
            </span>
            <span class="josh-about-machine-hit__gap" aria-hidden="true"></span>
            <span class="josh-about-machine-hit__face">
              <span class="josh-about-machine-hit__eyes-wrap">
                <svg class="josh-about-machine-hit__eyes" width="24" height="17" viewBox="0 0 24 17" fill="none" aria-hidden="true">
                  <circle class="josh-about-machine-hit__eye josh-about-machine-hit__eye--left" cx="7.5" cy="8.5" r="3.5"/>
                  <circle class="josh-about-machine-hit__eye josh-about-machine-hit__eye--right" cx="16.5" cy="8.5" r="3.5"/>
                </svg>
              </span>
              <span class="josh-about-machine-hit__power-wrap">
                <span class="josh-about-machine-hit__power-btn">
                  ${joshAboutMachinePowerSvg()}
                </span>
              </span>
            </span>
          </span>
        </span>
        <span class="josh-visually-hidden">Toggle Power</span>
      </button>
    </div>
  </div>`;
}

function joshAboutNameSpeakMarkup() {
  return `<span class="josh-about-name-speak" id="josh-about-name-speak" aria-hidden="true">
    <svg class="josh-about-name-wave josh-about-name-wave--outer" viewBox="0 0 14 16" width="14" height="16">
      <path d="M2 8 C2 4 5 2 8 2 C11 2 12 5 12 8 C12 11 11 14 8 14 C5 14 2 12 2 8" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
    <svg class="josh-about-name-wave josh-about-name-wave--inner" viewBox="0 0 10 12" width="10" height="12">
      <path d="M2 6 C2 3.5 4 2 6 2 C8 2 9 4 9 6 C9 8 8 10 6 10 C4 10 2 8.5 2 6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
    <svg class="josh-about-name-speaker" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
    </svg>
  </span>`;
}

function joshAboutGridMarkup() {
  const stats = joshAboutPostStats();
  const { postCount } = stats;
  return `<div class="josh-about-grid">
    <article class="josh-about-card josh-about-card--map" style="--josh-about-area: map">
      ${joshAboutMapWidgetMarkup()}
      <p id="josh-about-map-copy">我来自深圳。按 IP 粗算，我们相距约 <em id="josh-about-map-distance-inline">…</em>。</p>
      ${joshAboutMapCopyMarkup()}
    </article>

    <article class="josh-about-card josh-about-card--tall" style="--josh-about-area: tall">
      ${joshAboutArcMarkup()}
    </article>

    <article class="josh-about-card josh-about-card--job" style="--josh-about-area: job">
      <img class="josh-about-job-mascot josh-about-job-mascot--react" src="https://www.joshwcomeau.com/images/newsletter/joy-of-react-mascot.png" alt="" width="260" height="228">
      <img class="josh-about-job-mascot josh-about-job-mascot--css" src="https://www.joshwcomeau.com/images/css-for-js-mascot-light.png" alt="" width="200" height="178">
      <p class="josh-about-job-lead">工作室做 LLM 落地；博客和 side project，是我一直对外说话的地方。</p>
      <p>写给已经会调模型、还想知道 <strong>为什么这样设计、踩坑后怎么改</strong> 的人——不是入门课，也不是软文。</p>
      <p>目前沉淀了 <em><span class="josh-about-stat" id="josh-about-stat" data-target="${postCount}">0</span> 篇</em> 长文。</p>
      ${joshAboutJobCategoriesMarkup()}
    </article>

    <article class="josh-about-card josh-about-card--gay" style="--josh-about-area: gay">
      ${joshAboutGayMarkup(0)}
    </article>

    <article class="josh-about-card josh-about-card--conf" style="--josh-about-area: conf">
      ${joshAboutConfCoverMarkup()}
      <p>想先感受我写的东西，从下面三段开始：</p>
      ${joshAboutHotPostsMarkup(3)}
    </article>

    <article class="josh-about-card josh-about-card--pkg" style="--josh-about-area: pkg">
      ${joshAboutDrumsMarkup()}
      ${joshAboutPkgCopyMarkup()}
    </article>

    <article class="josh-about-card josh-about-card--desk" style="--josh-about-area: desk">
      ${joshAboutDeskMarkup()}
      <p id="josh-about-desk-text">演示环境 1.9 秒，老板点头；灰度一周，P95 到了 4.2 秒，工单来了。</p>
      <p class="small" id="josh-about-desk-sub">先画延迟瀑布，别在 Embedding 上空转。</p>
    </article>

    <article class="josh-about-card josh-about-card--side" style="--josh-about-area: side">
      <div class="josh-about-side-layout">
        <div class="josh-about-machine-wrap">
          ${joshAboutMachineMarkup()}
        </div>
        <div class="josh-about-side-copy">
          ${joshAboutSideCopyMarkup()}
        </div>
      </div>
      <div class="josh-about-viz" id="josh-about-viz" aria-hidden="true">
        <svg class="josh-about-viz__scene josh-about-viz__scene--sigmoid" viewBox="0 0 320 160" fill="none">
          <line class="josh-about-viz__axis" x1="30" y1="130" x2="300" y2="130" stroke="#555" stroke-width="1"/>
          <line class="josh-about-viz__axis" x1="30" y1="20" x2="30" y2="130" stroke="#555" stroke-width="1"/>
          <text x="15" y="28" fill="#888" font-size="9" font-family="monospace">1</text>
          <text x="15" y="80" fill="#888" font-size="9" font-family="monospace">.5</text>
          <text x="15" y="133" fill="#888" font-size="9" font-family="monospace">0</text>
          <line x1="28" y1="75" x2="300" y2="75" stroke="#444" stroke-width="0.5" stroke-dasharray="3,3"/>
          <polyline class="josh-about-viz__sigmoid-curve" points="" stroke="#32FF98" stroke-width="2.5" stroke-linecap="round" fill="none"/>
          <text class="josh-about-viz__label" x="160" y="155" fill="#aaa" font-size="10" font-family="monospace" text-anchor="middle">σ(x) = 1 / (1 + e⁻ˣ)</text>
        </svg>
        <svg class="josh-about-viz__scene josh-about-viz__scene--attention" viewBox="0 0 320 160" fill="none">
          <text class="josh-about-viz__label" x="160" y="155" fill="#aaa" font-size="10" font-family="monospace" text-anchor="middle">Q · Kᵀ → attention scores</text>
          <g class="josh-about-viz__heatmap" transform="translate(0,0)"></g>
        </svg>
        <svg class="josh-about-viz__scene josh-about-viz__scene--aggregate" viewBox="0 0 320 160" fill="none">
          <g class="josh-about-viz__v-vectors" transform="translate(0,0)"></g>
          <g class="josh-about-viz__attn-lines"></g>
          <g class="josh-about-viz__output-vector" transform="translate(0,0)"></g>
        </svg>
      </div>
    </article>

    <article class="josh-about-card josh-about-card--cat" style="--josh-about-area: cat">
      ${joshAboutCatMarkup()}
    </article>

    <article class="josh-about-card josh-about-card--name" style="--josh-about-area: name">
      <svg class="josh-about-quote-mark josh-about-quote-mark--open" width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M10 11H6C6 7.5 7.5 6 10 5L11 7C9.5 7.5 9 8 9 9H10C11 9 12 10 12 11V13C12 14 11 15 10 15H8C7 15 6 14 6 13V11ZM20 11H16C16 7.5 17.5 6 20 5L21 7C19.5 7.5 19 8 19 9H20C21 9 22 10 22 11V13C22 14 21 15 20 15H18C17 15 16 14 16 13V11Z" fill="currentColor"/>
      </svg>
      <button type="button" class="josh-about-name-btn" id="josh-about-name-btn" aria-label="播放一句激励名言。">
        <span>Do what you can't.</span>
        ${joshAboutNameSpeakMarkup()}
      </button>
      <p>— Kathrine Switzer</p>
      <p class="small">做别人说不可能的事。</p>
      <svg class="josh-about-quote-mark josh-about-quote-mark--close" width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M14 13H18C18 16.5 16.5 18 14 19L13 17C14.5 16.5 15 16 15 15H14C13 15 12 14 12 13V11C12 10 13 9 14 9H16C17 9 18 10 18 11V13ZM4 13H8C8 16.5 6.5 18 4 19L3 17C4.5 16.5 5 16 5 15H4C3 15 2 14 2 13V11C2 10 3 9 4 9H6C7 9 8 10 8 11V13Z" fill="currentColor"/>
      </svg>
    </article>
  </div>`;
}

function joshAboutSkillsSectionMarkup() {
  if (typeof renderAboutSkillStack !== 'function') return '';
  return `<section class="josh-about-section" aria-labelledby="josh-about-skills-heading">
    <h2 class="josh-about-section__title" id="josh-about-skills-heading">技能与文章</h2>
    <p class="josh-page-desc">高亮标签表示博客已有对应主题的文章，点击可跳转阅读。</p>
    ${renderAboutSkillStack()}
  </section>`;
}

function buildJoshAboutPageShell(heroHtml, gridHtml, activeHref) {
  const bodyWaveHtml = typeof joshAboutBodyWaveMarkup === 'function' ? joshAboutBodyWaveMarkup() : '';
  const bodyNavHtml = buildJoshInnerHeaderMarkup(activeHref, { aboutBodySticky: true });
  return `<div class="josh-page josh-page--about">
    ${heroHtml}
    <div class="josh-about-body">
      ${bodyWaveHtml}
      <div class="josh-about-body__sticky" aria-hidden="true">
        <div class="josh-about-body__sticky-inner">
          <div class="josh-about-body__nav-bg" aria-hidden="true"></div>
          ${bodyNavHtml}
        </div>
      </div>
      <div class="josh-about-body__gap" aria-hidden="true"></div>
      <div class="josh-about-body__inner">
        ${gridHtml}
      </div>
    </div>
    ${buildJoshFooterMarkup({ aboutPage: true })}
  </div>`;
}

function joshMountAboutPage(app, heroHtml, gridHtml, activeHref) {
  app.innerHTML = buildJoshAboutPageShell(heroHtml, gridHtml, activeHref);
  queueMicrotask(() => initJoshSiteInteractions(app));
}

function joshHaversineKm(lat1, lon1, lat2, lon2) {
  const toRad = (deg) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function joshAboutFormatDistance(km) {
  if (!Number.isFinite(km)) return null;
  const miles = km * 0.621371;
  if (miles < 10) return `${miles.toFixed(1)} miles`;
  return `${Math.round(miles).toLocaleString('en-US')} miles`;
}

function joshAboutFormatDistanceKm(km) {
  if (!Number.isFinite(km)) return null;
  if (km >= 1000) return `约 ${Math.round(km).toLocaleString('zh-CN')} 公里`;
  return `约 ${Math.round(km)} 公里`;
}

function joshAboutAnimateStat(el, target, duration = 1200) {
  if (!el || !Number.isFinite(target)) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    el.textContent = target.toLocaleString('en-US');
    return;
  }
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - (1 - t) ** 3;
    el.textContent = Math.round(target * eased).toLocaleString('en-US');
    if (t < 1) requestAnimationFrame(tick);
    else el.textContent = target.toLocaleString('en-US');
  };
  requestAnimationFrame(tick);
}

async function joshAboutHydrateMapDistance(distanceEl) {
  const inlineEl = document.getElementById('josh-about-map-distance-inline');
  const target = distanceEl || inlineEl;
  if (!target) return;
  const { lat: authorLat, lon: authorLon } = JOSH_ABOUT_AUTHOR_GEO;
  const fallback = '很远';
  try {
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timeoutId = controller ? window.setTimeout(() => controller.abort(), 5000) : null;
    const response = await fetch('https://ipwho.is/', controller ? { signal: controller.signal } : undefined);
    if (timeoutId) window.clearTimeout(timeoutId);
    if (!response.ok) throw new Error('geo unavailable');
    const data = await response.json();
    if (!data.success || !Number.isFinite(data.latitude) || !Number.isFinite(data.longitude)) {
      throw new Error('geo invalid');
    }
    const placeZh = await joshAboutPlaceLabelZh(data.latitude, data.longitude, data);
    joshAboutUpdateMapRoute(data.longitude, data.latitude, data, placeZh);
    const km = joshHaversineKm(authorLat, authorLon, data.latitude, data.longitude);
    const display = joshAboutFormatDistanceKm(km) || fallback;
    const distancePill = document.getElementById('josh-about-map-distance-pill');
    if (distancePill) {
      distancePill.textContent = display;
      distancePill.hidden = false;
    }
    if (inlineEl) inlineEl.textContent = display;
    if (distanceEl && distanceEl !== inlineEl) distanceEl.textContent = display;
  } catch {
    if (inlineEl) inlineEl.textContent = fallback;
    if (distanceEl && distanceEl !== inlineEl) distanceEl.textContent = fallback;
    const visitorPlace = document.getElementById('josh-about-map-visitor-place');
    if (visitorPlace) visitorPlace.textContent = '未能定位';
  }
}

function initJoshAboutInteractions(app) {
  const cleanups = [];
  const arcUnits = joshAboutArcUnits();

  const flagBtn = app.querySelector('#josh-about-flag-btn');
  const flagStrip = app.querySelector('#josh-about-flag-strip');
  const flagText = app.querySelector('#josh-about-flag-text');
  const flagSub = app.querySelector('#josh-about-flag-sub');
  if (flagBtn && flagStrip) {
    let flagIndex = 0;
    let flagSwapTimer = null;
    const onFlag = () => {
      flagIndex = (flagIndex + 1) % JOSH_ABOUT_CREDOS.length;
      const credo = JOSH_ABOUT_CREDOS[flagIndex];
      flagStrip.classList.remove('is-swapping');
      void flagStrip.offsetWidth;
      flagStrip.innerHTML = joshAboutFlagColumnsMarkup(credo);
      flagStrip.classList.add('is-swapping');
      if (flagSwapTimer) window.clearTimeout(flagSwapTimer);
      flagSwapTimer = window.setTimeout(() => flagStrip.classList.remove('is-swapping'), 1000);
      flagBtn.setAttribute('aria-label', credo.label);
      if (flagText) flagText.textContent = credo.text;
      if (flagSub) flagSub.textContent = credo.subtitle;
      if (typeof joshPlaySound === 'function') joshPlaySound('toggle');
    };
    flagBtn.addEventListener('click', onFlag);
    cleanups.push(() => {
      flagBtn.removeEventListener('click', onFlag);
      if (flagSwapTimer) window.clearTimeout(flagSwapTimer);
    });
  }

  const deskCard = app.querySelector('.josh-about-card--desk');
  const deskText = app.querySelector('#josh-about-desk-text');
  const deskSub = app.querySelector('#josh-about-desk-sub');
  if (deskCard && deskText) {
    let deskIndex = 0;
    deskCard.style.cursor = 'pointer';
    deskCard.setAttribute('role', 'button');
    deskCard.setAttribute('tabindex', '0');
    deskCard.setAttribute('aria-label', '点击切换工程信条');
    const onDesk = () => {
      deskIndex = (deskIndex + 1) % JOSH_ABOUT_DESK_CREDOS.length;
      const credo = JOSH_ABOUT_DESK_CREDOS[deskIndex];
      joshAboutSwapText(deskText, credo.text);
      if (deskSub) joshAboutSwapText(deskSub, credo.subtitle);
      if (typeof joshPlaySound === 'function') joshPlaySound('click');
    };
    deskCard.addEventListener('click', onDesk);
    deskCard.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onDesk(); }
    });
    if ('IntersectionObserver' in window) {
      const radarObs = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            deskCard.classList.add('is-visible');
            radarObs.unobserve(entry.target);
          }
        });
      }, { threshold: 0.3 });
      radarObs.observe(deskCard);
      cleanups.push(() => radarObs.disconnect());
    } else {
      deskCard.classList.add('is-visible');
    }
    cleanups.push(() => {
      deskCard.removeEventListener('click', onDesk);
    });
  }

  const nameBtn = app.querySelector('#josh-about-name-btn');
  const nameSpeak = app.querySelector('#josh-about-name-speak');
  if (nameBtn && 'speechSynthesis' in window) {
    let nameSpeakTimer = null;
    const stopNameSpeakAnim = () => {
      if (nameSpeakTimer) window.clearTimeout(nameSpeakTimer);
      nameSpeakTimer = null;
      nameSpeak?.classList.remove('is-speaking');
    };
    const startNameSpeakAnim = () => {
      if (!nameSpeak) return;
      nameSpeak.classList.remove('is-speaking');
      void nameSpeak.offsetWidth;
      nameSpeak.classList.add('is-speaking');
      if (nameSpeakTimer) window.clearTimeout(nameSpeakTimer);
      nameSpeakTimer = window.setTimeout(stopNameSpeakAnim, 1200);
    };
    const onName = () => {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance("Do what you can't.");
      utterance.lang = 'en-US';
      utterance.rate = 0.85;
      utterance.onstart = startNameSpeakAnim;
      utterance.onend = stopNameSpeakAnim;
      utterance.onerror = stopNameSpeakAnim;
      window.speechSynthesis.speak(utterance);
      if (typeof joshPlaySound === 'function') joshPlaySound('click');
    };
    nameBtn.addEventListener('click', onName);
    cleanups.push(() => {
      nameBtn.removeEventListener('click', onName);
      stopNameSpeakAnim();
    });
  }

  const arcBtn = app.querySelector('#josh-about-arc-btn');
  const arcValue = app.querySelector('#josh-about-arc-value');
  const arcPrefix = app.querySelector('#josh-about-arc-prefix');
  const arcSuffix = app.querySelector('#josh-about-arc-suffix');
  if (arcBtn && arcValue && arcPrefix && arcSuffix) {
    let arcIndex = 0;
    const onArc = () => {
      arcIndex = (arcIndex + 1) % arcUnits.length;
      const unit = arcUnits[arcIndex];
      arcBtn.style.width = unit.width;
      joshAboutSwapText(arcValue, unit.value);
      arcPrefix.textContent = unit.prefix;
      arcSuffix.textContent = unit.suffix;
      if (typeof joshPlaySound === 'function') joshPlaySound('click');
    };
    arcBtn.addEventListener('click', onArc);
    cleanups.push(() => arcBtn.removeEventListener('click', onArc));
  }

  const machineBtn = app.querySelector('#josh-about-machine-btn');
  const machineEl = app.querySelector('#josh-about-machine');
  if (machineBtn && machineEl) {
    const sideCard = machineEl.closest('.josh-about-card--side');
    const onMachine = () => {
      const next = machineBtn.getAttribute('aria-pressed') !== 'true';
      machineBtn.setAttribute('aria-pressed', String(next));
      joshAboutSetMachinePower(machineEl, next);
      if (sideCard) sideCard.classList.toggle('is-active', next);
      if (!next) {
        joshAboutVizStop();
        const vizEl = sideCard?.querySelector('#josh-about-viz');
        if (vizEl) vizEl.querySelectorAll('.josh-about-viz__scene').forEach(s => s.classList.remove('is-visible'));
        if (sideCard) sideCard.removeAttribute('data-viz-scene');
      }
      if (typeof joshPlaySound === 'function') joshPlaySound('toggle');
    };
    const onMachineDown = () => {
      if (typeof joshPlaySound === 'function') joshPlaySound('click');
    };
    machineBtn.addEventListener('click', onMachine);
    machineBtn.addEventListener('pointerdown', onMachineDown);
    cleanups.push(() => {
      machineBtn.removeEventListener('click', onMachine);
      machineBtn.removeEventListener('pointerdown', onMachineDown);
      joshAboutSetMachinePower(machineEl, false);
      joshAboutStopMachineAnimations(machineEl);
      joshAboutVizStop();
      if (sideCard) {
        sideCard.classList.remove('is-active');
        sideCard.removeAttribute('data-viz-scene');
      }
    });
  }

  const catBtn = app.querySelector('#josh-about-cat-btn');
  const catBodyGroup = app.querySelector('.josh-about-cat-body-group');
  if (catBtn && catBodyGroup) {
    const catPet = joshAboutCreateCatPetController(catBtn, catBodyGroup);
    const onCat = () => catPet.pet();
    catBtn.addEventListener('click', onCat);
    cleanups.push(() => {
      catBtn.removeEventListener('click', onCat);
      catPet.cleanup();
    });
  }

  const triggerDrum = (fader) => {
    if (!fader) return;
    const kind = fader.dataset.drum;
    const hue = getComputedStyle(fader).getPropertyValue('--fader-hue').trim() || '195';

    fader.classList.remove('is-hit');
    void fader.offsetWidth;
    fader.classList.add('is-hit');
    window.setTimeout(() => fader.classList.remove('is-hit'), 400);

    joshAboutSpawnWaves(fader, hue);
    joshAboutSpawnParticles(fader, hue);

    if (typeof joshPlayDrum === 'function') joshPlayDrum(kind);
  };

  app.querySelectorAll('.josh-about-fader').forEach((fader) => {
    const btn = fader.querySelector('.josh-about-fader__btn');
    if (!btn) return;
    const onPad = () => triggerDrum(fader);
    btn.addEventListener('click', onPad);
    cleanups.push(() => btn.removeEventListener('click', onPad));
  });

  const onDrumKeydown = (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target;
    if (target && (target.closest('input, textarea, select, [contenteditable="true"]'))) return;
    const fader = app.querySelector(`.josh-about-fader[data-drum-key="${event.key.toLowerCase()}"]`);
    if (!fader) return;
    event.preventDefault();
    triggerDrum(fader);
  };
  window.addEventListener('keydown', onDrumKeydown);
  cleanups.push(() => window.removeEventListener('keydown', onDrumKeydown));

  joshAboutInitLeafletMap(app).then(() => {
    joshAboutHydrateMapDistance(app.querySelector('#josh-about-map-distance-inline'));
  }).catch(() => {
    joshAboutHydrateMapDistance(app.querySelector('#josh-about-map-distance-inline'));
  });
  cleanups.push(() => joshAboutDestroyLeafletMap());

  const statEl = app.querySelector('#josh-about-stat');
  if (statEl) {
    const target = Number(statEl.dataset.target);
    const runStat = () => joshAboutAnimateStat(statEl, target);
    if ('IntersectionObserver' in window) {
      const statObs = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          runStat();
          statObs.disconnect();
        }
      }, { threshold: 0.35 });
      statObs.observe(statEl);
      cleanups.push(() => statObs.disconnect());
    } else {
      runStat();
    }
  }

  return () => cleanups.forEach((fn) => fn());
}

function joshHydratePostListViewCounts(postList) {
  if (!postList?.length || typeof hydrateListViewCounts !== 'function') return;
  queueMicrotask(() => {
    hydrateListViewCounts(postList, 'view-list-post', 'post', false);
  });
}

function renderJoshCategories(app) {
  updateMetaTags({
    title: '知识分类 - Tangentllm Notes',
    description: '按大模型知识体系组织的文章分类，每个分类都是一个值得深入的方向。',
    url: absolutePageUrl('categories'),
    type: 'website',
  });

  const mainHtml = `
    <header class="josh-categories-page__intro">
      <h1 class="josh-categories-page__title">知识分类</h1>
      <p class="josh-page-desc josh-categories-page__desc">按大模型知识体系组织的文章分类，每个分类都是一个值得深入的方向。</p>
    </header>
    <div class="josh-categories-page__grid">
      ${joshCategoryCardGridMarkup(categories)}
    </div>
  `;

  joshMountBlogArchivePage(app, mainHtml, Routes.categories(), {
    pageClass: ' josh-page--categories',
  });
}

function renderJoshCategoryPosts(app, categoryName) {
  const cat = categories.find((c) => c.name === categoryName);
  const catPosts = posts
    .filter((p) => p.category === categoryName)
    .sort((a, b) => new Date(b.date) - new Date(a.date));

  updateMetaTags({
    title: `${categoryName} - Tangentllm Notes`,
    description: cat?.desc || `${categoryName}相关文章列表`,
    url: absolutePageUrl(`category/${encodeURIComponent(categoryName)}`),
    type: 'website',
  });

  const countLabel = `${catPosts.length} 篇文章`;
  const mainHtml = `
    ${joshBlogArchiveDetailHeaderMarkup(categoryName, countLabel, Routes.categories(), '返回分类')}
    ${joshPostBlogArchiveGridMarkup(catPosts, {
      sparse: false,
      showSubtitle: false,
      showMeta: true,
      showTags: true,
      maxTags: 3,
      emptyText: '该分类下暂无文章，持续更新中',
      emptyBackHref: Routes.categories(),
      emptyBackLabel: '返回分类',
    })}
  `;

  joshMountBlogArchivePage(app, mainHtml, Routes.categories(), {
    pageClass: ' josh-page--category',
  });
}

function renderJoshTags(app) {
  const tagMap = new Map();
  posts.forEach((post) => {
    post.tags.forEach((tag) => {
      if (!tagMap.has(tag)) tagMap.set(tag, []);
      tagMap.get(tag).push(post);
    });
  });

  const sortedTags = Array.from(tagMap.entries()).sort((a, b) => b[1].length - a[1].length);
  const tagCount = sortedTags.length;
  const postCount = posts.length;
  const hottestCount = sortedTags[0]?.[1].length || 0;
  const avgCount = tagCount ? (postCount / tagCount).toFixed(1) : '0';

  updateMetaTags({
    title: '标签云 - Tangentllm Notes',
    description: '通过标签快速找到感兴趣的主题，每个标签代表一个技术领域或知识点。',
    url: absolutePageUrl('tags'),
    type: 'website',
  });

  const mainHtml = `
    <header class="josh-tags-page__intro">
      <h1 class="josh-tags-page__title">标签云</h1>
      <p class="josh-page-desc josh-tags-page__desc">通过标签快速找到感兴趣的主题，每个标签代表一个技术领域或知识点。</p>
    </header>
    ${joshTagsStatsMarkup({ tagCount, postCount, hottestCount, avgCount })}
    ${joshTagsHotSectionMarkup(sortedTags, 6)}
    <h2 class="josh-section-label josh-tags-page__cloud-title" id="josh-tags-cloud-heading">全部标签</h2>
    ${joshTagCloudMarkup(sortedTags)}
  `;

  joshMountBlogArchivePage(app, mainHtml, Routes.tags(), {
    pageClass: ' josh-page--tags',
  });
}

function renderJoshTagPosts(app, tag) {
  const tagPosts = posts
    .filter((post) => post.tags.includes(tag))
    .sort((a, b) => new Date(b.date) - new Date(a.date));

  updateMetaTags({
    title: `#${tag} - Tangentllm Notes`,
    description: `标签「${tag}」下的文章列表`,
    url: absolutePageUrl(`tag/${encodeURIComponent(tag)}`),
    type: 'website',
  });

  const mainHtml = `
    ${joshBlogArchiveDetailHeaderMarkup(`#${tag}`, `${tagPosts.length} 篇文章`, Routes.tags(), '返回标签')}
    ${joshPostBlogArchiveGridMarkup(tagPosts, {
      sparse: false,
      showSubtitle: false,
      showMeta: true,
      showCategory: true,
      showTags: true,
      omitTags: [tag],
      maxTags: 2,
      emptyText: '该标签下暂无文章',
      emptyBackHref: Routes.tags(),
      emptyBackLabel: '返回标签',
    })}
  `;

  joshMountBlogArchivePage(app, mainHtml, Routes.tags(), {
    pageClass: ' josh-page--tag',
  });
}

function renderJoshProjects(app, queryParams) {
  const hashQuery = queryParams instanceof URLSearchParams
    ? queryParams
    : new URLSearchParams(location.search || '');
  const statusFilter = hashQuery.get('status') || '全部';
  const tagFilter = hashQuery.get('tag') || '全部';

  const statusOptions = ['全部', ...new Set(projectsData.map((project) => project.status))];
  const tagOptions = ['全部', ...new Set(projectsData.flatMap((project) => project.tags || []))];

  const filteredProjects = projectsData.filter((project) => {
    const statusMatched = statusFilter === '全部' || project.status === statusFilter;
    const tagMatched = tagFilter === '全部' || (project.tags || []).includes(tagFilter);
    return statusMatched && tagMatched;
  });

  updateMetaTags({
    title: '作品集 - Tangentllm Notes',
    description: '大模型实战项目作品集，含架构设计、关键能力与效果展示。',
    url: absolutePageUrl('projects'),
    type: 'website',
  });

  const showFilters = projectsData.length > 5;
  const filterHtml = showFilters && projectsData.length > 0 ? `
    ${statusOptions.length > 2 ? joshProjectStatusFilterMarkup(statusOptions, statusFilter, tagFilter) : ''}
    ${tagOptions.length > 2 ? joshProjectTagFilterMarkup(tagOptions, statusFilter, tagFilter) : ''}
  ` : '';

  const listHtml = projectsData.length === 0
    ? `<div class="josh-empty-state">
        <p class="josh-empty-state__text">作品集正在筹备中，敬请期待</p>
      </div>`
    : `${filterHtml}
      ${joshBlogArchiveGridMarkup(filteredProjects, {
        sparse: false,
        emptyText: '没有匹配项目，试试调整筛选条件。',
        emptyBackHref: Routes.projects(),
        emptyBackLabel: '清除筛选',
        emptyBackClass: 'josh-empty-state__action',
        emptyBackShowArrow: false,
      })}`;

  const mainHtml = projectsData.length === 0
    ? listHtml
    : `${joshBlogArchiveHeaderMarkup('作品集', `${projectsData.length} 个项目`)}
      ${listHtml}`;

  joshMountBlogArchivePage(app, mainHtml, Routes.projects(), {
    pageClass: ' josh-page--projects',
  });

  queueMicrotask(() => bindJoshProjectFilters(app));
}

function renderJoshProjectDetail(app, slug) {
  const project = projectsData.find((item) => item.slug === slug);
  if (!project) {
    app.innerHTML = buildJoshPageShell('<p class="josh-post-meta">作品不存在</p>', Routes.projects());
    queueMicrotask(() => initJoshSiteInteractions(app));
    return;
  }

  const baseUrl = absolutePageUrl();
  updateMetaTags({
    title: `${project.title} - Tangentllm Notes`,
    description: project.summary || project.subtitle,
    keywords: project.tags.join(','),
    url: absolutePageUrl(`project/${project.slug}`),
    image: coverUrlForOpenGraph(project.cover, baseUrl),
    type: 'article',
    article: {
      headline: project.title,
      datePublished: project.period || '',
      author: 'tangentllm',
    },
  });

  const proseHtml = joshProjectProseMarkup(project);
  const { headings, contentWithIds } = processPostProseHeadings(proseHtml);
  const tocHtml = joshPostTocMarkup(headings, { heartSlug: slug });
  const actionsHtml = joshProjectLinkButtonsMarkup(project);
  const tailStatsHtml = joshPostTailStatsMarkup(slug, {
    updated: joshProjectUpdatedLabel(project),
  });
  const showCover = joshShouldShowProjectCover(project);
  const coverSrc = joshProjectCoverSrc(project);
  const projectIndex = projectsData.findIndex((item) => item.slug === slug);
  const prevProject = projectIndex > 0 ? projectsData[projectIndex - 1] : null;
  const nextProject = projectIndex >= 0 && projectIndex < projectsData.length - 1
    ? projectsData[projectIndex + 1]
    : null;

  const mainHtml = `
    <article class="josh-post-article">
      ${buildJoshInnerHeaderMarkup(Routes.projects())}
      <div class="josh-post-hero">
        <div class="josh-post-hero__band">
          <div class="josh-post-hero__sky-blocker" aria-hidden="true"></div>
          <div class="josh-post-hero__inner">
            <header class="josh-post-header">
              ${joshPostHeroTitleMarkup(project.title, project.subtitle, { includeSubtitle: false })}
              ${joshProjectMetaMarkup(project)}
            </header>
          </div>
          ${typeof joshPostHeroWaveMarkup === 'function' ? joshPostHeroWaveMarkup() : ''}
        </div>
      </div>
      <div class="josh-post-white">
        <div class="josh-post-layout__blocker" id="josh-post-blocker" data-is-stuck="false" aria-hidden="true"></div>
        <div class="josh-post-layout">
        ${tocHtml}
        ${showCover ? `
        <div class="josh-post-cover josh-post-cover--project">
          <img src="${coverSrc}" alt="" loading="lazy" data-no-inline-fallback="1">
        </div>` : ''}
        <div class="josh-post-body">
          ${actionsHtml}
          <div class="josh-prose">${contentWithIds}</div>
        </div>
        <div class="josh-post-tail">
            ${tailStatsHtml}
            <div class="josh-post-tags" role="list" aria-label="项目标签">
              ${project.tags.map((tag) => joshPostTagMarkup(tag)).join('')}
            </div>
            <hr class="josh-post-divider">
            ${joshPostNavMarkup(
              prevProject ? { href: Routes.project(prevProject.slug), title: prevProject.title } : null,
              nextProject ? { href: Routes.project(nextProject.slug), title: nextProject.title } : null,
              { prevLabel: '上一个作品', nextLabel: '下一个作品', ariaLabel: '作品导航' },
            )}
          </div>
        </div>
      </div>
    </article>
  `;

  app.innerHTML = buildJoshPageShell(mainHtml, Routes.projects(), {
    pageClass: ' josh-page--post josh-page--project-detail',
    mainClass: 'josh-inner-main josh-inner-main--post',
    omitHeader: true,
  });

  queueMicrotask(() => {
    initJoshSiteInteractions(app);
    if (typeof initPostToc === 'function') initPostToc();
    initJoshHeadingAnchors(app);
    if (typeof initJoshPostHeart === 'function') {
      initJoshPostHeart(app, project.slug);
    }
    if (typeof initJoshHitCounters === 'function') initJoshHitCounters(app);
    if (typeof displayViewCountMulti === 'function') {
      displayViewCountMulti(project.slug, 'project', [
        `view-count-tail-${project.slug}`,
      ]);
    } else if (typeof displayViewCount === 'function') {
      displayViewCount(project.slug, `view-count-tail-${project.slug}`, 'project');
    }
    if (typeof bindJoshReadingProgress === 'function') bindJoshReadingProgress(slug);
  });

  setTimeout(() => {
    if (typeof updateViewCount === 'function') updateViewCount(project.slug, 'project');
  }, 500);
}

function renderJoshAbout(app) {
  updateMetaTags({
    title: '关于我 - Tangentllm Notes',
    description: '深圳客户端出身，转向 LLM 应用落地。博客记录上线前后的判断与踩坑复盘。',
    url: absolutePageUrl('about'),
    type: 'website',
  });

  const heroHtml = joshAboutHeroMarkup({
    title: '你好，我是 Tangentllm。',
    paragraphs: [
      '人在深圳，做了十年客户端。2024 年开始把主要精力放在大模型应用上——RAG、Agent、推理链路，能进生产的那种。',
      '平时通过个人工作室接一些落地项目；博客是我主要的公开输出，side project 也会写在这里。',
      '我不写「十分钟上手」类教程。更想留下的是：当时怎么判断、上线后哪里出了问题、以及为什么最后改成现在这样。',
    ],
    activeHref: Routes.about(),
  });

  const bodyHtml = joshAboutGridMarkup();

  joshMountAboutPage(app, heroHtml, bodyHtml, Routes.about());
}

function renderJosh404(app) {
  updateMetaTags({
    title: '页面未找到 - Tangentllm Notes',
    description: '抱歉，您访问的页面不存在。',
    url: absolutePageUrl(),
    type: 'website',
  });

  const suggestPosts = posts.slice(0, 3);
  const mainHtml = `
    <div class="josh-not-found">
      <p class="josh-not-found__code" aria-hidden="true">404</p>
      <h1 class="josh-page-title">页面未找到</h1>
      <p class="josh-page-desc">抱歉，您访问的页面不存在。可能是链接已失效，或者页面已被移除。</p>
      <div class="josh-not-found__actions">
        <a class="josh-btn" href="${Routes.home()}">返回首页</a>
        <button type="button" class="josh-btn" onclick="openSearch()">搜索内容</button>
      </div>
      <div class="josh-not-found__suggest">
        <h2 class="josh-not-found__suggest-title">您可能感兴趣的内容</h2>
        <div class="josh-not-found__links">
          ${suggestPosts.map((post) => `
            <a class="josh-not-found__link" href="${Routes.post(post.slug)}">
              <p class="josh-not-found__link-title">${post.title}</p>
              <p class="josh-not-found__link-meta">${post.category} · ${post.readTime} · <span id="view-404-post-${post.slug}">…</span></p>
            </a>
          `).join('')}
        </div>
      </div>
    </div>
  `;

  joshMountListPage(app, mainHtml, Routes.home());

  if (suggestPosts.length > 0 && typeof hydrateListViewCounts === 'function') {
    queueMicrotask(() => hydrateListViewCounts(suggestPosts, 'view-404-post', 'post', false));
  }
}

let joshSiteCleanup = null;

function initJoshSiteInteractions(app) {
  if (typeof joshSiteCleanup === 'function') {
    joshSiteCleanup();
    joshSiteCleanup = null;
  }

  const mobileTogglePairs = [
    ['#josh-mobile-toggle', '#josh-mobile-menu'],
    ['#josh-about-body-mobile-toggle', '#josh-about-body-mobile-menu'],
  ].map(([toggleSel, menuSel]) => {
    const toggle = app.querySelector(toggleSel);
    const menu = app.querySelector(menuSel);
    return toggle && menu ? { toggle, menu } : null;
  }).filter(Boolean);
  const innerHeader = app.querySelector('#josh-inner-header');
  const aboutBodySticky = app.querySelector('.josh-about-body__sticky');
  const aboutBodyHeader = app.querySelector('#josh-about-body-header');
  const postBlocker = app.querySelector('#josh-post-blocker');
  const isPostPage = Boolean(app.querySelector('.josh-page--post'));
  const skyListPage = app.querySelector('.josh-page--sky-list');
  const aboutPage = app.querySelector('.josh-page--about');
  const pageHero = app.querySelector('.josh-page-hero');
  const aboutSky = app.querySelector('#josh-about-sky');
  const navScrollHero = pageHero || aboutSky;
  let navScrollRaf = null;

  const syncNavScrollState = () => {
    if (!innerHeader) return;

    if (aboutPage && aboutBodySticky && aboutBodyHeader && navScrollHero && !isPostPage) {
      let { progress, over } = joshAboutBodyNavScrollMetrics(aboutBodySticky);

      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        progress = over ? 1 : 0;
      }

      aboutBodySticky.style.opacity = String(progress);
      aboutBodySticky.style.pointerEvents = progress > 0 ? 'auto' : 'none';
      aboutBodySticky.setAttribute('aria-hidden', progress > 0 ? 'false' : 'true');
      aboutBodyHeader.setAttribute('data-is-over-threshold', String(over));

      if (innerHeader) {
        innerHeader.style.removeProperty('--josh-nav-scroll-progress');
        innerHeader.style.removeProperty('--josh-post-sky-progress');
        innerHeader.classList.remove('is-scrolled');
        innerHeader.removeAttribute('data-is-over-threshold');
      }
      return;
    }

    if (skyListPage && navScrollHero && !isPostPage) {
      let { whiteProgress, skyProgress, over } = joshSkyListNavScrollMetrics(innerHeader, navScrollHero);

      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        whiteProgress = over ? 1 : 0;
        skyProgress = over ? 0 : skyProgress;
      }

      innerHeader.style.setProperty('--josh-nav-scroll-progress', String(whiteProgress));
      innerHeader.style.setProperty('--josh-post-sky-progress', String(skyProgress));
      innerHeader.classList.toggle('is-scrolled', over);
      innerHeader.setAttribute('data-is-over-threshold', String(over));
      return;
    }

    if (isPostPage && postBlocker) {
      const skyBlocker = app.querySelector('.josh-post-hero__sky-blocker');
      let { whiteProgress, skyProgress, over } = joshPostNavScrollMetrics(
        innerHeader,
        app.querySelector('.josh-post-hero'),
        postBlocker,
      );

      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        over = window.scrollY > 8;
        whiteProgress = over ? 1 : 0;
        skyProgress = over ? 0 : skyProgress;
      }

      innerHeader.style.setProperty('--josh-nav-scroll-progress', String(whiteProgress));
      innerHeader.style.setProperty('--josh-post-sky-progress', String(skyProgress));
      innerHeader.classList.toggle('is-scrolled', over);
      innerHeader.setAttribute('data-is-over-threshold', String(over));
      postBlocker.setAttribute('data-is-stuck', String(over));
      if (skyBlocker) skyBlocker.setAttribute('data-is-stuck', String(over));
      return;
    }

    innerHeader.classList.toggle('is-scrolled', window.scrollY > 8);
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

  initJoshFooterMascotAnimation(app);

  mobileTogglePairs.forEach(({ toggle, menu }) => {
    toggle.addEventListener('click', () => {
      const open = menu.hasAttribute('hidden');
      if (open) {
        menu.removeAttribute('hidden');
        toggle.setAttribute('aria-expanded', 'true');
      } else {
        menu.setAttribute('hidden', '');
        toggle.setAttribute('aria-expanded', 'false');
      }
    });
  });

  if (typeof initJoshInteractions === 'function') {
    initJoshInteractions(app);
  }

  let cleanupAbout = null;
  if (app.querySelector('.tangent-about-page')) {
    if (typeof initTangentAboutInteractions === 'function') {
      cleanupAbout = initTangentAboutInteractions(app);
    }
  } else if (app.querySelector('.josh-page--about')) {
    cleanupAbout = initJoshAboutInteractions(app);
  }

  joshSiteCleanup = () => {
    if (navScrollRaf !== null) {
      cancelAnimationFrame(navScrollRaf);
      navScrollRaf = null;
    }
    window.removeEventListener('scroll', scheduleNavScrollState);
    window.removeEventListener('resize', syncNavScrollState);
    if (typeof joshFooterMascotCleanup === 'function') {
      joshFooterMascotCleanup();
      joshFooterMascotCleanup = null;
    }
    if (typeof cleanupAbout === 'function') cleanupAbout();
    if (typeof joshInteractionsCleanup === 'function') joshInteractionsCleanup();
  };
}

function joshActiveNavHref(route) {
  if (!route) return Routes.home();
  if (route.type === 'home' || route.type === 'post') return Routes.home();
  if (route.type === 'categories' || route.type === 'category') return Routes.categories();
  if (route.type === 'tags' || route.type === 'tag') return Routes.tags();
  if (route.type === 'projects' || route.type === 'project') return Routes.projects();
  if (route.type === 'about') return Routes.about();
  return Routes.home();
}

function renderJoshPost(app, slug) {
  const post = posts.find((p) => p.slug === slug);
  if (!post) {
    app.innerHTML = buildJoshPageShell(
      '<p class="josh-post-meta">文章不存在</p>',
      Routes.home(),
    );
    queueMicrotask(() => initJoshSiteInteractions(app));
    return;
  }

  const baseUrl = absolutePageUrl();
  updateMetaTags({
    title: `${post.title} - Tangentllm Notes`,
    description: post.excerpt || post.title,
    keywords: post.tags.join(','),
    url: absolutePageUrl(`post/${post.slug}`),
    image: coverUrlForOpenGraph(post.cover, baseUrl),
    type: 'article',
    article: {
      headline: post.title,
      datePublished: post.date,
      author: 'tangentllm',
    },
  });

  const seriesSiblings = joshSeriesSiblingPosts(post);
  const prevPost = seriesSiblings.prev || posts[posts.indexOf(post) - 1] || null;
  const nextPost = seriesSiblings.next || posts[posts.indexOf(post) + 1] || null;

  const proseHtml = post.content
    ? stripDuplicatePostH1(post.content, post.title)
    : (post.format === 'html'
      ? '<p class="josh-post-meta">HTML 正文加载失败，请用本地静态服务打开（勿用 file:// 直接打开）。</p>'
      : '');
  const { headings, contentWithIds } = processPostProseHeadings(proseHtml);

  const showCover = joshShouldShowPostCover(post);

  const tocHtml = joshPostTocMarkup(headings, { heartSlug: slug });

  const seriesHtml = post.series ? `
    <div class="josh-series-banner">
      <strong>系列文章：${post.series}</strong>
      ${post.seriesOrder ? ` · 第 ${post.seriesOrder} 篇` : ''}
    </div>
  ` : '';

  const heroSubtitle = joshArticleSubtitle(post);
  const tailStatsHtml = joshPostTailStatsMarkup(slug, {
    updated: joshPostUpdatedLabel(post),
  });

  const mainHtml = `
    <article class="josh-post-article">
      ${buildJoshInnerHeaderMarkup(Routes.home(), { postNavLayers: true })}
      <div class="josh-post-hero">
        <div class="josh-post-hero__band">
          <div class="josh-post-hero__sky-blocker" aria-hidden="true"></div>
          <div class="josh-post-hero__inner">
            <header class="josh-post-header">
              ${seriesHtml}
              ${joshPostHeroTitleMarkup(post.title, heroSubtitle, { includeSubtitle: false })}
              <div class="josh-post-meta" role="contentinfo" aria-label="文章元信息">
                <span>收录于</span>
                <a class="josh-post-meta__link" href="${Routes.category(post.category)}">${post.category}</a>
                <span class="josh-post-meta__sep" aria-hidden="true">·</span>
                <span>发布于</span>
                <time datetime="${post.date}">${formatDate(post.date)}</time>
                <span class="josh-post-meta__sep" aria-hidden="true">·</span>
                <span>${post.readTime}</span>
              </div>
            </header>
          </div>
          ${typeof joshPostHeroWaveMarkup === 'function' ? joshPostHeroWaveMarkup() : ''}
        </div>
      </div>
      <div class="josh-post-white">
        <div class="josh-post-layout__blocker" id="josh-post-blocker" data-is-stuck="false" aria-hidden="true"></div>
        <div class="josh-post-layout">
        ${tocHtml}
        ${showCover ? `
        <div class="josh-post-cover">
          <img src="${post.cover}" alt="" loading="lazy" data-no-inline-fallback="1">
        </div>` : ''}
        <div class="josh-post-body">
          <div class="josh-prose">${contentWithIds}</div>
        </div>
        <div class="josh-post-tail">
            ${tailStatsHtml}
            <hr class="josh-post-divider">
            ${joshPostNavMarkup(
              prevPost ? { href: Routes.post(prevPost.slug), title: prevPost.title } : null,
              nextPost ? { href: Routes.post(nextPost.slug), title: nextPost.title } : null,
            )}
        </div>
        </div>
      </div>
    </article>
  `;

  app.innerHTML = buildJoshPageShell(mainHtml, Routes.home(), {
    pageClass: ' josh-page--post',
    mainClass: 'josh-inner-main josh-inner-main--post',
    omitHeader: true,
  });

  queueMicrotask(() => {
    initJoshSiteInteractions(app);
    if (typeof initPostToc === 'function') initPostToc();
    if (typeof initJoshPostHeart === 'function') {
      initJoshPostHeart(app, post.slug);
    }
    if (typeof initJoshHitCounters === 'function') initJoshHitCounters(app);
    if (typeof displayViewCountMulti === 'function') {
      displayViewCountMulti(post.slug, 'post', [`view-count-tail-${post.slug}`]);
    }
    if (typeof bindJoshReadingProgress === 'function') bindJoshReadingProgress(post.slug);
  });

  setTimeout(() => {
    if (typeof updateViewCount === 'function') updateViewCount(post.slug);
  }, 500);
}

const JOSH_HEADING_ANCHOR_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>`;

function initJoshHeadingAnchors(scope) {
  const root = scope || document;
  const prose = root.querySelector('.josh-prose');
  if (!prose) return;

  const headerOffset = 112;

  prose.querySelectorAll('h2[id], h3[id]').forEach((heading) => {
    if (heading.querySelector('.josh-heading-anchor')) return;

    heading.classList.add('josh-content-heading');

    const anchor = document.createElement('a');
    anchor.href = `#${heading.id}`;
    anchor.className = 'josh-heading-anchor';
    anchor.setAttribute('aria-label', `链接到：${heading.textContent.trim()}`);
    anchor.innerHTML = `<span class="josh-heading-anchor__inner">${JOSH_HEADING_ANCHOR_ICON}</span>`;

    anchor.addEventListener('click', (e) => {
      e.preventDefault();
      const y = heading.getBoundingClientRect().top + window.pageYOffset - headerOffset;
      window.scrollTo({ top: y, behavior: 'smooth' });
      history.replaceState(null, '', `#${heading.id}`);
    });

    heading.insertBefore(anchor, heading.firstChild);
  });
}

/* === Markdown playground fences (```playground) === */
function joshEncodePlaygroundAttr(config) {
  return btoa(unescape(encodeURIComponent(JSON.stringify(config))));
}

function joshNormalizePlaygroundConfig(raw) {
  const files = {};
  Object.entries(raw.files || {}).forEach(([path, content]) => {
    const key = path.startsWith('/') ? path : `/${path}`;
    files[key] = String(content ?? '');
  });

  return {
    template: raw.template || 'vanilla',
    files,
    options: {
      editorHeight: 420,
      showTabs: true,
      showNavigator: false,
      closableTabs: false,
      editorWidthPercentage: 52,
      ...(raw.options || {}),
    },
  };
}

function joshParsePlaygroundFence(text) {
  const trimmed = String(text || '').trim();
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return joshNormalizePlaygroundConfig(parsed);
    }
  } catch (_) {
    /* shorthand — treat fence body as index.html */
  }

  return joshNormalizePlaygroundConfig({
    template: 'vanilla',
    files: {
      '/index.html': trimmed,
      '/styles.css': '',
      '/index.js': '',
    },
  });
}

function installJoshMarkedPlaygroundRenderer() {
  if (typeof marked === 'undefined' || installJoshMarkedPlaygroundRenderer.done) return;
  installJoshMarkedPlaygroundRenderer.done = true;

  marked.use({
    renderer: {
      code({ text, lang }) {
        if (String(lang || '').trim().toLowerCase() !== 'playground') return false;
        const config = joshParsePlaygroundFence(text);
        const encoded = joshEncodePlaygroundAttr(config);
        return `<div class="josh-playground" data-josh-playground data-config="${encoded}"><div class="josh-playground__shell"><p class="josh-playground__loading">Loading playground…</p></div></div>\n`;
      },
    },
  });
}

installJoshMarkedPlaygroundRenderer();
initJoshSearchWave();
