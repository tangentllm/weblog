/* Josh-style micro-interactions — nav flyouts, logo draw, sounds, scroll reveal */

const JOSH_SOUND_STORAGE_KEY = 'josh-sounds-enabled';

let joshSoundsEnabled = localStorage.getItem(JOSH_SOUND_STORAGE_KEY) !== 'false';
let joshAudioCtx = null;
let joshInteractionsCleanup = null;

function joshGetAudioContext() {
  if (!joshAudioCtx) {
    joshAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (joshAudioCtx.state === 'suspended') {
    joshAudioCtx.resume();
  }
  return joshAudioCtx;
}

function joshPlaySound(kind = 'click') {
  if (!joshSoundsEnabled) return;
  try {
    const ctx = joshGetAudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const now = ctx.currentTime;
    const presets = {
      click: { freq: 720, dur: 0.07, vol: 0.06 },
      open: { freq: 520, dur: 0.1, vol: 0.05 },
      toggle: { freq: 880, dur: 0.09, vol: 0.05 },
      'cat-press': { freq: 640, dur: 0.08, vol: 0.07 },
      'cat-on': { freq: 420, dur: 0.14, vol: 0.08 },
    };
    const preset = presets[kind] || presets.click;
    osc.type = 'sine';
    osc.frequency.setValueAtTime(preset.freq, now);
    gain.gain.setValueAtTime(preset.vol, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + preset.dur);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + preset.dur);
  } catch {
    /* audio unavailable */
  }
}

function joshPlayDrum(kind = 'kick') {
  if (!joshSoundsEnabled) return;
  try {
    const ctx = joshGetAudioContext();
    const now = ctx.currentTime;
    const master = ctx.createGain();
    master.gain.setValueAtTime(0.22, now);
    master.gain.exponentialRampToValueAtTime(0.0001, now + 0.35);
    master.connect(ctx.destination);

    if (kind === 'kick') {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(150, now);
      osc.frequency.exponentialRampToValueAtTime(42, now + 0.12);
      gain.gain.setValueAtTime(1, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);
      osc.connect(gain);
      gain.connect(master);
      osc.start(now);
      osc.stop(now + 0.2);
      return;
    }

    if (kind === 'hihat') {
      const bufferSize = ctx.sampleRate * 0.06;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i += 1) {
        data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
      }
      const noise = ctx.createBufferSource();
      noise.buffer = buffer;
      const filter = ctx.createBiquadFilter();
      filter.type = 'highpass';
      filter.frequency.setValueAtTime(7000, now);
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.35, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.05);
      noise.connect(filter);
      filter.connect(gain);
      gain.connect(master);
      noise.start(now);
      noise.stop(now + 0.06);
      return;
    }

    if (kind === 'snare') {
      const osc = ctx.createOscillator();
      const oscGain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(220, now);
      oscGain.gain.setValueAtTime(0.25, now);
      oscGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.08);
      osc.connect(oscGain);
      oscGain.connect(master);
      osc.start(now);
      osc.stop(now + 0.1);

      const bufferSize = ctx.sampleRate * 0.12;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i += 1) {
        data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
      }
      const noise = ctx.createBufferSource();
      noise.buffer = buffer;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.45, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);
      noise.connect(gain);
      gain.connect(master);
      noise.start(now);
      noise.stop(now + 0.14);
      return;
    }

    if (kind === 'cowbell') {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.setValueAtTime(560, now);
      gain.gain.setValueAtTime(0.12, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.09);
      osc.connect(gain);
      gain.connect(master);
      osc.start(now);
      osc.stop(now + 0.1);
    }
  } catch {
    /* audio unavailable */
  }
}

function joshSyncSoundToggleUi(scope) {
  scope.querySelectorAll('.josh-sound-toggle').forEach((btn) => {
    const on = joshSoundsEnabled;
    btn.setAttribute('aria-pressed', String(on));
    btn.setAttribute('aria-label', on ? '禁用音效' : '启用音效');
    btn.classList.toggle('is-muted', !on);
  });
}

function initJoshSoundToggle(scope) {
  joshSyncSoundToggleUi(scope);
  scope.querySelectorAll('.josh-sound-toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      joshSoundsEnabled = !joshSoundsEnabled;
      localStorage.setItem(JOSH_SOUND_STORAGE_KEY, String(joshSoundsEnabled));
      joshSyncSoundToggleUi(scope);
      if (joshSoundsEnabled) joshPlaySound('toggle');
    });
  });
}

function initJoshLogoDraw(scope) {
  scope.querySelectorAll('.josh-header .josh-logo').forEach((logo) => {
    const paths = [...logo.querySelectorAll('.josh-logo__w-wrap path')];
    if (!paths.length) return;

    paths.forEach((path) => {
      const len = path.getTotalLength();
      path.style.setProperty('--josh-path-len', String(len));
      path.style.strokeDasharray = String(len);
      path.style.strokeDashoffset = String(len);
    });

    const draw = () => {
      logo.classList.add('is-drawing');
      paths.forEach((path, i) => {
        path.style.transition = `stroke-dashoffset 0.55s cubic-bezier(0.4, 0, 0.2, 1) ${i * 0.08}s`;
        path.style.strokeDashoffset = '0';
      });
      window.setTimeout(() => {
        logo.classList.remove('is-drawing');
        logo.classList.add('is-drawn');
      }, 550 + paths.length * 80);
    };

    const reset = () => {
      if (logo.classList.contains('is-drawing')) return;
      logo.classList.remove('is-drawn');
      paths.forEach((path) => {
        path.style.transition = 'stroke-dashoffset 0.35s ease';
        path.style.strokeDashoffset = path.style.getPropertyValue('--josh-path-len') || path.getTotalLength();
      });
    };

    logo.addEventListener('mouseenter', draw);
    logo.addEventListener('mouseleave', reset);
    if (!logo.dataset.joshLogoDrawn) {
      logo.dataset.joshLogoDrawn = '1';
      requestAnimationFrame(() => requestAnimationFrame(draw));
    }
  });
}

function initJoshNavFlyouts(scope) {
  const layer = scope.querySelector('#josh-nav-flyout-layer');
  const panel = scope.querySelector('#josh-nav-flyout-panel');
  if (!layer || !panel) return () => {};

  const triggers = [...scope.querySelectorAll('[data-flyout-trigger]')];
  const panes = [...panel.querySelectorAll('[data-flyout-pane]')];
  let activeSlug = null;
  let closeTimer = null;
  let positionedTrigger = null;

  const positionPanel = (trigger) => {
    const rect = trigger.getBoundingClientRect();
    const panelWidth = panel.offsetWidth || 300;
    let left = rect.left + rect.width / 2 - panelWidth / 2;
    left = Math.max(12, Math.min(left, window.innerWidth - panelWidth - 12));
    panel.style.setProperty('--josh-flyout-top', `${rect.bottom + 8}px`);
    panel.style.setProperty('--josh-flyout-left', `${left}px`);
    panel.style.setProperty('--josh-flyout-tip-x', `${rect.left + rect.width / 2 - left}px`);
  };

  const setActivePane = (slug) => {
    panes.forEach((pane) => {
      const active = pane.getAttribute('data-flyout-pane') === slug;
      pane.hidden = !active;
      pane.classList.toggle('is-active', active);
    });
  };

  const openFlyout = (trigger, slug) => {
    window.clearTimeout(closeTimer);
    activeSlug = slug;
    positionedTrigger = trigger;
    setActivePane(slug);
    positionPanel(trigger);
    layer.hidden = false;
    layer.setAttribute('aria-hidden', 'false');
    panel.classList.add('is-open');
    triggers.forEach((t) => {
      const expanded = t === trigger;
      t.setAttribute('aria-expanded', String(expanded));
      t.classList.toggle('is-flyout-open', expanded);
    });
    joshPlaySound('open');
  };

  const closeFlyout = () => {
    activeSlug = null;
    positionedTrigger = null;
    layer.hidden = true;
    layer.setAttribute('aria-hidden', 'true');
    panel.classList.remove('is-open');
    triggers.forEach((t) => {
      t.setAttribute('aria-expanded', 'false');
      t.classList.remove('is-flyout-open');
    });
  };

  const scheduleClose = () => {
    window.clearTimeout(closeTimer);
    closeTimer = window.setTimeout(closeFlyout, 180);
  };

  const onTriggerEnter = (trigger) => {
    const slug = trigger.getAttribute('data-flyout-trigger');
    if (!slug) return;
    if (window.matchMedia('(hover: hover)').matches) {
      openFlyout(trigger, slug);
    }
  };

  triggers.forEach((trigger) => {
    trigger.addEventListener('mouseenter', () => onTriggerEnter(trigger));
    trigger.addEventListener('focus', () => onTriggerEnter(trigger));
    trigger.addEventListener('click', (e) => {
      e.preventDefault();
      const slug = trigger.getAttribute('data-flyout-trigger');
      if (!slug) return;
      if (activeSlug === slug && panel.classList.contains('is-open')) {
        closeFlyout();
      } else {
        openFlyout(trigger, slug);
      }
    });
    trigger.addEventListener('mouseleave', scheduleClose);
  });

  panel.addEventListener('mouseenter', () => window.clearTimeout(closeTimer));
  panel.addEventListener('mouseleave', scheduleClose);

  const onDocClick = (e) => {
    if (!panel.classList.contains('is-open')) return;
    if (panel.contains(e.target) || triggers.some((t) => t.contains(e.target))) return;
    closeFlyout();
  };

  const onKeyDown = (e) => {
    if (e.key === 'Escape') closeFlyout();
  };

  const onResize = () => {
    if (positionedTrigger) positionPanel(positionedTrigger);
  };

  document.addEventListener('click', onDocClick);
  document.addEventListener('keydown', onKeyDown);
  window.addEventListener('resize', onResize);

  panel.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => joshPlaySound('click'));
  });

  return () => {
    window.clearTimeout(closeTimer);
    document.removeEventListener('click', onDocClick);
    document.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('resize', onResize);
    closeFlyout();
  };
}

function initJoshReadMoreArrows(scope) {
  scope.querySelectorAll('.josh-read-more, .josh-blog-card__read-more').forEach((link) => {
    link.addEventListener('click', () => joshPlaySound('click'));
  });
}

const JOSH_ARTICLE_REVEAL_OPTIONS = { threshold: 0.05, rootMargin: '0px 0px 0px 0px' };

function joshArticleIsInViewport(el) {
  const rect = el.getBoundingClientRect();
  return rect.top < window.innerHeight && rect.bottom > 0;
}

function initJoshArticleScrollReveal(scope) {
  const page = scope.querySelector('.josh-page') || scope;
  const isHomePage = document.documentElement.classList.contains('josh-home-page');
  const isProjectsPage = Boolean(scope.querySelector('.josh-page--projects'));
  const isCategoryPostsPage = Boolean(scope.querySelector('.josh-page--category'));
  const isTagPostsPage = Boolean(scope.querySelector('.josh-page--tag'));
  const isAboutPage = Boolean(scope.querySelector('.josh-page--about'));
  const articleSelector = isAboutPage
    ? '.josh-article:not(.is-hidden), .josh-project-feature:not(.is-hidden), .josh-blog-card:not(.is-hidden), .josh-about-section, .tangent-about-bento__card'
    : '.josh-article:not(.is-hidden), .josh-project-feature:not(.is-hidden), .josh-blog-card:not(.is-hidden), .josh-category-card, .josh-tag-card, .josh-about-section, .josh-about-card, .tangent-about-bento__card';
  const allArticles = scope.querySelectorAll(articleSelector);
  if (!allArticles.length) return () => {};

  if (isHomePage) {
    scope.querySelectorAll('.josh-article:not(.is-hidden)').forEach((el) => {
      el.classList.add('is-revealed');
    });
  }

  if (isProjectsPage || isCategoryPostsPage || isTagPostsPage) {
    scope.querySelectorAll('.josh-blog-card:not(.is-hidden)').forEach((el) => {
      el.classList.add('is-revealed');
    });
  }

  const articles = [...allArticles].filter((el) => {
    if (isHomePage && el.classList.contains('josh-article')) return false;
    if (isProjectsPage && el.classList.contains('josh-blog-card')) return false;
    if (isCategoryPostsPage && el.classList.contains('josh-blog-card')) return false;
    if (isTagPostsPage && el.classList.contains('josh-blog-card')) return false;
    return true;
  });
  if (!articles.length) return () => {};

  page.classList.add('josh-reveal-ready');

  if (!('IntersectionObserver' in window)) {
    articles.forEach((el) => el.classList.add('is-revealed'));
    return () => page.classList.remove('josh-reveal-ready');
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-revealed');
      observer.unobserve(entry.target);
    });
  }, JOSH_ARTICLE_REVEAL_OPTIONS);

  articles.forEach((el, index) => {
    if (
      document.documentElement.classList.contains('josh-home-page')
      || scope.querySelector('.josh-page--blog-archive')
    ) {
      el.style.setProperty('--josh-article-reveal-delay', `${Math.min(index * 0.04, 0.24)}s`);
    } else if (
      el.classList.contains('josh-about-section')
      || el.classList.contains('josh-about-card')
      || el.classList.contains('tangent-about-bento__card')
    ) {
      el.style.setProperty('--josh-article-reveal-delay', `${Math.min(index * 0.06, 0.24)}s`);
    }
    if (joshArticleIsInViewport(el)) {
      el.classList.add('is-revealed');
      return;
    }
    observer.observe(el);
  });

  return () => {
    observer.disconnect();
    page.classList.remove('josh-reveal-ready');
  };
}

function initJoshCategoryPillHover(scope) {
  const cleanups = [];
  scope.querySelectorAll('.josh-pill').forEach((pill) => {
    const bg = pill.querySelector('.josh-pill__bg');
    if (!bg) return;

    const onEnter = () => {
      const { width, height } = pill.getBoundingClientRect();
      if (!width || !height) return;
      const scaleX = (width + 3) / width;
      const scaleY = (height + 3) / height;
      bg.style.transform = `scale(${scaleX}, ${scaleY})`;
      bg.style.opacity = '1';
      pill.style.zIndex = '1';
    };

    const onLeave = () => {
      bg.style.transform = 'scale(1, 1)';
      bg.style.opacity = '';
      pill.style.zIndex = '';
    };

    pill.addEventListener('mouseenter', onEnter);
    pill.addEventListener('mouseleave', onLeave);
    cleanups.push(() => {
      pill.removeEventListener('mouseenter', onEnter);
      pill.removeEventListener('mouseleave', onLeave);
    });
  });
  return () => cleanups.forEach((fn) => fn());
}

const JOSH_HEART_STORAGE_KEY = 'josh-post-hearts';

function joshHeartBaseCount(slug) {
  let hash = 0;
  const s = String(slug || '');
  for (let i = 0; i < s.length; i += 1) {
    hash = Math.imul(31, hash) + s.charCodeAt(i);
  }
  return 1800 + (Math.abs(hash) % 22000);
}

function joshReadHeartLiked(slug) {
  try {
    const data = JSON.parse(localStorage.getItem(JOSH_HEART_STORAGE_KEY) || '{}');
    return Boolean(data[slug]);
  } catch {
    return false;
  }
}

function joshWriteHeartLiked(slug, liked) {
  try {
    const data = JSON.parse(localStorage.getItem(JOSH_HEART_STORAGE_KEY) || '{}');
    if (liked) data[slug] = true;
    else delete data[slug];
    localStorage.setItem(JOSH_HEART_STORAGE_KEY, JSON.stringify(data));
  } catch {
    /* storage unavailable */
  }
}

function joshFormatHeartCount(n) {
  return Number(n || 0).toLocaleString('en-US');
}

function initJoshPostHeart(scope, slug) {
  const safeSlug = String(slug || '');
  const hearts = [...scope.querySelectorAll(`.josh-heart[data-slug="${safeSlug}"]`)];
  if (!hearts.length) return;

  const base = joshHeartBaseCount(slug);
  let liked = joshReadHeartLiked(slug);

  const render = () => {
    hearts.forEach((heart) => {
      const btn = heart.querySelector('.josh-heart__btn');
      const countEl = heart.querySelector('.josh-heart__count');
      if (!btn || !countEl) return;
      heart.classList.toggle('is-liked', liked);
      countEl.textContent = joshFormatHeartCount(base + (liked ? 1 : 0));
      btn.setAttribute('aria-pressed', String(liked));
    });
  };

  render();

  hearts.forEach((heart) => {
    const btn = heart.querySelector('.josh-heart__btn');
    if (!btn) return;
    btn.addEventListener('click', () => {
      liked = !liked;
      joshWriteHeartLiked(slug, liked);
      render();
      joshPlaySound(liked ? 'open' : 'click');
    });
  });
}

function initJoshThemeToggles(scope) {
  scope.querySelectorAll('.josh-theme-toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (typeof toggleTheme === 'function') {
        toggleTheme();
      }
      joshPlaySound('toggle');
    });
  });
}

function initJoshNavLinkSounds(scope) {
  const cleanups = [];
  const selectors = [
    '.josh-nav__link',
    '.josh-mobile-menu__link',
    '.josh-popular__link',
    '.josh-pill',
    '.josh-article__title',
    '.josh-blog-card__title',
    '.josh-mobile-toggle',
  ];
  selectors.forEach((selector) => {
    scope.querySelectorAll(selector).forEach((el) => {
      const onClick = () => joshPlaySound('click');
      el.addEventListener('click', onClick);
      cleanups.push(() => el.removeEventListener('click', onClick));
    });
  });
  return () => cleanups.forEach((fn) => fn());
}

function initJoshInteractions(scope = document) {
  if (typeof joshInteractionsCleanup === 'function') {
    joshInteractionsCleanup();
    joshInteractionsCleanup = null;
  }

  const cleanups = [
    initJoshNavFlyouts(scope),
    initJoshArticleScrollReveal(scope),
    initJoshCategoryPillHover(scope),
    initJoshNavLinkSounds(scope),
  ].filter((fn) => typeof fn === 'function');

  initJoshSoundToggle(scope);
  initJoshLogoDraw(scope);
  initJoshReadMoreArrows(scope);
  initJoshThemeToggles(scope);

  joshInteractionsCleanup = () => {
    cleanups.forEach((fn) => fn());
  };
}

function joshObserveNewArticles(scope, articles) {
  if (!('IntersectionObserver' in window)) {
    articles.forEach((el) => el.classList.add('is-revealed'));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-revealed');
      observer.unobserve(entry.target);
    });
  }, JOSH_ARTICLE_REVEAL_OPTIONS);
  articles.forEach((el) => {
    if (joshArticleIsInViewport(el)) {
      el.classList.add('is-revealed');
      return;
    }
    observer.observe(el);
  });
}
