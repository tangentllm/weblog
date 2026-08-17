/* Josh W. Comeau — Sandpack Code Playground for .josh-prose */

const JOSH_SANDPACK_REACT = 'https://esm.sh/@codesandbox/sandpack-react@2.19.0?deps=react@18.3.1,react-dom@18.3.1';
const JOSH_SANDPACK_THEMES = 'https://esm.sh/@codesandbox/sandpack-themes@2.0.21';

const joshPlaygroundMounts = new Map();
let joshSandpackDepsPromise = null;
let joshPlaygroundThemeObserver = null;

function joshPlaygroundIsDarkMode() {
  const html = document.documentElement;
  return html.classList.contains('dark') || html.getAttribute('data-color-mode') === 'dark';
}

function joshDecodePlaygroundAttr(encoded) {
  return JSON.parse(decodeURIComponent(escape(atob(encoded))));
}

function loadJoshSandpackDeps() {
  if (!joshSandpackDepsPromise) {
    joshSandpackDepsPromise = Promise.all([
      import('https://esm.sh/react@18.3.1'),
      import('https://esm.sh/react-dom@18.3.1/client'),
      import(JOSH_SANDPACK_REACT),
      import(JOSH_SANDPACK_THEMES),
    ]).then(([ReactMod, ReactDOMMod, SandpackMod, ThemesMod]) => ({
      React: ReactMod.default,
      createRoot: ReactDOMMod.createRoot,
      Sandpack: SandpackMod.Sandpack,
      sandpackDark: ThemesMod.sandpackDark,
      sandpackLight: ThemesMod.sandpackLight,
    }));
  }
  return joshSandpackDepsPromise;
}

function joshPlaygroundTheme(deps) {
  return joshPlaygroundIsDarkMode() ? deps.sandpackDark : deps.sandpackLight;
}

function joshPlaygroundRender(deps, root, config) {
  root.render(
    deps.React.createElement(deps.Sandpack, {
      template: config.template || 'vanilla',
      files: config.files,
      theme: joshPlaygroundTheme(deps),
      options: {
        ...config.options,
        recompileMode: 'delayed',
        recompileDelay: 300,
        autorun: true,
      },
    }),
  );
}

function ensureJoshPlaygroundThemeObserver() {
  if (joshPlaygroundThemeObserver) return;
  joshPlaygroundThemeObserver = new MutationObserver(() => {
    joshPlaygroundMounts.forEach(({ deps, root, config }) => {
      joshPlaygroundRender(deps, root, config);
    });
  });
  joshPlaygroundThemeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class', 'data-color-mode'],
  });
}

async function mountJoshPlayground(el, config) {
  const shell = el.querySelector('.josh-playground__shell') || el;
  const mountHost = document.createElement('div');
  mountHost.className = 'josh-playground__mount';
  shell.replaceChildren(mountHost);

  const deps = await loadJoshSandpackDeps();
  const root = deps.createRoot(mountHost);
  joshPlaygroundRender(deps, root, config);
  joshPlaygroundMounts.set(el, { deps, root, config });
  ensureJoshPlaygroundThemeObserver();
}

async function enhanceJoshPlaygrounds(scope = document) {
  const nodes = [...scope.querySelectorAll('[data-josh-playground]:not([data-josh-playground-ready])')];
  if (!nodes.length) return true;

  await loadJoshSandpackDeps();

  for (const el of nodes) {
    el.setAttribute('data-josh-playground-ready', 'true');
    try {
      const encoded = el.getAttribute('data-config');
      const config = joshDecodePlaygroundAttr(encoded);
      await mountJoshPlayground(el, config);
    } catch (err) {
      console.warn('[josh-playground] mount failed', err);
      const shell = el.querySelector('.josh-playground__shell');
      if (shell) {
        shell.innerHTML = '<p class="josh-playground__error">Playground failed to load.</p>';
      }
    }
  }

  return true;
}

function preloadJoshPlayground() {
  return loadJoshSandpackDeps();
}

window.enhanceJoshPlaygrounds = enhanceJoshPlaygrounds;
window.preloadJoshPlayground = preloadJoshPlayground;

if (document.documentElement.classList.contains('josh-site')) {
  preloadJoshPlayground();
}
