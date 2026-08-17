/* Josh W. Comeau — Shiki dual-theme code blocks for .josh-prose */

const JOSH_SHIKI_CDN = 'https://esm.sh/shiki@3.6.0';
const JOSH_SHIKI_LANGS = [
  'bash', 'css', 'html', 'javascript', 'json', 'markdown', 'plaintext',
  'python', 'sql', 'tsx', 'typescript', 'xml', 'yaml',
];

const JOSH_SHIKI_LANG_ALIASES = {
  js: 'javascript',
  ts: 'typescript',
  py: 'python',
  sh: 'bash',
  shell: 'bash',
  yml: 'yaml',
  md: 'markdown',
  text: 'plaintext',
  txt: 'plaintext',
};

const joshThemeLight = {
  name: 'josh-theme-light',
  type: 'light',
  bg: '#e8f1fc00',
  fg: '#292929',
  settings: [
    { settings: { foreground: '#292929' } },
    { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: '#5c6278', fontStyle: 'italic' } },
    { scope: ['keyword', 'storage.type', 'storage.modifier', 'keyword.control'], settings: { foreground: '#db0078', fontStyle: 'bold' } },
    { scope: ['string', 'constant.other.symbol', 'meta.attribute-value'], settings: { foreground: '#4d1cff' } },
    { scope: ['constant.numeric', 'constant.language', 'variable', 'support.constant'], settings: { foreground: '#566773' } },
    { scope: ['entity.name.function', 'support.function', 'meta.function-call'], settings: { foreground: '#403efa' } },
    { scope: ['entity.name.type', 'entity.name.class', 'support.type', 'support.class'], settings: { foreground: '#bd00bc' } },
    { scope: ['entity.name.tag', 'tag'], settings: { foreground: '#db0078', fontStyle: 'bold' } },
    { scope: ['entity.other.attribute-name'], settings: { foreground: '#4d1cff' } },
    { scope: ['punctuation', 'meta.brace'], settings: { foreground: '#566773' } },
    { scope: ['invalid', 'markup.deleted'], settings: { foreground: '#ff5555' } },
  ],
};

const joshThemeDark = {
  name: 'josh-theme-dark',
  type: 'dark',
  bg: '#0d0f1200',
  fg: '#ffffff',
  settings: [
    { settings: { foreground: '#ffffff' } },
    { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: '#6b8ba3', fontStyle: 'italic' } },
    { scope: ['keyword', 'storage.type', 'storage.modifier', 'keyword.control'], settings: { foreground: '#ff38a9' } },
    { scope: ['string', 'constant.other.symbol', 'meta.attribute-value'], settings: { foreground: '#9d71ff' } },
    { scope: ['constant.numeric', 'constant.language', 'variable', 'support.constant'], settings: { foreground: '#a3b0bd' } },
    { scope: ['entity.name.function', 'support.function', 'meta.function-call'], settings: { foreground: '#02c7ff' } },
    { scope: ['entity.name.type', 'entity.name.class', 'support.type', 'support.class'], settings: { foreground: '#d454ff' } },
    { scope: ['entity.name.tag', 'tag'], settings: { foreground: '#b9c4d0', fontStyle: 'bold' } },
    { scope: ['entity.other.attribute-name'], settings: { foreground: '#ff38a9' } },
    { scope: ['punctuation', 'meta.brace'], settings: { foreground: '#a3b0bd' } },
    { scope: ['invalid', 'markup.deleted'], settings: { foreground: '#ff5555' } },
  ],
};

let joshHighlighterPromise = null;

function joshShikiNormalizeLang(raw) {
  const lang = String(raw || 'plaintext').toLowerCase();
  return JOSH_SHIKI_LANG_ALIASES[lang] || lang;
}

function joshShikiInferLang(codeEl) {
  const cls = codeEl.className || '';
  const match = cls.match(/language-([\w+-]+)/i);
  if (match) return joshShikiNormalizeLang(match[1]);
  const parent = codeEl.closest('[data-language]');
  if (parent) return joshShikiNormalizeLang(parent.getAttribute('data-language'));
  return 'plaintext';
}

function joshShikiExtractCode(codeEl) {
  return codeEl.textContent.replace(/\n$/, '');
}

function getJoshHighlighter() {
  if (!joshHighlighterPromise) {
    joshHighlighterPromise = import(JOSH_SHIKI_CDN).then(({ createHighlighter }) =>
      createHighlighter({
        themes: [joshThemeLight, joshThemeDark],
        langs: JOSH_SHIKI_LANGS,
      }),
    );
  }
  return joshHighlighterPromise;
}

function preloadJoshShiki() {
  return getJoshHighlighter();
}

function joshCodeSnippetCopyMarkup() {
  return `<button type="button" class="josh-code-snippet__copy" data-josh-copy>
    <span class="josh-code-snippet__copy-label">Copy to clipboard</span>
  </button>`;
}

function joshCreateCodeSnippetShell() {
  const figure = document.createElement('figure');
  figure.className = 'josh-code-snippet';
  figure.innerHTML = `
    <div class="josh-code-snippet__toolbar" data-josh-copy-toolbar>
      ${joshCodeSnippetCopyMarkup()}
    </div>
    <div class="josh-code-snippet__body"></div>
  `;
  return figure;
}

function joshBuildCodeSnippetFigure(shikiHtml) {
  const figure = joshCreateCodeSnippetShell();
  figure.querySelector('.josh-code-snippet__body').innerHTML = shikiHtml;
  return figure;
}

function joshWrapPreInCodeSnippet(pre) {
  const figure = joshCreateCodeSnippetShell();
  const body = figure.querySelector('.josh-code-snippet__body');
  pre.classList.remove('josh-shiki-pending');
  body.appendChild(pre);
  joshBindCodeSnippetCopy(figure);
  return figure;
}

function joshBindCodeSnippetCopy(figure) {
  const btn = figure.querySelector('[data-josh-copy]');
  const pre = figure.querySelector('pre');
  if (!btn || !pre) return;

  btn.addEventListener('click', async () => {
    const text = pre.textContent || '';
    try {
      await navigator.clipboard.writeText(text);
      const label = btn.querySelector('.josh-code-snippet__copy-label');
      if (label) {
        const prev = label.textContent;
        label.textContent = 'Copied!';
        setTimeout(() => { label.textContent = prev; }, 1400);
      }
      if (typeof joshPlaySound === 'function') joshPlaySound('click');
    } catch {
      /* clipboard blocked */
    }
  });
}

async function joshHighlightCodeElement(highlighter, codeEl) {
  const langRaw = joshShikiInferLang(codeEl);
  const lang = highlighter.getLoadedLanguages().includes(langRaw) ? langRaw : 'plaintext';
  const code = joshShikiExtractCode(codeEl);
  const shikiHtml = highlighter.codeToHtml(code, {
    lang,
    themes: {
      light: 'josh-theme-light',
      dark: 'josh-theme-dark',
    },
    defaultColor: false,
  });

  const pre = codeEl.closest('pre');
  const host = pre?.closest('.code-block') || pre;
  if (!host) return;

  const figure = joshBuildCodeSnippetFigure(shikiHtml);
  host.replaceWith(figure);
  joshBindCodeSnippetCopy(figure);
}

function joshWrapPlainCodeSnippets(scope = document) {
  const pres = [];
  scope.querySelectorAll('.josh-prose pre').forEach((pre) => {
    if (pre.closest('.josh-code-snippet') || pre.closest('.josh-playground') || pre.closest('.sp-wrapper')) return;
    if (!pre.querySelector('code')) return;
    if (!pre.textContent.trim()) {
      pre.remove();
      return;
    }
    pres.push(pre);
  });

  pres.forEach((pre) => {
    pre.replaceWith(joshWrapPreInCodeSnippet(pre));
  });

  return pres.length;
}

async function highlightJoshProseCode(scope = document) {
  const roots = scope.querySelectorAll('.josh-prose');
  if (!roots.length) return false;

  const blocks = [];
  roots.forEach((root) => {
    root.querySelectorAll('pre code').forEach((codeEl) => {
      if (codeEl.closest('.josh-playground')) return;
      if (codeEl.closest('.josh-code-snippet')) return;
      if (codeEl.closest('pre.shiki')) return;
      blocks.push(codeEl);
    });
  });

  if (!blocks.length) return true;

  blocks.forEach((codeEl) => {
    const pre = codeEl.closest('pre');
    if (pre) pre.classList.add('josh-shiki-pending');
  });

  try {
    const highlighter = await getJoshHighlighter();
    for (const codeEl of blocks) {
      await joshHighlightCodeElement(highlighter, codeEl);
    }
    return true;
  } catch (err) {
    blocks.forEach((codeEl) => {
      codeEl.closest('pre')?.classList.remove('josh-shiki-pending');
    });
    console.warn('[josh-shiki] highlight failed, falling back to hljs', err);
    return false;
  }
}

window.preloadJoshShiki = preloadJoshShiki;
window.highlightJoshProseCode = highlightJoshProseCode;
window.joshWrapPlainCodeSnippets = joshWrapPlainCodeSnippets;

if (document.documentElement.classList.contains('josh-site')) {
  preloadJoshShiki();
}
