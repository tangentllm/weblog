/**
 * Pure CSS scoper for HTML magazine articles embedded into .josh-prose.
 * Run: node scripts/test-scope-html-article-css.mjs
 */
import assert from 'node:assert/strict';

export const JOSH_HTML_MAGAZINE_SCOPE = '.josh-prose .josh-html-magazine';

export function joshStripCssComments(css) {
  return String(css || '').replace(/\/\*[\s\S]*?\*\//g, '');
}

export function joshRewriteMagazineDarkThemeSelector(selector, scope = JOSH_HTML_MAGAZINE_SCOPE) {
  const s = String(selector || '').trim();
  if (!s) return s;
  const darkAttr = /^(?::root|html)\[data-theme=["']dark["']\](.*)$/i.exec(s);
  if (darkAttr) {
    const rest = darkAttr[1] || '';
    const scopedRest = rest
      ? (rest.startsWith(' ') || rest.startsWith(':') || rest.startsWith('.') || rest.startsWith('[')
        ? `${scope}${rest}`
        : `${scope} ${rest}`)
      : scope;
    return [
      `html.josh-site.dark ${scopedRest}`,
      `html.josh-site[data-color-mode='dark'] ${scopedRest}`,
    ].join(', ');
  }
  return null;
}

export function joshScopeSelectorList(selectorGroup, scope = JOSH_HTML_MAGAZINE_SCOPE) {
  return selectorGroup
    .split(',')
    .map((raw) => {
      const s = raw.trim();
      if (!s) return s;
      if (/^(from|to|\d+(\.\d+)?%)$/i.test(s)) return s;
      if (s === ':root' || s === 'html' || s === 'body') return scope;
      if (s === '*') return `${scope}, ${scope} *`;
      if (s === scope || s.startsWith(`${scope} `) || s.startsWith(`${scope}:`) || s.startsWith(`${scope}.`) || s.startsWith(`${scope}[`)) {
        return s;
      }
      const darkRewrite = joshRewriteMagazineDarkThemeSelector(s, scope);
      if (darkRewrite) return darkRewrite;
      if (s.startsWith(':root')) return `${scope}${s.slice(5)}`;
      if (/^html\b/.test(s)) return `${scope}${s.replace(/^html\b/, '')}`;
      if (/^body\b/.test(s)) return `${scope}${s.replace(/^body\b/, '')}`;
      return `${scope} ${s}`;
    })
    .join(', ');
}

function readBalancedBlock(text, openIndex) {
  let i = openIndex + 1;
  let depth = 1;
  const start = i;
  while (i < text.length && depth > 0) {
    const c = text[i];
    if (c === '"' || c === "'") {
      const q = c;
      i += 1;
      while (i < text.length && text[i] !== q) {
        if (text[i] === '\\') i += 1;
        i += 1;
      }
      i += 1;
      continue;
    }
    if (c === '{') depth += 1;
    else if (c === '}') depth -= 1;
    i += 1;
  }
  return {
    body: text.slice(start, i - 1),
    end: i,
  };
}

export function joshScopeHtmlArticleCss(css, scope = JOSH_HTML_MAGAZINE_SCOPE) {
  const text = joshStripCssComments(css);

  const processRules = (chunk) => {
    let j = 0;
    let out = '';

    const skipWs = () => {
      while (j < chunk.length && /\s/.test(chunk[j])) j += 1;
    };

    while (j < chunk.length) {
      skipWs();
      if (j >= chunk.length) break;

      if (chunk[j] === '@') {
        const atMatch = chunk.slice(j).match(/^@[a-zA-Z-]+/);
        const atName = (atMatch ? atMatch[0] : '@').toLowerCase();
        const atStart = j;
        j += atName.length;

        if (atName === '@keyframes' || atName === '@font-face' || atName === '@page') {
          while (j < chunk.length && chunk[j] !== '{') j += 1;
          if (chunk[j] === '{') {
            const { body, end } = readBalancedBlock(chunk, j);
            out += `${chunk.slice(atStart, j)}{${body}}`;
            j = end;
          } else {
            out += chunk.slice(atStart, j);
          }
          continue;
        }

        if (atName === '@media' || atName === '@supports' || atName === '@container') {
          const preludeStart = j;
          while (j < chunk.length && chunk[j] !== '{') j += 1;
          const prelude = chunk.slice(preludeStart, j).trim();
          if (chunk[j] === '{') {
            const { body, end } = readBalancedBlock(chunk, j);
            out += `${atName} ${prelude}{${processRules(body)}}`;
            j = end;
          }
          continue;
        }

        while (j < chunk.length && chunk[j] !== '{' && chunk[j] !== ';') j += 1;
        if (chunk[j] === ';') {
          j += 1;
          out += chunk.slice(atStart, j);
        } else if (chunk[j] === '{') {
          const { body, end } = readBalancedBlock(chunk, j);
          out += `${chunk.slice(atStart, j)}{${body}}`;
          j = end;
        } else {
          out += chunk.slice(atStart, j);
        }
        continue;
      }

      const selStart = j;
      while (j < chunk.length && chunk[j] !== '{') j += 1;
      const selectorGroup = chunk.slice(selStart, j).trim();
      if (chunk[j] !== '{') {
        out += chunk.slice(selStart, j);
        break;
      }
      const { body, end } = readBalancedBlock(chunk, j);
      j = end;
      if (!selectorGroup) continue;
      out += `${joshScopeSelectorList(selectorGroup, scope)}{${body}}`;
    }

    return out;
  };

  return processRules(text);
}

const SCOPE = JOSH_HTML_MAGAZINE_SCOPE;
const esc = SCOPE.replace(/\./g, '\\.');

{
  const scoped = joshScopeHtmlArticleCss(':root{--ink:#111} body{color:red} .card{display:grid} *{margin:0}', SCOPE);
  assert.match(scoped, new RegExp(`${esc}\\{--ink:#111\\}`));
  assert.match(scoped, new RegExp(`${esc}\\{color:red\\}`));
  assert.match(scoped, new RegExp(`${esc} \\.card\\{display:grid\\}`));
  assert.match(scoped, new RegExp(`${esc}, ${esc} \\*\\{margin:0\\}`));
}

{
  const scoped = joshScopeHtmlArticleCss('@media (max-width:640px){h1.title{font-size:30px} .grid{gap:8px}}', SCOPE);
  assert.match(scoped, /@media \(max-width:640px\)\{/);
  assert.match(scoped, new RegExp(`${esc} h1\\.title\\{font-size:30px\\}`));
  assert.match(scoped, new RegExp(`${esc} \\.grid\\{gap:8px\\}`));
}

{
  const scoped = joshScopeHtmlArticleCss('@keyframes fade{from{opacity:0} to{opacity:1}} .x{animation:fade 1s}', SCOPE);
  assert.match(scoped, /@keyframes fade\{from\{opacity:0\} to\{opacity:1\}\}/);
  assert.match(scoped, new RegExp(`${esc} \\.x\\{animation:fade 1s\\}`));
}

{
  const scoped = joshScopeHtmlArticleCss(
    ':root[data-theme="dark"]{--rust-bg:#2C1F19} :root[data-theme="dark"] .callout.note{color:#B9D2DE}',
    SCOPE,
  );
  assert.match(
    scoped,
    /html\.josh-site\.dark .josh-prose \.josh-html-magazine, html\.josh-site\[data-color-mode='dark'\] .josh-prose \.josh-html-magazine\{--rust-bg:#2C1F19\}/,
  );
  assert.match(
    scoped,
    /html\.josh-site\.dark .josh-prose \.josh-html-magazine \.callout\.note, html\.josh-site\[data-color-mode='dark'\] .josh-prose \.josh-html-magazine \.callout\.note\{color:#B9D2DE\}/,
  );
}

console.log('ok: joshScopeHtmlArticleCss');
