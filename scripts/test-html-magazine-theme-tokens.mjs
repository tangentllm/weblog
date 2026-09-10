/**
 * Magazine theme token isolation for .josh-html-magazine.
 * Run: node scripts/test-html-magazine-theme-tokens.mjs
 */
import assert from 'node:assert/strict';

export const JOSH_HTML_MAGAZINE_SCOPE = '.josh-prose .josh-html-magazine';

/**
 * Build CSS that remaps magazine surface/ink tokens onto Josh theme tokens.
 * Must NOT force --accent / --accent-2 / --accent-color onto Josh primary.
 */
export function joshBuildHtmlMagazineIsolationCss(scope = JOSH_HTML_MAGAZINE_SCOPE) {
  return [
    `${scope}{`,
    'max-width:100%;',
    'background:transparent!important;',
    'color:var(--josh-color-text)!important;',
    '--paper:var(--josh-color-background)!important;',
    '--paper-deep:color-mix(in srgb,var(--josh-color-cloud-300) 14%,var(--josh-color-background))!important;',
    '--paper-alt:color-mix(in srgb,var(--josh-color-cloud-300) 10%,var(--josh-color-background))!important;',
    '--bg:var(--josh-color-background)!important;',
    '--bg-card:color-mix(in srgb,var(--josh-color-cloud-300) 55%,var(--josh-color-background))!important;',
    '--surface:color-mix(in srgb,var(--josh-color-cloud-300) 55%,var(--josh-color-background))!important;',
    '--border:color-mix(in srgb,var(--josh-color-text) 16%,transparent)!important;',
    '--ink:var(--josh-color-text)!important;',
    '--ink-soft:var(--josh-color-gray-700)!important;',
    '--ink-mute:var(--josh-color-gray-500)!important;',
    '--ink-faint:var(--josh-color-gray-500)!important;',
    '--text:var(--josh-color-text)!important;',
    '--text-muted:var(--josh-color-gray-500)!important;',
    '--fg:var(--josh-color-text)!important;',
    '--line:color-mix(in srgb,var(--josh-color-text) 16%,transparent)!important;',
    '--line-soft:color-mix(in srgb,var(--josh-color-text) 10%,transparent)!important;',
    '--rule:color-mix(in srgb,var(--josh-color-text) 16%,transparent)!important;',
    '--card:color-mix(in srgb,var(--josh-color-cloud-300) 55%,var(--josh-color-background))!important;',
    '--code-bg:var(--josh-color-code-bg)!important;',
    '--code-ink:var(--josh-syntax-txt)!important;',
    '}',
    `${scope} .page,`,
    `${scope} .wrap,`,
    `${scope} .paper,`,
    `${scope} .container,`,
    `${scope} main{`,
    'max-width:100%;',
    'margin-left:0;',
    'margin-right:0;',
    'padding-left:0;',
    'padding-right:0;',
    'background:transparent!important;',
    'color:inherit!important;',
    '}',
  ].join('');
}

const css = joshBuildHtmlMagazineIsolationCss();

assert.match(css, /--paper:var\(--josh-color-background\)/);
assert.match(css, /--ink:var\(--josh-color-text\)/);
assert.match(css, /--card:color-mix/);
assert.match(css, /--code-bg:var\(--josh-color-code-bg\)/);
assert.match(css, /--bg:var\(--josh-color-background\)/);
assert.match(css, /--surface:color-mix/);
assert.match(css, /--text:var\(--josh-color-text\)/);
assert.match(css, /--text-muted:var\(--josh-color-gray-500\)/);
assert.doesNotMatch(css, /--accent\s*:/);
assert.doesNotMatch(css, /--accent-2\s*:/);
assert.doesNotMatch(css, /--accent-color\s*:/);
assert.doesNotMatch(css, /--accent\s*:\s*var\(--josh-color-primary\)/);

console.log('ok: joshBuildHtmlMagazineIsolationCss');
