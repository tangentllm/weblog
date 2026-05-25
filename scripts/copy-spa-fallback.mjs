import { copyFileSync } from 'node:fs';

copyFileSync('index.html', '404.html');
console.log('404.html updated from index.html (GitHub Pages SPA fallback)');
