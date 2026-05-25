/**
 * 校验 Supabase 真实 PV 与前台注水公式是否一致。
 * 运行: node scripts/verify-page-views.mjs
 */
import { readFileSync } from 'node:fs';

const SUPABASE_URL = 'https://iklrasypjkwijgubymux.supabase.co';
const ANON_KEY = 'sb_publishable_53c0VZ2Rvn38ck8yasgk3A_GRHzw2c5';

const VIEW_DISPLAY_INFLATE_MIN = 100;
const VIEW_DISPLAY_INFLATE_MAX = 5000;

function stableViewInflateBoost(slug, kind = 'post') {
  const s = `${kind || 'post'}:${String(slug || '')}`;
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  const span = VIEW_DISPLAY_INFLATE_MAX - VIEW_DISPLAY_INFLATE_MIN + 1;
  return VIEW_DISPLAY_INFLATE_MIN + (h % span);
}

function displayViewCountValue(realCount, slug, kind = 'post') {
  const real = Math.max(0, Number(realCount) || 0);
  return real + stableViewInflateBoost(slug, kind);
}

const headers = {
  apikey: ANON_KEY,
  Authorization: `Bearer ${ANON_KEY}`,
  'Content-Type': 'application/json',
};

async function fetchAllViews() {
  const url = `${SUPABASE_URL}/rest/v1/page_views?select=slug,kind,count,updated_at&order=count.desc`;
  const r = await fetch(url, { headers });
  if (!r.ok) throw new Error(`page_views query failed: ${r.status} ${await r.text()}`);
  return r.json();
}

async function main() {
  const rows = await fetchAllViews();
  console.log('=== Supabase page_views (真实 PV) ===\n');
  if (!rows.length) {
    console.log('(表为空 — 尚无访问记录或 RPC 未部署)\n');
  } else {
    console.table(rows.map((row) => ({
      slug: row.slug,
      kind: row.kind,
      real_pv: row.count,
      display_pv: displayViewCountValue(row.count, row.slug, row.kind),
      boost: stableViewInflateBoost(row.slug, row.kind),
      updated_at: row.updated_at,
    })));
  }

  const manifest = JSON.parse(readFileSync('content/posts/manifest.json', 'utf-8'));
  console.log('\n=== manifest 文章（含未入库 slug）===\n');
  const inDb = new Set(rows.map((r) => `${r.kind}:${r.slug}`));
  for (const post of manifest.slice(0, 5)) {
    const row = rows.find((r) => r.slug === post.slug && r.kind === 'post');
    const real = row ? row.count : 0;
    console.log(
      `${post.slug}: 真实=${real}, 展示=${displayViewCountValue(real, post.slug, 'post')}, 偏移=${stableViewInflateBoost(post.slug, 'post')}`,
    );
    if (!inDb.has(`post:${post.slug}`) && !row) {
      console.log(`  (尚未有 Supabase 记录，展示数字仅为偏移 ${stableViewInflateBoost(post.slug, 'post')})`);
    }
  }
  console.log('\n公式: 展示 PV = 真实 PV + stableViewInflateBoost(slug, kind)，偏移范围 [100, 5000]');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
