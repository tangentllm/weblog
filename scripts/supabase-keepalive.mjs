/**
 * Supabase 免费项目保活：每日一次只读 RPC，避免 7 天无活动被自动暂停。
 * 运行: node scripts/supabase-keepalive.mjs
 */
const SUPABASE_URL = (
  process.env.SUPABASE_URL || 'https://iklrasypjkwijgubymux.supabase.co'
).replace(/\/$/, '');
const ANON_KEY = process.env.SUPABASE_ANON_KEY || 'sb_publishable_53c0VZ2Rvn38ck8yasgk3A_GRHzw2c5';

const headers = {
  apikey: ANON_KEY,
  Authorization: `Bearer ${ANON_KEY}`,
  'Content-Type': 'application/json',
};

async function ping() {
  const url = `${SUPABASE_URL}/rest/v1/rpc/get_page_view`;
  const r = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify({ p_slug: '_keepalive', p_kind: 'post' }),
    cache: 'no-store',
  });
  const text = await r.text();
  if (!r.ok) {
    throw new Error(`keepalive failed: ${r.status} ${text}`);
  }
  return text;
}

ping()
  .then((body) => {
    console.log(`Supabase keepalive OK (${new Date().toISOString()}): ${body.trim()}`);
  })
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
