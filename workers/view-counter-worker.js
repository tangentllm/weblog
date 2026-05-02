/**
 * Cloudflare Workers + KV：页面浏览量计数（可选部署）
 *
 * wrangler.toml 示例：
 * name = "blog-view-counter"
 * main = "view-counter-worker.js"
 * kv_namespaces = [{ binding = "VIEWS", id = "<your-kv-id>" }]
 *
 * 部署后将 Workers URL 填到 index.html 的 siteMeta.viewCounterEndpoint（勿带末尾 /）
 */
export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '*';
    const cors = {
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: cors });
    }

    const url = new URL(request.url);

    if (request.method === 'POST' && url.pathname.endsWith('/hit')) {
      let body = {};
      try {
        body = await request.json();
      } catch {
        return json({ error: 'invalid json' }, 400, cors);
      }
      const slug = typeof body.slug === 'string' ? body.slug.slice(0, 200) : '';
      const kind = body.kind === 'project' ? 'project' : 'post';
      if (!slug) return json({ error: 'missing slug' }, 400, cors);

      const key = `${kind}:${slug}`;
      const prev = parseInt((await env.VIEWS.get(key)) || '0', 10);
      const next = prev + 1;
      await env.VIEWS.put(key, String(next));
      return json({ count: next }, 200, cors);
    }

    if (request.method === 'GET' && url.pathname.endsWith('/count')) {
      const slug = url.searchParams.get('slug') || '';
      const kind = url.searchParams.get('kind') === 'project' ? 'project' : 'post';
      if (!slug) return json({ error: 'missing slug' }, 400, cors);
      const key = `${kind}:${slug.slice(0, 200)}`;
      const cur = parseInt((await env.VIEWS.get(key)) || '0', 10);
      return json({ count: cur }, 200, cors);
    }

    return new Response('Not Found', { status: 404, headers: cors });
  },
};

function json(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...extraHeaders,
    },
  });
}
