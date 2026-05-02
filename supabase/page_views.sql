-- 在 Supabase：SQL Editor 中整段执行一次即可。
-- 前端使用 anon key + RLS；写入仅允许通过 SECURITY DEFINER 的 RPC。

create table if not exists public.page_views (
  slug text not null,
  kind text not null check (kind in ('post', 'project')),
  count bigint not null default 0,
  updated_at timestamptz not null default now(),
  primary key (slug, kind)
);

alter table public.page_views enable row level security;

drop policy if exists "Allow public read counts" on public.page_views;
create policy "Allow public read counts"
  on public.page_views for select
  to anon, authenticated
  using (true);

drop policy if exists "No direct inserts" on public.page_views;
create policy "No direct inserts"
  on public.page_views for insert
  to anon, authenticated
  with check (false);

drop policy if exists "No direct updates" on public.page_views;
create policy "No direct updates"
  on public.page_views for update
  to anon, authenticated
  using (false);

drop policy if exists "No direct deletes" on public.page_views;
create policy "No direct deletes"
  on public.page_views for delete
  to anon, authenticated
  using (false);

create or replace function public.increment_page_view(p_slug text, p_kind text)
returns bigint
language plpgsql
security definer
set search_path = public
as $$
declare
  v_count bigint;
begin
  if p_slug is null or length(trim(p_slug)) = 0 or length(p_slug) > 200 then
    raise exception 'invalid slug';
  end if;
  if p_kind not in ('post', 'project') then
    raise exception 'invalid kind';
  end if;

  insert into public.page_views (slug, kind, count)
  values (trim(p_slug), p_kind, 1)
  on conflict (slug, kind)
  do update set count = public.page_views.count + 1, updated_at = now()
  returning count into v_count;

  return v_count;
end;
$$;

create or replace function public.get_page_view(p_slug text, p_kind text)
returns bigint
language sql
security definer
set search_path = public
stable
as $$
  select coalesce(
    (select pv.count from public.page_views pv
     where pv.slug = trim(p_slug) and pv.kind = p_kind limit 1),
    0::bigint
  );
$$;

revoke insert, update, delete on public.page_views from anon, authenticated;

grant usage on schema public to anon, authenticated;
grant select on public.page_views to anon, authenticated;
grant execute on function public.increment_page_view(text, text) to anon, authenticated;
grant execute on function public.get_page_view(text, text) to anon, authenticated;
