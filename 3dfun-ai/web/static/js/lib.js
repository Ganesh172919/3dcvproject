// Core helpers: API client, DOM utilities, state, formatting, toasts.

export const API = '/api/v1';

export const state = {
  token: localStorage.getItem('3dfun_token') || null,
  user: null,
  meta: null,
};

// ---------------------------------------------------------------- API client
let refreshPromise;

export async function api(path, { method = 'GET', body, raw = false, form = false, retry = true } = {}) {
  const headers = {};
  if (state.token) headers.authorization = `Bearer ${state.token}`;
  // Double-submit CSRF: echo the non-HttpOnly cookie on state-changing calls so
  // cookie-authenticated requests (no bearer header) are still accepted.
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method.toUpperCase())) {
    const csrf = document.cookie.split('; ').find((c) => c.startsWith('3dfun_csrf='));
    if (csrf) headers['x-csrf-token'] = decodeURIComponent(csrf.split('=')[1]);
  }
  let payload = body;
  if (body && !form && !(body instanceof Blob) && !(body instanceof FormData)) {
    headers['content-type'] = 'application/json';
    payload = JSON.stringify(body);
  }
  const res = await fetch(API + path, { method, headers, body: payload, credentials: 'include' });
  if (res.status === 401 && retry && state.token && (!path.startsWith('/auth/') || path === '/auth/me')) {
    refreshPromise ??= api('/auth/refresh', { method: 'POST', retry: false })
      .then(data => { setToken(data.access_token); return true; })
      .catch(() => false).finally(() => { refreshPromise = null; });
    if (await refreshPromise) return api(path, { method, body, raw, form, retry: false });
  }
  if (raw) {
    if (!res.ok) throw new ApiError(`HTTP ${res.status}`, res.status, null);
    return res;
  }
  let json = null;
  try { json = await res.json(); } catch { /* empty body */ }
  if (!res.ok || (json && json.success === false)) {
    const err = json && json.error ? json.error : { code: 'HTTP_' + res.status, message: 'Request failed' };
    throw new ApiError(err.message, res.status, err);
  }
  return json ? json.data : null;
}

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail || {};
    this.code = this.detail.code || `HTTP_${status}`;
  }
}

// ---------------------------------------------------------------- DOM helpers
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'disabled' || k === 'checked' || k === 'selected' || k === 'multiple') el[k] = !!v;
    else if (k === 'value') el.value = v;
    else el.setAttribute(k, v);
  }
  for (const child of children.flat(3)) {
    if (child === null || child === undefined || child === false) continue;
    el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
export function mount(el, ...nodes) { clear(el).append(...nodes.flat(3).filter(Boolean)); return el; }
export function esc(str) {
  return String(str ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---------------------------------------------------------------- formatting
export const fmt = {
  n(v) { return Number(v ?? 0).toLocaleString(); },
  bytes(b) {
    const v = Number(b || 0);
    if (!v) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.min(units.length - 1, Math.floor(Math.log(v) / Math.log(1024)));
    return `${(v / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
  },
  ms(v) {
    const n = Number(v || 0);
    if (!n) return '—';
    return n < 1000 ? `${Math.round(n)} ms` : `${(n / 1000).toFixed(1)} s`;
  },
  date(v) {
    if (!v) return '—';
    const d = new Date(v);
    return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  },
  ago(v) {
    if (!v) return '—';
    const s = (Date.now() - new Date(v).getTime()) / 1000;
    if (s < 60) return `${Math.round(s)}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return `${Math.round(s / 86400)}d ago`;
  },
  pct(v) { return `${Math.round(Number(v || 0))}%`; },
};

// ---------------------------------------------------------------- toasts
export function toast(message, kind = 'info', ms = 4200) {
  const box = document.getElementById('toasts');
  const el = h('div', { class: `toast ${kind}`, role: 'status' }, message);
  box.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 260); }, ms);
  return el;
}

// ---------------------------------------------------------------- auth/session
export async function refreshUser() {
  if (!state.token) { setAuthUI(false); return null; }
  try {
    state.user = await api('/auth/me');
    const credits = await api('/credits/balance').catch(() => null);
    if (credits) state.user.credits = credits.available;
    setAuthUI(true);
    return state.user;
  } catch {
    logout(false);
    return null;
  }
}

export function setAuthUI(authed) {
  const wallet = document.getElementById('wallet');
  const login = document.getElementById('auth-login');
  const out = document.getElementById('auth-logout');
  const admin = document.getElementById('nav-admin');
  wallet.classList.toggle('hidden', !authed);
  login.classList.toggle('hidden', authed);
  out.classList.toggle('hidden', !authed);
  const isAdmin = authed && state.user && state.user.role === 'admin';
  if (admin) admin.classList.toggle('hidden', !isAdmin);
  if (authed && state.user) {
    document.getElementById('wallet-value').textContent = fmt.n(state.user.credits ?? 0);
  }
}

export function setToken(token) {
  state.token = token;
  if (token) localStorage.setItem('3dfun_token', token);
  else localStorage.removeItem('3dfun_token');
}

export function logout(redirect = true) {
  setToken(null);
  state.user = null;
  setAuthUI(false);
  api('/auth/logout', { method: 'POST' }).catch(() => {});
  if (redirect) navigate('/app/login');
}

// ---------------------------------------------------------------- router
const routes = [];
export function route(pattern, handler) { routes.push({ pattern, handler }); }

export function navigate(path, replace = false) {
  history[replace ? 'replaceState' : 'pushState']({}, '', path);
  renderRoute();
}

export async function renderRoute() {
  state.routeCleanup?.();
  state.routeCleanup = null;
  state._viewer?.dispose();
  state._viewer = null;
  const path = location.pathname.replace(/\/$/, '') || '/app';
  const main = document.getElementById('main');
  $$('#nav a').forEach(a => {
    const selected = a.getAttribute('href') === path || path.startsWith(`${a.getAttribute('href')}/`);
    a.classList.toggle('active', selected);
    if (selected) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  for (const r of routes) {
    const m = matchRoute(r.pattern, path);
    if (m) {
      try {
        await r.handler(m.params, main);
      } catch (err) {
        console.error(err);
        renderError(main, err);
      }
      window.scrollTo({ top: 0 });
      return;
    }
  }
  renderError(main, new ApiError('Page not found', 404));
}

function matchRoute(pattern, path) {
  const pp = pattern.split('/');
  const sp = path.split('/');
  if (pp.length !== sp.length) return null;
  const params = {};
  for (let i = 0; i < pp.length; i++) {
    if (pp[i].startsWith(':')) params[pp[i].slice(1)] = decodeURIComponent(sp[i]);
    else if (pp[i] !== sp[i]) return null;
  }
  return { params };
}

export function renderError(main, err) {
  mount(main, h('div', { class: 'card center' },
    h('h2', {}, err.status === 404 ? 'Not found' : 'Something went wrong'),
    h('p', { class: 'muted' }, err.message || String(err)),
    err.code ? h('p', { class: 'mono muted' }, `code: ${err.code}`) : null,
    h('a', { class: 'btn primary', href: '/app/studio', 'data-link': '' }, 'Back to studio')));
}

export function initRouter() {
  const nav = $('#nav');
  const menu = $('#nav-toggle');
  function toggleMenu(open) {
    nav.classList.toggle('open', open);
    menu.setAttribute('aria-expanded', String(open));
  }
  menu.addEventListener('click', () => toggleMenu(!nav.classList.contains('open')));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && nav.classList.contains('open')) { toggleMenu(false); menu.focus(); }
  });
  window.addEventListener('popstate', () => { toggleMenu(false); renderRoute(); });
  document.addEventListener('click', (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    if (!e.target.closest('#nav, #nav-toggle')) toggleMenu(false);
    const a = e.target.closest('a[data-link]');
    if (!a) return;
    const href = a.getAttribute('href');
    if (!href || href.startsWith('http') || a.target === '_blank') return;
    e.preventDefault();
    toggleMenu(false);
    navigate(href);
  });
}

// ---------------------------------------------------------------- misc
export function debounce(fn, ms = 250) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

export function statusBadge(status) {
  const s = String(status || '').toLowerCase();
  const cls = ['completed', 'failed', 'cancelled'].includes(s) ? s
    : ['queued', 'draft'].includes(s) ? 'queued' : 'running';
  const label = String(status || '').replaceAll('_', ' ').toLowerCase();
  return h('span', { class: `badge ${cls}` }, label);
}

export function emptyState(icon, title, sub, action) {
  return h('div', { class: 'empty' },
    h('div', { class: 'ic', 'aria-hidden': 'true' }, icon),
    h('h3', {}, title),
    h('p', {}, sub),
    action || null);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('Copied to clipboard');
  } catch {
    const ta = h('textarea', { style: 'position:fixed;opacity:0' }, text);
    document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); ta.remove();
    toast('Copied to clipboard');
  }
}

export function skeletonBlock(height = 80) {
  return h('div', { class: 'skeleton', style: `height:${height}px` });
}
