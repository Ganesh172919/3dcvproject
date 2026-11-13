// 3DFUN AI Single-Page Application: Router + Views.
// Course: 3D Computer Vision Course Project · IIIT Dharwad
// Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
// Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
//          Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)
import {
  ApiError, $, $$, api, clear, copyText, debounce, emptyState, esc, fmt, h, initRouter,
  logout, mount, navigate, refreshUser, renderError, renderRoute, route, setAuthUI, state, statusBadge,
  toast,
} from './lib.js';
import { renderChat } from './chat.js';
import { renderStudio } from './studio.js';
import { renderTeam } from './team.js';
import { renderWork } from './work.js';
import { experimentsPanel, inferenceLabel } from './experiments.js';

// ---------------------------------------------------------------------------
// shared components
// ---------------------------------------------------------------------------
function statCard(label, value, detail, tone) {
  return h('div', { class: 'card tight stat' },
    h('div', { class: 'k' }, label),
    h('div', { class: 'v', style: tone ? `color:${tone}` : '' }, value),
    detail ? h('div', { class: 'd' }, detail) : null);
}

function scrollTable(label, ...sections) {
  return h('div', { class: 'table-scroll', role: 'region', 'aria-label': label, tabindex: '0' },
    h('table', { class: 'tbl' }, sections));
}

function requireAuth(main) {
  if (!state.token) {
    const next = new URLSearchParams({ next: location.pathname + location.search });
    mount(main, h('div', { class: 'card auth-card center' },
      h('h2', {}, 'Sign in to continue'),
      h('p', { class: 'muted' }, 'You need an account to generate and store 3D models.'),
      h('div', { class: 'row', style: 'justify-content:center' },
        h('a', { class: 'btn primary', href: `/app/login?${next}`, 'data-link': '' }, 'Sign in'),
        h('a', { class: 'btn', href: `/app/register?${next}`, 'data-link': '' }, 'Create account'))));
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// auth views
// ---------------------------------------------------------------------------
function renderAuth(main, initialMode) {
  const next = new URLSearchParams(location.search).get('next');
  const destination = next && next.startsWith('/app/') && !next.includes('\\') ? next : '/app/studio';
  let mode = 'login';
  const email = h('input', { type: 'email', id: 'email', required: 'required', autocomplete: 'email', placeholder: 'you@example.com' });
  const password = h('input', { type: 'password', id: 'password', required: 'required', autocomplete: 'current-password' });
  const name = h('input', { type: 'text', id: 'name', placeholder: 'Display name (optional)', maxlength: '80' });
  const mfa = h('input', { type: 'text', id: 'mfa', placeholder: '6-digit code (if enabled)', maxlength: '6', inputmode: 'numeric' });
  const errorBox = h('div', { class: 'form-error', role: 'alert' });
  const submit = h('button', { class: 'btn primary block', type: 'submit' }, 'Sign in');
  const nameField = h('label', { class: 'field hidden' }, h('span', {}, 'Display name'), name);
  const mfaField = h('label', { class: 'field hidden' }, h('span', {}, 'Authenticator code'), mfa);
  const switchBtn = h('a', { class: 'btn ghost sm', 'data-link': '' });
  const demo = h('button', { class: 'btn block mt8', type: 'button', onclick: () => {
    email.value = 'demo@3dfun.dev';
    password.value = 'DemoPassword123!';
    form.requestSubmit();
  } }, 'Use demo account');

  const form = h('form', { class: 'card auth-card' },
    h('h1', { style: 'font-size:24px' }, 'Welcome back'),
    h('p', { class: 'muted' }, 'Sign in to generate 3D models from photos.'),
    errorBox,
    h('label', { class: 'field' }, h('span', {}, 'Email'), email),
    nameField,
    h('label', { class: 'field' }, h('span', {}, 'Password'), password),
    mfaField,
    submit, state.meta?.features?.demo_account ? demo : null,
    h('div', { class: 'row mt16', style: 'justify-content:space-between;align-items:center' },
      switchBtn,
      h('a', { class: 'muted', href: '/app/forgot', 'data-link': '', style: 'font-size:13px' }, 'Forgot password?')));

  function setMode(nextMode) {
    mode = nextMode;
    form.querySelector('h1').textContent = mode === 'login' ? 'Welcome back' : 'Create your studio';
    form.querySelector('p').textContent = mode === 'login'
      ? 'Sign in to generate 3D models from photos.'
      : 'New accounts start with free generation credits.';
    submit.textContent = mode === 'login' ? 'Sign in' : 'Create account';
    nameField.classList.toggle('hidden', mode === 'login');
    demo.hidden = mode !== 'login';
    switchBtn.textContent = mode === 'login' ? 'Create an account' : 'Sign in instead';
    switchBtn.href = `${mode === 'login' ? '/app/register' : '/app/login'}?${new URLSearchParams({ next: destination })}`;
    password.setAttribute('autocomplete', mode === 'login' ? 'current-password' : 'new-password');
    password.minLength = mode === 'login' ? 0 : 10;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorBox.textContent = '';
    submit.disabled = demo.disabled = true;
    submit.innerHTML = '<span class="spinner"></span> Working…';
    try {
      if (mode === 'login') {
        const data = await api('/auth/login', {
          method: 'POST',
          body: { email: email.value, password: password.value, mfa_code: mfa.value || null },
        });
        localStorage.setItem('3dfun_token', data.access_token);
        state.token = data.access_token;
        await refreshUser();
        toast(`Signed in as ${data.user.display_name || data.user.email}`);
        navigate(destination);
      } else {
        const data = await api('/auth/register', {
          method: 'POST',
          body: { email: email.value, password: password.value, display_name: name.value, accept_terms: true },
        });
        localStorage.setItem('3dfun_token', data.access_token);
        state.token = data.access_token;
        await refreshUser();
        toast(`Account created · ${data.user.credits} credits added`);
        if (data.verification_token) {
          toast('Dev mode: email verification token printed to the API console', 'warn', 6000);
        }
        navigate(destination);
      }
    } catch (err) {
      errorBox.textContent = err.message + (err.detail && err.detail.details && err.detail.details[0]
        ? ` — ${err.detail.details[0].issue || err.detail.details[0].field}` : '');
      if (err.code === 'MFA_REQUIRED') { mfaField.classList.remove('hidden'); mfa.focus(); }
    } finally {
      submit.disabled = demo.disabled = false;
      submit.textContent = mode === 'login' ? 'Sign in' : 'Create account';
    }
  });
  setMode(initialMode);
  mount(main, form);
}

route('/app/login', (_p, main) => renderAuth(main, 'login'));
route('/app/register', (_p, main) => renderAuth(main, 'register'));

route('/app/forgot', async (_p, main) => {
  const email = h('input', { type: 'email', required: 'required', placeholder: 'you@example.com' });
  const out = h('div', { class: 'mt16' });
  const form = h('form', { class: 'card', style: 'max-width:430px;margin:24px auto' },
    h('h1', { style: 'font-size:24px' }, 'Reset your password'),
    h('p', { class: 'muted' }, 'We will email a reset link if that address exists.'),
    h('label', { class: 'field' }, h('span', {}, 'Email'), email),
    h('button', { class: 'btn primary block', type: 'submit' }, 'Send reset link'),
    out,
    h('p', { class: 'mt16' }, h('a', { href: '/app/login', 'data-link': '' }, '← Back to sign in')));
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      const data = await api('/auth/forgot-password', { method: 'POST', body: { email: email.value } });
      mount(out, h('div', { class: 'card tight' },
        h('p', { class: 'mb0' }, data.message),
        data.reset_token ? h('a', { class: 'btn sm mt8',
          href: `/app/reset-password?token=${encodeURIComponent(data.reset_token)}`, 'data-link': '' }, 'Open local reset link') : null));
    } catch (err) { toast(err.message, 'error'); }
  });
  mount(main, form);
});

route('/app/reset-password', (_p, main) => {
  const token = new URLSearchParams(location.search).get('token');
  if (!token) { renderError(main, new ApiError('Open the link from your password reset request.', 400)); return; }
  const password = h('input', { type: 'password', autocomplete: 'new-password', minlength: '10', maxlength: '200', required: true });
  const feedback = h('p', { class: 'form-error', role: 'alert' });
  const submit = h('button', { type: 'submit', class: 'btn primary block' }, 'Update password');
  const form = h('form', { class: 'card auth-card' }, h('h1', {}, 'Choose a new password'),
    h('label', { class: 'field' }, h('span', {}, 'New password'), password), feedback, submit);
  form.addEventListener('submit', async event => {
    event.preventDefault();
    submit.disabled = true;
    feedback.textContent = '';
    try {
      await api('/auth/reset-password', { method: 'POST', body: { token, password: password.value } });
      toast('Password updated. Sign in with your new password.');
      logout();
    } catch (err) {
      feedback.textContent = err.detail?.details?.[0]?.issue || err.message;
      submit.disabled = false;
    }
  });
  mount(main, form);
});

route('/app/verify-email', async (_p, main) => {
  const token = new URLSearchParams(location.search).get('token');
  const feedback = h('p', { role: 'status' }, 'Verifying your email…');
  mount(main, h('section', { class: 'card auth-card' }, h('h1', {}, 'Verify email'), feedback,
    h('a', { class: 'btn', href: '/app/studio', 'data-link': '' }, 'Open studio')));
  if (!token) { feedback.textContent = 'Open the verification link from your email.'; return; }
  try {
    await api('/auth/verify-email', { method: 'POST', body: { token } });
    feedback.textContent = 'Email verified.';
  } catch (err) { feedback.textContent = err.message; }
});

// ---------------------------------------------------------------------------
// studio
// ---------------------------------------------------------------------------
route('/app/studio', async (params, main) => {
  if (requireAuth(main)) await renderStudio(params, main);
});

route('/app/chat', async (params, main) => {
  if (requireAuth(main)) await renderChat(params, main);
});

// ---------------------------------------------------------------------------
// dashboard
// ---------------------------------------------------------------------------
route('/app/dashboard', async (_p, main) => {
  if (!requireAuth(main)) return;
  mount(main, h('div', { class: 'empty' }, h('div', { class: 'spinner' }), h('p', { class: 'mt16' }, 'Loading dashboard…')));
  const [gens, balance, tx] = await Promise.all([
    api('/generations?page_size=100'),
    api('/credits/balance'),
    api('/credits/transactions?limit=25'),
  ]);
  const items = gens.items;
  const completed = items.filter((g) => g.status === 'COMPLETED');
  const failed = items.filter((g) => g.status === 'FAILED');
  const durations = completed.map((g) => g.duration_ms || 0).filter(Boolean);
  const avg = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : 0;
  const totalCredits = completed.reduce((a, g) => a + (g.credits_charged || 0), 0);
  const finished = completed.length + failed.length;
  const successRate = finished ? completed.length / finished * 100 : 0;

  // daily volume for the last 14 days
  const days = [];
  for (let i = 13; i >= 0; i--) {
    const d = new Date(); d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    const bucket = items.filter((g) => String(g.created_at).slice(0, 10) === key);
    days.push({ key, total: bucket.length, ok: bucket.filter((g) => g.status === 'COMPLETED').length });
  }
  const maxDay = Math.max(1, ...days.map((d) => d.total));

  const chart = h('div', { class: 'bar', role: 'img', 'aria-label': `Generations per day for the last 14 days. Peak ${maxDay} per day.` },
    days.map((d) => h('i', {
      style: `height:${Math.max(3, (d.total / maxDay) * 100)}%;opacity:${d.total ? 1 : 0.25}`,
      title: `${d.key}: ${d.total} total, ${d.ok} completed`,
    })));

  mount(main,
    h('div', { class: 'spread wrap' },
      h('div', {}, h('h1', {}, 'Dashboard'),
        h('p', { class: 'muted mt0' }, `Signed in as ${esc(state.user?.email || '')}`)),
      h('div', { class: 'row gap8' },
        h('a', { class: 'btn', href: '/app/studio', 'data-link': '' }, 'New generation'),
        h('a', { class: 'btn', href: '/app/assets', 'data-link': '' }, 'My models'))),
    h('div', { class: 'grid c4 mt16' },
      statCard('Generations', fmt.n(items.length), `${completed.length} completed`),
      statCard('Success rate', fmt.pct(successRate), `${failed.length} failed`,
        successRate >= 90 ? 'var(--ok)' : successRate >= 70 ? 'var(--amber)' : 'var(--danger)'),
      statCard('Avg duration', fmt.ms(avg), 'wall clock, end to end'),
      statCard('Credits used', fmt.n(totalCredits), `${fmt.n(balance.available)} available`)),
    experimentsPanel(items, gens.total, state.meta?.inference, scrollTable),
    h('div', { class: 'grid c2 mt16' },
      h('div', { class: 'card' },
        h('div', { class: 'section-title' }, h('h2', {}, 'Volume · last 14 days'), h('span', { class: 'hint' }, 'generations per day')),
        chart,
        h('div', { class: 'legend mt8' }, h('span', {}, `${days.reduce((a, d) => a + d.total, 0)} total`))),
      h('div', { class: 'card' },
        h('div', { class: 'section-title' }, h('h2', {}, 'Credit ledger'), h('span', { class: 'hint' }, 'immutable, append-only')),
        tx.length === 0 ? emptyState('◆', 'No transactions yet', 'Your ledger fills up as you generate.')
          : scrollTable('Credit ledger',
            h('thead', {}, h('tr', {}, h('th', {}, 'When'), h('th', {}, 'Reason'), h('th', { class: 'num' }, 'Amount'), h('th', { class: 'num' }, 'Balance'))),
            h('tbody', {}, tx.slice(0, 8).map((t) => h('tr', {},
              h('td', {}, fmt.ago(t.created_at)),
              h('td', {}, t.reason.replaceAll('_', ' ').toLowerCase()),
              h('td', { class: 'num', style: t.amount < 0 ? 'color:var(--danger)' : 'color:var(--ok)' }, t.amount > 0 ? `+${t.amount}` : String(t.amount)),
              h('td', { class: 'num' }, fmt.n(t.balance_after)))))))),
    h('div', { class: 'card mt16' },
      h('div', { class: 'section-title' }, h('h2', {}, 'Recent generations'),
        h('a', { class: 'hint', href: '/app/assets', 'data-link': '' }, 'View all models →')),
      items.length === 0
        ? emptyState('◈', 'No generations yet', 'Upload a photo in the studio to create your first model.',
          h('a', { class: 'btn primary mt16', href: '/app/studio', 'data-link': '' }, 'Open studio'))
        : scrollTable('Recent generations',
          h('thead', {}, h('tr', {},
            h('th', {}, 'Created'), h('th', {}, 'Preset'), h('th', {}, 'Status'),
            h('th', { class: 'num' }, 'Progress'), h('th', { class: 'num' }, 'Duration'),
            h('th', { class: 'num' }, 'Credits'), h('th', {}, ''))),
          h('tbody', {}, items.slice(0, 12).map((g) => h('tr', {},
            h('td', {}, fmt.ago(g.created_at)),
            h('td', {}, g.preset.replaceAll('_', ' ')),
            h('td', {}, statusBadge(g.status)),
            h('td', { class: 'num' }, `${g.progress}%`),
            h('td', { class: 'num' }, fmt.ms(g.duration_ms)),
            h('td', { class: 'num' }, fmt.n(g.credits_charged)),
            h('td', {}, g.asset_id
              ? h('a', { class: 'btn sm', href: `/app/assets/${g.asset_id}`, 'data-link': '' }, 'Open')
              : h('a', { class: 'btn sm ghost', href: `/app/studio?generation=${g.id}`, 'data-link': '' }, 'Track')))))),
    h('div', { class: 'grid c2 mt16' },
      h('div', { class: 'card' },
        h('h3', {}, 'Credits'),
        h('div', { class: 'grid c2', style: 'gap:8px' },
          statCard('Balance', fmt.n(balance.balance), 'current wallet'),
          statCard('Reserved', fmt.n(balance.reserved), 'held for running jobs')),
        h('div', { class: 'row wrap gap8 mt16' },
          ['starter', 'studio', 'scale'].map((pack) => h('button', {
            class: 'btn sm', onclick: async () => {
              try {
                const r = await api('/billing/checkout', { method: 'POST', body: { pack } });
                toast(r.mode === 'development'
                  ? `Development checkout: ${r.credits} credits added`
                  : 'Redirecting to Stripe…');
                await refreshUser();
                navigate('/app/dashboard');
              } catch (err) { toast(err.message, 'error'); }
            },
          }, `${pack} pack`)))),
      h('div', { class: 'card' },
        h('h3', {}, 'Account security'),
        h('dl', { class: 'kv' },
          h('dt', {}, 'Email'), h('dd', {}, esc(state.user?.email || '')),
          h('dt', {}, 'Role'), h('dd', {}, esc(state.user?.role || '')),
          h('dt', {}, '2FA'), h('dd', {}, state.user?.mfa_enabled ? 'enabled' : 'not enabled'),
          h('dt', {}, 'Email verified'), h('dd', {}, state.user?.email_verified ? 'yes' : 'no')),
        h('div', { class: 'row wrap gap8 mt16' },
          h('a', { class: 'btn sm', href: '/app/settings', 'data-link': '' }, 'Manage sessions & 2FA'),
          h('button', { class: 'btn sm', onclick: () => logout() }, 'Sign out'))))));
});

// ---------------------------------------------------------------------------
// my models
// ---------------------------------------------------------------------------
route('/app/assets', async (_p, main) => {
  if (!requireAuth(main)) return;
  const search = h('input', { type: 'search', placeholder: 'Search models…', style: 'max-width:280px' });
  const sort = h('select', { style: 'max-width:170px' },
    [['newest', 'Newest'], ['oldest', 'Oldest'], ['polygons', 'Polygons'], ['likes', 'Likes']]
      .map(([v, l]) => h('option', { value: v }, l)));
  const grid = h('div', { class: 'grid auto mt16' });

  async function load() {
    mount(grid, h('div', { class: 'skeleton', style: 'height:220px' }), h('div', { class: 'skeleton', style: 'height:220px' }), h('div', { class: 'skeleton', style: 'height:220px' }));
    const data = await api(`/assets?page_size=60&q=${encodeURIComponent(search.value)}&sort=${sort.value}`);
    if (!data.items.length) {
      mount(grid, emptyState('▦', 'No models yet', 'Generate one from a photo in the studio.',
        h('a', { class: 'btn primary mt16', href: '/app/studio', 'data-link': '' }, 'Open studio')));
      return;
    }
    mount(grid, data.items.map(assetCard));
  }

  search.addEventListener('input', debounce(load, 300));
  sort.addEventListener('change', load);
  mount(main,
    h('div', { class: 'spread' },
      h('div', {}, h('h1', {}, 'My models'), h('p', { class: 'muted mt0' }, 'Every asset you have generated, newest first.')),
      h('div', { class: 'row gap8' }, search, sort)),
    grid);
  await load();
});

function assetCard(asset) {
  return h('div', {
    class: 'card asset-card hover', role: 'button', tabindex: '0',
    'aria-label': `${asset.name}, ${fmt.n(asset.polygons)} triangles`,
    onclick: () => navigate(`/app/assets/${asset.id}`),
    onkeydown: (e) => { if (e.key === 'Enter') navigate(`/app/assets/${asset.id}`); },
  },
    h('img', { src: asset.preview_url || '', alt: `Preview of ${asset.name}`, loading: 'lazy' }),
    h('div', { class: 'body' },
      h('div', { class: 'title truncate' }, asset.name),
      h('div', { class: 'sub' },
        h('span', {}, `${fmt.n(asset.polygons)} tris`),
        h('span', {}, fmt.bytes(asset.size_bytes)),
        h('span', {}, asset.watertight ? 'watertight' : 'open'),
        h('span', {}, fmt.ago(asset.created_at)))));
}

// ---------------------------------------------------------------------------
// asset detail
// ---------------------------------------------------------------------------
route('/app/assets/:id', async function assetDetail({ id }, main) {
  if (!requireAuth(main)) return;
  const asset = await api(`/assets/${id}`);
  const { ModelViewer } = await import('./viewer.js');
  if (state._viewer) { state._viewer.dispose(); state._viewer = null; }

  const m = asset.metrics || {};
  const eng = asset.manifest.engine || {};
  const num = (v, d = 3) => (typeof v === 'number' ? v.toFixed(d) : '—');

  const nameInput = h('input', { type: 'text', value: asset.name, maxlength: '140' });
  const descInput = h('textarea', { rows: '3', maxlength: '2000' }, asset.description || '');
  const tagsInput = h('input', { type: 'text', value: (asset.tags || []).join(', '), placeholder: 'product, sneaker, teal' });
  const visSelect = h('select', {}, ['PRIVATE', 'UNLISTED', 'PUBLIC'].map((v) =>
    h('option', { value: v, selected: asset.visibility === v ? 'selected' : null }, v.toLowerCase())));

  const saveBtn = h('button', { class: 'btn primary sm', onclick: async () => {
    try {
      await api(`/assets/${id}`, { method: 'PATCH', body: {
        name: nameInput.value,
        description: descInput.value,
        tags: tagsInput.value.split(',').map((t) => t.trim()).filter(Boolean),
        visibility: visSelect.value,
      } });
      toast('Saved');
      navigate(`/app/assets/${id}`);
    } catch (err) { toast(err.message, 'error'); }
  } }, 'Save changes');

  const downloadLinks = (asset.downloads || []).map((d) => h('a', {
    class: 'btn', href: d.url, style: 'justify-content:space-between',
  }, h('span', {}, d.label), h('small', { class: 'mono muted' }, fmt.bytes(d.size_bytes))));

  const timingRows = Object.entries(asset.manifest.timings || {}).map(([k, v]) =>
    h('tr', {}, h('td', {}, k), h('td', { class: 'num' }, `${(v * 1000).toFixed(0)} ms`)));

  const warnings = (asset.manifest.warnings || []).length
    ? h('div', { class: 'card tight mt16', style: 'border-color:#efd5aa' },
        h('b', {}, 'Warnings'),
        h('ul', { style: 'margin:8px 0 0 18px;padding:0;color:var(--amber)' },
          asset.manifest.warnings.map((w) => h('li', {}, w))))
    : null;

  const report = h('dl', { class: 'kv' },
    h('dt', {}, 'Watertight'), h('dd', {}, asset.watertight ? 'yes' : 'no'),
    h('dt', {}, 'Boundary edges'), h('dd', {}, fmt.n(m.boundary_edges)),
    h('dt', {}, 'Non-manifold edges'), h('dd', {}, fmt.n(m.nonmanifold_edges)),
    h('dt', {}, 'Euler characteristic'), h('dd', {}, fmt.n(m.euler_characteristic)),
    h('dt', {}, 'Aspect ratio p95'), h('dd', {}, num(m.aspect_ratio_p95, 2)),
    h('dt', {}, 'Surface area'), h('dd', {}, num(m.surface_area, 4)),
    h('dt', {}, 'Colour'), h('dd', {}, m.vertex_colors ? 'Vertex colours' : m.texture_size ? 'Texture atlas' : '—'),
    typeof m.uv_coverage === 'number' ? [h('dt', {}, 'Atlas coverage'), h('dd', {}, fmt.pct(m.uv_coverage * 100))] : null,
    h('dt', {}, 'Engine'), h('dd', {}, eng.name || '—'),
    h('dt', {}, 'Inference'), h('dd', {}, inferenceLabel(asset.manifest.hardware)),
    eng.depth_model ? [h('dt', {}, 'Depth model'), h('dd', {}, eng.depth_model)] : null,
    h('dt', {}, 'Config hash'), h('dd', { class: 'mono' }, (asset.manifest.config_hash || '—').slice(0, 16)));

  const viewerHost = h('div');
  const sourceHost = h('div', { class: 'muted' }, asset.generation_id ? 'Loading source…' : 'No linked generation.');

  mount(main,
    h('div', { class: 'spread wrap' },
      h('div', {},
        h('h1', {}, asset.name),
        h('p', { class: 'muted mt0' },
          `${fmt.n(asset.polygons)} triangles · ${fmt.n(asset.vertices)} vertices · `
          + `${fmt.bytes(asset.size_bytes)} · ${fmt.date(asset.created_at)}`)),
      h('div', { class: 'row gap8' },
        h('a', { class: 'btn primary', href: `/api/v1/assets/${id}/download?format=glb` }, 'Download GLB'),
        h('a', { class: 'btn', href: '/app/assets', 'data-link': '' }, 'Back'))),

    h('div', { class: 'grid asset-detail-grid mt16' },
      h('div', { class: 'col' },
        h('div', { class: 'card', style: 'padding:0;overflow:hidden' }, viewerHost),
        h('div', { class: 'card' },
          h('h2', {}, 'Downloads'),
          h('div', { class: 'grid c2', style: 'gap:8px' }, downloadLinks)),
        asset.turntable_url
          ? h('div', { class: 'card' }, h('h3', {}, 'Turntable'),
              asset.turntable_url.endsWith('.gif')
                ? h('img', { src: asset.turntable_url, alt: 'Animated model turntable', style: 'width:100%;border-radius:10px' })
                : h('video', { src: asset.turntable_url, controls: 'controls', loop: 'loop', muted: 'muted', playsinline: 'playsinline', style: 'width:100%;border-radius:10px' }))
          : null,
        h('div', { class: 'card' },
          h('h3', {}, 'Geometry & texture report'),
          report,
          timingRows.length ? h('div', { class: 'mt16' }, h('h4', {}, 'Stage timings'), h('table', { class: 'tbl' }, h('tbody', {}, timingRows))) : null,
          warnings)),

      h('div', { class: 'col' },
        h('div', { class: 'card' },
          h('h3', {}, 'Metadata'),
          h('label', { class: 'field' }, h('span', {}, 'Name'), nameInput),
          h('label', { class: 'field' }, h('span', {}, 'Description'), descInput),
          h('label', { class: 'field' }, h('span', {}, 'Tags'), tagsInput),
          h('label', { class: 'field' }, h('span', {}, 'Visibility'), visSelect),
          saveBtn),
        h('div', { class: 'card' },
          h('h3', {}, 'Share'),
          h('dl', { class: 'kv' },
            h('dt', {}, 'Share link'), h('dd', {}, asset.visibility === 'PRIVATE' ? 'Available when unlisted or public'
              : h('a', { href: `/m/${asset.share_slug}` }, 'Open share link')),
            h('dt', {}, 'Visibility'), h('dd', {}, String(asset.visibility).toLowerCase()),
            h('dt', {}, 'Likes'), h('dd', {}, fmt.n(asset.like_count))),
          h('div', { class: 'row gap8 mt16' },
            h('button', { class: 'btn sm', disabled: asset.visibility === 'PRIVATE', onclick: () => copyText(`${location.origin}/m/${asset.share_slug}`) }, 'Copy link'),
            h('button', { class: 'btn sm', disabled: asset.visibility !== 'PUBLIC', onclick: async () => {
              try { const r = await api(`/assets/${id}/embed`); copyText(r.embed_html); }
              catch (err) { toast(err.message, 'error'); }
            } }, 'Copy embed code'))),
        h('div', { class: 'card' }, h('h3', {}, 'Source image'), sourceHost),
        h('div', { class: 'card' },
          h('h3', {}, 'Danger zone'),
          h('button', { class: 'btn danger sm', onclick: async () => {
            if (!confirm('Delete this model? You can restore it for 7 days.')) return;
            try {
              await api(`/assets/${id}`, { method: 'DELETE' });
              toast('Model deleted');
              navigate('/app/assets');
            } catch (err) { toast(err.message, 'error'); }
          } }, 'Delete model'),
          h('small', { class: 'muted', style: 'display:block;margin-top:8px' },
            'Deletion is soft for 7 days, then purged by the cleanup job.')))));

  if (asset.generation_id) {
    api(`/generations/${asset.generation_id}`).then(generation => {
      mount(sourceHost, generation.source_image_url ? h('img', {
        class: 'asset-source', src: generation.source_image_url, alt: 'Reference image used to generate the model',
      }) : 'The source image is unavailable.');
    }).catch(() => { sourceHost.textContent = 'The source image is unavailable.'; });
  }

  state._viewer = new ModelViewer(viewerHost, {
    modelUrl: asset.model_url,
    previewUrl: asset.preview_url,
    altText: `Interactive 3D model: ${asset.name}. ${fmt.n(asset.polygons)} triangles.`,
    height: 460,
  });
});

// ---------------------------------------------------------------------------
// gallery
// ---------------------------------------------------------------------------
route('/app/gallery', async (_p, main) => {
  const search = h('input', { type: 'search', placeholder: 'Search public models…', style: 'max-width:300px' });
  const sort = h('select', { style: 'max-width:160px' },
    [['newest', 'Newest'], ['popular', 'Most liked'], ['trending', 'Trending']]
      .map(([v, l]) => h('option', { value: v }, l)));
  const grid = h('div', { class: 'grid auto mt16' });
  async function load() {
    const data = await api(`/gallery?page_size=48&q=${encodeURIComponent(search.value)}&sort=${sort.value}`);
    if (!data.items.length) {
      mount(grid, emptyState('◍', 'The gallery is empty', 'Publish a model from its detail page to appear here.'));
      return;
    }
    mount(grid, data.items.map((a) => h('div', { class: 'card asset-card hover', role: 'button', tabindex: '0',
      onclick: () => navigate(`/app/assets/${a.id}`),
      onkeydown: (e) => { if (e.key === 'Enter') navigate(`/app/assets/${a.id}`); } },
      h('img', { src: a.preview_url || '', alt: `Preview of ${a.name}`, loading: 'lazy' }),
      h('div', { class: 'body' },
        h('div', { class: 'title truncate' }, a.name),
        h('div', { class: 'sub' },
          h('span', {}, `by ${a.owner_name}`),
          h('span', {}, `♥ ${a.like_count}`),
          h('span', {}, `${fmt.n(a.polygons)} tris`))))));
  }
  search.addEventListener('input', debounce(load, 300));
  sort.addEventListener('change', load);
  mount(main,
    h('div', { class: 'spread' },
      h('div', {}, h('h1', {}, 'Gallery'), h('p', { class: 'muted mt0' }, 'Public models from the community.')),
      h('div', { class: 'row gap8' }, search, sort)),
    grid);
  await load();
});

// ---------------------------------------------------------------------------
// settings
// ---------------------------------------------------------------------------
route('/app/settings', async (_p, main) => {
  if (!requireAuth(main)) return;
  const sessionsHost = h('div', { class: 'mt16' });
  const mfaHost = h('div', { class: 'mt16' });
  async function loadSessions() {
    const sessions = await api('/auth/sessions');
    mount(sessionsHost, h('table', { class: 'tbl' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Device'), h('th', {}, 'IP'), h('th', {}, 'Last seen'), h('th', {}, ''))),
      h('tbody', {}, sessions.map((s) => h('tr', {},
        h('td', {}, s.current ? h('b', {}, 'This device') : h('span', { class: 'truncate' }, s.user_agent.slice(0, 48) || 'unknown')),
        h('td', { class: 'mono' }, s.ip || '—'),
        h('td', {}, fmt.ago(s.last_seen_at)),
        h('td', {}, s.current ? null : h('button', {
          class: 'btn sm danger', onclick: async () => {
            await api(`/auth/sessions/${s.id}`, { method: 'DELETE' });
            toast('Session revoked');
            loadSessions();
          },
        }, 'Revoke')))))));
  }
  const setupOut = h('div');
  mount(mfaHost,
    h('p', { class: 'muted' }, state.user?.mfa_enabled
      ? 'Two-factor authentication is enabled.'
      : 'Protect your account with a TOTP authenticator app.'),
    h('button', {
      class: 'btn sm', onclick: async () => {
        const data = await api('/auth/mfa/setup', { method: 'POST' });
        mount(setupOut,
          h('p', { class: 'mono', style: 'background:var(--bg-2);padding:10px;border-radius:8px;word-break:break-all' }, data.secret),
          h('p', { class: 'muted' }, 'Add this secret to your authenticator, then confirm with a code:'),
          h('div', { class: 'row gap8' },
            h('input', { type: 'text', id: 'mfa-code', placeholder: '123456', maxlength: '6', style: 'max-width:140px' }),
            h('button', {
              class: 'btn sm primary', onclick: async () => {
                try {
                  await api('/auth/mfa/enable', { method: 'POST', body: { code: $('#mfa-code').value } });
                  toast('Two-factor enabled');
                  await refreshUser();
                  navigate('/app/settings');
                } catch (err) { toast(err.message, 'error'); }
              },
            }, 'Enable')));
      },
    }, 'Set up two-factor'),
    setupOut);
  mount(main,
    h('h1', {}, 'Account settings'),
    h('div', { class: 'grid c2 mt16' },
      h('div', { class: 'card' },
        h('h3', {}, 'Profile'),
        h('dl', { class: 'kv' },
          h('dt', {}, 'Email'), h('dd', {}, esc(state.user?.email || '')),
          h('dt', {}, 'Display name'), h('dd', {}, esc(state.user?.display_name || '')),
          h('dt', {}, 'Role'), h('dd', {}, esc(state.user?.role || '')),
          h('dt', {}, 'Member since'), h('dd', {}, fmt.date(state.user?.created_at)))),
      h('div', { class: 'card' }, h('h3', {}, 'Two-factor authentication'), mfaHost)),
    h('div', { class: 'card mt16' }, h('h3', {}, 'Active sessions'), sessionsHost),
    h('div', { class: 'card mt16' },
      h('h3', {}, 'Data & deletion'),
      h('p', { class: 'muted' }, 'Deleting your account deactivates it immediately and purges assets after the retention window.'),
      h('button', {
        class: 'btn danger sm', onclick: async () => {
          if (!confirm('Delete your account? This cannot be undone after the retention window.')) return;
          try {
            await api('/auth/me', { method: 'DELETE' });
            logout(false);
            toast('Account deleted');
            navigate('/app/login');
          } catch (err) { toast(err.message, 'error'); }
        },
      }, 'Delete account')));
  await loadSessions();
});

// ---------------------------------------------------------------------------
// admin
// ---------------------------------------------------------------------------
route('/app/admin', async function adminConsole(_p, main) {
  if (!requireAuth(main)) return;
  if (state.user?.role !== 'admin') {
    mount(main, emptyState('⛨', 'Administrators only', 'Your account does not have the admin role.'));
    return;
  }

  const [overview, users, jobs, providers, flags, audit, mod] = await Promise.all([
    api('/admin/overview'), api('/admin/users?page_size=20'), api('/admin/jobs?page_size=15'),
    api('/admin/providers'), api('/admin/feature-flags'), api('/admin/audit-logs?page_size=12'),
    api('/admin/moderation'),
  ]);

  const metrics = overview.metrics || {};
  const latency = Object.entries(metrics.histograms || {})
    .filter(([k]) => k.startsWith('http_request_duration'))
    .map(([k, v]) => h('tr', {},
      h('td', { class: 'mono' }, k.slice(0, 42)),
      h('td', { class: 'num' }, fmt.n(v.count)),
      h('td', { class: 'num' }, `${v.p50} ms`),
      h('td', { class: 'num' }, `${v.p95} ms`),
      h('td', { class: 'num' }, `${v.p99} ms`)));

  const latencyTable = h('table', { class: 'tbl' },
    h('thead', {}, h('tr', {},
      h('th', {}, 'Metric'), h('th', { class: 'num' }, 'n'),
      h('th', { class: 'num' }, 'p50'), h('th', { class: 'num' }, 'p95'), h('th', { class: 'num' }, 'p99'))),
    h('tbody', {}, latency.length
      ? latency
      : h('tr', {}, h('td', { colspan: '5', class: 'muted' }, 'No requests measured yet.'))));

  const providerRows = providers.map((p) => h('tr', {},
    h('td', {}, h('b', {}, p.display_name), h('div', { class: 'mono muted' }, p.slug)),
    h('td', {}, p.kind),
    h('td', {}, p.healthy
      ? h('span', { class: 'badge completed' }, 'healthy')
      : h('span', { class: 'badge failed' }, 'down')),
    h('td', { class: 'num' }, p.weight),
    h('td', { class: 'num' }, fmt.n(p.generations)),
    h('td', { class: 'num' }, p.success_rate === null ? '—' : `${p.success_rate}%`),
    h('td', { class: 'muted' }, p.notes || '—')));

  const usersTable = h('table', { class: 'tbl' },
    h('thead', {}, h('tr', {},
      h('th', {}, 'Email'), h('th', {}, 'Role'),
      h('th', { class: 'num' }, 'Credits'), h('th', { class: 'num' }, 'Jobs'), h('th', {}, 'Active'))),
    h('tbody', {}, users.items.map((u) => h('tr', {},
      h('td', { class: 'truncate' }, u.email),
      h('td', {}, u.role),
      h('td', { class: 'num' }, fmt.n(u.credits)),
      h('td', { class: 'num' }, fmt.n(u.generations)),
      h('td', {}, u.is_active ? 'yes' : 'no')))));

  const flagRows = flags.map((f) => h('tr', {},
    h('td', {}, h('div', {}, f.key), h('small', { class: 'muted' }, f.description)),
    h('td', {}, h('button', {
      class: `btn sm ${f.enabled ? 'primary' : ''}`,
      onclick: async (e) => {
        try {
          await api(`/admin/feature-flags/${f.id}`, { method: 'PATCH', body: { enabled: !f.enabled } });
          e.currentTarget.textContent = f.enabled ? 'off' : 'on';
          toast(`${f.key} toggled`);
        } catch (err) { toast(err.message, 'error'); }
      },
    }, f.enabled ? 'on' : 'off')),
    h('td', { class: 'num' }, `${f.rollout_pct}%`)));

  const reportCard = (r) => h('div', { class: 'card tight mt8' },
    h('b', {}, r.reason),
    h('p', { class: 'mb0 muted' }, r.details || 'No details'),
    h('div', { class: 'row gap8 mt8' },
      h('button', { class: 'btn sm', onclick: async () => {
        try {
          await api(`/admin/moderation/${r.asset_id}`, { method: 'POST', body: { decision: 'APPROVE' } });
          toast('Approved'); navigate('/app/admin');
        } catch (err) { toast(err.message, 'error'); }
      } }, 'Approve'),
      h('button', { class: 'btn sm danger', onclick: async () => {
        try {
          await api(`/admin/moderation/${r.asset_id}`, { method: 'POST', body: { decision: 'REJECT' } });
          toast('Rejected'); navigate('/app/admin');
        } catch (err) { toast(err.message, 'error'); }
      } }, 'Reject')));

  const moderationBody = (mod.reports.length === 0 && mod.pending_assets.length === 0)
    ? h('p', { class: 'muted' }, 'Queue is clear.')
    : h('div', {}, mod.reports.map(reportCard));

  const auditTable = h('table', { class: 'tbl' },
    h('thead', {}, h('tr', {}, h('th', {}, 'When'), h('th', {}, 'Action'), h('th', {}, 'Target'))),
    h('tbody', {}, audit.items.map((a) => h('tr', {},
      h('td', {}, fmt.ago(a.created_at)),
      h('td', { class: 'mono' }, a.action),
      h('td', { class: 'mono truncate' }, `${a.target_type}:${a.target_id}`.slice(0, 26))))));

  const jobsTable = h('table', { class: 'tbl' },
    h('thead', {}, h('tr', {},
      h('th', {}, 'Created'), h('th', {}, 'User'), h('th', {}, 'Status'),
      h('th', {}, 'Stage'), h('th', { class: 'num' }, 'Progress'), h('th', { class: 'num' }, 'Duration'))),
    h('tbody', {}, jobs.items.map((j) => h('tr', {},
      h('td', {}, fmt.ago(j.created_at)),
      h('td', { class: 'truncate' }, j.user_email),
      h('td', {}, statusBadge(j.status)),
      h('td', { class: 'muted' }, j.stage),
      h('td', { class: 'num' }, `${j.progress}%`),
      h('td', { class: 'num' }, fmt.ms(j.duration_ms))))));

  mount(main,
    h('div', { class: 'spread' },
      h('div', {}, h('h1', {}, 'Administration'), h('p', { class: 'muted mt0' }, 'Live state of this deployment.')),
      h('div', { class: 'row gap8' },
        h('a', { class: 'btn sm', href: '/api/v1/metrics', target: '_blank' }, 'Prometheus metrics'),
        h('a', { class: 'btn sm', href: '/docs', target: '_blank' }, 'OpenAPI docs'))),
    h('div', { class: 'grid c4 mt16' },
      statCard('Users', fmt.n(overview.users), `${overview.assets} assets`),
      statCard('Generations', fmt.n(overview.generations), `${overview.completed} completed`),
      statCard('Success rate', fmt.pct(overview.success_rate), `${overview.failed} failed`),
      statCard('Storage', fmt.bytes(overview.storage_bytes), `${overview.runner.backend.kind} runner`)),
    h('div', { class: 'grid c2 mt16' },
      h('div', { class: 'card' },
        h('h3', {}, 'Queue'),
        h('dl', { class: 'kv' },
          h('dt', {}, 'Queued'), h('dd', {}, fmt.n(overview.queue.queued)),
          h('dt', {}, 'Running'), h('dd', {}, fmt.n(overview.queue.running)),
          h('dt', {}, 'Completed today'), h('dd', {}, fmt.n(overview.queue.completed_today)),
          h('dt', {}, 'Failed today'), h('dd', {}, fmt.n(overview.queue.failed_today)),
          h('dt', {}, 'In-flight (worker)'), h('dd', {}, fmt.n(overview.runner.inflight)),
          h('dt', {}, 'Backend'), h('dd', { class: 'mono' }, overview.runner.backend.kind),
          h('dt', {}, 'Redis configured'), h('dd', {}, String(overview.runner.backend.redis_configured)))),
      h('div', { class: 'card' }, h('h3', {}, 'Request latency'), latencyTable)),
    h('div', { class: 'card mt16' },
      h('div', { class: 'section-title' }, h('h2', {}, 'Providers'), h('span', { class: 'hint' }, 'routing + health probes')),
      h('table', { class: 'tbl' },
        h('thead', {}, h('tr', {},
          h('th', {}, 'Provider'), h('th', {}, 'Kind'), h('th', {}, 'Health'),
          h('th', { class: 'num' }, 'Weight'), h('th', { class: 'num' }, 'Jobs'),
          h('th', { class: 'num' }, 'Success'), h('th', {}, 'Notes'))),
        h('tbody', {}, providerRows))),
    h('div', { class: 'grid c2 mt16' },
      h('div', { class: 'card' }, h('h3', {}, 'Users'), usersTable),
      h('div', { class: 'card' },
        h('h3', {}, 'Feature flags'),
        h('table', { class: 'tbl' },
          h('thead', {}, h('tr', {},
            h('th', {}, 'Key'), h('th', {}, 'Enabled'), h('th', { class: 'num' }, 'Rollout'))),
          h('tbody', {}, flagRows)))),
    h('div', { class: 'grid c2 mt16' },
      h('div', { class: 'card' }, h('h3', {}, 'Moderation'), moderationBody),
      h('div', { class: 'card' }, h('h3', {}, 'Audit log'), auditTable)),
    h('div', { class: 'card mt16' },
      h('div', { class: 'section-title' }, h('h2', {}, 'Recent jobs'), h('span', { class: 'hint' }, 'all users')),
      jobsTable));
});

// ---------------------------------------------------------------------------
// public share page
// ---------------------------------------------------------------------------
async function renderSharedModel({ slug }, main) {
  const asset = await api(`/gallery/share/${encodeURIComponent(slug)}`).catch(err => {
    if (err.status === 404) return null;
    throw err;
  });
  if (!asset) {
    mount(main, emptyState('◇', 'Model not available', 'This link may be private or removed.'));
    return;
  }
  const { ModelViewer } = await import('./viewer.js');
  const host = h('div');
  mount(main,
    h('div', { class: 'spread' },
      h('div', {}, h('h1', {}, asset.name), h('p', { class: 'muted mt0' }, `by ${asset.owner_name} · ${fmt.date(asset.created_at)}`)),
      h('a', { class: 'btn primary', href: `/api/v1/assets/${asset.id}/download?format=glb` }, 'Download GLB')),
    h('div', { class: 'card mt16', style: 'padding:0;overflow:hidden' }, host),
    h('p', { class: 'alt-desc' }, `${fmt.n(asset.polygons)} triangles · ${fmt.bytes(asset.size_bytes)}`));
  state._viewer = new ModelViewer(host, { modelUrl: asset.model_url, previewUrl: asset.preview_url, height: 520 });
}

route('/m/:slug', renderSharedModel);
route('/embed/:slug', renderSharedModel);


// ---------------------------------------------------------------------------
// benchmark / results page  (every number comes from results/, never typed)
// ---------------------------------------------------------------------------
route('/app/benchmark', async (_p, main) => {
  const data = await api('/results/summary').catch(() => ({ available: false }));
  if (!data.available) {
    mount(main,
      h('h1', {}, 'Benchmarks'),
      h('p', { class: 'muted' }, 'No results yet. Run the harness, then reload this page.'),
      h('pre', { class: 'code-block' }, 'make benchmark'),
      emptyState('◇', 'results/ is empty', 'make benchmark runs 25 images x 3 methods x 3 repetitions.'));
    return;
  }
  const METHOD_LABEL = { proposed: '3DFUN AI (ours)', tsdf: 'TSDF fusion', visual_hull: 'Visual hull' };
  const num = (v, digits = 3) => (v === undefined || v === null || v === '' || Number.isNaN(Number(v))
    ? '—' : Number(v).toFixed(digits));
  const methods = data.methods || [];
  const best = (key, lower = true) => {
    const vals = methods.map((m) => Number(m[key])).filter((v) => Number.isFinite(v));
    if (!vals.length) return null;
    return lower ? Math.min(...vals) : Math.max(...vals);
  };
  const cdBest = best('chamfer_mean', true);
  const fsBest = best('f_score_mean', false);
  const ssimBest = best('ssim_mean', false);

  const table = scrollTable('Quantitative comparison',
    h('thead', {}, h('tr', {},
      h('th', {}, 'Method'),
      h('th', { class: 'num' }, 'Chamfer ↓'),
      h('th', { class: 'num' }, 'F-score ↑'),
      h('th', { class: 'num' }, 'IoU ↑'),
      h('th', { class: 'num' }, 'NC ↑'),
      h('th', { class: 'num' }, 'SSIM ↑'),
      h('th', { class: 'num' }, 'Silhouette IoU'),
      h('th', { class: 'num' }, 'Time (s)'),
      h('th', { class: 'num' }, 'Watertight')),
    h('tbody', {}, methods.map((m) => h('tr', {},
      h('td', {}, h('span', { class: `method-tag ${m.method}` }, METHOD_LABEL[m.method] || m.method)),
      h('td', { class: 'num' }, `${num(m.chamfer_mean, 4)} ± ${num(m.chamfer_std, 4)}`),
      h('td', { class: 'num' }, num(m.f_score_mean)),
      h('td', { class: 'num' }, num(m.iou_mean)),
      h('td', { class: 'num' }, num(m.normal_consistency_mean)),
      h('td', { class: 'num' }, num(m.ssim_mean)),
      h('td', { class: 'num' }, num(m.silhouette_iou_mean)),
      h('td', { class: 'num' }, `${num(m.seconds_mean, 2)} ± ${num(m.seconds_std, 2)}`),
      h('td', { class: 'num' }, `${num(m.watertight_pct, 0)}%`))))));

  const chart = (key, lower, valueLabel) => {
    const values = methods.map((m) => Number(m[key]) || 0);
    const max = Math.max(...values) || 1;
    return h('div', { class: 'bars' }, methods.map((m, i) => h('div', {
      class: `bar-row${((lower && values[i] === Math.min(...values.filter(Boolean))) || (!lower && values[i] === max)) ? ' win' : ''}`,
    },
    h('span', { class: `method-tag ${m.method}` }, METHOD_LABEL[m.method] || m.method),
    h('span', { class: 'bar-track' }, h('i', { style: `width:${(values[i] / max) * 100}%` })),
    h('span', { class: 'val' }, `${num(values[i], lower ? 4 : 3)}`))));
  };

  const ablation = (data.ablation || []).map((row) => h('tr', {},
    h('td', {}, row.config),
    h('td', { class: 'num' }, `${num(row.chamfer_mean, 4)} ± ${num(row.chamfer_std, 4)}`),
    h('td', { class: 'num' }, num(row.f_score_mean)),
    h('td', { class: 'num' }, num(row.iou_mean)),
    h('td', { class: 'num' }, num(row.ssim_mean)),
    h('td', { class: 'num' }, `${num(row.seconds_mean, 2)}`)));

  const stages = data.stage_timings || [];
  const stageTotal = stages.reduce((acc, s) => acc + Number(s.mean_s || 0), 0) || 1;

  mount(main,
    h('div', { class: 'spread wrap' },
      h('div', {}, h('h1', {}, 'Benchmarks'),
        h('p', { class: 'muted mt0' },
          `Dataset v${(data.dataset || {}).version} · ${(data.dataset || {}).scenes?.total || 0} images · `
          + `seed ${(data.dataset || {}).seed} · checksum ${(data.dataset || {}).checksum}`)),
      h('a', { class: 'btn', href: '/api/v1/results/summary', target: '_blank' }, 'Raw JSON')),
    h('div', { class: 'grid c4 mt16' },
      statCard('Chamfer (CD)', num(cdBest, 4), 'lower is better · ICT-normalised'),
      statCard('F-score @ 1%', num(fsBest, 3), 'harmonic mean of precision/recall'),
      statCard('Best SSIM', num(ssimBest, 3), 'reference-view render'),
      statCard('Winner', methods.find((m) => Number(m.chamfer_mean) === cdBest)?.method === 'proposed' ? '3DFUN AI' : 'baseline', 'by Chamfer distance')),
    h('div', { class: 'card mt16' }, h('div', { class: 'section-title' },
      h('h3', { class: 'mt0 mb0' }, 'Quantitative comparison'), h('span', { class: 'hint' }, 'mean ± std over repetitions')),
      table),
    h('div', { class: 'grid c2 mt16' },
      h('div', { class: 'card' }, h('div', { class: 'section-title' }, h('h3', { class: 'mt0 mb0' }, 'Chamfer distance')), chart('chamfer_mean', true)),
      h('div', { class: 'card' }, h('div', { class: 'section-title' }, h('h3', { class: 'mt0 mb0' }, 'F-score @ 1%')), chart('f_score_mean', false))),
    h('div', { class: 'grid c2 mt16' },
      h('div', { class: 'card' }, h('div', { class: 'section-title' }, h('h3', { class: 'mt0 mb0' }, 'Volumetric IoU')), chart('iou_mean', false)),
      h('div', { class: 'card' }, h('div', { class: 'section-title' }, h('h3', { class: 'mt0 mb0' }, 'Render SSIM')), chart('ssim_mean', false))),
    ablation.length ? h('div', { class: 'card mt16' },
      h('div', { class: 'section-title' }, h('h3', { class: 'mt0 mb0' }, 'Ablation'),
        h('span', { class: 'hint' }, 'same engine, one setting changed at a time')),
      scrollTable('Ablation results',
        h('thead', {}, h('tr', {}, h('th', {}, 'Configuration'), h('th', { class: 'num' }, 'Chamfer ↓'),
          h('th', { class: 'num' }, 'F-score ↑'), h('th', { class: 'num' }, 'IoU ↑'),
          h('th', { class: 'num' }, 'SSIM ↑'), h('th', { class: 'num' }, 'Time (s)'))),
        h('tbody', {}, ablation))) : null,
    stages.length ? h('div', { class: 'card mt16' },
      h('div', { class: 'section-title' }, h('h3', { class: 'mt0 mb0' }, 'Where the time goes'), h('span', { class: 'hint' }, 'proposed pipeline, per stage')),
      h('div', { class: 'bars' }, stages.map((s) => h('div', { class: 'bar-row' },
        h('span', { class: 'mono' }, s.stage),
        h('span', { class: 'bar-track' }, h('i', { style: `width:${(Number(s.mean_s) / stageTotal) * 100}%` })),
        h('span', { class: 'val' }, `${num(s.mean_s, 2)} s`))))) : null,
    (data.failures && data.failures.length) ? h('div', { class: 'card mt16' },
      h('div', { class: 'section-title' }, h('h3', { class: 'mt0 mb0' }, 'Failure cases'),
        h('span', { class: 'hint' }, `${data.failures.length} runs below the stated thresholds`)),
      scrollTable('Failure cases',
        h('thead', {}, h('tr', {}, h('th', {}, 'Scene'), h('th', {}, 'Method'), h('th', {}, 'Why'),
          h('th', { class: 'num' }, 'Chamfer'), h('th', { class: 'num' }, 'F-score'))),
        h('tbody', {}, data.failures.map((f) => h('tr', {},
          h('td', { class: 'mono' }, f.entry_id), h('td', {}, f.method), h('td', { class: 'muted' }, f.reason),
          h('td', { class: 'num' }, num(f.chamfer, 4)), h('td', { class: 'num' }, num(f.f_score))))))) : null,
    data.environment ? h('div', { class: 'card mt16' },
      h('div', { class: 'section-title' }, h('h3', { class: 'mt0 mb0' }, 'Environment'), h('span', { class: 'hint' }, 'recorded with the run')),
      h('dl', { class: 'kv' },
        h('dt', {}, 'Host'), h('dd', { class: 'mono' }, `${data.environment.cpu} · ${data.environment.cpu_count} cores`),
        h('dt', {}, 'Memory'), h('dd', { class: 'mono' }, data.environment.memory?.MemTotal || '—'),
        h('dt', {}, 'GPU'), h('dd', { class: 'mono' }, data.environment.gpu),
        h('dt', {}, 'Python'), h('dd', { class: 'mono' }, data.environment.python),
        h('dt', {}, 'Commit'), h('dd', { class: 'mono' }, data.environment.git_commit),
        h('dt', {}, 'Seed'), h('dd', { class: 'mono' }, String(data.environment.seed)),
        Object.entries(data.environment.libraries || {}).slice(0, 8).map(([k, v]) =>
          [h('dt', {}, k), h('dd', { class: 'mono' }, v)]))) : null,
    h('p', { class: 'alt-desc mt16' },
      'Reproduce with one command: ', h('code', {}, 'make benchmark'),
      ' — every figure on this page is read from results/summary/*.csv.'));
});

route('/app/team', async (_p, main) => { renderTeam(main); });
route('/app/work', async (_p, main) => { renderWork(main); });

route('/app/', async () => { navigate('/app/studio'); });
route('/app', async () => { navigate('/app/studio'); });

// ---------------------------------------------------------------------------
// bootstrap
// ---------------------------------------------------------------------------
async function boot() {
  initRouter();
  document.getElementById('auth-logout').addEventListener('click', () => logout());
  try { state.meta = await api('/meta'); } catch { /* offline */ }
  await refreshUser();
  const path = location.pathname.replace(/\/$/, '');
  if (path === '' || path === '/app') navigate('/app/studio', true);
  else await renderRoute();
  // keep wallet fresh when returning to the tab
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && state.token) refreshUser();
  });
}

boot();

export { ApiError };
