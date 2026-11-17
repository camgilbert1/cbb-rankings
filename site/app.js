/* CBB Model — static site. Data comes from data/*.json written nightly by BuildSite.py. */
'use strict';

const MODELS = [
  { key: 'formula', label: 'Formula', color: '#3b9eff', note: 'KenPom efficiency × tempo' },
  { key: 'v2', label: 'v2', color: '#22c55e', note: 'XGBoost: KenPom, rest, form' },
  { key: 'v3', label: 'v3', color: '#a78bfa', note: 'Box-score ratings + v2 blend' },
];
const S = { meta: null, teams: {}, games: [], ui: {} };
const $app = document.getElementById('app');

/* ---------------- helpers ---------------- */
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
const sgn = (v, d = 1) => (v === null || v === undefined) ? '—' : (v > 0 ? '+' : '') + Number(v).toFixed(d);
const fmtLine = x => x === null || x === undefined ? '—' : x === 0 ? 'PK' : (x > 0 ? '+' : '') + x;
const team = id => S.teams[String(id)];
const initials = n => (n || '?').split(/\s+/).slice(0, 2).map(w => w[0]).join('').toUpperCase();

function logo(side, cls = 'logo') {
  if (!side || !side.id) return `<span class="logo-fallback">${esc(initials(side && side.name))}</span>`;
  const src = side.logo || `https://a.espncdn.com/i/teamlogos/ncaa/500/${side.id}.png`;
  return `<img class="${cls}" loading="lazy" src="${src}" alt="" onerror="this.outerHTML='<span class=&quot;logo-fallback&quot;>${esc(initials(side.name))}</span>'">`;
}
function shortName(side) {
  if (!side) return '?';
  if (side.short) return side.short;
  const t = team(side.id);
  if (t) return t.name.split(' ').length > 1 ? schoolOf(t.name) : t.name;
  return side.name;
}
// ESPN display names are "School Mascot"; drop the mascot using the KenPom school name length as a guide
function schoolOf(name) {
  const w = name.split(' ');
  if (w.length <= 2) return w[0];
  const twoWordMascots = ['Blue Devils', 'Tar Heels', 'Horned Frogs', 'Red Storm', 'Golden Eagles', 'Fighting Irish', 'Crimson Tide',
    'Yellow Jackets', 'Demon Deacons', 'Golden Gophers', 'Nittany Lions', 'Scarlet Knights', 'Sun Devils', 'Rainbow Warriors',
    'Fighting Illini', 'Red Raiders', 'Blue Hens', 'Mean Green', 'Golden Hurricane', 'Black Knights', 'Thundering Herd',
    'Ragin\' Cajuns', 'Red Foxes', 'Purple Aces', 'River Hawks', 'Mountain Hawks', 'Blue Raiders', 'Golden Bears', 'Golden Flashes',
    'Running Rebels', 'Fighting Hawks', 'Golden Griffins', 'Big Green', 'Blue Demons', 'Red Wolves', 'Great Danes', 'Green Wave',
    'Golden Lions', 'Golden Panthers', 'Wolf Pack', 'Runnin\' Bulldogs', 'Black Bears', 'Golden Grizzlies', 'Sea Wolves', 'Lady Bears',
    'Delta Devils', 'Mountain Hawks', 'Red Flash', 'Purple Eagles', 'Golden Rams', 'Bearkats', 'Fighting Camels', 'Blue Hose'];
  for (const m of twoWordMascots) if (name.endsWith(' ' + m)) return name.slice(0, -m.length - 1);
  return w.slice(0, -1).join(' ');
}
function etTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' }) + ' ET';
}
function niceDate(ymd, opts = { weekday: 'short', month: 'short', day: 'numeric' }) {
  const [y, m, d] = ymd.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d, 12)).toLocaleDateString('en-US', { ...opts, timeZone: 'UTC' });
}

/* ATS pick for a model on a game: {home: bool, edge, label, conf} or null */
function pick(g, key) {
  const m = g[key];
  if (!m || m.margin === null || g.spread === null || g.spread === undefined) return null;
  const v = m.margin + g.spread;
  if (v === 0) return null;
  const home = v > 0;
  const label = home ? `${shortName(g.home)} ${fmtLine(g.spread)}` : `${shortName(g.away)} ${fmtLine(-g.spread)}`;
  const conf = key === 'formula' ? (m.cover_confidence && m.cover_confidence !== 'N/A' ? m.cover_confidence : null) : m.confidence;
  return { home, edge: Math.abs(v), label, conf };
}
/* graded result for a model's ATS pick: 'W' | 'L' | 'P' | null */
function result(g, key) {
  const p = pick(g, key);
  if (!p || !g.final) return null;
  const stored = g[key] && g[key].ats;
  if (stored === 1) return 'W';
  if (stored === 0) return 'L';
  const c = (g.final.home - g.final.away) + g.spread;
  if (c === 0) return 'P';
  return (c > 0) === p.home ? 'W' : 'L';
}
const suCorrect = (g, key) => g.final && g[key] && g[key].margin ? (Math.sign(g[key].margin) === Math.sign(g.final.home - g.final.away)) : null;
const projScore = g => (g.v3 && g.v3.home !== null) ? g.v3 : (g.v2 && g.v2.home !== null) ? g.v2 : g.formula;
const hasModel = g => MODELS.some(m => g[m.key]);

/* ---------------- data ---------------- */
async function load() {
  const get = p => fetch(`data/${p}.json`, { cache: 'no-store' }).then(r => { if (!r.ok) throw new Error(p); return r.json(); });
  const [meta, teams, games] = await Promise.all([get('meta'), get('teams'), get('games')]);
  S.meta = meta; S.teams = teams; S.games = games.filter(hasModel);
  const t = new Date(meta.generated_at);
  document.getElementById('updated').textContent =
    'Updated ' + t.toLocaleString('en-US', { timeZone: 'America/New_York', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) + ' ET';
}

/* ---------------- router ---------------- */
const routes = { picks: renderPicks, scorecard: renderScorecard, ratings: renderRatings, results: renderResults, team: renderTeam };
function route() {
  const [, name = 'picks', arg] = (location.hash || '#/picks').split('/');
  document.querySelectorAll('#tabs a').forEach(a => a.classList.toggle('active', a.dataset.route === (name === 'team' ? 'ratings' : name)));
  (routes[name] || renderPicks)(arg);
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', route);

/* ================= PICKS ================= */
function gameCard(g) {
  const live = g.state === 'in', done = !!g.final;
  const ps = projScore(g);
  const top = `<span>${done ? '<span class="final">FINAL</span>' : live ? `<span class="live">● ${esc(g.status || 'LIVE')}</span>` : esc(etTime(g.time))}
               ${g.neutral ? '<span class="tag">NEUTRAL</span>' : ''}</span>
               <span>${g.spread === null || g.spread === undefined ? 'No line' : 'Line ' + esc(shortName(g.spread <= 0 ? g.home : g.away)) + ' ' + fmtLine(-Math.abs(g.spread))}</span>`;
  const row = (side, isHome) => {
    const t = team(side.id);
    const rk = t ? `<span class="rk">${t.kenpom.rank}</span>` : '';
    const line = g.spread === null || g.spread === undefined ? '' : fmtLine(isHome ? g.spread : -g.spread);
    let sc;
    if (done) {
      const mine = isHome ? g.final.home : g.final.away, theirs = isHome ? g.final.away : g.final.home;
      sc = `<div class="score num ${mine > theirs ? 'win' : 'loss'}">${mine}</div>`;
    } else {
      sc = `<div class="score proj num" title="Projected">${ps ? fmt(isHome ? ps.home : ps.away, 0) : ''}</div>`;
    }
    const href = t ? `href="#/team/${side.id}"` : '';
    return `<div class="team-row">${logo(side)}
      <div style="min-width:0"><a ${href} class="tname">${rk}${esc(shortName(side))}</a>
      <div class="tmeta">${esc(side.record || (t && t.record) || '')}${t ? ' · ' + esc(t.conf) : ''}</div></div>
      <div class="line num">${line}</div>${sc}</div>`;
  };
  const models = MODELS.map(m => {
    const mm = g[m.key];
    if (!mm) return `<div class="model none"><div class="mname">${m.label}</div><div class="pick">—</div><div class="proj">no pick</div></div>`;
    const p = pick(g, m.key), r = result(g, m.key);
    const fav = mm.margin >= 0 ? shortName(g.home) : shortName(g.away);
    return `<div class="model"><div class="mname"><span style="color:${m.color}">${m.label}</span>
        ${r ? `<span class="res ${r}">${r === 'P' ? 'PUSH' : r}</span>` : p && p.conf ? `<span class="conf ${p.conf}">${p.conf.toUpperCase()}</span>` : ''}</div>
      <div class="pick">${p ? esc(p.label) : '<span class="mut">no line</span>'}</div>
      <div class="proj num">${esc(fav)} by ${fmt(Math.abs(mm.margin))}${p ? ` · edge ${fmt(p.edge)}` : ''}</div></div>`;
  }).join('');
  let blend = '';
  if (g.v3 && g.v3.box !== null && g.v3.kp !== null) {
    const share = Math.round((g.v3.box_share || 0) * 100);
    blend = `<div class="blendbar">v3 blend · box-score model <b class="num">${sgn(g.v3.box)}</b> · KenPom model <b class="num">${sgn(g.v3.kp)}</b> · ${share}% box
      <div class="bar"><span style="width:${share}%"></span></div></div>`;
  }
  const picks = MODELS.map(m => pick(g, m.key)).filter(Boolean);
  const dis = picks.length > 1 && new Set(picks.map(p => p.home)).size > 1;
  return `<div class="card">${`<div class="card-top">${top}</div>`}<div class="teams">${row(g.away, false)}${row(g.home, true)}</div>
    <div class="models">${models}</div>${blend}${dis ? '<div class="blendbar disagree">⚡ Models disagree on the cover</div>' : ''}</div>`;
}

function renderPicks() {
  const dates = [...new Set(S.games.map(g => g.date))].sort().reverse();
  const ui = S.ui.picks = S.ui.picks || { date: dates.includes(S.meta.today) ? S.meta.today : dates[0], filter: 'all', sort: 'time' };
  let gs = S.games.filter(g => g.date === ui.date);
  const counts = {
    all: gs.length, line: gs.filter(g => g.spread !== null && g.spread !== undefined).length,
    high: gs.filter(g => ['v2', 'v3'].some(k => (pick(g, k) || {}).conf === 'High')).length,
    dis: gs.filter(g => { const p = MODELS.map(m => pick(g, m.key)).filter(Boolean); return p.length > 1 && new Set(p.map(x => x.home)).size > 1; }).length,
  };
  if (ui.filter === 'line') gs = gs.filter(g => g.spread !== null && g.spread !== undefined);
  if (ui.filter === 'high') gs = gs.filter(g => ['v2', 'v3'].some(k => (pick(g, k) || {}).conf === 'High'));
  if (ui.filter === 'dis') gs = gs.filter(g => { const p = MODELS.map(m => pick(g, m.key)).filter(Boolean); return p.length > 1 && new Set(p.map(x => x.home)).size > 1; });
  if (ui.sort === 'edge') gs = [...gs].sort((a, b) => ((pick(b, 'v3') || pick(b, 'v2') || {}).edge || -1) - ((pick(a, 'v3') || pick(a, 'v2') || {}).edge || -1));
  const day = S.games.filter(g => g.date === ui.date);
  const graded = MODELS.map(m => { const rs = day.map(g => result(g, m.key)).filter(r => r === 'W' || r === 'L'); return { m, w: rs.filter(r => r === 'W').length, l: rs.filter(r => r === 'L').length }; });
  $app.innerHTML = `
    <div class="page-head"><div><h1>${ui.date === S.meta.today ? "Tonight's Picks" : 'Picks'}</h1>
      <div class="sub">${niceDate(ui.date, { weekday: 'long', month: 'long', day: 'numeric' })} · ${day.length} games</div></div>
      <select class="select" id="date">${dates.map(d => `<option value="${d}" ${d === ui.date ? 'selected' : ''}>${niceDate(d)}</option>`).join('')}</select></div>
    <div class="strip">
      <div class="stat"><div class="k">Games</div><div class="v num">${counts.all}</div><div class="d">${counts.line} with a Vegas line</div></div>
      <div class="stat"><div class="k">High-confidence</div><div class="v num">${counts.high}</div><div class="d">v2/v3 edge 8+ pts</div></div>
      <div class="stat"><div class="k">Models disagree</div><div class="v num">${counts.dis}</div><div class="d">different ATS side</div></div>
      ${graded.map(x => `<div class="stat"><div class="k" style="color:${x.m.color}">${x.m.label} today</div><div class="v num">${x.w}-${x.l}</div><div class="d">ATS graded</div></div>`).join('')}
    </div>
    <div class="controls">
      ${[['all', 'All'], ['line', 'With line'], ['high', 'High confidence'], ['dis', 'Models disagree']]
        .map(([k, l]) => `<button class="chip ${ui.filter === k ? 'on' : ''}" data-f="${k}">${l} <span class="mut">${counts[k]}</span></button>`).join('')}
      <span class="spacer"></span>
      <select class="select" id="sort"><option value="time" ${ui.sort === 'time' ? 'selected' : ''}>Sort: tip-off</option>
        <option value="edge" ${ui.sort === 'edge' ? 'selected' : ''}>Sort: biggest edge</option></select>
    </div>
    ${gs.length ? `<div class="cards">${gs.map(gameCard).join('')}</div>` : '<div class="empty">No games match this filter.</div>'}`;
  document.getElementById('date').onchange = e => { ui.date = e.target.value; renderPicks(); };
  document.getElementById('sort').onchange = e => { ui.sort = e.target.value; renderPicks(); };
  document.querySelectorAll('[data-f]').forEach(b => b.onclick = () => { ui.filter = b.dataset.f; renderPicks(); });
}

/* ================= SCORECARD ================= */
function modelStats(key, games) {
  const st = { w: 0, l: 0, p: 0, conf: {}, su: [0, 0], err: [], daily: {} };
  for (const g of games) {
    if (!g[key] || !g.final) continue;
    const s = suCorrect(g, key);
    if (s !== null) { st.su[0] += s ? 1 : 0; st.su[1] += 1; }
    if (g[key].margin !== null) st.err.push(Math.abs(g[key].margin - (g.final.home - g.final.away)));
    const r = result(g, key);
    if (!r) continue;
    const c = (pick(g, key) || {}).conf || 'Low';
    st.conf[c] = st.conf[c] || { w: 0, l: 0 };
    if (r === 'W') { st.w++; st.conf[c].w++; } else if (r === 'L') { st.l++; st.conf[c].l++; } else st.p++;
    st.daily[g.date] = (st.daily[g.date] || 0) + (r === 'W' ? 1 : r === 'L' ? -1.1 : 0);
  }
  st.n = st.w + st.l;
  st.pct = st.n ? st.w / st.n * 100 : null;
  st.units = st.w - 1.1 * st.l;
  st.mae = st.err.length ? st.err.reduce((a, b) => a + b, 0) / st.err.length : null;
  return st;
}

function unitsChart(stats) {
  const dates = [...new Set(Object.values(stats).flatMap(s => Object.keys(s.daily)))].sort();
  if (dates.length < 2) return '<div class="empty">The chart appears once there are two graded days.</div>';
  const series = MODELS.map(m => {
    let c = 0; const pts = [];
    for (const d of dates) { if (stats[m.key].daily[d] !== undefined) c += stats[m.key].daily[d]; pts.push(stats[m.key].n || Object.keys(stats[m.key].daily).length ? c : null); }
    const first = dates.findIndex(d => stats[m.key].daily[d] !== undefined);
    return { m, pts: pts.map((v, i) => i < first ? null : v) };
  });
  const all = series.flatMap(s => s.pts.filter(v => v !== null)).concat([0]);
  const lo = Math.min(...all) - 1, hi = Math.max(...all) + 1;
  const W = 1000, H = 260, P = { l: 44, r: 16, t: 14, b: 28 };
  const x = i => P.l + i / (dates.length - 1) * (W - P.l - P.r);
  const y = v => P.t + (hi - v) / (hi - lo) * (H - P.t - P.b);
  const ticks = [lo, (lo + hi) / 2, hi].map(v => Math.round(v));
  const lines = series.map(s => {
    const d = s.pts.map((v, i) => v === null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`).filter(Boolean).join(' ');
    return d ? `<polyline fill="none" stroke="${s.m.color}" stroke-width="2.5" stroke-linejoin="round" points="${d}"/>` : '';
  }).join('');
  const xl = [0, Math.floor((dates.length - 1) / 2), dates.length - 1].map(i =>
    `<text x="${x(i)}" y="${H - 8}" fill="#5c6b7a" font-size="12" text-anchor="middle">${niceDate(dates[i], { month: 'short', day: 'numeric' })}</text>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    ${ticks.map(v => `<line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}" stroke="#1f2b38"/><text x="${P.l - 8}" y="${y(v) + 4}" fill="#5c6b7a" font-size="12" text-anchor="end">${v > 0 ? '+' : ''}${v}u</text>`).join('')}
    <line x1="${P.l}" x2="${W - P.r}" y1="${y(0)}" y2="${y(0)}" stroke="#2a3a4b" stroke-dasharray="4 4"/>
    ${lines}${xl}</svg>`;
}

function renderScorecard() {
  const be = S.meta.breakeven || 52.4;
  const stats = Object.fromEntries(MODELS.map(m => [m.key, modelStats(m.key, S.games)]));
  const card = m => {
    const s = stats[m.key];
    const delta = s.pct === null ? null : s.pct - be;
    return `<div class="mcard" style="--c:${m.color}"><div class="t"><span>${m.label}</span><small>${esc(m.note)}</small></div>
      <div class="big num">${s.n ? `${s.w}-${s.l}${s.p ? '-' + s.p : ''}` : '0-0'}</div>
      <div class="delta num ${delta === null ? 'mut' : delta >= 0 ? 'pos' : 'neg'}">${s.pct === null ? 'no graded picks yet' : `${fmt(s.pct)}% ATS · ${sgn(delta)} vs ${be}% break-even`}</div>
      <div class="rows">
        <div><span>Units (−110)</span><b class="num ${s.units >= 0 ? 'pos' : 'neg'}">${sgn(s.units)}u</b></div>
        ${['High', 'Medium', 'Low'].map(c => s.conf[c] ? `<div><span>${c} confidence</span><b class="num">${s.conf[c].w}-${s.conf[c].l} (${fmt(s.conf[c].w / (s.conf[c].w + s.conf[c].l) * 100, 0)}%)</b></div>` : '').join('')}
        <div><span>Straight-up winners</span><b class="num">${s.su[1] ? `${s.su[0]}/${s.su[1]} (${fmt(s.su[0] / s.su[1] * 100, 0)}%)` : '—'}</b></div>
        <div><span>Avg margin miss</span><b class="num">${s.mae === null ? '—' : fmt(s.mae) + ' pts'}</b></div>
      </div></div>`;
  };
  // head-to-head on games every model picked
  const common = S.games.filter(g => MODELS.every(m => result(g, m.key) === 'W' || result(g, m.key) === 'L'));
  const h2h = MODELS.map(m => { const w = common.filter(g => result(g, m.key) === 'W').length; return { m, w, l: common.length - w }; });
  $app.innerHTML = `
    <div class="page-head"><div><h1>Model Scorecard</h1><div class="sub">Against the spread, ${S.meta.season - 1}-${String(S.meta.season).slice(2)} season · break-even at −110 is ${be}%</div></div></div>
    <div class="models-grid">${MODELS.map(card).join('')}</div>
    <div class="panel"><h2>Cumulative units <small>1 unit per pick at −110</small></h2><div class="body">
      <div class="legend">${MODELS.map(m => `<span><i style="background:${m.color}"></i>${m.label}</span>`).join('')}</div>${unitsChart(stats)}</div></div>
    <div class="panel"><h2>Head-to-head <small>${common.length} games graded for all three models</small></h2><div class="body">
      ${common.length ? `<div class="strip" style="margin:0">${h2h.map(x => `<div class="stat"><div class="k" style="color:${x.m.color}">${x.m.label}</div>
        <div class="v num">${x.w}-${x.l}</div><div class="d">${fmt(x.w / common.length * 100)}% ATS</div></div>`).join('')}</div>`
        : '<div class="mut">Appears once v3 has graded picks (v3 started Nov 17).</div>'}</div></div>
    <div class="panel"><h2>Notes</h2><div class="body mut">
      High / Medium / Low = model edge vs the line of 8+, 4–8 and under 4 points (formula uses its own cover confidence).
      v3's box-score share is ${esc(S.meta.trust || 'fixed at 25% for now')}. Small samples swing a lot — judge after a few hundred picks.</div></div>`;
}

/* ================= RATINGS ================= */
const RCOLS = {
  kenpom: [
    ['kprank', 'KP #', t => t.kenpom.rank, 'r'], ['team', 'Team', null], ['conf', 'Conf', t => t.conf], ['rec', 'Record', t => t.record, 'r'],
    ['kpem', 'KenPom EM', t => t.kenpom.em, 'r', 'em'], ['kpo', 'Off', t => t.kenpom.o, 'r'], ['kpd', 'Def', t => t.kenpom.d, 'r', 'low'],
    ['kpt', 'Tempo', t => t.kenpom.tempo, 'r'], ['bxem', 'Box EM', t => t.box && t.box.em, 'r', 'em'], ['bxrk', 'Box #', t => t.box && t.box.rank, 'r', 'low'],
    ['prior', 'Roster prior', t => t.box && t.box.prior_em, 'r', 'em'], ['gp', 'GP', t => t.box && t.box.gp, 'r'],
  ],
  four: [
    ['kprank', 'KP #', t => t.kenpom.rank, 'r'], ['team', 'Team', null], ['conf', 'Conf', t => t.conf],
    ['oefg', 'eFG% O', t => t.box && t.box.o_efg, 'r'], ['defg', 'eFG% D', t => t.box && t.box.d_efg, 'r', 'low'],
    ['otov', 'TO% O', t => t.box && t.box.o_tov, 'r', 'low'], ['dtov', 'TO% D', t => t.box && t.box.d_tov, 'r'],
    ['oorb', 'OReb% O', t => t.box && t.box.o_orb, 'r'], ['dorb', 'OReb% D', t => t.box && t.box.d_orb, 'r', 'low'],
    ['oftr', 'FTR O', t => t.box && t.box.o_ftr, 'r'], ['dftr', 'FTR D', t => t.box && t.box.d_ftr, 'r', 'low'],
    ['bxt', 'Box tempo', t => t.box && t.box.tempo, 'r'],
  ],
};
function heat(v, kind) {
  if (v === null || v === undefined || kind !== 'em') return fmt(v);
  const a = Math.min(1, Math.abs(v) / 30) * .55;
  const c = v >= 0 ? `rgba(34,197,94,${a})` : `rgba(240,82,82,${a})`;
  return `<span class="heat" style="background:${c}">${sgn(v)}</span>`;
}
function renderRatings() {
  const ui = S.ui.ratings = S.ui.ratings || { view: 'kenpom', sort: 'kpem', dir: -1, q: '', conf: '' };
  const cols = RCOLS[ui.view];
  if (!cols.find(c => c[0] === ui.sort)) { ui.sort = 'kprank'; ui.dir = 1; }
  const confs = [...new Set(Object.values(S.teams).map(t => t.conf))].sort();
  let rows = Object.values(S.teams).filter(t => (!ui.conf || t.conf === ui.conf) && (!ui.q || t.name.toLowerCase().includes(ui.q.toLowerCase())));
  const col = cols.find(c => c[0] === ui.sort);
  if (col && col[2]) rows.sort((a, b) => { const x = col[2](a), y = col[2](b); if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1; return (x < y ? -1 : 1) * ui.dir; });
  else rows.sort((a, b) => a.name.localeCompare(b.name) * ui.dir);
  $app.innerHTML = `
    <div class="page-head"><div><h1>Team Ratings</h1><div class="sub">KenPom adjusted efficiency next to our box-score ratings (points per 100 possessions, opponent-adjusted${S.meta.box_through ? ', box scores through ' + niceDate(S.meta.box_through) : ''})</div></div></div>
    <div class="controls">
      <button class="chip ${ui.view === 'kenpom' ? 'on' : ''}" data-v="kenpom">Efficiency</button>
      <button class="chip ${ui.view === 'four' ? 'on' : ''}" data-v="four">Four factors</button>
      <span class="spacer"></span>
      <select class="select" id="conf"><option value="">All conferences</option>${confs.map(c => `<option ${c === ui.conf ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>
      <input class="search" id="q" placeholder="Search teams…" value="${esc(ui.q)}">
    </div>
    <div class="panel"><div class="table-wrap"><table><thead><tr>${cols.map(c =>
      `<th class="sortable ${c[3] || ''} ${ui.sort === c[0] ? 'sorted' : ''}" data-s="${c[0]}">${c[1]}${ui.sort === c[0] ? (ui.dir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(t => `<tr class="link" data-id="${t.id}">${cols.map(c => c[0] === 'team'
        ? `<td><div class="teamcell">${logo({ id: t.id, name: t.name, logo: t.logo }, 'logo-sm')}${esc(t.name)}</div></td>`
        : `<td class="${c[3] || ''} num">${c[4] === 'em' ? heat(c[2](t), 'em') : (typeof c[2](t) === 'number' && !Number.isInteger(c[2](t)) ? fmt(c[2](t)) : esc(c[2](t) ?? '—'))}</td>`).join('')}</tr>`).join('')}
      </tbody></table></div></div>
    <div class="mut" style="font-size:12px">EM = offense minus defense. Box EM is our own rating from this season's box scores, re-centered so the average D1 team is 0.
      Roster prior = the starting rating from returning players, D1 transfers and newcomers, before this season's games take over.</div>`;
  document.querySelectorAll('[data-v]').forEach(b => b.onclick = () => { ui.view = b.dataset.v; renderRatings(); });
  document.querySelectorAll('[data-s]').forEach(th => th.onclick = () => {
    const k = th.dataset.s; const c = cols.find(x => x[0] === k);
    if (ui.sort === k) ui.dir *= -1; else { ui.sort = k; ui.dir = (c[4] === 'low' || ['kprank', 'team', 'conf'].includes(k)) ? 1 : -1; }
    renderRatings();
  });
  document.getElementById('conf').onchange = e => { ui.conf = e.target.value; renderRatings(); };
  const q = document.getElementById('q');
  q.oninput = e => { ui.q = e.target.value; const pos = e.target.selectionStart; renderRatings(); const n = document.getElementById('q'); n.focus(); n.setSelectionRange(pos, pos); };
  document.querySelectorAll('tr[data-id]').forEach(tr => tr.onclick = () => { location.hash = `#/team/${tr.dataset.id}`; });
}

/* ================= TEAM ================= */
function renderTeam(id) {
  const t = team(id);
  if (!t) { $app.innerHTML = '<div class="empty">Team not found.</div>'; return; }
  const sched = S.games.filter(g => String(g.home.id) === String(id) || String(g.away.id) === String(id)).sort((a, b) => a.date.localeCompare(b.date));
  const box = t.box || {};
  const avg = k => { const v = Object.values(S.teams).map(x => x.box && x.box[k]).filter(v => typeof v === 'number'); return v.reduce((a, b) => a + b, 0) / (v.length || 1); };
  const ff = [
    ['Shooting (eFG%)', 'o_efg', 'd_efg', true], ['Turnovers (per 100)', 'o_tov', 'd_tov', false],
    ['Off. rebounding %', 'o_orb', 'd_orb', true], ['Free throw rate', 'o_ftr', 'd_ftr', true],
  ];
  const meter = (v, a, goodHigh) => {
    if (typeof v !== 'number') return '<div class="meter"><span class="mid"></span></div>';
    const diff = (v - a) * (goodHigh ? 1 : -1), w = Math.min(50, Math.abs(diff) / a * 250);
    const left = diff >= 0 ? 50 : 50 - w;
    return `<div class="meter"><span style="left:${left}%;width:${w}%;background:${diff >= 0 ? 'var(--green)' : 'var(--red)'}"></span><span class="mid"></span></div>`;
  };
  const rows = sched.map(g => {
    const home = String(g.home.id) === String(id), opp = home ? g.away : g.home;
    const where = g.neutral ? 'N' : home ? 'vs' : '@';
    let res = `<span class="mut">${esc(etTime(g.time))}</span>`;
    if (g.final) {
      const me = home ? g.final.home : g.final.away, them = home ? g.final.away : g.final.home;
      res = `<b class="${me > them ? 'pos' : 'neg'}">${me > them ? 'W' : 'L'}</b> <span class="num">${me}-${them}</span>`;
    }
    const line = g.spread === null || g.spread === undefined ? '—' : fmtLine(home ? g.spread : -g.spread);
    const cells = MODELS.map(m => {
      const p = pick(g, m.key), r = result(g, m.key);
      if (!g[m.key]) return '<td class="mut">—</td>';
      const mine = home ? g[m.key].margin : -g[m.key].margin;
      return `<td class="num">${sgn(mine)} ${p ? `<span class="mut">(${p.home === home ? 'cover' : 'fade'})</span>` : ''} ${r ? `<span class="res ${r}">${r}</span>` : ''}</td>`;
    }).join('');
    const ot = team(opp.id);
    return `<tr><td class="num">${niceDate(g.date, { month: 'short', day: 'numeric' })}</td>
      <td><div class="teamcell"><span class="mut" style="width:20px">${where}</span>${logo(opp, 'logo-sm')}
        ${ot ? `<a href="#/team/${opp.id}">${ot.kenpom.rank ? `<span class="mut">${ot.kenpom.rank}</span> ` : ''}${esc(shortName(opp))}</a>` : esc(opp.name)}</div></td>
      <td>${res}</td><td class="r num">${line}</td>${cells}</tr>`;
  }).join('');
  $app.innerHTML = `
    <a class="back" href="#/ratings">← All teams</a>
    <div class="team-hero">${logo({ id: t.id, name: t.name, logo: t.logo }, 'logo-xl')}
      <div><h1>${esc(t.name)}</h1><div class="meta">${esc(t.conf)} · ${esc(t.record)}</div></div>
      <div class="ranks">
        <div class="rankbox"><div class="v num">#${t.kenpom.rank}</div><div class="k">KenPom</div></div>
        <div class="rankbox"><div class="v num">${box.rank ? '#' + box.rank : '—'}</div><div class="k">Box score</div></div>
        <div class="rankbox"><div class="v num">${sgn(t.kenpom.em)}</div><div class="k">KenPom EM</div></div>
        <div class="rankbox"><div class="v num">${fmt(t.kenpom.tempo)}</div><div class="k">Tempo #${t.kenpom.t_rank}</div></div>
      </div></div>
    <div class="models-grid">
      <div class="mcard" style="--c:#3b9eff"><div class="t"><span>KenPom</span><small>adjusted, per 100 poss.</small></div>
        <div class="rows"><div><span>Offense</span><b class="num">${fmt(t.kenpom.o)} (#${t.kenpom.o_rank})</b></div>
        <div><span>Defense</span><b class="num">${fmt(t.kenpom.d)} (#${t.kenpom.d_rank})</b></div>
        <div><span>Margin</span><b class="num">${sgn(t.kenpom.em)}</b></div></div></div>
      <div class="mcard" style="--c:#a78bfa"><div class="t"><span>Box score</span><small>${box.gp ? box.gp + ' games' : 'no games yet'}</small></div>
        <div class="rows"><div><span>Offense</span><b class="num">${fmt(box.o_eff)}</b></div>
        <div><span>Defense</span><b class="num">${fmt(box.d_eff)}</b></div>
        <div><span>Margin</span><b class="num">${sgn(box.em)}</b></div>
        <div><span>Roster prior</span><b class="num">${sgn(box.prior_em)}</b></div></div></div>
    </div>
    <div class="panel"><h2>Four factors <small>vs D1 average · green = better</small></h2><div class="body"><div class="ff">
      ${ff.map(([lab, o, d, goodHigh]) => `<div class="ffrow"><div class="h">${lab}</div>
        <div class="h" style="margin-top:8px"><span>Offense</span><b class="num">${fmt(box[o])}</b></div>${meter(box[o], avg(o), goodHigh)}
        <div class="h" style="margin-top:10px"><span>Defense allowed</span><b class="num">${fmt(box[d])}</b></div>${meter(box[d], avg(d), !goodHigh)}</div>`).join('')}
    </div></div></div>
    <div class="panel"><h2>Schedule &amp; model picks <small>margins from ${esc(shortName({ id: t.id, name: t.name }))}'s side</small></h2>
      <div class="table-wrap"><table><thead><tr><th>Date</th><th>Opponent</th><th>Result</th><th class="r">Line</th>${MODELS.map(m => `<th style="color:${m.color}">${m.label}</th>`).join('')}</tr></thead>
      <tbody>${rows || '<tr><td colspan="7" class="mut">No predicted games yet.</td></tr>'}</tbody></table></div></div>`;
}

/* ================= RESULTS ================= */
function renderResults() {
  const dates = [...new Set(S.games.filter(g => g.final).map(g => g.date))].sort().reverse();
  if (!dates.length) { $app.innerHTML = '<div class="empty">No graded games yet.</div>'; return; }
  const ui = S.ui.results = S.ui.results || { date: dates[0], conf: 'all' };
  let gs = S.games.filter(g => g.date === ui.date && g.final);
  if (ui.conf === 'high') gs = gs.filter(g => MODELS.some(m => (pick(g, m.key) || {}).conf === 'High'));
  const tally = key => { const rs = gs.map(g => result(g, key)); return [rs.filter(r => r === 'W').length, rs.filter(r => r === 'L').length]; };
  const last7 = new Set(dates.slice(0, 7));
  const wk = MODELS.map(m => { const s = modelStats(m.key, S.games.filter(g => last7.has(g.date))); return { m, s }; });
  $app.innerHTML = `
    <div class="page-head"><div><h1>Results</h1><div class="sub">${niceDate(ui.date, { weekday: 'long', month: 'long', day: 'numeric' })} · ${gs.length} graded games</div></div>
      <select class="select" id="rdate">${dates.map(d => `<option value="${d}" ${d === ui.date ? 'selected' : ''}>${niceDate(d)}</option>`).join('')}</select></div>
    <div class="strip">${MODELS.map(m => { const [w, l] = tally(m.key); return `<div class="stat"><div class="k" style="color:${m.color}">${m.label} this day</div><div class="v num">${w}-${l}</div><div class="d">${w + l ? fmt(w / (w + l) * 100, 0) + '% ATS' : 'no graded picks'}</div></div>`; }).join('')}
      ${wk.map(x => `<div class="stat"><div class="k" style="color:${x.m.color}">${x.m.label} last ${last7.size} days</div><div class="v num">${x.s.w}-${x.s.l}</div><div class="d">${x.s.pct === null ? '—' : fmt(x.s.pct, 0) + '% · ' + sgn(x.s.units) + 'u'}</div></div>`).join('')}</div>
    <div class="controls"><button class="chip ${ui.conf === 'all' ? 'on' : ''}" data-c="all">All games</button>
      <button class="chip ${ui.conf === 'high' ? 'on' : ''}" data-c="high">Any high-confidence pick</button></div>
    <div class="panel"><div class="table-wrap"><table><thead><tr><th>Matchup</th><th class="r">Final</th><th class="r">Line</th>
      ${MODELS.map(m => `<th style="color:${m.color}">${m.label} pick</th>`).join('')}</tr></thead><tbody>
      ${gs.map(g => `<tr><td><div class="teamcell">${logo(g.away, 'logo-sm')}${esc(shortName(g.away))} <span class="mut">${g.neutral ? 'vs' : '@'}</span> ${logo(g.home, 'logo-sm')}${esc(shortName(g.home))}</div></td>
        <td class="r num"><b>${g.final.away}-${g.final.home}</b></td>
        <td class="r num">${g.spread === null || g.spread === undefined ? '—' : esc(shortName(g.home)) + ' ' + fmtLine(g.spread)}</td>
        ${MODELS.map(m => { const p = pick(g, m.key), r = result(g, m.key);
          return `<td>${p ? `${esc(p.label)} ${r ? `<span class="res ${r}">${r}</span>` : ''} ${p.conf ? `<span class="conf ${p.conf}">${p.conf[0]}</span>` : ''}` : '<span class="mut">—</span>'}</td>`; }).join('')}</tr>`).join('')}
    </tbody></table></div></div>`;
  document.getElementById('rdate').onchange = e => { ui.date = e.target.value; renderResults(); };
  document.querySelectorAll('[data-c]').forEach(b => b.onclick = () => { ui.conf = b.dataset.c; renderResults(); });
}

/* ---------------- boot ---------------- */
load().then(route).catch(e => {
  $app.innerHTML = `<div class="empty">Could not load the data files (${esc(e.message)}).<br>
    Run <code>python BuildSite.py</code>, then serve the folder: <code>python -m http.server -d site 8000</code>.</div>`;
});
