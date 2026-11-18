/* CBB Model — static site. Data comes from data/*.json written nightly by BuildSite.py. */
'use strict';

// v3 is the official model (its bets are the "official picks"); Formula and v2 are benchmarks that
// still make their own bets, analysed on the Models tab.
const MODELS = [
  { key: 'v3', label: 'v3', color: '#a78bfa', note: 'Box-score ratings + v2 blend' },
  { key: 'v2', label: 'v2', color: '#22c55e', note: 'XGBoost: KenPom, rest, form' },
  { key: 'formula', label: 'Formula', color: '#3b9eff', note: 'KenPom efficiency × tempo' },
];
const OFFICIAL = 'v3';
const WIN_PAYOUT = 100 / 110;   // a 1u bet at -110 wins 0.91u
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

const hasModel = g => MODELS.some(m => g[m.key]);

/* ---------------- bets ----------------
   A model's bet is stored per game (BuildSite: g[model].bet = {side, label, edge, prob, units, retro}).
   Records count ONLY bets. A bet's result is the stored ATS grade (it grades the model's side, which is
   always the bet side); a push or a game graded only on the site falls back to the final score. */
// Two markets, tracked separately everywhere: against the spread and over/under.
// g[model].bet / .ats = spread bet and grade; g[model].ou_bet / .ou = O/U bet and grade.
const MARKETS = {
  ats: { bet: 'bet', grade: 'ats', label: 'ATS' },
  ou: { bet: 'ou_bet', grade: 'ou', label: 'O/U' },
};
const betOf = (g, key, mk = 'ats') => (g[key] && g[key][MARKETS[mk].bet]) || null;
function betResult(g, key, mk = 'ats') {
  const b = betOf(g, key, mk);
  if (!b || !g.final) return null;
  const stored = g[key][MARKETS[mk].grade];
  if (stored === 1) return 'W';
  if (stored === 0) return 'L';
  if (mk === 'ou') {
    if (g.total_line === null || g.total_line === undefined) return null;
    const tot = g.final.home + g.final.away;
    if (tot === g.total_line) return 'P';
    return (tot > g.total_line) === (b.side === 'over') ? 'W' : 'L';
  }
  if (g.spread === null || g.spread === undefined) return null;
  const c = (g.final.home - g.final.away) + g.spread;
  if (c === 0) return 'P';
  return (c > 0) === (b.side === 'home') ? 'W' : 'L';
}
const betUnits = (b, r) => r === 'W' ? (b.units || 1) * WIN_PAYOUT : r === 'L' ? -(b.units || 1) : 0;

function betStats(key, games, mk = 'ats') {
  const st = { bets: 0, w: 0, l: 0, p: 0, pending: 0, units: 0, risked: 0, probSum: 0, retro: 0, live: 0, daily: {} };
  for (const g of games) {
    const b = betOf(g, key, mk);
    if (!b) continue;
    st.bets++; st.probSum += b.prob || 0; b.retro ? st.retro++ : st.live++;
    const r = betResult(g, key, mk);
    if (!r) { st.pending++; continue; }
    if (r === 'W') st.w++; else if (r === 'L') st.l++; else st.p++;
    const u = betUnits(b, r);
    st.units += u; st.risked += r === 'P' ? 0 : (b.units || 1);
    st.daily[g.date] = (st.daily[g.date] || 0) + u;
  }
  st.n = st.w + st.l;
  st.pct = st.n ? st.w / st.n * 100 : null;
  st.avgProb = st.bets ? st.probSum / st.bets * 100 : null;
  st.roi = st.risked ? st.units / st.risked * 100 : null;
  return st;
}
const recStr = s => `${s.w}-${s.l}${s.p ? '-' + s.p : ''}`;
const resChip = r => r ? `<span class="res ${r}">${r === 'P' ? 'PUSH' : r}</span>` : '<span class="res pend">PENDING</span>';
const retroTag = b => b && b.retro ? '<span class="tag retro" title="Bet applied after the fact to the stored pre-game prediction and line">RETRO</span>' : '';
const unitsStr = u => `${u > 0 ? '+' : ''}${u.toFixed(2)}u`;
/* ATS | O/U toggle; `ui` holds the page's chosen market in ui.mk */
const marketToggle = (ui, attr = 'data-mk') => `<div class="seg">${Object.entries(MARKETS).map(([k, m]) =>
  `<button class="chip ${(ui.mk || 'ats') === k ? 'on' : ''}" ${attr}="${k}">${m.label}</button>`).join('')}</div>`;

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
const routes = { picks: renderPicks, results: renderResults, models: renderModels, scorecard: renderModels,
                 ratings: renderRatings, team: renderTeam };
function route() {
  const [, name = 'picks', arg] = (location.hash || '#/picks').split('/');
  const tab = name === 'team' ? 'ratings' : name === 'scorecard' ? 'models' : name;
  document.querySelectorAll('#tabs a').forEach(a => a.classList.toggle('active', a.dataset.route === tab));
  (routes[name] || renderPicks)(arg);
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', route);

/* ================= PICKS (v3 only) ================= */
// Picks is a pre-game view: the model runs once and posts its picks, so cards show predictions only
// (no final scores, live status or W/L). Results and Models grade the picks from saved results.
function gameCard(g) {
  const v3 = g[OFFICIAL], bet = betOf(g, OFFICIAL), ouBet = betOf(g, OFFICIAL, 'ou');
  const top = `<span>${esc(etTime(g.time))}
               ${g.neutral ? '<span class="tag">NEUTRAL</span>' : ''}</span>
               <span>${g.spread === null || g.spread === undefined ? 'No line' : 'Line ' + esc(shortName(g.spread <= 0 ? g.home : g.away)) + ' ' + fmtLine(-Math.abs(g.spread))}</span>`;
  const row = (side, isHome) => {
    const t = team(side.id);
    const rk = t ? `<span class="rk">${t.kenpom.rank}</span>` : '';
    const line = g.spread === null || g.spread === undefined ? '' : fmtLine(isHome ? g.spread : -g.spread);
    const isBetSide = bet && (bet.side === 'home') === isHome;
    const pred = v3 && v3.home !== null && v3.home !== undefined ? fmt(isHome ? v3.home : v3.away, 0) : '—';
    const href = t ? `href="#/team/${side.id}"` : '';
    return `<div class="team-row${isBetSide ? ' bet-side' : ''}">${logo(side)}
      <div style="min-width:0"><a ${href} class="tname">${rk}${esc(shortName(side))}</a>
      <div class="tmeta">${esc(side.record || (t && t.record) || '')}${t ? ' · ' + esc(t.conf) : ''}</div></div>
      <div class="line num">${line}</div><div class="pred num">${pred}</div></div>`;
  };
  const colsHead = `<div class="cols-head"><span></span><span></span><span>Line</span><span>Pred</span></div>`;
  let detail;
  if (!v3) {
    detail = '<div class="v3line mut">No v3 prediction for this game</div>';
  } else {
    const fav = v3.margin >= 0 ? shortName(g.home) : shortName(g.away);
    const edge = g.spread === null || g.spread === undefined ? null : Math.abs(v3.margin + g.spread);
    const predTotal = v3.home !== null && v3.home !== undefined ? v3.home + v3.away : null;
    const ou = g.total_line === null || g.total_line === undefined ? null : g.total_line;
    detail = `<div class="v3line"><span><b style="color:var(--violet)">v3</b> ${esc(fav)} by ${fmt(Math.abs(v3.margin))}</span>
      <span class="mut num">${edge === null ? 'no line' : 'edge ' + fmt(edge)}${bet ? '' : ' · no bet'}</span></div>
      <div class="v3line totals"><span>Total <b class="num" style="color:var(--violet)">${predTotal === null ? '—' : fmt(predTotal, 0)}</b> predicted</span>
      <span class="mut num">${ou === null ? 'O/U line not posted' : 'O/U ' + fmt(ou, 1).replace(/\.0$/, '')}${ou === null || ouBet ? '' : ' · no bet'}</span></div>`;
  }
  let banner = '';
  if (bet) {
    banner += `<div class="official-banner">
      <div class="ob-top"><span class="ob-label">OFFICIAL PICK · ATS</span>${retroTag(bet)}</div>
      <div class="ob-pick">${esc(bet.label)}</div>
      <div class="ob-meta num">${fmt(bet.prob * 100, 0)}% to cover · edge ${fmt(bet.edge)} · ${fmt(bet.units, 0)}U</div></div>`;
  }
  if (ouBet) {
    banner += `<div class="official-banner">
      <div class="ob-top"><span class="ob-label">OFFICIAL PICK · O/U</span>${retroTag(ouBet)}</div>
      <div class="ob-pick">${esc(ouBet.label)}</div>
      <div class="ob-meta num">${fmt(ouBet.prob * 100, 0)}% · edge ${fmt(ouBet.edge)} pts · ${fmt(ouBet.units, 0)}U</div></div>`;
  }
  let blend = '';
  if (v3 && v3.box !== null && v3.kp !== null && v3.box !== undefined) {
    const share = Math.round((v3.box_share || 0) * 100);
    blend = `<div class="blendbar">Box-score model <b class="num">${sgn(v3.box)}</b> · KenPom model <b class="num">${sgn(v3.kp)}</b> · ${share}% box
      <div class="bar"><span style="width:${share}%"></span></div></div>`;
  }
  return `<div class="card${bet || ouBet ? ' official' : ''}"><div class="card-top">${top}</div>${banner}
    <div class="teams">${colsHead}${row(g.away, false)}${row(g.home, true)}</div>${detail}${blend}</div>`;
}

function renderPicks() {
  const dates = [...new Set(S.games.map(g => g.date))].sort().reverse();
  const ui = S.ui.picks = S.ui.picks || { date: dates.includes(S.meta.today) ? S.meta.today : dates[0], filter: 'all' };
  const day = S.games.filter(g => g.date === ui.date);
  const isPick = g => !!(betOf(g, OFFICIAL) || betOf(g, OFFICIAL, 'ou'));
  const picks = day.filter(isPick);
  const nAts = day.filter(g => betOf(g, OFFICIAL)).length, nOu = day.filter(g => betOf(g, OFFICIAL, 'ou')).length;
  const v3Ran = (S.meta.v3_dates || []).includes(ui.date) || day.some(g => g[OFFICIAL]);
  let gs = ui.filter === 'picks' ? picks : day;
  gs = [...gs].sort((a, b) => (isPick(b) ? 1 : 0) - (isPick(a) ? 1 : 0) || (a.time || '').localeCompare(b.time || ''));
  const season = betStats(OFFICIAL, S.games), seasonOu = betStats(OFFICIAL, S.games, 'ou');
  const tracker = (label, s) => `<div class="stat"><div class="k">${label} record</div><div class="v num">${recStr(s)}</div>
      <div class="d">${s.pct === null ? (s.bets ? s.bets + ' pending' : 'no graded picks yet') : `${fmt(s.pct)}% · <span class="${s.units >= 0 ? 'pos' : 'neg'}">${unitsStr(s.units)}</span> · ${s.bets} bets`}</div></div>`;
  $app.innerHTML = `
    <div class="page-head"><div><h1>${ui.date === S.meta.today ? "Tonight's Picks" : 'Picks'}</h1>
      <div class="sub">${niceDate(ui.date, { weekday: 'long', month: 'long', day: 'numeric' })} · ${day.length} games · official picks are v3 bets at ${fmt((S.meta.bet_threshold || .6) * 100, 0)}%+ to cover</div></div>
      <select class="select" id="date">${dates.map(d => `<option value="${d}" ${d === ui.date ? 'selected' : ''}>${niceDate(d)}</option>`).join('')}</select></div>
    ${v3Ran ? '' : `<div class="notice">v3 picks unavailable for this day${S.meta.box_through ? ` (box-score data through ${esc(niceDate(S.meta.box_through))})` : ''}.</div>`}
    <div class="strip">
      <div class="stat accent"><div class="k">Official picks</div><div class="v num">${nAts + nOu}</div><div class="d">${nAts} ATS · ${nOu} O/U</div></div>
      ${tracker('ATS', season)}
      ${tracker('O/U', seasonOu)}
    </div>
    <div class="controls">
      <button class="chip ${ui.filter === 'all' ? 'on' : ''}" data-f="all">All games <span class="mut">${day.length}</span></button>
      <button class="chip ${ui.filter === 'picks' ? 'on' : ''}" data-f="picks">Official picks only <span class="mut">${picks.length}</span></button>
    </div>
    ${gs.length ? `<div class="cards">${gs.map(gameCard).join('')}</div>` : `<div class="empty">${ui.filter === 'picks' ? 'No official picks this day.' : 'No games this day.'}</div>`}`;
  document.getElementById('date').onchange = e => { ui.date = e.target.value; renderPicks(); };
  document.querySelectorAll('[data-f]').forEach(b => b.onclick = () => { ui.filter = b.dataset.f; renderPicks(); });
}

/* ================= MODELS (analysis of every model's bets) ================= */
const MMK = () => (S.ui.models && S.ui.models.mk) || 'ats';   // market shown on the Models tab
/* Every-game ATS view (secondary): how each model's lean did on every game with a line */
function pick(g, key) {
  const m = g[key];
  if (!m || m.margin === null || g.spread === null || g.spread === undefined) return null;
  const v = m.margin + g.spread;
  if (v === 0) return null;
  const home = v > 0;
  const label = home ? `${shortName(g.home)} ${fmtLine(g.spread)}` : `${shortName(g.away)} ${fmtLine(-g.spread)}`;
  return { home, edge: Math.abs(v), label };
}
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
function everyGameStats(key) {
  const st = { w: 0, l: 0, su: [0, 0], err: [] };
  for (const g of S.games) {
    if (!g[key] || !g.final) continue;
    const s = suCorrect(g, key);
    if (s !== null) { st.su[0] += s ? 1 : 0; st.su[1] += 1; }
    if (g[key].margin !== null) st.err.push(Math.abs(g[key].margin - (g.final.home - g.final.away)));
    const r = result(g, key);
    if (r === 'W') st.w++; else if (r === 'L') st.l++;
  }
  st.mae = st.err.length ? st.err.reduce((a, b) => a + b, 0) / st.err.length : null;
  return st;
}

function unitsChart(stats) {
  const dates = [...new Set(Object.values(stats).flatMap(s => Object.keys(s.daily)))].sort();
  if (dates.length < 2) return '<div class="empty">The chart appears once there are two graded days.</div>';
  const series = MODELS.map(m => {
    const daily = stats[m.key].daily, first = dates.findIndex(d => daily[d] !== undefined);
    let c = 0;
    return { m, pts: dates.map((d, i) => { if (daily[d] !== undefined) c += daily[d]; return first === -1 || i < first ? null : c; }) };
  });
  const all = series.flatMap(s => s.pts.filter(v => v !== null)).concat([0]);
  const lo = Math.floor(Math.min(...all) - 1), hi = Math.ceil(Math.max(...all) + 1);
  const W = 1000, H = 260, P = { l: 48, r: 16, t: 14, b: 28 };
  const x = i => P.l + i / (dates.length - 1) * (W - P.l - P.r);
  const y = v => P.t + (hi - v) / (hi - lo) * (H - P.t - P.b);
  const ticks = [...new Set([lo, Math.round((lo + hi) / 2), hi])];
  const lines = series.map(s => {
    const pts = s.pts.map((v, i) => v === null ? null : [x(i), y(v)]).filter(Boolean);
    if (!pts.length) return '';
    const end = pts[pts.length - 1];
    return `<polyline fill="none" stroke="${s.m.color}" stroke-width="2.5" stroke-linejoin="round" points="${pts.map(p => p.map(n => n.toFixed(1)).join(',')).join(' ')}"/>
      <circle cx="${end[0].toFixed(1)}" cy="${end[1].toFixed(1)}" r="4" fill="${s.m.color}"/>`;
  }).join('');
  const xl = [...new Set([0, Math.floor((dates.length - 1) / 2), dates.length - 1])].map(i =>
    `<text x="${x(i)}" y="${H - 8}" fill="#5c6b7a" font-size="12" text-anchor="middle">${niceDate(dates[i], { month: 'short', day: 'numeric' })}</text>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Cumulative units by model">
    ${ticks.map(v => `<line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}" stroke="#1f2b38"/><text x="${P.l - 8}" y="${y(v) + 4}" fill="#5c6b7a" font-size="12" text-anchor="end">${v > 0 ? '+' : ''}${v}u</text>`).join('')}
    <line x1="${P.l}" x2="${W - P.r}" y1="${y(0)}" y2="${y(0)}" stroke="#2a3a4b" stroke-dasharray="4 4"/>
    ${lines}${xl}</svg>`;
}

/* rows of [label, test(bet)] -> table of each model's record in each bucket */
function bucketTable(title, sub, buckets, valueLabel) {
  const cell = (key, test) => {
    const gs = S.games.filter(g => { const b = betOf(g, key, MMK()); return b && test(b); });
    const s = betStats(key, gs, MMK());
    if (!s.bets) return '<td class="mut">—</td>';
    return `<td class="num">${recStr(s)}${s.n ? ` <span class="mut">${fmt(s.pct, 0)}%</span>` : ''}${s.pending ? ` <span class="mut">+${s.pending}</span>` : ''}
      <div class="${s.units >= 0 ? 'pos' : 'neg'} small">${s.n ? unitsStr(s.units) : ''}</div></td>`;
  };
  return `<div class="panel"><h2>${title} <small>${sub}</small></h2><div class="table-wrap"><table>
    <thead><tr><th>${valueLabel}</th>${MODELS.map(m => `<th style="color:${m.color}">${m.label}</th>`).join('')}</tr></thead>
    <tbody>${buckets.map(([lab, test]) => `<tr><td>${lab}</td>${MODELS.map(m => cell(m.key, test)).join('')}</tr>`).join('')}</tbody></table></div></div>`;
}

function calibrationPanel() {
  const buckets = [[60, 65], [65, 70], [70, 80], [80, 101]];
  const rows = buckets.map(([lo, hi]) => {
    const cells = MODELS.map(m => {
      const gs = S.games.filter(g => { const b = betOf(g, m.key, MMK()); return b && b.prob * 100 >= lo && b.prob * 100 < hi; });
      const s = betStats(m.key, gs, MMK());
      if (!s.n) return `<td class="mut">${s.bets ? s.bets + ' pending' : '—'}</td>`;
      const stated = gs.filter(g => betResult(g, m.key, MMK()) === 'W' || betResult(g, m.key, MMK()) === 'L').reduce((a, g) => a + betOf(g, m.key, MMK()).prob, 0) / s.n * 100;
      const diff = s.pct - stated;
      return `<td class="num">${fmt(s.pct, 0)}% <span class="mut">won · said ${fmt(stated, 0)}%</span>
        <div class="calib"><span class="said" style="left:${Math.min(stated, 100)}%"></span><span class="won ${diff >= 0 ? 'ok' : 'bad'}" style="width:${Math.min(s.pct, 100)}%"></span></div>
        <div class="mut small">${s.n} graded</div></td>`;
    }).join('');
    return `<tr><td>${lo}–${hi > 100 ? '100' : hi}%</td>${cells}</tr>`;
  }).join('');
  return `<div class="panel"><h2>Does 60% really win 60%? <small>stated cover probability vs actual win rate on each model's bets</small></h2>
    <div class="table-wrap"><table><thead><tr><th>Stated</th>${MODELS.map(m => `<th style="color:${m.color}">${m.label}</th>`).join('')}</tr></thead>
    <tbody>${rows}</tbody></table></div>
    <div class="body mut small">A bar reaching the tick means the model's confidence held up. This is what decides when to switch to the Vegas rule (track-record-adjusted probability vs break-even), after about 150 graded bets per model.</div></div>`;
}

function disagreementsPanel() {
  const games = S.games.filter(g => {
    const sides = MODELS.map(m => betOf(g, m.key, MMK())).filter(Boolean).map(b => b.side);
    return new Set(sides).size > 1;
  }).sort((a, b) => b.date.localeCompare(a.date));
  const body = games.length ? games.map(g => `<tr><td class="num">${niceDate(g.date, { month: 'short', day: 'numeric' })}</td>
      <td><div class="teamcell">${logo(g.away, 'logo-sm')}${esc(shortName(g.away))} <span class="mut">${g.neutral ? 'vs' : '@'}</span> ${logo(g.home, 'logo-sm')}${esc(shortName(g.home))}</div></td>
      ${MODELS.map(m => { const b = betOf(g, m.key, MMK()); return `<td>${b ? `${esc(b.label)} ${g.final ? resChip(betResult(g, m.key, MMK())) : ''}` : '<span class="mut">no bet</span>'}</td>`; }).join('')}</tr>`).join('')
    : `<tr><td colspan="${2 + MODELS.length}" class="mut">No games yet where models bet opposite sides.</td></tr>`;
  return `<div class="panel"><h2>Model disagreements <small>games where two models bet opposite sides</small></h2>
    <div class="table-wrap"><table><thead><tr><th>Date</th><th>Matchup</th>${MODELS.map(m => `<th style="color:${m.color}">${m.label}</th>`).join('')}</tr></thead>
    <tbody>${body}</tbody></table></div></div>`;
}

function betLog() {
  const ui = S.ui.log = S.ui.log || { model: 'all', res: 'all' };
  let rows = [];
  for (const g of S.games) for (const m of MODELS) {
    const b = betOf(g, m.key, MMK());
    if (b) rows.push({ g, m, b, r: betResult(g, m.key, MMK()) });
  }
  const counts = { all: rows.length, ...Object.fromEntries(MODELS.map(m => [m.key, rows.filter(x => x.m.key === m.key).length])) };
  if (ui.model !== 'all') rows = rows.filter(x => x.m.key === ui.model);
  if (ui.res === 'W' || ui.res === 'L') rows = rows.filter(x => x.r === ui.res);
  if (ui.res === 'pending') rows = rows.filter(x => !x.r);
  rows.sort((a, b) => b.g.date.localeCompare(a.g.date) || (b.g.time || '').localeCompare(a.g.time || ''));
  return `<div class="panel" id="betlog"><h2>Bet log <small>${rows.length} bets</small></h2>
    <div class="body"><div class="controls" style="margin:0">
      <button class="chip ${ui.model === 'all' ? 'on' : ''}" data-lm="all">All models <span class="mut">${counts.all}</span></button>
      ${MODELS.map(m => `<button class="chip ${ui.model === m.key ? 'on' : ''}" data-lm="${m.key}">${m.label} <span class="mut">${counts[m.key]}</span></button>`).join('')}
      <span class="spacer"></span>
      ${[['all', 'Any result'], ['W', 'Wins'], ['L', 'Losses'], ['pending', 'Pending']].map(([k, l]) => `<button class="chip ${ui.res === k ? 'on' : ''}" data-lr="${k}">${l}</button>`).join('')}
    </div></div>
    <div class="table-wrap"><table><thead><tr><th>Date</th><th>Model</th><th>Matchup</th><th>Bet</th><th class="r">Prob</th><th class="r">Edge</th><th>Result</th><th class="r">Units</th></tr></thead>
    <tbody>${rows.length ? rows.map(({ g, m, b, r }) => `<tr><td class="num">${niceDate(g.date, { month: 'short', day: 'numeric' })}</td>
      <td><b style="color:${m.color}">${m.label}</b></td>
      <td><div class="teamcell">${logo(g.away, 'logo-sm')}${esc(shortName(g.away))} <span class="mut">${g.neutral ? 'vs' : '@'}</span> ${logo(g.home, 'logo-sm')}${esc(shortName(g.home))}</div></td>
      <td><b>${esc(b.label)}</b> ${retroTag(b)}</td><td class="r num">${fmt(b.prob * 100, 0)}%</td><td class="r num">${fmt(b.edge)}</td>
      <td>${resChip(r)}${g.final ? ` <span class="mut num">${g.final.away}-${g.final.home}</span>` : ''}</td>
      <td class="r num ${r === 'W' ? 'pos' : r === 'L' ? 'neg' : 'mut'}">${r ? unitsStr(betUnits(b, r)) : '—'}</td></tr>`).join('')
      : '<tr><td colspan="8" class="mut">No bets match.</td></tr>'}</tbody></table></div></div>`;
}

function renderModels() {
  const ui = S.ui.models = S.ui.models || { mk: 'ats' };
  const mk = MMK(), M = MARKETS[mk], other = mk === 'ats' ? 'ou' : 'ats';
  const be = S.meta.breakeven || 52.4;
  const stats = Object.fromEntries(MODELS.map(m => [m.key, betStats(m.key, S.games, MMK())]));
  const card = m => {
    const s = stats[m.key], delta = s.pct === null ? null : s.pct - be;
    const o = betStats(m.key, S.games, other);
    return `<div class="mcard${m.key === OFFICIAL ? ' official-model' : ''}" style="--c:${m.color}">
      <div class="t"><span>${m.label}${m.key === OFFICIAL ? ' <span class="tag official-tag">OFFICIAL</span>' : ''}</span><small>${esc(m.note)}</small></div>
      <div class="mk-label">${M.label} record</div>
      <div class="big num">${s.bets ? recStr(s) : '0-0'}</div>
      <div class="delta num ${delta === null ? 'mut' : delta >= 0 ? 'pos' : 'neg'}">${s.pct === null ? (s.bets ? 'no graded bets yet' : 'no bets yet') : `${fmt(s.pct)}% · ${sgn(delta)} vs ${be}% break-even`}</div>
      <div class="rows">
        <div><span>Units (1U per bet, −110)</span><b class="num ${s.units >= 0 ? 'pos' : 'neg'}">${unitsStr(s.units)}</b></div>
        <div><span>Return on risk</span><b class="num">${s.roi === null ? '—' : sgn(s.roi) + '%'}</b></div>
        <div><span>Bets</span><b class="num">${s.bets}${s.pending ? ` <span class="mut">(${s.pending} pending)</span>` : ''}</b></div>
        <div><span>Avg stated probability</span><b class="num">${s.avgProb === null ? '—' : fmt(s.avgProb, 0) + '%'}</b></div>
        <div><span>Live / retroactive</span><b class="num">${s.live} / ${s.retro}</b></div>
        <div class="other-mk"><span>${MARKETS[other].label} record</span><b class="num">${o.bets ? `${recStr(o)} · <span class="${o.units >= 0 ? 'pos' : 'neg'}">${unitsStr(o.units)}</span>` : 'no bets yet'}</b></div>
      </div></div>`;
  };
  const every = MODELS.map(m => ({ m, s: everyGameStats(m.key) }));
  $app.innerHTML = `
    <div class="page-head"><div><h1>Models</h1><div class="sub">Each model bets games it rates ${fmt((S.meta.bet_threshold || .6) * 100, 0)}%+ · ATS and O/U tracked separately · records count only bets · ${S.meta.season - 1}-${String(S.meta.season).slice(2)} season</div></div>
      ${marketToggle(ui)}</div>
    ${mk === 'ou' ? '<div class="notice">v2 and v3 share the same predicted total (KenPom tempo formula), so their O/U bets are identical until v3 gets its own total (#23).</div>' : ''}
    <div class="models-grid">${MODELS.map(card).join('')}</div>
    <div class="panel"><h2>${M.label} units over time <small>cumulative, bets only</small></h2><div class="body">
      <div class="legend">${MODELS.map(m => `<span><i style="background:${m.color}"></i>${m.label} <b class="num ${stats[m.key].units >= 0 ? 'pos' : 'neg'}">${unitsStr(stats[m.key].units)}</b></span>`).join('')}</div>${unitsChart(stats)}</div></div>
    ${calibrationPanel()}
    ${bucketTable('Bets by edge size', 'points between the model and the line · record, win %, +pending, units',
      [['Under 4 pts', b => b.edge < 4], ['4–6 pts', b => b.edge >= 4 && b.edge < 6], ['6–10 pts', b => b.edge >= 6 && b.edge < 10], ['10+ pts', b => b.edge >= 10]], 'Edge')}
    ${disagreementsPanel()}
    ${betLog()}
    <div class="panel"><h2>Every game, not just bets <small>for accuracy tracking; not a betting record</small></h2><div class="table-wrap"><table>
      <thead><tr><th>Model</th><th class="r">ATS lean, all games</th><th class="r">Straight-up winners</th><th class="r">Avg margin miss</th></tr></thead>
      <tbody>${every.map(({ m, s }) => `<tr><td><b style="color:${m.color}">${m.label}</b></td>
        <td class="r num">${s.w}-${s.l}${s.w + s.l ? ` <span class="mut">${fmt(s.w / (s.w + s.l) * 100)}%</span>` : ''}</td>
        <td class="r num">${s.su[1] ? `${s.su[0]}/${s.su[1]} <span class="mut">${fmt(s.su[0] / s.su[1] * 100, 0)}%</span>` : '—'}</td>
        <td class="r num">${s.mae === null ? '—' : fmt(s.mae) + ' pts'}</td></tr>`).join('')}</tbody></table></div></div>
    <div class="mut small" style="margin-bottom:8px">RETRO = a bet applied after the fact to the prediction and line stored before tip-off. Games predicted after tip-off never count as bets. Small samples swing a lot; judge after a few hundred bets.</div>`;
  document.querySelectorAll('[data-lm]').forEach(b => b.onclick = () => { S.ui.log.model = b.dataset.lm; rerenderLog(); });
  document.querySelectorAll('[data-lr]').forEach(b => b.onclick = () => { S.ui.log.res = b.dataset.lr; rerenderLog(); });
  document.querySelectorAll('[data-mk]').forEach(b => b.onclick = () => { ui.mk = b.dataset.mk; renderModels(); });
}
function rerenderLog() {
  document.getElementById('betlog').outerHTML = betLog();
  document.querySelectorAll('[data-lm]').forEach(b => b.onclick = () => { S.ui.log.model = b.dataset.lm; rerenderLog(); });
  document.querySelectorAll('[data-lr]').forEach(b => b.onclick = () => { S.ui.log.res = b.dataset.lr; rerenderLog(); });
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


/* ================= RESULTS (official picks) ================= */
function renderResults() {
  const ui = S.ui.results = S.ui.results || { mk: 'ats' };
  const mk = ui.mk || 'ats', M = MARKETS[mk];
  const dates = [...new Set(S.games.filter(g => betOf(g, OFFICIAL, mk)).map(g => g.date))].sort().reverse();
  const head = `<div class="page-head"><div><h1>Results</h1><div class="sub">Official ${M.label} picks (v3)${dates.length ? ' · ' + niceDate(dates.includes(ui.date) ? ui.date : dates[0], { weekday: 'long', month: 'long', day: 'numeric' }) : ''} · other models on the <a href="#/models" class="link-inline">Models</a> tab</div></div>
      <div class="controls" style="margin:0">${marketToggle(ui)}${dates.length ? `<select class="select" id="rdate">${dates.map(d => `<option value="${d}" ${d === ui.date ? 'selected' : ''}>${niceDate(d)}</option>`).join('')}</select>` : ''}</div></div>`;
  const wire = () => document.querySelectorAll('[data-mk]').forEach(b => b.onclick = () => { ui.mk = b.dataset.mk; ui.date = null; renderResults(); });
  if (!dates.length) { $app.innerHTML = head + `<div class="empty">No official ${M.label} picks yet.</div>`; wire(); return; }
  if (!dates.includes(ui.date)) ui.date = dates[0];
  const gs = S.games.filter(g => g.date === ui.date && betOf(g, OFFICIAL, mk))
    .sort((a, b) => (a.time || '').localeCompare(b.time || ''));
  const day = betStats(OFFICIAL, gs, mk), season = betStats(OFFICIAL, S.games, mk);
  const last7 = new Set(dates.slice(0, 7)), wk = betStats(OFFICIAL, S.games.filter(g => last7.has(g.date)), mk);
  $app.innerHTML = `
    ${head}
    <div class="strip">
      <div class="stat accent"><div class="k">${M.label} this day</div><div class="v num">${recStr(day)}</div><div class="d">${day.pending ? day.pending + ' pending · ' : ''}${day.n ? unitsStr(day.units) : 'not graded yet'}</div></div>
      <div class="stat"><div class="k">${M.label} last ${last7.size} pick days</div><div class="v num">${recStr(wk)}</div><div class="d">${wk.n ? fmt(wk.pct, 0) + '% · ' + unitsStr(wk.units) : '—'}</div></div>
      <div class="stat"><div class="k">Season ${M.label} record</div><div class="v num">${recStr(season)}</div><div class="d">${season.pct === null ? 'no graded picks yet' : fmt(season.pct) + '% · break-even ' + S.meta.breakeven + '%'}</div></div>
      <div class="stat"><div class="k">Season ${M.label} units</div><div class="v num ${season.units >= 0 ? 'pos' : 'neg'}">${unitsStr(season.units)}</div><div class="d">${season.bets} picks · ${season.pending} pending</div></div>
    </div>
    <div class="panel"><div class="table-wrap"><table><thead><tr><th>Matchup</th><th class="r">Final</th><th>Official ${M.label} pick</th><th class="r">${mk === 'ou' ? 'Prob' : 'To cover'}</th><th class="r">Edge</th><th>Result</th><th class="r">Units</th></tr></thead><tbody>
      ${gs.map(g => { const b = betOf(g, OFFICIAL, mk), r = betResult(g, OFFICIAL, mk);
        const fin = g.final ? `<b>${g.final.away}-${g.final.home}</b>${mk === 'ou' ? ` <span class="mut">(${g.final.home + g.final.away})</span>` : ''}` : `<span class="mut">${esc(etTime(g.time))}</span>`;
        return `<tr><td><div class="teamcell">${logo(g.away, 'logo-sm')}${esc(shortName(g.away))} <span class="mut">${g.neutral ? 'vs' : '@'}</span> ${logo(g.home, 'logo-sm')}${esc(shortName(g.home))}</div></td>
        <td class="r num">${fin}</td>
        <td><b>${esc(b.label)}</b> ${retroTag(b)}</td><td class="r num">${fmt(b.prob * 100, 0)}%</td><td class="r num">${fmt(b.edge)}</td>
        <td>${resChip(r)}</td><td class="r num ${r === 'W' ? 'pos' : r === 'L' ? 'neg' : 'mut'}">${r ? unitsStr(betUnits(b, r)) : '—'}</td></tr>`; }).join('')}
    </tbody></table></div></div>`;
  document.getElementById('rdate').onchange = e => { ui.date = e.target.value; renderResults(); };
  wire();
}

/* ---------------- boot ---------------- */
load().then(route).catch(e => {
  $app.innerHTML = `<div class="empty">Could not load the data files (${esc(e.message)}).<br>
    Run <code>python BuildSite.py</code>, then serve the folder: <code>python -m http.server -d site 8000</code>.</div>`;
});
