"""
Point-in-time, opponent-adjusted team ratings built from ESPN box scores.

For every game date D, ratings are solved using ONLY that season's games played before D,
so a training row for a November game sees November-quality ratings — the same situation
as predicting tonight's games. (Season-end KenPom, which the v2 model trains on for
2024-2026, does not have this property.)

In-season model (one ridge regression per stat, per date):
    stat[team i vs opponent j] = mu + O_i + D_j + H * home_sign
  home_sign = +1 home, -1 away, 0 neutral. O_i = team i's offense above average,
  D_j = what opponent j allows above average. Tempo is its own model:
    possessions = mu_t + T_i + T_j
Stats (per team-game, offense perspective):
    eff = points per 100 possessions      efg = effective FG%
    tov = turnovers per 100 possessions   orb = offensive rebound %
    ftr = free throw attempts per FGA
Older games can count less (HALF_LIFE_DAYS), so a team that improves during the season moves.

Starting point (prior) for each team, which fades as games accumulate (PRIOR_GAMES):
  - eff offense/defense: the ROSTER model (RosterPrior). Players who have played for the team
    so far this season x their value from last season's box stats (returning players and D1
    transfers), a newcomer penalty for minutes going to players with no D1 history, plus last
    season's team rating as a program/coaching effect. Refreshed every date as minutes accrue.
    Before a team's first game: program effect only.
  - four factors and tempo (style): last season's team values x CARRYOVER.
  Non-D1 teams start at last season's non-D1 average.

Data: nightly Parquet export of ESPN box scores (github.com/WalrusQuant/sportsleader-data).
"""

import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import LinearRegression, Ridge

MANIFEST_URL = 'https://github.com/WalrusQuant/sportsleader-data/releases/download/manifest/manifest.json'
CACHE_DIR = os.getenv('BOX_CACHE_DIR',
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'box_cache'))
CURRENT_MAX_AGE_HOURS = 6

# Tuned Nov 2026 on 2024-2025 only (EvaluateBoxModel.py --sweep). With the roster prior:
# 1 game / no decay = raw margin RMSE 12.40. Recency half-lives of 45 and 25 days were worse at
# every prior weight and every stage of the season (early-season games stay informative).
PRIOR_GAMES = 1.0     # prior weight, in games
CARRYOVER = 1.0       # share of last season's style ratings (four factors, tempo) kept at season start
HALF_LIFE_DAYS = None # recency weighting of in-season games (None = all games count equally)
ROSTER_PRIOR = True   # eff offense/defense prior from the roster model (False = last season x CARRYOVER)
MU_PENALTY = 0.5      # league-average terms: weak pull toward last season's average
HOME_PENALTY = 300.0  # home edge anchored to last season's value (early-season buy games
                      # otherwise get read as a huge home advantage)
D1_MIN_GAMES = 15     # a team with fewer games in a completed season is treated as non-D1
METRICS = ['eff', 'efg', 'tov', 'orb', 'ftr']
FOUR_FACTORS = ['efg', 'tov', 'orb', 'ftr']

PLAYER_STATS = ['points', 'fg_att', 'fg3_att', 'ft_att', 'off_rebounds', 'def_rebounds',
                'assists', 'turnovers', 'steals', 'blocks', 'fouls']
MIN_PLAYER_MINUTES = 40   # last-season minutes needed for a player to carry a value
ROSTER_FIT_GAMES = 3      # roster-model coefficients are fit on minutes from each team's first N games
ROSTER_FEATURES = ['ret_q{k}', 'ret_adj{k}', 'tr_q{k}', 'tr_adj{k}', 'share_new', 'carry_{k}']


# =============================================================================
# DATA
# =============================================================================

def download_box(seasons, current_season=None, datasets=('games', 'team_game_stats', 'player_game_logs')):
    """Download (or reuse cached) Parquet files. Returns {(dataset, season): path}."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest, paths = None, {}
    for s in seasons:
        for ds in datasets:
            p = os.path.join(CACHE_DIR, f'{ds}_{s}.parquet')
            stale = (not os.path.exists(p) or
                     (s == current_season and time.time() - os.path.getmtime(p) > CURRENT_MAX_AGE_HOURS * 3600))
            if stale:
                if manifest is None:
                    manifest = requests.get(MANIFEST_URL, timeout=60).json()
                f = next(f for f in manifest['files']
                         if f['league'] == 'cbb' and f['dataset'] == ds and f['season'] == s)
                r = requests.get(f['url'], timeout=300)
                r.raise_for_status()
                with open(p, 'wb') as fh:
                    fh.write(r.content)
            paths[(ds, s)] = p
    return paths


def load_season(season, paths):
    """Completed team-game rows (2 per game) with per-row stats and home sign."""
    g = pd.read_parquet(paths[('games', season)])
    t = pd.read_parquet(paths[('team_game_stats', season)])
    g = g[g['completed'].fillna(False).astype(bool)]
    g['neutral_site'] = g['neutral_site'].fillna(False).astype(bool)
    t = t.merge(g[['game_id', 'neutral_site']], on='game_id', how='inner')
    t = t[t['possessions'].notna() & (t['possessions'] > 0) & (t['fg_att'] > 0)]
    t = t[t.groupby('game_id')['game_id'].transform('size') == 2].copy()
    t['game_date'] = pd.to_datetime(t['game_date'])
    t['eff'] = 100 * t['score'] / t['possessions']
    t['efg'] = 100 * (t['fg_made'] + 0.5 * t['fg3_made']) / t['fg_att']
    t['tov'] = 100 * t['turnovers'] / t['possessions']
    t['orb'] = 100 * t['off_rebounds'] / (t['off_rebounds'] + t['opponent_def_rebounds']).clip(lower=1)
    t['ftr'] = 100 * t['ft_att'] / t['fg_att']
    t['hsign'] = np.where(t['neutral_site'], 0, np.where(t['is_home'].astype(bool), 1, -1))
    return t.sort_values(['game_date', 'game_id', 'is_home']).reset_index(drop=True)


def load_player_logs(season, paths):
    """Player game rows for players who actually played."""
    x = pd.read_parquet(paths[('player_game_logs', season)],
                        columns=['game_id', 'game_date', 'team_id', 'player_id', 'did_not_play', 'minutes']
                        + PLAYER_STATS)
    x = x[~x['did_not_play'].fillna(False).astype(bool) & (x['minutes'] > 0)].copy()
    x['game_date'] = pd.to_datetime(x['game_date'])
    x['team_game_no'] = x.groupby('team_id')['game_date'].rank(method='dense')
    return x


# =============================================================================
# SOLVER
# =============================================================================

class SeasonSolver:
    """Accumulates normal equations for one season; solve() gives ratings from games added so far."""

    def __init__(self, team_ids, prior):
        self.ids = list(team_ids)
        self.idx = {tid: i for i, tid in enumerate(self.ids)}
        N, K = len(self.ids), len(METRICS)
        self.N = N
        n = 2 * N + 2                                     # [mu, H, O_0..O_N-1, D_0..D_N-1]
        self.A = np.zeros((n, n)); self.B = np.zeros((n, K))
        self.At = np.zeros((N + 1, N + 1)); self.bt = np.zeros(N + 1)   # [mu_t, T_0..]
        self.pen = np.full(n, PRIOR_GAMES); self.pen[0] = MU_PENALTY
        self.pen[1] = HOME_PENALTY if prior['has_prev'] else 1.0
        self.pen_t = np.full(N + 1, PRIOR_GAMES); self.pen_t[0] = MU_PENALTY
        self.prior = np.zeros((n, K)); self.prior_t = np.zeros(N + 1)
        self.prior[0] = prior['mu']; self.prior[1] = prior['H']
        self.prior[2:2 + N] = prior['O']; self.prior[2 + N:] = prior['D']
        self.prior_t[0] = prior['mu_t']; self.prior_t[1:] = prior['T']
        self.last_date = None

    def set_eff_prior(self, team_ids, o_vals, d_vals):
        """Overwrite the eff offense/defense prior for these teams (metric index 0)."""
        i = np.array([self.idx[t] for t in team_ids], dtype=int)
        if len(i):
            self.prior[2 + i, 0] = o_vals
            self.prior[2 + self.N + i, 0] = d_vals

    def _decay(self, date):
        if HALF_LIFE_DAYS and self.last_date is not None:
            f = 0.5 ** ((date - self.last_date).days / HALF_LIFE_DAYS)
            self.A *= f; self.B *= f; self.At *= f; self.bt *= f
        self.last_date = date

    def add(self, rows, date=None):
        if date is not None:
            self._decay(date)
        i = rows['team_id'].map(self.idx).values
        j = rows['opponent_id'].map(self.idx).values
        r = len(rows); n = 2 * self.N + 2
        ri = np.repeat(np.arange(r), 4)
        ci = np.column_stack([np.zeros(r, int), np.ones(r, int), 2 + i, 2 + self.N + j]).ravel()
        v = np.column_stack([np.ones(r), rows['hsign'].values, np.ones(r), np.ones(r)]).ravel()
        X = sparse.csr_matrix((v, (ri, ci)), shape=(r, n))
        G = (X.T @ X).tocoo()
        np.add.at(self.A, (G.row, G.col), G.data)          # in place: no dense n x n temporaries
        self.B += X.T @ rows[METRICS].values
        # tempo: one row per game
        gm = rows.drop_duplicates('game_id')
        gi = gm['team_id'].map(self.idx).values; gj = gm['opponent_id'].map(self.idx).values
        m = len(gm)
        ri = np.repeat(np.arange(m), 3)
        ci = np.column_stack([np.zeros(m, int), 1 + gi, 1 + gj]).ravel()
        Xt = sparse.csr_matrix((np.ones(3 * m), (ri, ci)), shape=(m, self.N + 1))
        Gt = (Xt.T @ Xt).tocoo()
        np.add.at(self.At, (Gt.row, Gt.col), Gt.data)
        self.bt += Xt.T @ gm['possessions'].values

    @staticmethod
    def _ridge(A, pen, rhs):
        """Solve (A + diag(pen)) x = rhs, reusing one work array."""
        M = A.copy()
        M[np.diag_indices_from(M)] += pen
        return cho_solve(cho_factor(M, overwrite_a=True, check_finite=False), rhs, check_finite=False)

    def solve(self):
        beta = self._ridge(self.A, self.pen, self.B + self.pen[:, None] * self.prior)
        bt = self._ridge(self.At, self.pen_t, self.bt + self.pen_t * self.prior_t)
        N = self.N
        return {'mu': beta[0], 'H': beta[1], 'O': beta[2:2 + N], 'D': beta[2 + N:],
                'mu_t': bt[0], 'T': bt[1:], 'ids': self.ids, 'idx': self.idx}


def make_prior(team_ids, prev):
    """Prior means for a new season from last season's final ratings (zeros if none)."""
    N, K = len(team_ids), len(METRICS)
    pr = {'mu': np.zeros(K), 'H': np.zeros(K), 'O': np.zeros((N, K)), 'D': np.zeros((N, K)),
          'mu_t': 0.0, 'T': np.zeros(N), 'has_prev': prev is not None}
    if prev is None:
        return pr
    f, d1 = prev['final'], prev['d1']
    pr['H'] = f['H'].copy()
    non = [f['idx'][t] for t in f['ids'] if t not in d1]
    non_O = f['O'][non].mean(axis=0) if non else np.zeros(K)
    non_D = f['D'][non].mean(axis=0) if non else np.zeros(K)
    non_T = f['T'][non].mean() if non else 0.0
    pr['mu'], pr['mu_t'] = f['mu'].copy(), float(f['mu_t'])
    for k, tid in enumerate(team_ids):
        if tid in d1:
            p = f['idx'][tid]
            pr['O'][k], pr['D'][k], pr['T'][k] = CARRYOVER * f['O'][p], CARRYOVER * f['D'][p], CARRYOVER * f['T'][p]
        else:
            pr['O'][k], pr['D'][k], pr['T'][k] = non_O, non_D, non_T
    return pr


def _team_ids(t, prev):
    ids = set(t['team_id']) | set(t['opponent_id'])
    if prev is not None:
        ids |= prev['d1']
    return sorted(ids)


def season_finals(seasons, paths):
    """Full-season ratings for each season (one solve per season). Used to fit the roster model."""
    prev, out = None, {}
    for s in seasons:
        t = load_season(s, paths)
        team_ids = _team_ids(t, prev)
        solver = SeasonSolver(team_ids, make_prior(team_ids, prev))
        solver.add(t)
        counts = t['team_id'].value_counts()
        prev = {'final': solver.solve(), 'd1': set(counts[counts >= D1_MIN_GAMES].index)}
        out[s] = prev
    return out


# =============================================================================
# ROSTER PRIOR
# =============================================================================

class RosterPrior:
    """
    Turns last season's player box stats + this season's minutes into each team's starting
    eff offense/defense rating. Coefficients for season s are always fit on OTHER seasons.
    """

    def __init__(self, finals, paths, seasons):
        self.finals = finals
        self.logs = {s: load_player_logs(s, paths) for s in seasons}
        self.complete = [s for s in seasons if finals[s]['d1']]      # seasons with season-end targets
        self.tables = {s: self._team_table(s) for s in self.complete}

    def _team_table(self, s):
        f, d1 = self.finals[s]['final'], self.finals[s]['d1']
        ids = sorted(d1)
        ix = [f['idx'][t] for t in ids]
        return pd.DataFrame({'O': f['O'][ix, 0], 'D': f['D'][ix, 0]}, index=pd.Index(ids, name='team_id'))

    # ---- player value: box stats -> points per 100 possessions, weights fit on team-seasons ----
    def _value_weights(self, exclude):
        F, Y = [], []
        for s in self.complete:
            if s == exclude:
                continue
            tt = self.tables[s]
            x = self.logs[s][self.logs[s]['team_id'].isin(tt.index)]
            team = x.groupby('team_id')[PLAYER_STATS + ['minutes']].sum()
            F.append(team[PLAYER_STATS].div(team['minutes'], axis=0) * 40)
            Y.append(tt.loc[team.index])
        F, Y = pd.concat(F), pd.concat(Y)
        mu = F.mean()
        return {'mu': mu, 'O': Ridge(alpha=1.0).fit(F - mu, Y['O']), 'D': Ridge(alpha=1.0).fit(F - mu, Y['D'])}

    def _player_values(self, s, W):
        """Season-s value per player: box-stat value + share of team residual (scheme/defense/schedule)."""
        tt = self.tables[s]
        x = self.logs[s]
        g = x.groupby('player_id')
        ps = g[PLAYER_STATS + ['minutes']].sum()
        ps['prev_team'] = g['team_id'].agg(lambda v: v.mode().iloc[0])
        ps = ps[ps['prev_team'].isin(tt.index) & (ps['minutes'] >= MIN_PLAYER_MINUTES)].copy()
        r40 = ps[PLAYER_STATS].div(ps['minutes'], axis=0) * 40 - W['mu']
        ps['qO'] = W['O'].predict(r40); ps['qD'] = W['D'].predict(r40)
        share = ps['minutes'] / ps.groupby('prev_team')['minutes'].transform('sum')
        for k in ('O', 'D'):
            team_pred = (share * ps[f'q{k}']).groupby(ps['prev_team']).sum()
            ps[f'adj{k}'] = ps['prev_team'].map(tt[k] - team_pred)
        return ps[['prev_team', 'qO', 'qD', 'adjO', 'adjD']]

    @staticmethod
    def _features(minutes, pv, prev_tt, carry_fill):
        """minutes: DataFrame(team_id, player_id, minutes) for the current season so far."""
        m = minutes.join(pv, on='player_id')
        m['share'] = m['minutes'] / m.groupby('team_id')['minutes'].transform('sum')
        kind = np.where(m['prev_team'].isna(), 'new', np.where(m['prev_team'] == m['team_id'], 'ret', 'tr'))
        out = pd.DataFrame(index=pd.Index(sorted(m['team_id'].unique()), name='team_id'))
        for kd in ('ret', 'tr'):
            sub = m[kind == kd]
            for k in ('O', 'D'):
                out[f'{kd}_q{k}'] = (sub['share'] * sub[f'q{k}']).groupby(sub['team_id']).sum()
                out[f'{kd}_adj{k}'] = (sub['share'] * sub[f'adj{k}']).groupby(sub['team_id']).sum()
        new = m[kind == 'new']
        out['share_new'] = new['share'].groupby(new['team_id']).sum()
        out = out.fillna(0.0)
        out['carry_O'] = prev_tt['O'].reindex(out.index).fillna(carry_fill['O'])
        out['carry_D'] = prev_tt['D'].reindex(out.index).fillna(carry_fill['D'])
        return out

    def season_model(self, s):
        """Fitted roster model for season s (None if last season's data is missing)."""
        if (s - 1) not in self.tables or s not in self.logs:
            return None
        W = self._value_weights(exclude=s)
        X, Y = [], []
        for u in self.complete:
            if u == s or (u - 1) not in self.tables:
                continue
            prev_tt = self.tables[u - 1]
            fill = prev_tt.quantile(0.1)
            early = self.logs[u][self.logs[u]['team_game_no'] <= ROSTER_FIT_GAMES]
            mins = early.groupby(['team_id', 'player_id'], as_index=False)['minutes'].sum()
            mins = mins[mins['team_id'].isin(self.tables[u].index)]
            feats = self._features(mins, self._player_values(u - 1, W), prev_tt, fill)
            X.append(feats); Y.append(self.tables[u].loc[feats.index])
        X, Y = pd.concat(X), pd.concat(Y)
        cols = {k: [c.format(k=k) for c in ROSTER_FEATURES] for k in ('O', 'D')}
        return SeasonRoster(
            pv=self._player_values(s - 1, W), prev_tt=self.tables[s - 1], fill=self.tables[s - 1].quantile(0.1),
            cols=cols,
            full={k: LinearRegression().fit(X[cols[k]], Y[k]) for k in ('O', 'D')},
            carry={k: LinearRegression().fit(X[[f'carry_{k}']], Y[k]) for k in ('O', 'D')},
            logs=self.logs[s])


class SeasonRoster:
    def __init__(self, pv, prev_tt, fill, cols, full, carry, logs):
        self.pv, self.prev_tt, self.fill, self.cols = pv, prev_tt, fill, cols
        self.full, self.carry, self.logs = full, carry, logs
        self.by_date = {d: g[['team_id', 'player_id', 'minutes']] for d, g in logs.groupby('game_date')}

    def eligible(self, team_ids):
        """Teams that were D1 last season get the roster model; others keep the non-D1 start."""
        return [t for t in team_ids if t in self.prev_tt.index]

    def program_only(self, team_ids):
        c = self.prev_tt.loc[team_ids]
        return (self.carry['O'].predict(c[['O']].rename(columns={'O': 'carry_O'})),
                self.carry['D'].predict(c[['D']].rename(columns={'D': 'carry_D'})))

    def priors(self, minutes, team_ids):
        """Roster priors for teams in team_ids that have minutes so far."""
        mins = minutes[minutes['team_id'].isin(team_ids)]
        if mins.empty:
            return [], np.array([]), np.array([])
        f = RosterPrior._features(mins, self.pv, self.prev_tt, self.fill)
        return (list(f.index), self.full['O'].predict(f[self.cols['O']]),
                self.full['D'].predict(f[self.cols['D']]))


# =============================================================================
# POINT-IN-TIME GAME FRAME
# =============================================================================

def _side_cols(r, i, prefix):
    out = {}
    for k, m in enumerate(METRICS):
        out[f'{prefix}_o_{m}'] = r['O'][i, k]
        out[f'{prefix}_d_{m}'] = r['D'][i, k]
    out[f'{prefix}_t'] = r['T'][i]
    return out


def build_asof(seasons, current_season=None, verbose=True):
    """
    Returns (games_df, state).
      games_df: one row per completed game with ratings as of the morning of that game
                (first season in `seasons` is a warm-up whose games are still returned).
      state:    {'final': ratings after the last game, 'names': team_id -> name,
                 'played': team_id -> games this season, 'season': last season, 'finals': ...,
                 'roster': SeasonRoster for the last season, 'minutes': its minutes so far}
    """
    paths = download_box(seasons, current_season=current_season)
    finals_full = season_finals(seasons, paths) if ROSTER_PRIOR else None
    roster = RosterPrior(finals_full, paths, seasons) if ROSTER_PRIOR else None
    prev, frames, finals = None, [], {}
    sr, minutes = None, None
    for s in seasons:
        t0 = time.time()
        t = load_season(s, paths)
        team_ids = _team_ids(t, prev)
        solver = SeasonSolver(team_ids, make_prior(team_ids, prev))
        sr = roster.season_model(s) if roster is not None else None
        eligible = sr.eligible(team_ids) if sr is not None else []
        if eligible:
            o, d = sr.program_only(eligible)
            solver.set_eff_prior(eligible, o, d)
        minutes = pd.DataFrame(columns=['team_id', 'player_id', 'minutes'])
        played = Counter()
        recs = []
        for d, day in t.groupby('game_date', sort=True):
            if eligible and len(minutes):
                ids, o, dd = sr.priors(minutes, eligible)
                solver.set_eff_prior(ids, o, dd)
            r = solver.solve()
            home = day[day['is_home'].astype(bool)]
            for row in home.itertuples(index=False):
                hi, ai = r['idx'][row.team_id], r['idx'][row.opponent_id]
                rec = {'game_id': row.game_id, 'season': s, 'game_date': d,
                       'home_team_id': row.team_id, 'away_team_id': row.opponent_id,
                       'home_team': row.team_name, 'away_team': row.opponent_name,
                       'neutral_site': bool(row.neutral_site),
                       'home_score': row.score, 'away_score': row.opponent_score, 'poss': row.possessions,
                       'h_gp': played[row.team_id], 'a_gp': played[row.opponent_id], 'mu_t': r['mu_t'],
                       'h_prior_o': solver.prior[2 + hi, 0], 'h_prior_d': solver.prior[2 + solver.N + hi, 0],
                       'a_prior_o': solver.prior[2 + ai, 0], 'a_prior_d': solver.prior[2 + solver.N + ai, 0]}
                for k, m in enumerate(METRICS):
                    rec[f'mu_{m}'] = r['mu'][k]; rec[f'H_{m}'] = r['H'][k]
                rec.update(_side_cols(r, hi, 'h')); rec.update(_side_cols(r, ai, 'a'))
                recs.append(rec)
            solver.add(day, date=d)
            for tid in day['team_id']:
                played[tid] += 1
            if sr is not None and d in sr.by_date:
                minutes = (pd.concat([minutes, sr.by_date[d]], ignore_index=True)
                           .groupby(['team_id', 'player_id'], as_index=False)['minutes'].sum())
        if eligible and len(minutes):
            ids, o, dd = sr.priors(minutes, eligible)
            solver.set_eff_prior(ids, o, dd)
        final = solver.solve()
        final['prior_o'] = solver.prior[2:2 + solver.N, 0].copy()      # starting point as of now
        final['prior_d'] = solver.prior[2 + solver.N:, 0].copy()
        d1 = {tid for tid, c in played.items() if c >= D1_MIN_GAMES}
        prev = {'final': final, 'd1': d1}
        finals[s] = prev
        frames.append(pd.DataFrame(recs))
        names = dict(zip(t['team_id'], t['team_name']))
        if verbose:
            print(f"  box ratings {s}: {len(recs)} games, {len(team_ids)} teams, "
                  f"mu eff {final['mu'][0]:.1f}, mu poss {final['mu_t']:.1f}, H eff {final['H'][0]:.2f}, "
                  f"roster prior {'on' if eligible else 'off'}  ({time.time() - t0:.0f}s)")
    state = {'final': prev['final'], 'names': names, 'played': dict(played), 'season': seasons[-1],
             'finals': finals, 'roster': sr, 'minutes': minutes}
    return pd.concat(frames, ignore_index=True), state


def upcoming_rows(games, state):
    """
    Ratings rows for upcoming games from the end-of-data state (this morning's ratings).
    games: iterable of dicts with home_id, away_id, neutral_site (+ any passthrough keys).
    Games with a team the ratings have never seen are skipped.
    """
    r, played = state['final'], state['played']
    rows = []
    for g in games:
        hi, ai = r['idx'].get(g.get('home_id')), r['idx'].get(g.get('away_id'))
        if hi is None or ai is None:
            continue
        rec = {**g, 'h_gp': played.get(g['home_id'], 0), 'a_gp': played.get(g['away_id'], 0), 'mu_t': r['mu_t'],
               'h_prior_o': r['prior_o'][hi], 'h_prior_d': r['prior_d'][hi],
               'a_prior_o': r['prior_o'][ai], 'a_prior_d': r['prior_d'][ai]}
        for k, m in enumerate(METRICS):
            rec[f'mu_{m}'] = r['mu'][k]; rec[f'H_{m}'] = r['H'][k]
        rec.update(_side_cols(r, hi, 'h')); rec.update(_side_cols(r, ai, 'a'))
        rows.append(rec)
    return add_box_features(pd.DataFrame(rows)) if rows else pd.DataFrame()


def add_box_features(df):
    """Matchup features from the point-in-time ratings (works for historical and upcoming rows)."""
    df = df.copy()
    hs = np.where(df['neutral_site'].astype(bool), 0.0, 1.0)
    for m in METRICS:
        df[f'exp_h_{m}'] = df[f'h_o_{m}'] + df[f'a_d_{m}'] + df[f'H_{m}'] * hs     # relative to league avg
        df[f'exp_a_{m}'] = df[f'a_o_{m}'] + df[f'h_d_{m}'] - df[f'H_{m}'] * hs
    df['exp_poss'] = df['mu_t'] + df['h_t'] + df['a_t']
    df['exp_poss_rel'] = df['h_t'] + df['a_t']
    h_eff = df['mu_eff'] + df['exp_h_eff']
    a_eff = df['mu_eff'] + df['exp_a_eff']
    df['box_margin_est'] = df['exp_poss'] * (h_eff - a_eff) / 100
    df['box_total_est'] = df['exp_poss'] * (h_eff + a_eff) / 100
    df['box_em_diff'] = (df['h_o_eff'] - df['h_d_eff']) - (df['a_o_eff'] - df['a_d_eff'])
    df['box_min_gp'] = np.minimum(df['h_gp'], df['a_gp'])
    # starting-point (roster/program) margin: how much the prior alone says
    df['prior_em_diff'] = (df['h_prior_o'] - df['h_prior_d']) - (df['a_prior_o'] - df['a_prior_d'])
    return df
