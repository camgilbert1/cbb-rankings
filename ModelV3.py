"""
v3 model: point-in-time box-score ratings (BoxRatings.py) blended with the v2 KenPom model.

  v3 margin = w(gp) * box_model + (1 - w(gp)) * v2_model
    gp = games played by the less-experienced team.
    w(gp) is a per-band share learned from past seasons' honest component predictions (fit_trust, Nov 22 2026);
    BOX_SHARE_DEFAULT only if fewer than 2 past seasons are available.
  v3 total = KenPom tempo formula (same as v2; box totals lost to it in Nov 2026 evaluation).

Evaluation: EvaluateBoxModel.py. Nov 2026, 195 leak-free 2027 games: v2 12.96 RMSE,
box model 13.26, 25/75 blend 12.92, Vegas 12.40.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import BoxRatings as B

SEASONS = [2023, 2024, 2025, 2026, 2027]   # 2023 = warm-up only (player values / priors for 2024)
MODEL_FAMILY = 'v3'
BOX_SHARE_DEFAULT = 0.25
DYN_MIN_GAMES = 400

FOUR_FACTOR_FEATURES = [f'exp_{s}_{m}' for s in 'ha' for m in B.FOUR_FACTORS]
# Four factors (matchup-level: home offense vs away defense, and vice versa) added Nov 18 2026: honest
# 2025-26 backtest, v3 margin RMSE 11.525 -> 11.515 (DraftKings ~11.17).
# Our own point-in-time Elo (Elo.py) added to BOTH halves Nov 18 2026: honest v3 RMSE 11.515 -> 11.503 (test
# 2025-26), 11.405 -> 11.390 (test 2024-25). v3's KenPom half = v2's features + these, trained separately from v2.
# Player availability change (PlayerAvailability.py) added to both halves Nov 19 2026: honest v3 RMSE
# 11.503 -> 11.482 (test 2025-26), 11.390 -> 11.385 (test 2024-25).
AVAIL_FEATURES = ['home_chg_gs_last', 'away_chg_gs_last', 'chg_gs_last_diff']
KP_EXTRA_FEATURES = ['elo_diff'] + AVAIL_FEATURES
BOX_FEATURES = ['box_margin_est', 'box_em_diff', 'exp_h_eff', 'exp_a_eff', 'exp_poss_rel',
                'prior_em_diff', 'home_court', 'home_rest', 'away_rest', 'box_min_gp'] + FOUR_FACTOR_FEATURES \
               + KP_EXTRA_FEATURES


def merge_frames(v2, box, name_key, verbose=True):
    """Attach box ratings to the v2 training frame by date + team names (fallback: date + final score)."""
    k = lambda df, h, a: list(zip(df['game_date'], df[h].map(name_key), df[a].map(name_key)))
    v2 = v2.copy(); box = box.copy()
    v2['k'] = k(v2, 'home_team', 'away_team')
    box['k'] = k(box, 'home_team', 'away_team')
    box_cols = [c for c in box.columns if c not in v2.columns or c == 'k']
    val_cols = [c for c in box_cols if c != 'k']
    m = v2.merge(box[box_cols].drop_duplicates('k'), on='k', how='left')
    miss = m['box_margin_est'].isna()
    if miss.any():
        v2s = v2.loc[miss.values].copy()
        v2s['k2'] = list(zip(v2s.game_date, v2s.actual_home_score.astype(int), v2s.actual_away_score.astype(int)))
        box['k2'] = list(zip(box.game_date, box.home_score.astype(int), box.away_score.astype(int)))
        uniq = box[~box.duplicated('k2', keep=False)]
        fill = v2s.merge(uniq[val_cols + ['k2']], on='k2', how='left')
        m.loc[miss.values, val_cols] = fill[val_cols].values
    still = m['box_margin_est'].isna()
    if verbose:
        print(f"  box ratings joined to {(~still).sum()} of {len(m)} training games "
              f"({miss.sum() - still.sum()} by score fallback)")
    m = m[~still].copy()
    for c in val_cols:
        if m[c].dtype == object:
            m[c] = pd.to_numeric(m[c], errors='coerce')
    return add_availability(m, verbose)


def add_availability(m, verbose=True):
    """Player availability change (PlayerAvailability.py) for every game; 0 (= no change) if unavailable."""
    import PlayerAvailability as PA
    try:
        seasons = sorted(int(s) for s in m['season'].dropna().unique())
        paths = B.download_box(seasons, datasets=('player_game_logs',))
        return PA.attach(m, {s: paths[('player_game_logs', s)] for s in seasons})
    except Exception as e:
        print(f"::warning::player availability unavailable ({type(e).__name__}: {e}); using 0 (no change)")
        m = m.copy()
        for f in PA.FEATURES:
            m[f] = 0.0
        return m


# v3's own total: blend the KenPom tempo-formula total with a box-score total model (our pace + efficiency),
# the same 25/75 pattern as the margin. The box total runs high this season, so its bias on this season's
# graded games is measured each run and subtracted. The margin is unaffected (separate prediction).
BOX_TOTAL_FEATURES = ['box_total_est', 'exp_poss', 'exp_h_eff', 'exp_a_eff', 'mu_eff', 'home_court', 'box_min_gp'] \
                     + FOUR_FACTOR_FEATURES      # 'box ML total + 4F' in EvaluateBoxModel.py
BOX_TOTAL_SHARE = 0.25
TOTAL_BIAS_DAYS = 30


def fit_box_total_model(df, weights, xgb_params):
    m = XGBRegressor(**xgb_params)
    m.fit(df[BOX_TOTAL_FEATURES], df['total'], sample_weight=weights)
    return m


def total_bias(model, df, current_season):
    """Mean (predicted - actual) total of the box total model on this season's last TOTAL_BIAS_DAYS days."""
    cur = df[df['season'] == current_season]
    if cur.empty:
        return 0.0, 0
    cur = cur[cur['game_date'] >= cur['game_date'].max() - pd.Timedelta(days=TOTAL_BIAS_DAYS)]
    return float(np.mean(model.predict(cur[BOX_TOTAL_FEATURES]) - cur['total'].values)), len(cur)


# Season scoring-level correction for the KenPom half of the v3 total: the formula has no constant, so it
# lags a season that scores higher/lower than usual. Each run subtracts its mean miss on this season's
# last KP_TOTAL_BIAS_DAYS days, shrunk by n / (n + KP_TOTAL_BIAS_SHRINK) while the sample is small.
# Honest backtest (Nov 2026): total RMSE 17.432 -> 17.307 (2025-26), 16.717 -> 16.587 (2024-25); flat to Dec 15.
KP_TOTAL_BIAS_DAYS = 14
KP_TOTAL_BIAS_SHRINK = 100


def kp_total_bias(df, current_season):
    """Shrunk mean (KenPom formula total - actual) on this season's last KP_TOTAL_BIAS_DAYS days, and n."""
    cur = df[(df['season'] == current_season)].dropna(subset=['kp_total_est', 'total'])
    if cur.empty:
        return 0.0, 0
    cur = cur[cur['game_date'] >= cur['game_date'].max() - pd.Timedelta(days=KP_TOTAL_BIAS_DAYS)]
    n = len(cur)
    raw = float(np.mean(cur['kp_total_est'].values.astype(float) - cur['total'].values.astype(float)))
    return raw * n / (n + KP_TOTAL_BIAS_SHRINK), n


# First-half predictions (1H tab, added Nov 18 2026). Honest tests (train 2 seasons, test 2025-26 / 2024-25):
#   1H margin = v3 margin x measured 1H margin share (~0.535): RMSE 9.27 / 9.26 (a separate 1H model was worse)
#   1H total  = average of (v3 total x 1H scoring share ~0.469) and a direct v3-style 1H total model:
#               RMSE 10.30 / 9.91 (plain "half the full game" 11.00 / 10.59)
# Shares are re-measured every run on all completed games with first-half scores (BackfillHalves.py).
HALVES_TABLE = 'workspace.default.game_halves'


def load_halves(cursor):
    cursor.execute(f"SELECT game_date, home_team, away_team, home_h1, away_h1 FROM {HALVES_TABLE}")
    h = pd.DataFrame(cursor.fetchall(), columns=['game_date', 'home_team', 'away_team', 'home_h1', 'away_h1'])
    h['game_date'] = pd.to_datetime(h['game_date'])
    return h


def fit_h1(df, halves, weight_fn, xgb_params, total_features):
    """df: v3 training frame (one row per D1 game). Returns the pieces predict_h1 needs."""
    d = df.merge(halves, on=['game_date', 'home_team', 'away_team'], how='inner')
    d['h1_margin'] = d['home_h1'] - d['away_h1']
    d['h1_total'] = d['home_h1'] + d['away_h1']
    w, _ = weight_fn(d)
    return {
        'margin_share': float(np.sum(d['h1_margin'] * d['margin']) / np.sum(d['margin'] ** 2)),
        'total_share': float(d['h1_total'].sum() / d['total'].sum()),
        'box_total': XGBRegressor(**xgb_params).fit(d[BOX_TOTAL_FEATURES], d['h1_total'], sample_weight=w),
        'kp_total': XGBRegressor(**xgb_params).fit(d[total_features], d['h1_total'], sample_weight=w),
        'total_features': total_features, 'n': len(d),
    }


def predict_h1(h1, X, v3_margin, v3_total):
    """(1H home scores, 1H away scores) for rows X given v3's full-game margin and total."""
    direct = (BOX_SHARE_DEFAULT * h1['box_total'].predict(X[BOX_TOTAL_FEATURES])
              + (1 - BOX_SHARE_DEFAULT) * h1['kp_total'].predict(X[h1['total_features']]))
    tot = 0.5 * (h1['total_share'] * np.asarray(v3_total, dtype=float) + direct)
    mar = h1['margin_share'] * np.asarray(v3_margin, dtype=float)
    return (tot + mar) / 2, (tot - mar) / 2


def fit_box_model(df, weights, xgb_params):
    m = XGBRegressor(**xgb_params)
    m.fit(df[BOX_FEATURES], df['margin'], sample_weight=weights)
    return m


TRUST_BINS = [(0, 2), (3, 5), (6, 9), (10, 14), (15, 20), (21, 999)]   # games played by the less-experienced team
TRUST_MIN_GAMES = 200                                                  # per bin; fewer -> BOX_SHARE_DEFAULT


def fit_trust(df, current_season, v2_features, xgb_params, season_weights):
    """
    Box share by games-played band, learned from PAST seasons (Nov 22 2026; replaced the straight line fit on the
    current season's first 400 games, which extrapolated to 100% box score by 15 games played).
    Each past season (day-of KenPom) is predicted by box + KenPom models trained on the OTHER past seasons only,
    then per band w = argmin error of w*box + (1-w)*kp, clipped to [0, 1].
    Honest backtest (test 2025-26 and 2024-25): margin RMSE 11.419 vs 11.432 old dynamic line, 11.434 fixed 25%.
    Returns (fn(gp array) -> share array, description).
    """
    past = df[(df['season'] < current_season) & df['ratings_point_in_time'].astype(bool)]
    seasons = sorted(past['season'].unique())
    if len(seasons) < 2:
        return (lambda gp: np.full(len(gp), BOX_SHARE_DEFAULT),
                f"fixed {BOX_SHARE_DEFAULT:.2f} box share (need 2+ past seasons with day-of KenPom)")
    parts = []
    for s in seasons:
        tr, te = past[past['season'] != s], past[past['season'] == s]
        w = tr['season'].map(season_weights).fillna(0.1)
        parts.append(pd.DataFrame({
            'box': XGBRegressor(**xgb_params).fit(tr[BOX_FEATURES], tr['margin'], sample_weight=w).predict(te[BOX_FEATURES]),
            'kp': XGBRegressor(**xgb_params).fit(tr[v2_features], tr['margin'], sample_weight=w).predict(te[v2_features]),
            'margin': te['margin'].values, 'gp': te['box_min_gp'].values}))
    oof = pd.concat(parts, ignore_index=True)
    shares = []
    for lo, hi in TRUST_BINS:
        d = oof[(oof['gp'] >= lo) & (oof['gp'] <= hi)]
        diff = d['box'] - d['kp']
        shares.append(float(np.clip((diff * (d['margin'] - d['kp'])).sum() / (diff ** 2).sum(), 0.0, 1.0))
                      if len(d) >= TRUST_MIN_GAMES else BOX_SHARE_DEFAULT)

    def fn(gp):
        g = np.asarray(gp, dtype=float)
        out = np.full(len(g), shares[-1])
        for (lo, hi), s in zip(TRUST_BINS, shares):
            out[(g >= lo) & (g <= hi)] = s
        return out
    desc = "box share by games played (learned from " + '-'.join(map(str, [seasons[0], seasons[-1]])) + "): " + \
           ', '.join(f"{lo}-{hi if hi < 999 else '+'}: {s:.2f}".replace('-+', '+') for (lo, hi), s in zip(TRUST_BINS, shares))
    return fn, desc
