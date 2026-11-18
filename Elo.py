"""
Our own point-in-time Elo rating, rebuilt every run from game results (ml_game_features, incl. non-D1 rows).
A game's elo_diff uses only games played on EARLIER dates, so it is safe for training and backtests.

  Margin-of-victory Elo (FiveThirtyEight style): home court = HCA_ELO points, K = K_FACTOR,
  ratings regress CARRY toward 1500 between seasons; non-D1 teams start at NON_D1_ELO.
  Feature: elo_diff = home Elo - away Elo + home court (0 at neutral sites).

Used by v3 only (both the box half and v3's own KenPom half). Adopted Nov 18 2026 on honest backtests:
v3 margin RMSE 11.515 -> 11.503 (test 2025-26) and 11.405 -> 11.390 (test 2024-25). Our own SRS was tested
too and did not help inside v3 (it duplicates the box-score rating), so it is not used.
"""

import numpy as np
import pandas as pd

FEATURE = 'elo_diff'
HCA_ELO, K_FACTOR, CARRY, NON_D1_ELO = 80.0, 25.0, 0.70, 1250.0


def build(games, key=lambda s: s):
    """
    games: season, game_date, home_team, away_team, actual_home_score, actual_away_score, neutral_site,
    home_adj_em, away_adj_em (NaN adj_em = non-D1 side). Team names go through key() (e.g. TrainModel.name_key).
    Returns (elo_diff Series aligned to games.index, state for upcoming()).
    """
    g = pd.DataFrame({
        'season': games['season'].values,
        'game_date': pd.to_datetime(games['game_date']).values,
        'h': [key(t) for t in games['home_team']], 'a': [key(t) for t in games['away_team']],
        'margin': (pd.to_numeric(games['actual_home_score'], errors='coerce')
                   - pd.to_numeric(games['actual_away_score'], errors='coerce')).values,
        'home': (~games['neutral_site'].fillna(False).astype(bool)).astype(float).values,
        'd1_h': games['home_adj_em'].notna().values, 'd1_a': games['away_adj_em'].notna().values,
    }, index=games.index).sort_values('game_date', kind='stable')
    out = pd.Series(np.nan, index=g.index, name=FEATURE)
    elo, last = {}, None
    for _, day in g.groupby('game_date', sort=True):      # ratings frozen within a day
        s = day['season'].iloc[0]
        if s != last:
            if last is not None:
                elo = _regress(elo)
            last = s
        upd = {}
        for i, x in zip(day.index, day.itertuples()):
            rh = elo.get(x.h, 1500.0 if x.d1_h else NON_D1_ELO)
            ra = elo.get(x.a, 1500.0 if x.d1_a else NON_D1_ELO)
            diff = rh - ra + HCA_ELO * x.home
            out.at[i] = diff
            if np.isnan(x.margin):
                continue
            win = x.margin > 0
            exp = 1 / (1 + 10 ** (-diff / 400))
            mult = np.log(abs(x.margin) + 1) * 2.2 / ((diff if win else -diff) * 0.001 + 2.2)
            delta = K_FACTOR * mult * ((1.0 if win else 0.0) - exp)
            upd[x.h] = upd.get(x.h, rh) + delta
            upd[x.a] = upd.get(x.a, ra) - delta
        elo.update(upd)
    return out.loc[games.index], {'elo': elo, 'season': last}


def _regress(elo):
    return {t: 1500 + CARRY * (v - 1500) for t, v in elo.items()}


def upcoming(state, season, home_key, away_key, neutral, home_d1=True, away_d1=True):
    """Pre-game elo_diff for a game not yet played (ratings after every completed game)."""
    if state is None:
        return np.nan
    elo = state['elo'] if season == state['season'] else _regress(state['elo'])
    rh = elo.get(home_key, 1500.0 if home_d1 else NON_D1_ELO)
    ra = elo.get(away_key, 1500.0 if away_d1 else NON_D1_ELO)
    return rh - ra + (0.0 if neutral else HCA_ELO)
