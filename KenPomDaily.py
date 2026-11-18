"""
Day-of KenPom ratings (workspace.default.kenpom_daily, from KenPom's archive via BackfillKenPomDaily.py).

apply_day_of(games, season) swaps a season's KenPom columns (home_/away_ adj_o, adj_d, adj_em, tempo) for
the ratings KenPom showed on each game's date, so training data matches what the live pipeline sees each
morning (#5). Shared by the training table update, BackfillHistoricalData.py and BacktestBets.py.

Rules: a date without an archive page uses the latest earlier date (ratings only change when games are
played); a game before the first archive date uses the earliest one (KenPom's preseason ratings); never
end-of-season ratings. Non-D1 sides (not in d1_teams.py for that season) stay blank.
"""

import numpy as np
import pandas as pd

from d1_teams import D1_TEAMS

TABLE = 'workspace.default.kenpom_daily'
COLS = ('adj_o', 'adj_d', 'adj_em', 'tempo')


def _key(name):
    return str(name).replace("'", "").replace("’", "").lower()


def load_daily(cursor, season):
    cursor.execute(f"SELECT rating_date, team_name, adj_o, adj_d, adj_em, tempo FROM {TABLE} WHERE season = {int(season)}")
    df = cursor.fetchall_arrow().to_pandas()
    df['rating_date'] = pd.to_datetime(df['rating_date'])
    return df


def apply_day_of(games, season, daily, verbose=True):
    """Return a copy of `games` with `season`'s KenPom columns replaced by day-of ratings."""
    if daily is None or daily.empty or season not in D1_TEAMS:
        if verbose:
            print(f"  no day-of KenPom for {season}; ratings unchanged")
        return games
    key = {(r.rating_date, _key(r.team_name)): r for r in daily.itertuples()}
    espn2kp = {_key(e): kp for e, kp in D1_TEAMS[season].items()}
    dates = np.array(sorted(daily['rating_date'].unique()))

    def as_of(d):
        i = np.searchsorted(dates, np.datetime64(d), side='right') - 1
        return pd.Timestamp(dates[max(i, 0)])          # before the first archive date -> preseason ratings

    g = games.copy()
    hit = blank = 0
    for i in g.index[g['season'] == season]:
        d = as_of(pd.Timestamp(g.at[i, 'game_date']))
        for side in ('home', 'away'):
            kp = espn2kp.get(_key(g.at[i, f'{side}_team']))
            r = key.get((d, _key(kp))) if kp else None
            if r is None:
                for c in COLS:
                    g.at[i, f'{side}_{c}'] = np.nan
                blank += 1
                continue
            hit += 1
            for c in COLS:
                g.at[i, f'{side}_{c}'] = float(getattr(r, c))
    if verbose:
        print(f"  {season}: day-of KenPom on {hit} team-sides, {blank} blank (non-D1 or unmatched)")
    return g
