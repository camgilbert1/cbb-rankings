"""
Player availability change for v3 (added Nov 19 2026).

Team ratings (KenPom, box score, Elo) are built from the games a team has played, so they already reflect
who usually plays. What they miss is a CHANGE: a key player newly out, or one coming back. For each team
before each game, over its rotation players (3+ games, 10+ minutes per game this season):

    chg_gs_last = sum over players of (played the team's previous game: 1/0 - share of earlier games played)
                  x the player's season-average game score (Hollinger's box-score production number)

Negative = less production available than the ratings were built on (a starter just went out); positive =
more (a regular back after missing games). It is a proxy: a player who sat the previous game also sits the
next one ~75% of the time, so a player returning tonight from a one-game absence still counts as out.

Honest v3 backtests: RMSE 11.503 -> 11.482 (test 2025-26) and 11.390 -> 11.385 (test 2024-25). Ceiling with
perfect knowledge of who plays: -0.026 / -0.022, so lineups are a small part of the gap to the books.
"""

import numpy as np
import pandas as pd

ROT_GAMES, ROT_MIN = 3, 10.0
FEATURES = ['home_chg_gs_last', 'away_chg_gs_last', 'chg_gs_last_diff']
COLS = ['game_id', 'game_date', 'team_id', 'player_id', 'did_not_play', 'minutes', 'points', 'fg_made', 'fg_att',
        'ft_made', 'ft_att', 'off_rebounds', 'def_rebounds', 'steals', 'assists', 'blocks', 'fouls', 'turnovers']


def _logs(path):
    x = pd.read_parquet(path, columns=COLS)
    x['played'] = ~x['did_not_play'].fillna(False).astype(bool) & (x['minutes'].fillna(0) > 0)
    f = lambda c: x[c].fillna(0)
    x['gs'] = (f('points') + 0.4 * f('fg_made') - 0.7 * f('fg_att') - 0.4 * (f('ft_att') - f('ft_made'))
               + 0.7 * f('off_rebounds') + 0.3 * f('def_rebounds') + f('steals') + 0.7 * f('assists')
               + 0.7 * f('blocks') - 0.4 * f('fouls') - f('turnovers'))
    return x.sort_values(['team_id', 'game_date', 'game_id'])


def _change(tot, last_played, n):
    """Availability change for the next game, given season-to-date totals and who played the last game."""
    if not n:
        return 0.0
    return float(sum((float(p in last_played) - v[0] / n) * v[2] / v[0]
                     for p, v in tot.items() if v[0] >= ROT_GAMES and v[1] / v[0] >= ROT_MIN))


def season_values(path):
    """({(game_id, team_id): value before that game}, {team_id: value for the team's NEXT game})."""
    per_game, upcoming = {}, {}
    for tid, tg in _logs(path).groupby('team_id', sort=False):
        tot, last, n = {}, set(), 0
        for (_, gid), gg in tg.groupby(['game_date', 'game_id'], sort=True):
            per_game[(gid, tid)] = _change(tot, last, n)
            played = gg[gg.played]
            for r in played.itertuples(index=False):
                v = tot.setdefault(r.player_id, [0, 0.0, 0.0]); v[0] += 1; v[1] += r.minutes; v[2] += r.gs
            last, n = set(played['player_id']), n + 1
        upcoming[tid] = _change(tot, last, n)
    return per_game, upcoming


def attach(m, paths):
    """Add FEATURES to a v3 frame (needs season, game_id, home_team_id, away_team_id). paths: {season: logs path}.
    Games without player logs get 0 (no change)."""
    vals = {}
    for s, p in paths.items():
        vals.update(season_values(p)[0])
    m = m.copy()
    for side in ('home', 'away'):
        m[f'{side}_chg_gs_last'] = [vals.get((g, t), 0.0) for g, t in zip(m['game_id'], m[f'{side}_team_id'])]
    m['chg_gs_last_diff'] = m['home_chg_gs_last'] - m['away_chg_gs_last']
    return m


def attach_upcoming(X, path):
    """Add FEATURES to today's rows (home_id / away_id = ESPN team ids) from the current season's logs."""
    up = season_values(path)[1]
    X = X.copy()
    X['home_chg_gs_last'] = [up.get(int(i), 0.0) if pd.notna(i) else 0.0 for i in X['home_id']]
    X['away_chg_gs_last'] = [up.get(int(i), 0.0) if pd.notna(i) else 0.0 for i in X['away_id']]
    X['chg_gs_last_diff'] = X['home_chg_gs_last'] - X['away_chg_gs_last']
    return X
