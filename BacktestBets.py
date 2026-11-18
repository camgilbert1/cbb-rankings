"""
Honest backtest of the bet rule on the 2025-26 season (#12 / #20): do v2/v3 edges vs the line pay off,
and should big-spread edges be shrunk?

  - v2 / v3 trained on 2023-24 + 2024-25 exactly as production trains on past seasons
  - every 2025-26 game predicted with KenPom AS OF THAT DAY (kenpom_daily, from KenPom's archive) and
    point-in-time box-score ratings, so nothing "knows" how the season turned out
  - lines: ESPN/DraftKings (historical_lines)

For each model and line size (<10, 10-20, 20+):
  share real  = how much of the model's disagreement with the line shows up in results
                (slope of result-vs-line on model-vs-line; 1 = all real, 0 = none)
  current     = the live rule: bet at 60%+ cover probability
  shrunk      = edge x (share real for that line size), shares fit on Nov-Jan only, scored on Feb-Apr

    python BacktestBets.py
"""

import numpy as np
import pandas as pd
from statistics import NormalDist

import BoxRatings as B
import Betting
import ModelV3 as V3
import TrainModel as T
from d1_teams import D1_TEAMS
from EvaluateModel import fit_predict

TEST_SEASON = 2026
TRAIN_SEASONS = [2024, 2025]
SIZES = [('<10', 0, 10), ('10-20', 10, 20), ('20+', 20, 999)]
phi = NormalDist().cdf


def load_table(sql):
    conn = T.db_connect(); cur = conn.cursor(); cur.execute(sql)
    df = cur.fetchall_arrow().to_pandas(); cur.close(); conn.close()
    return df


def day_of_kenpom(games):
    """Replace the test season's end-of-season KenPom with KenPom as of each game's date."""
    daily = load_table(f"SELECT rating_date, team_name, adj_o, adj_d, adj_em, tempo FROM workspace.default.kenpom_daily WHERE season = {TEST_SEASON}")
    daily['rating_date'] = pd.to_datetime(daily['rating_date'])
    key = {(r.rating_date, T.name_key(r.team_name).lower()): r for r in daily.itertuples()}
    espn2kp = {T.name_key(e): kp for e, kp in D1_TEAMS[TEST_SEASON].items()}
    # KenPom only changes when games are played: a date without an archive page uses the latest earlier
    # date's ratings. Never fall back to end-of-season ratings (that would leak the season's outcome).
    archive_dates = sorted(daily['rating_date'].unique())
    def as_of(d):
        i = np.searchsorted(archive_dates, np.datetime64(d), side='right') - 1
        return pd.Timestamp(archive_dates[i]) if i >= 0 else None
    g = games.copy()
    hit = 0
    for i in g.index[g['season'] == TEST_SEASON]:
        d = as_of(pd.Timestamp(g.at[i, 'game_date']))
        for side in ('home', 'away'):
            kp = espn2kp.get(T.name_key(g.at[i, f'{side}_team']))
            r = key.get((d, T.name_key(kp).lower())) if kp and d is not None else None
            if r is None:                      # non-D1 side, or no rating yet: no KenPom (never season-end)
                for c in ('adj_o', 'adj_d', 'adj_em', 'tempo'):
                    g.at[i, f'{side}_{c}'] = np.nan
                continue
            hit += 1
            g.at[i, f'{side}_adj_o'], g.at[i, f'{side}_adj_d'] = r.adj_o, r.adj_d
            g.at[i, f'{side}_adj_em'], g.at[i, f'{side}_tempo'] = r.adj_em, r.tempo
    n = 2 * int((g['season'] == TEST_SEASON).sum())
    print(f"  day-of KenPom attached to {hit} of {n} {TEST_SEASON} team-sides")
    return g


def attach_lines(test):
    lines = load_table(f"SELECT game_date, home_team, away_team, home_spread FROM workspace.default.historical_lines WHERE season = {TEST_SEASON} AND home_spread IS NOT NULL")
    lines['game_date'] = pd.to_datetime(lines['game_date'])
    same = {(r.game_date, T.name_key(r.home_team), T.name_key(r.away_team)): r.home_spread for r in lines.itertuples()}
    test = test.copy()
    sp = []
    for r in test.itertuples():
        k = (pd.Timestamp(r.game_date), T.name_key(r.home_team), T.name_key(r.away_team))
        s = same.get(k)
        if s is None:                                            # ESPN relabelled home/away
            s2 = same.get((k[0], k[2], k[1]))
            s = None if s2 is None else -s2 + 0.0
        sp.append(s)
    test['home_spread'] = sp
    print(f"  lines attached to {test['home_spread'].notna().sum()} of {len(test)} test games")
    return test[test['home_spread'].notna()].reset_index(drop=True)


def share_real(edge, y):
    """Least-squares slope of (result - line) on (model - line) through 0, with its standard error."""
    w = float(edge @ y / (edge @ edge))
    resid = y - w * edge
    se = float(np.sqrt(resid @ resid / (len(y) - 1) / (edge @ edge)))
    return w, se


def bets(df, edge, scatter, w_by_size=None, threshold=Betting.BET_THRESHOLD):
    """Record of the bet rule: returns dict size -> (bets, wins, losses, units)."""
    out = {}
    line = df['home_spread'].abs().values
    for name, lo, hi in SIZES + [('all', 0, 999)]:
        m = (line >= lo) & (line < hi)
        e = edge[m].copy()
        if w_by_size is not None:
            e = np.array([x * w_by_size[sz] for x, sz in zip(e, df.loc[m, 'size'])])
        sd = scatter[0] + scatter[1] * line[m]
        p_home = np.array([phi(x / s) for x, s in zip(e, sd)])
        bet_home, bet_away = p_home >= threshold, (1 - p_home) >= threshold
        y = df.loc[m, 'result_vs_line'].values
        win = (bet_home & (y > 0)) | (bet_away & (y < 0))
        lose = (bet_home & (y < 0)) | (bet_away & (y > 0))
        n = int((bet_home | bet_away).sum()); w_, l_ = int(win.sum()), int(lose.sum())
        out[name] = (n, w_, l_, w_ * 100 / 110 - l_)
    return out


def fmt_rec(r):
    n, w, l, u = r
    return f"{w}-{l} ({w / (w + l):.1%}) {u:+.1f}u" if w + l else "no bets"


def main():
    print("Building the backtest frame...")
    box, _ = B.build_asof(V3.SEASONS, current_season=T.CURRENT_SEASON, verbose=False)
    box = B.add_box_features(box)
    games = day_of_kenpom(T.load_games())
    v2, _ = T.build_training_frame(games)
    df = V3.merge_frames(v2, box, T.name_key)
    tr = df[df['season'].isin(TRAIN_SEASONS)].reset_index(drop=True)
    te = df[df['season'] == TEST_SEASON].reset_index(drop=True)
    print(f"  train {len(tr)} games ({TRAIN_SEASONS}) | test {len(te)} games ({TEST_SEASON}, day-of KenPom)")

    preds = {'v2': fit_predict(tr, te, T.MARGIN_FEATURES, 'margin')}
    box_pred = fit_predict(tr, te, V3.BOX_FEATURES, 'margin')
    preds['v3'] = V3.BOX_SHARE_DEFAULT * box_pred + (1 - V3.BOX_SHARE_DEFAULT) * preds['v2']
    for k, p in preds.items():
        te[f'pred_{k}'] = p
    te = attach_lines(te)
    te['result_vs_line'] = te['margin'] + te['home_spread']
    te['size'] = pd.cut(te['home_spread'].abs(), [-0.1, 9.99, 19.99, 999], labels=[s[0] for s in SIZES]).astype(str)
    te['game_date'] = pd.to_datetime(te['game_date'])


    # scatter of results around the line (same a + b|line| fit as production), from the test season
    graded = pd.DataFrame({'game_date': te['game_date'], 'home_spread': te['home_spread'], 'actual_margin': te['margin']})
    scatter, desc = Betting.fit_scatter(graded, te['game_date'].max() + pd.Timedelta(days=1))
    print(f"  result scatter around the line: {scatter[0]} + {scatter[1]} x |line| ({desc})")
    vegas_rmse = np.sqrt(np.mean(te['result_vs_line'] ** 2))
    for k in preds:
        print(f"  {k} margin RMSE {np.sqrt(np.mean((te[f'pred_{k}'] - te['margin']) ** 2)):.2f} vs Vegas {vegas_rmse:.2f} ({len(te)} games)")

    for k in preds:
        edge = (te[f'pred_{k}'] + te['home_spread']).values
        y = te['result_vs_line'].values
        print(f"\n===== {k}: {len(te)} games with a line =====")
        print("  line size | games | share of edge that is real (+/- 2 se) | current 60% rule (full season)")
        cur = bets(te, edge, scatter)
        for name, lo, hi in SIZES:
            m = (te['size'] == name).values
            w, se = share_real(edge[m], y[m])
            print(f"  {name:>9} | {m.sum():5d} | {w:5.2f}  ({w - 2 * se:+.2f} to {w + 2 * se:+.2f})          | {fmt_rec(cur[name])}")
        print(f"  {'all':>9} | {len(te):5d} |                                        | {fmt_rec(cur['all'])}")
        # Two-way split across the whole season (big spreads are mostly December): fit the shares on odd
        # days, score even days, then swap, so every game is scored by shares it wasn't fit on.
        odd = (te['game_date'].dt.day % 2 == 1).values
        totals = {name: [[0, 0, 0, 0.0], [0, 0, 0, 0.0]] for name, _, _ in SIZES + [('all', 0, 0)]}
        fitted = []
        for fit_mask in (odd, ~odd):
            w_fit = {name: max(0.0, min(1.0, share_real(edge[(te['size'] == name).values & fit_mask],
                                                        y[(te['size'] == name).values & fit_mask])[0])) for name, _, _ in SIZES}
            fitted.append(w_fit)
            part = te[~fit_mask].reset_index(drop=True)
            e_part = (part[f'pred_{k}'] + part['home_spread']).values
            c, s = bets(part, e_part, scatter), bets(part, e_part, scatter, w_by_size=w_fit)
            for name in totals:
                for j, res in enumerate((c[name], s[name])):
                    totals[name][j] = [a + b for a, b in zip(totals[name][j], res)]
        print("  out-of-sample, whole season (shares fit on the other half of the dates: "
              + '; '.join(', '.join(f"{s} {f[s]:.2f}" for s, _, _ in SIZES) for f in fitted) + "):")
        for name in totals:
            print(f"     {name:>6}  current {fmt_rec(tuple(totals[name][0])):28s} shrunk {fmt_rec(tuple(totals[name][1]))}")


if __name__ == '__main__':
    main()
