"""
Offline evaluation of the box-score point-in-time ratings (BoxRatings.py). Run locally;
read-only on Databricks.

  python EvaluateBoxModel.py --sweep   # tune PRIOR_GAMES / CARRYOVER on 2024-2025 only
  python EvaluateBoxModel.py           # model comparison

Test sets:
  A) train 2024-2025 -> test 2026. Box features are point-in-time on both sides; the v2 model's
     KenPom features are season-end values (leaked), so v2 looks better here than it really is.
  B) train 2024-2026 -> test 2027 games that have point-in-time KenPom (the honest head-to-head),
     plus formula model and Vegas on the same games.
"""

import sys

import numpy as np
import pandas as pd

import BoxRatings as B
import ModelV3 as V3
import TrainModel as T
from EvaluateModel import fit_predict, load_outcome_baselines, mae, rmse

SEASONS = V3.SEASONS                           # 2023 = warm-up only (priors for 2024)

FF = V3.FOUR_FACTOR_FEATURES
BOX_M = V3.BOX_FEATURES                        # the production v3 box model's features
BOX_M_FF = BOX_M + FF
BOX_M_FF_R = BOX_M_FF + ['rating_diff']
BOX_T = ['box_total_est', 'exp_poss', 'exp_h_eff', 'exp_a_eff', 'mu_eff', 'home_court', 'box_min_gp']
BOX_T_FF = BOX_T + FF
GP_BINS = [(0, 0), (1, 5), (6, 15), (16, 999)]


def sweep():
    """
    Tune the ratings on 2024-2025 only (2026/2027 untouched).
    History: carryover-only prior, games {0.5..8} x carry {.5..1} -> 1 game / 1.0 (Nov 2026).
    Now: roster prior on; prior weight x recency half-life.
    """
    print("Ratings sweep — raw box estimate, seasons 2024-2025 (2026/2027 untouched)")
    print(f"  {'games':>5} {'half-life':>9}   all     gp0    gp1-5  gp6-15  gp16+   total")
    for pg in (1.0, 2.0, 4.0, 6.0):
        for hl in (None, 45, 25):
            B.PRIOR_GAMES, B.HALF_LIFE_DAYS = pg, hl
            df, _ = B.build_asof([2023, 2024, 2025], verbose=False)
            x = B.add_box_features(df)
            x = x[x.season.isin([2024, 2025])]
            m, t = x.home_score - x.away_score, x.home_score + x.away_score
            cells = []
            for lo, hi in GP_BINS:
                s = (x.box_min_gp >= lo) & (x.box_min_gp <= hi)
                cells.append(rmse(x.box_margin_est[s], m[s]))
            print(f"  {pg:5.0f} {str(hl):>9}   {rmse(x.box_margin_est, m):6.3f} " +
                  ' '.join(f"{c:6.3f}" for c in cells) + f"  {rmse(x.box_total_est, t):6.3f}", flush=True)


def merge_frames(v2, box):
    return V3.merge_frames(v2, box, T.name_key)


DYN_MIN_HISTORY = 60   # games with point-in-time KenPom needed before trust weights are fit


def dynamic_blend(test, box_pred, kp_pred):
    """
    Trust weights by games played, fit rolling-origin: for each date, a small ridge stack
    margin ~ box + kp + box*gp + kp*gp on point-in-time games from EARLIER dates only.
    Before DYN_MIN_HISTORY games exist, 50/50.
    """
    from sklearn.linear_model import Ridge
    gp = test['box_min_gp'].values / 10.0
    X = np.column_stack([box_pred, kp_pred, box_pred * gp, kp_pred * gp])
    y = test['margin'].values
    dates = test['game_date'].values
    out = 0.5 * box_pred + 0.5 * kp_pred
    for d in np.unique(dates):
        hist, now = dates < d, dates == d
        if hist.sum() >= DYN_MIN_HISTORY:
            m = Ridge(alpha=5.0, fit_intercept=False).fit(X[hist], y[hist])
            out[now] = m.predict(X[now])
    return out


def table(title, rows, y_col='margin'):
    print(f"\n{title}")
    print(f"  {'model':48} {'n':>5} {'RMSE':>7} {'MAE':>7}")
    for name, p, y in rows:
        print(f"  {name:48} {len(y):5d} {rmse(p, y):7.2f} {mae(p, y):7.2f}")


def by_gp(label, test, preds):
    print(f"\n  {label} — margin RMSE by games played (lower of the two teams)")
    print(f"  {'':12}" + ''.join(f"{name:>14}" for name in preds))
    for lo, hi in GP_BINS:
        s = (test['box_min_gp'] >= lo) & (test['box_min_gp'] <= hi)
        if s.sum() < 20:
            continue
        cells = ''.join(f"{rmse(np.asarray(p)[s.values], test.loc[s, 'margin']):14.2f}" for p in preds.values())
        print(f"  gp {lo:>2}-{hi if hi < 999 else '+':<3} n={s.sum():<5}" + cells)


def ats(test, preds):
    v = test['vegas_margin'].notna() & (test['margin'] != test['vegas_margin'])
    if v.sum() < 20:
        return
    print(f"\n  ATS vs Vegas line (n={int(v.sum())}, pushes removed)  — break-even 52.4%")
    for name, p in preds.items():
        p = np.asarray(p)[v.values]
        pick_home = p > test.loc[v, 'vegas_margin'].values
        home_cov = test.loc[v, 'margin'].values > test.loc[v, 'vegas_margin'].values
        edge = np.abs(p - test.loc[v, 'vegas_margin'].values)
        hit = pick_home == home_cov
        big = edge >= 4
        print(f"    {name:28} all {hit.mean():6.1%}   edge>=4: {hit[big].mean() if big.any() else float('nan'):6.1%} (n={big.sum()})")


def main():
    print("Building point-in-time box ratings...")
    box, _ = B.build_asof(SEASONS, current_season=T.CURRENT_SEASON)
    box = B.add_box_features(box)

    games = T.load_games()
    v2, _ = T.build_training_frame(games)
    df = merge_frames(v2, box)
    df = df[df.season >= 2024].reset_index(drop=True)

    # ---------------- Split A ----------------
    tr, te = df[df.season.isin([2024, 2025])], df[df.season == 2026]
    print("\n" + "=" * 78)
    print(f"SPLIT A: train 2024-2025 -> test 2026   (train n={len(tr)}, test n={len(te)})")
    print("  note: v2 uses season-end KenPom here (leaked); box features are point-in-time")
    print("=" * 78)
    pa = {'v2 (leaky KenPom)': fit_predict(tr, te, T.MARGIN_FEATURES, 'margin'),
          'box raw': te['box_margin_est'].values,
          'box ML': fit_predict(tr, te, BOX_M, 'margin'),
          'box ML + 4F': fit_predict(tr, te, BOX_M_FF, 'margin'),
          'box ML + 4F + rating': fit_predict(tr, te, BOX_M_FF_R, 'margin'),
          'box ridge + 4F': fit_predict(tr, te, BOX_M_FF, 'margin', kind='ridge')}
    table('MARGIN', [(k, v, te['margin']) for k, v in pa.items()])
    by_gp('2026', te, {k: pa[k] for k in ['v2 (leaky KenPom)', 'box raw', 'box ML + 4F']})
    ta = {'KenPom formula total (leaky)': te['kp_total_est'].values,
          'box raw total': te['box_total_est'].values,
          'box ML total': fit_predict(tr, te, BOX_T, 'total'),
          'box ML total + 4F': fit_predict(tr, te, BOX_T_FF, 'total'),
          'box ridge total + 4F': fit_predict(tr, te, BOX_T_FF, 'total', kind='ridge')}
    table('TOTAL', [(k, v, te['total']) for k, v in ta.items()])
    for k, v in ta.items():
        print(f"    bias {k:30} {np.mean(v - te['total']):+.2f}")

    # ---------------- Split B ----------------
    tr = df[df.season.isin([2024, 2025, 2026])]
    cur = df[df.season == T.CURRENT_SEASON].copy()
    base = load_outcome_baselines()
    base['k'] = list(zip(base.game_date, base.home_team.map(T.name_key), base.away_team.map(T.name_key)))
    cur['k'] = list(zip(cur.game_date, cur.home_team.map(T.name_key), cur.away_team.map(T.name_key)))
    cur = cur.merge(base[['k', 'formula_margin', 'vegas_margin']].drop_duplicates('k'), on='k', how='left')
    pit = cur['ratings_point_in_time'].astype(bool).values

    print("\n" + "=" * 78)
    print(f"SPLIT B: train 2024-2026 -> test {T.CURRENT_SEASON}   (train n={len(tr)}, all 2027 n={len(cur)}, "
          f"with point-in-time KenPom n={pit.sum()})")
    print("=" * 78)
    v2p = fit_predict(tr, cur, T.MARGIN_FEATURES, 'margin')
    boxp = fit_predict(tr, cur, BOX_M_FF, 'margin')
    boxb = fit_predict(tr, cur, BOX_M, 'margin')
    boxr = fit_predict(tr, cur, BOX_M_FF_R, 'margin')
    # stacked: box features + KenPom features in one model, trained on the leaky history
    both = fit_predict(tr, cur, BOX_M_FF + [f for f in T.MARGIN_FEATURES if f not in BOX_M_FF], 'margin')

    t = cur[pit].reset_index(drop=True)
    sel = lambda a: np.asarray(a)[pit]
    order = np.argsort(t['game_date'].values, kind='stable')
    t = t.iloc[order].reset_index(drop=True)
    sel = lambda a, _s=sel: np.asarray(_s(a))[order]
    preds = {'v2 (current prod)': sel(v2p), 'box raw': t['box_margin_est'].values,
             'box ML': sel(boxb), 'box ML + 4F': sel(boxp), 'box ML + 4F + rating': sel(boxr),
             'box+KenPom one model': sel(both)}
    for w in (0.25, 0.5, 0.75):
        preds[f'blend {w:.2f} box / {1 - w:.2f} v2'] = w * sel(boxp) + (1 - w) * sel(v2p)
    preds['DYNAMIC: trust by games played'] = dynamic_blend(t, sel(boxp), sel(v2p))
    late = t['game_date'] >= t['game_date'].iloc[min(DYN_MIN_HISTORY, len(t) - 1)]
    print(f"  (dynamic weights are fit from {DYN_MIN_HISTORY} games on; {int(late.sum())} test games after that point)")
    table(f'MARGIN on {T.CURRENT_SEASON} point-in-time games (honest head-to-head)',
          [(k, v, t['margin']) for k, v in preds.items()])
    f = t['formula_margin'].notna().values
    vg = t['vegas_margin'].notna().values
    rows = [('formula model (GamePredictions)', t.loc[f, 'formula_margin'], t.loc[f, 'margin'])]
    rows += [(f'{k}  [formula subset]', np.asarray(v)[f], t.loc[f, 'margin']) for k, v in preds.items()
             if k in ('v2 (current prod)', 'box ML + 4F', 'blend 0.50 box / 0.50 v2')]
    rows += [('Vegas', t.loc[vg, 'vegas_margin'], t.loc[vg, 'margin'])]
    rows += [(f'{k}  [Vegas subset]', np.asarray(v)[vg], t.loc[vg, 'margin']) for k, v in preds.items()
             if k in ('v2 (current prod)', 'box ML + 4F', 'blend 0.50 box / 0.50 v2', 'blend 0.25 box / 0.75 v2',
                      'DYNAMIC: trust by games played')]
    table('Baselines on the same games', rows)
    lv = late.values
    table(f'Games after the dynamic weights start (n={int(lv.sum())})',
          [(k, np.asarray(v)[lv], t.loc[lv, 'margin']) for k, v in preds.items()
           if k in ('v2 (current prod)', 'box ML + 4F', 'blend 0.50 box / 0.50 v2', 'DYNAMIC: trust by games played')])
    by_gp(f'{T.CURRENT_SEASON} point-in-time', t, {k: preds[k] for k in
                                                     ['v2 (current prod)', 'box ML + 4F', 'blend 0.50 box / 0.50 v2']})
    ats(t, {k: preds[k] for k in ['v2 (current prod)', 'box ML + 4F', 'blend 0.50 box / 0.50 v2',
                                   'DYNAMIC: trust by games played']})
    table(f'MARGIN on ALL {T.CURRENT_SEASON} games (box models only; v2 uses non-PIT KenPom on some rows)',
          [('v2', v2p, cur['margin']), ('box raw', cur['box_margin_est'], cur['margin']),
           ('box ML + 4F', boxp, cur['margin'])])

    tt = {'KenPom formula total (prod)': t['kp_total_est'].values,
          'box raw total': t['box_total_est'].values,
          'box ML total + 4F': sel(fit_predict(tr, cur, BOX_T_FF, 'total')),
          'box ridge total + 4F': sel(fit_predict(tr, cur, BOX_T_FF, 'total', kind='ridge'))}
    tt['blend 0.5 KenPom / 0.5 box raw'] = 0.5 * tt['KenPom formula total (prod)'] + 0.5 * tt['box raw total']
    table(f'TOTAL on {T.CURRENT_SEASON} point-in-time games', [(k, v, t['total']) for k, v in tt.items()])
    for k, v in tt.items():
        print(f"    bias {k:32} {np.mean(v - t['total']):+.2f}")


if __name__ == "__main__":
    sweep() if '--sweep' in sys.argv else main()
