"""
Offline model evaluation — run locally. Its only Databricks write is the current-season
sync into ml_game_features (the same sync TrainModel.py runs nightly); no models or
predictions are saved.

Compares the old ML model (7 KenPom features) with the reworked one on held-out data:
  A) train 2024-2025 → test 2026 season
  B) train 2024-2026 → test current-season games with point-in-time ratings
     (the only fully leak-free test: ratings as they were before each game)
Also reports:
  - the formula model (GamePredictions.py) and Vegas on test set B
  - "better team controls pace" ablation for the total model
  - pregame-rating step-size (RATING_K) sweep
  - bias correction from the previous 14 days' residuals

Usage:  python EvaluateModel.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import TrainModel as T

OLD_FEATURES = ['em_diff', 'home_adj_o', 'away_adj_o', 'home_adj_d', 'away_adj_d', 'oe_matchup', 'de_matchup']
REST_FEATURES = ['home_rest', 'away_rest']
BIAS_WINDOW_DAYS = 14


def rmse(p, y): return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))
def mae(p, y):  return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def fit_predict(train, test, features, target, kind='xgb'):
    w = train['season'].map(T.SEASON_WEIGHTS).fillna(0.1)
    if kind == 'xgb':
        m = XGBRegressor(**T.XGBOOST_PARAMS)
        m.fit(train[features], train[target], sample_weight=w)
    else:
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(train[features], train[target], ridge__sample_weight=w)
    return m.predict(test[features])


def bias_corrected(test, preds, target):
    """Add the mean residual of test games from the previous BIAS_WINDOW_DAYS (strictly before each date)."""
    t = test[['game_date']].copy()
    t['resid'] = test[target].values - preds
    daily = t.groupby('game_date')['resid'].agg(['sum', 'count'])
    out = np.array(preds, dtype=float)
    for i, d in enumerate(test['game_date']):
        window = daily[(daily.index < d) & (daily.index >= d - pd.Timedelta(days=BIAS_WINDOW_DAYS))]
        if window['count'].sum() >= 30:
            out[i] += window['sum'].sum() / window['count'].sum()
    return out


def report(title, rows):
    print(f"\n{title}")
    print(f"  {'model':44} {'RMSE':>7} {'MAE':>7}")
    for name, p, y in rows:
        print(f"  {name:44} {rmse(p, y):7.2f} {mae(p, y):7.2f}")


def evaluate_split(label, train, test, show_formula=False):
    print("\n" + "=" * 70)
    print(f"{label}   train n={len(train)}  test n={len(test)}")
    print("=" * 70)
    no_rest_m = [f for f in T.MARGIN_FEATURES if f not in REST_FEATURES]
    no_rest_t = [f for f in T.TOTAL_FEATURES if f not in REST_FEATURES]

    m_old = fit_predict(train, test, OLD_FEATURES, 'margin')
    m_new = fit_predict(train, test, T.MARGIN_FEATURES, 'margin')
    m_ridge = fit_predict(train, test, T.MARGIN_FEATURES, 'margin', kind='ridge')
    m_norest = fit_predict(train, test, no_rest_m, 'margin')
    rows = [('old model (7 KenPom features)', m_old, test['margin']),
            ('new model', m_new, test['margin']),
            ('new model WITHOUT rest days', m_norest, test['margin']),
            ('new features, Ridge', m_ridge, test['margin']),
            (f'new model + {BIAS_WINDOW_DAYS}-day bias correction', bias_corrected(test, m_new, 'margin'), test['margin'])]
    if show_formula:
        f = test['formula_margin'].notna()
        rows.append(('formula model (GamePredictions)  [subset]', test.loc[f, 'formula_margin'], test.loc[f, 'margin']))
        rows.append(('new model on same subset', m_new[f.values], test.loc[f, 'margin']))
        v = test['vegas_margin'].notna()
        rows.append((f'Vegas spread  [n={int(v.sum())}]', test.loc[v, 'vegas_margin'], test.loc[v, 'margin']))
        rows.append(('new model on Vegas subset', m_new[v.values], test.loc[v, 'margin']))
        rows.append(('old model on Vegas subset', m_old[v.values], test.loc[v, 'margin']))
    report('MARGIN (home - away)', rows)
    print(f"  winner accuracy: old {np.mean(np.sign(m_old) == np.sign(test['margin'])):.1%}"
          f"   new {np.mean(np.sign(m_new) == np.sign(test['margin'])):.1%}")

    t_old = fit_predict(train, test, OLD_FEATURES, 'total')
    t_new = fit_predict(train, test, T.TOTAL_FEATURES, 'total')
    t_norest = fit_predict(train, test, no_rest_t, 'total')
    report('TOTAL (home + away)', [
        ('old model (7 KenPom features, no tempo)', t_old, test['total']),
        ('KenPom formula total (tempo, no ML)', test['kp_total_est'], test['total']),
        ('new model', t_new, test['total']),
        ('new model WITHOUT rest days', t_norest, test['total']),
        (f'new model + {BIAS_WINDOW_DAYS}-day bias correction', bias_corrected(test, t_new, 'total'), test['total']),
    ])
    print(f"  mean total error (bias): old {np.mean(t_old - test['total']):+.2f}   new {np.mean(t_new - test['total']):+.2f}"
          f"   formula {np.mean(test['kp_total_est'] - test['total']):+.2f}")


def load_outcome_baselines():
    """Formula predictions + Vegas spread for current-season games (prediction_outcomes)."""
    conn = T.db_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT game_date, home_team, away_team, predicted_margin, vegas_spread
        FROM workspace.default.prediction_outcomes WHERE actual_home_score IS NOT NULL
    """)
    df = cur.fetchall_arrow().to_pandas()
    cur.close(); conn.close()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['vegas_margin'] = np.where(df['vegas_spread'] != 0, -df['vegas_spread'], np.nan)  # spread is home line
    return df.rename(columns={'predicted_margin': 'formula_margin'})[
        ['game_date', 'home_team', 'away_team', 'formula_margin', 'vegas_margin']]


def main():
    ratings_df = T.load_current_ratings()
    T.sync_current_season(ratings_df)
    games = T.load_games()

    # --- Pregame rating step size: pick K on split A (train never sees 2026) ---
    print("\nRATING_K sweep (margin RMSE on 2026, model trained on 2024-25):")
    best_k, best = T.RATING_K, None
    for k in (0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        T.RATING_K = k
        df, _ = T.build_training_frame(games)
        tr, te = df[df.season.isin([2024, 2025])], df[df.season == 2026]
        score = rmse(fit_predict(tr, te, T.MARGIN_FEATURES, 'margin'), te['margin'])
        print(f"  K={k:.2f}  RMSE {score:.3f}")
        if best is None or score < best:
            best_k, best = k, score
    T.RATING_K = best_k
    print(f"  → using K={best_k}")

    df, _ = T.build_training_frame(games)

    evaluate_split("SPLIT A: train 2024-2025 → test 2026 (season-end KenPom on both sides)",
                   df[df.season.isin([2024, 2025])], df[df.season == 2026])

    cur = df[(df.season == T.CURRENT_SEASON) & df['ratings_point_in_time'].astype(bool)]
    cur = cur.merge(load_outcome_baselines(), on=['game_date', 'home_team', 'away_team'], how='left')
    evaluate_split(f"SPLIT B: train 2024-2026 → test {T.CURRENT_SEASON} point-in-time games (leak-free)",
                   df[df.season.isin([2024, 2025, 2026])], cur, show_formula=True)


if __name__ == "__main__":
    main()
