"""
XGBoost ML Model — Nightly Retraining
Trains on completed-game outcomes using KenPom efficiency features.
Stores trained models as base64 pickle in Databricks.
Generates today's ML predictions after training.
"""

import pickle
import base64
import json
import numpy as np
import pandas as pd
import requests
from databricks import sql
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os
from datetime import datetime
from dotenv import load_dotenv
import pytz

from team_names import find_team_match

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

CURRENT_SEASON = 2027

SEASON_WEIGHTS = {2027: 3.0, 2026: 1.5, 2025: 0.75, 2024: 0.30}
BOOST_THRESHOLD = 100
BOOST_MULTIPLIER = 2.0

# Features derived from KenPom adj_o / adj_d / adj_em
# em_diff      = home_adj_em - away_adj_em
# oe_matchup   = home_adj_o  - away_adj_d   (home offense vs away defense)
# de_matchup   = away_adj_o  - home_adj_d   (away offense vs home defense)
MARGIN_FEATURES = ['em_diff', 'home_adj_o', 'away_adj_o', 'home_adj_d', 'away_adj_d',
                   'oe_matchup', 'de_matchup']
TOTAL_FEATURES  = ['home_adj_o', 'away_adj_o', 'home_adj_d', 'away_adj_d',
                   'em_diff', 'oe_matchup', 'de_matchup']

XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'verbosity': 0,
}


def db_connect():
    return sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
        _socket_timeout=900,
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data():
    """
    Load completed games with KenPom features from prediction_outcomes.
    Also tries historical_game_features if the table exists (populated by
    BackfillHistoricalData.py once you run it).
    """
    print("Loading training data...")
    frames = []

    try:
        conn = db_connect()
        cursor = conn.cursor()

        # Current season — prediction_outcomes already has KenPom features
        cursor.execute("""
            SELECT
                game_date,
                home_team,
                away_team,
                actual_home_score,
                actual_away_score,
                home_adj_em  AS home_adj_em,
                away_adj_em  AS away_adj_em,
                home_oe      AS home_adj_o,
                away_oe      AS away_adj_o,
                home_de      AS home_adj_d,
                away_de      AS away_adj_d
            FROM workspace.default.prediction_outcomes
            WHERE actual_home_score IS NOT NULL
              AND home_adj_em IS NOT NULL
        """)
        current_df = cursor.fetchall_arrow().to_pandas()
        current_df['season'] = CURRENT_SEASON
        if not current_df.empty:
            frames.append(current_df)
            print(f"  Current season: {len(current_df)} games")

        # Historical seasons (populated by BackfillHistoricalData.py)
        try:
            cursor.execute("""
                SELECT season, game_date, home_team, away_team,
                       actual_home_score, actual_away_score,
                       home_adj_em, away_adj_em,
                       home_adj_o, away_adj_o,
                       home_adj_d, away_adj_d
                FROM workspace.default.historical_game_features
                WHERE actual_home_score IS NOT NULL
                  AND home_adj_em IS NOT NULL
            """)
            hist_df = cursor.fetchall_arrow().to_pandas()
            if not hist_df.empty:
                frames.append(hist_df)
                for season, cnt in hist_df.groupby('season').size().items():
                    print(f"  Season {season}: {cnt} historical games")
        except Exception:
            pass  # table doesn't exist yet — that's fine

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"  Error loading training data: {e}")
        return pd.DataFrame()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(df)} games across {df['season'].nunique()} season(s)")
    return df


def load_current_ratings():
    """Load adj_o, adj_d, adj_em for all teams from the gold rankings table."""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT team_name,
                   adj_offensive_efficiency AS adj_o,
                   adj_defensive_efficiency AS adj_d,
                   adj_efficiency_margin    AS adj_em
            FROM workspace.default_gold.gold_team_rankings
        """)
        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        conn.close()
        print(f"  Loaded ratings for {len(df)} teams")
        return df
    except Exception as e:
        print(f"  Error loading current ratings: {e}")
        return pd.DataFrame()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_features(df):
    df = df.copy()
    df['margin']     = df['actual_home_score'] - df['actual_away_score']
    df['total']      = df['actual_home_score'] + df['actual_away_score']
    df['em_diff']    = df['home_adj_em']  - df['away_adj_em']
    df['oe_matchup'] = df['home_adj_o']   - df['away_adj_d']
    df['de_matchup'] = df['away_adj_o']   - df['home_adj_d']
    return df


def compute_weights(df):
    current_count = int((df['season'] == CURRENT_SEASON).sum())
    weights_map = dict(SEASON_WEIGHTS)

    if current_count >= BOOST_THRESHOLD:
        weights_map[CURRENT_SEASON] = SEASON_WEIGHTS[CURRENT_SEASON] * BOOST_MULTIPLIER
        print(f"  Auto-boost: {current_count} games >= {BOOST_THRESHOLD} → "
              f"current weight {SEASON_WEIGHTS[CURRENT_SEASON]:.1f}x → {weights_map[CURRENT_SEASON]:.1f}x")
    else:
        print(f"  Current season: {current_count} games (boost at {BOOST_THRESHOLD})")

    return df['season'].map(weights_map).fillna(0.1).reset_index(drop=True), weights_map


# =============================================================================
# TRAINING
# =============================================================================

def cv_rmse_xgb(X, y, weights, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []
    for tr, val in kf.split(X):
        m = XGBRegressor(**XGBOOST_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=weights.iloc[tr])
        preds = m.predict(X.iloc[val])
        rmses.append(float(np.sqrt(np.mean((preds - y.iloc[val].values) ** 2))))
    return float(np.mean(rmses))


def train_model(X, y, weights, model_name):
    print(f"\nTraining {model_name}...")
    valid = X.notna().all(axis=1) & y.notna()
    X_c = X[valid].reset_index(drop=True)
    y_c = y[valid].reset_index(drop=True)
    w_c = weights[valid].reset_index(drop=True)
    print(f"  {len(X_c)} samples  ({(~valid).sum()} dropped — NaN)")

    if len(X_c) < 15:
        print(f"  Not enough data yet. Skipping.")
        return None, None, None

    n_splits = min(5, max(3, len(X_c) // 10))
    xgb_rmse = cv_rmse_xgb(X_c, y_c, w_c, n_splits=n_splits)

    # Ridge unweighted (comparison baseline)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_c)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    ridge = Ridge(alpha=1.0)
    ridge_rmses = []
    for tr, val in kf.split(X_scaled):
        ridge.fit(X_scaled[tr], y_c.iloc[tr])
        preds = ridge.predict(X_scaled[val])
        ridge_rmses.append(float(np.sqrt(np.mean((preds - y_c.iloc[val].values) ** 2))))
    ridge_rmse = float(np.mean(ridge_rmses))

    print(f"  XGBoost CV RMSE : {xgb_rmse:.2f} pts")
    print(f"  Ridge CV RMSE   : {ridge_rmse:.2f} pts  (baseline)")

    final = XGBRegressor(**XGBOOST_PARAMS)
    final.fit(X_c, y_c, sample_weight=w_c)

    importances = pd.Series(final.feature_importances_, index=X_c.columns)
    top3 = importances.nlargest(3)
    print(f"  Top features: {', '.join(f'{k} ({v:.3f})' for k, v in top3.items())}")

    return final, xgb_rmse, ridge_rmse


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model, model_name, xgb_rmse, ridge_rmse, n_games, weights_used):
    if model is None:
        return
    model_b64 = base64.b64encode(pickle.dumps(model)).decode('utf-8')
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    weights_json = json.dumps({str(k): v for k, v in weights_used.items()}).replace("'", "''")

    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace.default.ml_models (
                model_name STRING, model_version STRING, model_data STRING,
                training_games INT, xgb_cv_rmse DOUBLE, ridge_cv_rmse DOUBLE,
                season_weights STRING, created_at TIMESTAMP
            ) USING DELTA
        """)
        cursor.execute(f"DELETE FROM workspace.default.ml_models WHERE model_name = '{model_name}'")
        cursor.execute(f"""
            INSERT INTO workspace.default.ml_models VALUES (
                '{model_name}', '{version}', '{model_b64}',
                {n_games}, {xgb_rmse:.4f}, {ridge_rmse:.4f},
                '{weights_json}', current_timestamp()
            )
        """)
        cursor.close()
        conn.close()
        print(f"  Saved {model_name} (RMSE {xgb_rmse:.2f})")
    except Exception as e:
        print(f"  Error saving {model_name}: {e}")


def load_models():
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_name, model_data, model_version, training_games, xgb_cv_rmse, ridge_cv_rmse
            FROM workspace.default.ml_models
            WHERE model_name IN ('margin_model', 'total_model')
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        models = {}
        for row in rows:
            name, data_b64, version, n_games, xgb_rmse, ridge_rmse = row
            models[name] = {
                'model': pickle.loads(base64.b64decode(data_b64)),
                'version': version,
                'training_games': n_games,
                'xgb_rmse': xgb_rmse,
                'ridge_rmse': ridge_rmse,
            }
        return models
    except Exception as e:
        print(f"  Error loading models: {e}")
        return {}


# =============================================================================
# PREDICTION GENERATION
# =============================================================================

def fetch_todays_games():
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).strftime('%Y%m%d')
    today_date = today[:4] + '-' + today[4:6] + '-' + today[6:]
    print(f"\nFetching today's games ({today_date})...")
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        resp = requests.get(url, params={'dates': today, 'limit': 300, 'groups': 50}, timeout=15)
        resp.raise_for_status()
        games = []
        for event in resp.json().get('events', []):
            if event['status']['type']['state'] == 'post':
                continue
            competitors = event['competitions'][0]['competitors']
            home = away = None
            for c in competitors:
                if c['homeAway'] == 'home':
                    home = c['team']['displayName']
                else:
                    away = c['team']['displayName']
            if home and away:
                games.append({'home_team': home, 'away_team': away, 'game_date': today_date})
        print(f"  {len(games)} upcoming games")
        return games
    except Exception as e:
        print(f"  Error fetching games: {e}")
        return []


def generate_predictions(models, games, ratings_df):
    if not models or not games:
        return []
    margin_model = models.get('margin_model', {}).get('model')
    total_model  = models.get('total_model',  {}).get('model')
    if not margin_model or not total_model:
        return []

    version = models.get('margin_model', {}).get('version', 'unknown')
    records = []

    for game in games:
        home_match = find_team_match(game['home_team'], ratings_df)
        away_match = find_team_match(game['away_team'], ratings_df)
        if not home_match or not away_match:
            continue

        h = ratings_df[ratings_df['team_name'] == home_match]
        a = ratings_df[ratings_df['team_name'] == away_match]
        if h.empty or a.empty:
            continue
        h, a = h.iloc[0], a.iloc[0]

        feat = {
            'em_diff':    float(h['adj_em']  - a['adj_em']),
            'home_adj_o': float(h['adj_o']),
            'away_adj_o': float(a['adj_o']),
            'home_adj_d': float(h['adj_d']),
            'away_adj_d': float(a['adj_d']),
            'oe_matchup': float(h['adj_o']   - a['adj_d']),
            'de_matchup': float(a['adj_o']   - h['adj_d']),
        }

        try:
            X_m = pd.DataFrame([[feat[k] for k in MARGIN_FEATURES]], columns=MARGIN_FEATURES)
            X_t = pd.DataFrame([[feat[k] for k in TOTAL_FEATURES]],  columns=TOTAL_FEATURES)
            pred_margin = float(margin_model.predict(X_m)[0])
            pred_total  = float(total_model.predict(X_t)[0])
            pred_home   = (pred_total + pred_margin) / 2
            pred_away   = (pred_total - pred_margin) / 2
            win_prob    = float(1 / (1 + np.exp(-pred_margin / 10)))

            records.append({
                'game_date':               game['game_date'],
                'home_team':               game['home_team'],
                'away_team':               game['away_team'],
                'ml_predicted_home_score': round(pred_home,   1),
                'ml_predicted_away_score': round(pred_away,   1),
                'ml_predicted_margin':     round(pred_margin, 1),
                'ml_predicted_total':      round(pred_total,  1),
                'ml_win_probability':      round(win_prob,    4),
                'model_version':           version,
            })
        except Exception as e:
            print(f"  Prediction error {game['home_team']} vs {game['away_team']}: {e}")

    return records


def upload_predictions(records):
    if not records:
        print("  No ML predictions to upload")
        return
    game_date = records[0]['game_date']
    print(f"Uploading {len(records)} ML predictions for {game_date}...")
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace.default.ml_predictions (
                game_date DATE, home_team STRING, away_team STRING,
                ml_predicted_home_score DOUBLE, ml_predicted_away_score DOUBLE,
                ml_predicted_margin DOUBLE, ml_predicted_total DOUBLE,
                ml_win_probability DOUBLE, model_version STRING, generated_at TIMESTAMP
            ) USING DELTA
        """)
        cursor.execute(f"DELETE FROM workspace.default.ml_predictions WHERE game_date = '{game_date}'")
        for rec in records:
            he = rec['home_team'].replace("'", "''")
            ae = rec['away_team'].replace("'", "''")
            cursor.execute(f"""
                INSERT INTO workspace.default.ml_predictions VALUES (
                    '{rec['game_date']}', '{he}', '{ae}',
                    {rec['ml_predicted_home_score']}, {rec['ml_predicted_away_score']},
                    {rec['ml_predicted_margin']}, {rec['ml_predicted_total']},
                    {rec['ml_win_probability']}, '{rec['model_version']}',
                    current_timestamp()
                )
            """)
        cursor.close()
        conn.close()
        print(f"  Uploaded {len(records)} predictions")
    except Exception as e:
        print(f"  Error uploading predictions: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ML MODEL — NIGHTLY RETRAINING")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%B %d, %Y %H:%M')}\n")

    if not all([DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN]):
        print("Databricks credentials not found. Exiting.")
        return

    # 1. Load training data
    df = load_training_data()

    if df.empty:
        print("\nNo training data yet. Skipping model training.")
    else:
        # 2. Build features
        df = build_features(df)
        weights, weights_used = compute_weights(df)

        # 3. Train margin model
        X_m = df[MARGIN_FEATURES]
        margin_model, margin_xgb, margin_ridge = train_model(
            X_m, df['margin'], weights, "Margin (home - away)"
        )

        # 4. Train total model
        X_t = df[TOTAL_FEATURES]
        total_model, total_xgb, total_ridge = train_model(
            X_t, df['total'], weights, "Total (home + away)"
        )

        n_games = int(X_m.notna().all(axis=1).sum())

        # 5. Save to Databricks
        print("\nSaving models...")
        if margin_model:
            save_model(margin_model, 'margin_model', margin_xgb, margin_ridge, n_games, weights_used)
        if total_model:
            save_model(total_model,  'total_model',  total_xgb,  total_ridge,  n_games, weights_used)

    # 6. Generate today's predictions from saved models
    models = load_models()
    if not models:
        print("\nNo trained models found. Run once more after games have been played.")
        return

    print("\nLoading current KenPom ratings for prediction...")
    ratings_df = load_current_ratings()
    if ratings_df.empty:
        print("Could not load ratings. Skipping predictions.")
        return

    games = fetch_todays_games()
    prediction_records = generate_predictions(models, games, ratings_df)
    upload_predictions(prediction_records)

    # 7. Summary
    print("\n" + "=" * 70)
    if prediction_records:
        print(f"ML PREDICTIONS — {prediction_records[0]['game_date']}")
        print("=" * 70)
        for rec in prediction_records:
            winner = rec['home_team'] if rec['ml_predicted_margin'] > 0 else rec['away_team']
            print(f"  {rec['away_team']} @ {rec['home_team']}")
            print(f"    {winner} wins  {rec['ml_win_probability']:.0%}  |  "
                  f"Score: {rec['ml_predicted_home_score']:.0f}-{rec['ml_predicted_away_score']:.0f}  "
                  f"Total: {rec['ml_predicted_total']:.0f}")
    else:
        print("No ML predictions generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
