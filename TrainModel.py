"""
XGBoost ML Model — Nightly Retraining
Trains on 3+ seasons of Torvik efficiency data to predict game margin and total.
Stores trained models as base64 pickle in Databricks so they persist across
ephemeral CI runners. Generates today's ML predictions after training.

Pipeline position: runs AFTER UpdateResults.py so each training cycle
includes last night's completed games.
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

# Base season weights — higher = more influence on model
# Auto-boost doubles the current-season weight once 100+ games are available
SEASON_WEIGHTS = {2027: 3.0, 2026: 1.5, 2025: 0.75, 2024: 0.30}
BOOST_THRESHOLD = 100   # current-season games needed to trigger boost
BOOST_MULTIPLIER = 2.0  # multiply current-season weight when boosted

# Features used for each target
# margin = actual_home_score - actual_away_score
# total  = actual_home_score + actual_away_score
MARGIN_FEATURES = [
    'em_diff', 'home_adj_o', 'away_adj_o', 'home_adj_d', 'away_adj_d',
    'home_barthag', 'away_barthag', 'oe_matchup', 'de_matchup',
]
TOTAL_FEATURES = [
    'avg_tempo', 'home_adj_o', 'away_adj_o', 'home_adj_d', 'away_adj_d',
    'avg_barthag', 'em_diff', 'oe_matchup', 'de_matchup',
]

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


# =============================================================================
# DATABASE
# =============================================================================

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

def load_historical_features():
    """Load pre-joined game features for seasons 2024-2026 from Databricks."""
    print("Loading historical game features (2024-2026)...")
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT season, game_date, home_team, away_team,
                   actual_home_score, actual_away_score,
                   home_barthag, home_adj_o, home_adj_d, home_adj_t,
                   home_efg_o, home_efg_d, home_tov_o, home_tov_d, home_orb,
                   away_barthag, away_adj_o, away_adj_d, away_adj_t,
                   away_efg_o, away_efg_d, away_tov_o, away_tov_d, away_orb
            FROM workspace.default.historical_game_features
            WHERE actual_home_score IS NOT NULL
              AND home_barthag IS NOT NULL
              AND away_barthag IS NOT NULL
        """)
        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        conn.close()
        print(f"  Loaded {len(df)} historical games")
        return df
    except Exception as e:
        print(f"  Error loading historical features: {e}")
        return pd.DataFrame()


def load_current_season_data():
    """
    Load this season's completed games from prediction_outcomes,
    then join with torvik_ratings_2027 to get Torvik features.
    """
    print(f"Loading current season ({CURRENT_SEASON}) data...")
    try:
        conn = db_connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT game_date, home_team, away_team,
                   actual_home_score, actual_away_score
            FROM workspace.default.prediction_outcomes
            WHERE actual_home_score IS NOT NULL
        """)
        outcomes_df = cursor.fetchall_arrow().to_pandas()

        cursor.execute("""
            SELECT team_name, barthag, adj_o, adj_d, adj_t,
                   efg_o, efg_d, tov_o, tov_d, orb
            FROM workspace.default.torvik_ratings_2027
        """)
        torvik_df = cursor.fetchall_arrow().to_pandas()

        cursor.close()
        conn.close()

        print(f"  {len(outcomes_df)} completed games,  {len(torvik_df)} Torvik teams")

        if outcomes_df.empty or torvik_df.empty:
            return pd.DataFrame()

        torvik_avail = [c for c in ['barthag', 'adj_o', 'adj_d', 'adj_t', 'efg_o', 'efg_d', 'tov_o', 'tov_d', 'orb']
                        if c in torvik_df.columns]
        records = []
        skipped = 0

        for _, row in outcomes_df.iterrows():
            home_match = find_team_match(row['home_team'], torvik_df)
            away_match = find_team_match(row['away_team'], torvik_df)
            if not home_match or not away_match:
                skipped += 1
                continue

            home_torvik = torvik_df[torvik_df['team_name'] == home_match]
            away_torvik = torvik_df[torvik_df['team_name'] == away_match]
            if home_torvik.empty or away_torvik.empty:
                skipped += 1
                continue

            home_torvik = home_torvik.iloc[0]
            away_torvik = away_torvik.iloc[0]

            rec = {
                'season': CURRENT_SEASON,
                'game_date': row['game_date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'actual_home_score': row['actual_home_score'],
                'actual_away_score': row['actual_away_score'],
            }
            for col in torvik_avail:
                rec[f'home_{col}'] = home_torvik.get(col)
                rec[f'away_{col}'] = away_torvik.get(col)
            records.append(rec)

        df = pd.DataFrame(records) if records else pd.DataFrame()
        print(f"  Matched {len(records)}/{len(outcomes_df)} games  ({skipped} skipped — no Torvik match)")
        return df

    except Exception as e:
        print(f"  Error loading current season: {e}")
        return pd.DataFrame()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_features(df):
    """Compute derived features from raw Torvik columns."""
    df = df.copy()

    df['margin'] = df['actual_home_score'] - df['actual_away_score']
    df['total']  = df['actual_home_score'] + df['actual_away_score']

    # Composite efficiency features
    df['em_diff']     = df['home_barthag'] - df['away_barthag']
    df['oe_matchup']  = df['home_adj_o']   - df['away_adj_d']   # home offense vs away defense
    df['de_matchup']  = df['away_adj_o']   - df['home_adj_d']   # away offense vs home defense
    df['avg_tempo']   = (df['home_adj_t']  + df['away_adj_t'])  / 2
    df['avg_barthag'] = (df['home_barthag'] + df['away_barthag']) / 2

    return df


def compute_weights(df):
    """Assign sample weights by season, with optional auto-boost for current season."""
    current_count = int((df['season'] == CURRENT_SEASON).sum())
    weights_map = dict(SEASON_WEIGHTS)

    if current_count >= BOOST_THRESHOLD:
        weights_map[CURRENT_SEASON] = SEASON_WEIGHTS[CURRENT_SEASON] * BOOST_MULTIPLIER
        print(f"  Auto-boost: {current_count} current-season games >= {BOOST_THRESHOLD} → weight {SEASON_WEIGHTS[CURRENT_SEASON]:.1f}x → {weights_map[CURRENT_SEASON]:.1f}x")
    else:
        print(f"  Current season: {current_count} games (boost kicks in at {BOOST_THRESHOLD})")

    return df['season'].map(weights_map).fillna(0.1).reset_index(drop=True), weights_map


# =============================================================================
# TRAINING
# =============================================================================

def cv_rmse_xgb(X, y, weights, n_splits=5):
    """5-fold CV RMSE for XGBoost, correctly subsetting sample weights per fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_tr = weights.iloc[train_idx]

        m = XGBRegressor(**XGBOOST_PARAMS)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        preds = m.predict(X_val)
        rmses.append(float(np.sqrt(np.mean((preds - y_val.values) ** 2))))
    return float(np.mean(rmses))


def train_model(X, y, weights, model_name):
    """
    Train XGBoost model with 5-fold CV.
    Also measures unweighted Ridge CV as a comparison baseline.

    Returns (model, xgb_rmse, ridge_rmse) or (None, None, None) if data too small.
    """
    print(f"\nTraining {model_name}...")

    valid_mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_mask].reset_index(drop=True)
    y_clean = y[valid_mask].reset_index(drop=True)
    w_clean = weights[valid_mask].reset_index(drop=True)

    dropped = int((~valid_mask).sum())
    print(f"  {len(X_clean)} samples  ({dropped} dropped — NaN features)")

    if len(X_clean) < 50:
        print(f"  Not enough data. Skipping.")
        return None, None, None

    # XGBoost 5-fold CV with correct per-fold weights
    xgb_rmse = cv_rmse_xgb(X_clean, y_clean, w_clean)

    # Ridge unweighted (comparison baseline only)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    ridge = Ridge(alpha=1.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_rmses = []
    for tr, val in kf.split(X_scaled):
        ridge.fit(X_scaled[tr], y_clean.iloc[tr])
        preds = ridge.predict(X_scaled[val])
        ridge_rmses.append(float(np.sqrt(np.mean((preds - y_clean.iloc[val].values) ** 2))))
    ridge_rmse = float(np.mean(ridge_rmses))

    print(f"  XGBoost CV RMSE : {xgb_rmse:.3f} pts")
    print(f"  Ridge CV RMSE   : {ridge_rmse:.3f} pts  (baseline, unweighted)")

    # Final model trained on all clean data with weights
    final_model = XGBRegressor(**XGBOOST_PARAMS)
    final_model.fit(X_clean, y_clean, sample_weight=w_clean)

    # Feature importances
    importances = pd.Series(final_model.feature_importances_, index=X_clean.columns)
    top3 = importances.nlargest(3)
    print(f"  Top features: {', '.join(f'{k} ({v:.3f})' for k, v in top3.items())}")

    return final_model, xgb_rmse, ridge_rmse


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model, model_name, xgb_rmse, ridge_rmse, n_games, weights_used):
    """Serialize model as base64 pickle and upsert into workspace.default.ml_models."""
    if model is None:
        return

    model_b64 = base64.b64encode(pickle.dumps(model)).decode('utf-8')
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    weights_json = json.dumps({str(k): v for k, v in weights_used.items()})

    print(f"  Saving {model_name} to Databricks (v{version})...")
    try:
        conn = db_connect()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace.default.ml_models (
                model_name STRING,
                model_version STRING,
                model_data STRING,
                training_games INT,
                xgb_cv_rmse DOUBLE,
                ridge_cv_rmse DOUBLE,
                season_weights STRING,
                created_at TIMESTAMP
            ) USING DELTA
        """)
        cursor.execute(f"DELETE FROM workspace.default.ml_models WHERE model_name = '{model_name}'")

        wj = weights_json.replace("'", "''")
        cursor.execute(f"""
            INSERT INTO workspace.default.ml_models VALUES (
                '{model_name}', '{version}', '{model_b64}',
                {n_games}, {xgb_rmse:.4f}, {ridge_rmse:.4f},
                '{wj}', current_timestamp()
            )
        """)
        cursor.close()
        conn.close()
        print(f"  Saved — RMSE {xgb_rmse:.3f}")

    except Exception as e:
        print(f"  Error saving {model_name}: {e}")


def load_models():
    """Load margin and total models from Databricks. Returns dict keyed by model_name."""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_name, model_data, model_version, training_games,
                   xgb_cv_rmse, ridge_cv_rmse
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
    """Fetch today's scheduled (not yet started) games from ESPN."""
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).strftime('%Y%m%d')
    print(f"\nFetching today's games ({today}) for ML predictions...")

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {'dates': today, 'limit': 300, 'groups': 50}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()

        games = []
        today_date = today[:4] + '-' + today[4:6] + '-' + today[6:]
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
        print(f"  Error fetching today's games: {e}")
        return []


def generate_predictions(models, games, torvik_df):
    """Run trained models on today's matchups. Returns list of prediction records."""
    if not models or not games:
        return []

    margin_model = models.get('margin_model', {}).get('model')
    total_model  = models.get('total_model',  {}).get('model')
    if not margin_model or not total_model:
        print("  Models missing — cannot predict")
        return []

    model_version = models.get('margin_model', {}).get('version', 'unknown')
    records = []

    for game in games:
        home_match = find_team_match(game['home_team'], torvik_df)
        away_match = find_team_match(game['away_team'], torvik_df)
        if not home_match or not away_match:
            continue

        home_row = torvik_df[torvik_df['team_name'] == home_match]
        away_row = torvik_df[torvik_df['team_name'] == away_match]
        if home_row.empty or away_row.empty:
            continue

        h = home_row.iloc[0]
        a = away_row.iloc[0]

        feat = {
            'em_diff':     float(h.get('barthag', 0.5) - a.get('barthag', 0.5)),
            'home_adj_o':  float(h.get('adj_o', 100)),
            'away_adj_o':  float(a.get('adj_o', 100)),
            'home_adj_d':  float(h.get('adj_d', 100)),
            'away_adj_d':  float(a.get('adj_d', 100)),
            'home_barthag': float(h.get('barthag', 0.5)),
            'away_barthag': float(a.get('barthag', 0.5)),
            'oe_matchup':  float(h.get('adj_o', 100) - a.get('adj_d', 100)),
            'de_matchup':  float(a.get('adj_o', 100) - h.get('adj_d', 100)),
            'avg_tempo':   float((h.get('adj_t', 68) + a.get('adj_t', 68)) / 2),
            'avg_barthag': float((h.get('barthag', 0.5) + a.get('barthag', 0.5)) / 2),
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
                'game_date':              game['game_date'],
                'home_team':              game['home_team'],
                'away_team':              game['away_team'],
                'ml_predicted_home_score': round(pred_home, 1),
                'ml_predicted_away_score': round(pred_away, 1),
                'ml_predicted_margin':     round(pred_margin, 1),
                'ml_predicted_total':      round(pred_total, 1),
                'ml_win_probability':      round(win_prob, 4),
                'model_version':           model_version,
            })
        except Exception as e:
            print(f"  Prediction error {game['home_team']} vs {game['away_team']}: {e}")

    return records


def upload_predictions(records):
    """Store today's ML predictions in Databricks, replacing any existing rows for today."""
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
                game_date DATE,
                home_team STRING,
                away_team STRING,
                ml_predicted_home_score DOUBLE,
                ml_predicted_away_score DOUBLE,
                ml_predicted_margin DOUBLE,
                ml_predicted_total DOUBLE,
                ml_win_probability DOUBLE,
                model_version STRING,
                generated_at TIMESTAMP
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
        print(f"  Error uploading ML predictions: {e}")


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

    # ── 1. Load data ─────────────────────────────────────────────────────────
    historical_df   = load_historical_features()
    current_df      = load_current_season_data()

    all_dfs = [df for df in [historical_df, current_df] if not df.empty]

    if not all_dfs:
        print("\nNo training data yet — run BackfillHistoricalData.py first.")
        print("Attempting to load existing model for today's predictions...\n")
    else:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal training data: {len(combined_df)} games")
        for season, count in combined_df.groupby('season').size().items():
            print(f"  Season {season}: {count} games")

        # ── 2. Build features ─────────────────────────────────────────────────
        combined_df = build_features(combined_df)
        weights, weights_used = compute_weights(combined_df)

        # ── 3. Train models ───────────────────────────────────────────────────
        X_margin = combined_df[MARGIN_FEATURES]
        y_margin = combined_df['margin']
        margin_model, margin_xgb_rmse, margin_ridge_rmse = train_model(
            X_margin, y_margin, weights, "Margin Model (home - away)"
        )

        X_total = combined_df[TOTAL_FEATURES]
        y_total = combined_df['total']
        total_model, total_xgb_rmse, total_ridge_rmse = train_model(
            X_total, y_total, weights, "Total Model (home + away)"
        )

        n_games = int(X_margin.notna().all(axis=1).sum())

        # ── 4. Persist models ─────────────────────────────────────────────────
        print("\nSaving models...")
        if margin_model:
            save_model(margin_model, 'margin_model', margin_xgb_rmse, margin_ridge_rmse, n_games, weights_used)
        if total_model:
            save_model(total_model, 'total_model', total_xgb_rmse, total_ridge_rmse, n_games, weights_used)

    # ── 5. Generate today's predictions ──────────────────────────────────────
    models = load_models()
    if not models:
        print("\nNo trained models available. Skipping prediction generation.")
        print("Run BackfillHistoricalData.py then re-run this script.")
        return

    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT team_name, barthag, adj_o, adj_d, adj_t
            FROM workspace.default.torvik_ratings_2027
        """)
        torvik_df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"\nError loading Torvik for predictions: {e}")
        return

    games = fetch_todays_games()
    prediction_records = generate_predictions(models, games, torvik_df)
    upload_predictions(prediction_records)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if prediction_records:
        print(f"ML PREDICTIONS — {prediction_records[0]['game_date']}")
        print("=" * 70)
        for rec in prediction_records:
            margin = rec['ml_predicted_margin']
            winner = rec['home_team'] if margin > 0 else rec['away_team']
            print(f"  {rec['away_team']} @ {rec['home_team']}")
            print(f"    ML Winner: {winner}  ({rec['ml_win_probability']:.0%})  "
                  f"Score: {rec['ml_predicted_home_score']:.0f}-{rec['ml_predicted_away_score']:.0f}  "
                  f"Total: {rec['ml_predicted_total']:.0f}")
    else:
        print("No ML predictions generated for today.")
    print("=" * 70)


if __name__ == "__main__":
    main()
