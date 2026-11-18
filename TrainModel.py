"""
XGBoost ML Model — Nightly Retraining
Trains on completed-game outcomes using KenPom efficiency + tempo features,
home court, rest, and a leak-free pregame rating computed from game results.
Stores trained models as base64 pickle in Databricks.
Generates today's ML predictions after training.

Training data: workspace.default.ml_game_features
  - Seasons 2024-2026 seeded by BackfillHistoricalData.py (season-end KenPom ratings)
  - Current season appended nightly by sync_current_season() (point-in-time ratings
    from prediction_history where available)
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
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import pytz

from team_names import find_team_match
import Betting
from BackfillHistoricalData import (FEATURE_TABLE, fetch_espn_games_for_date, team_ratings,
                                    upload_features)

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

CURRENT_SEASON = 2027
CURRENT_SEASON_START = date(2026, 11, 1)
EASTERN = pytz.timezone('US/Eastern')

SEASON_WEIGHTS = {2027: 3.0, 2026: 1.5, 2025: 0.75, 2024: 0.30}
BOOST_THRESHOLD = 100
BOOST_MULTIPLIER = 2.0

# Tag stored with each prediction so the dashboard can track this model's ATS record separately
MODEL_FAMILY = 'v2'

# Pregame margin rating (online, leak-free): each game nudges both teams toward the observed margin.
RATING_K = 0.30            # update step size (best of sweep in EvaluateModel.py, Nov 2026)
RATING_HCA = 3.0           # home-court points assumed by the rating update
RATING_CARRYOVER = 0.6     # fraction of last season's rating kept at season start
RATING_NON_D1_START = -20.0
RATING_ERR_CAP = 25.0      # cap per-game surprise so blowouts don't swing ratings
REST_CAP = 10

# margin = home - away ; total = home + away
# Ratings/tempo enter as *_rel = value minus that season's team average, because the
# scale drifts between seasons (e.g. early-2027 KenPom tempo avg ~70 vs ~67.5 historically).
# Tested and dropped: "better team controls pace" features, rest days for totals (no holdout improvement).
MARGIN_FEATURES = ['em_diff', 'home_o_rel', 'away_o_rel', 'home_d_rel', 'away_d_rel',
                   'oe_matchup', 'de_matchup', 'home_court', 'rating_diff',
                   'home_rest', 'away_rest', 'min_games_played']
TOTAL_FEATURES  = ['home_o_rel', 'away_o_rel', 'home_d_rel', 'away_d_rel',
                   'home_tempo_rel', 'away_tempo_rel', 'poss_est', 'kp_total_est', 'home_court',
                   'abs_em_diff', 'min_games_played']

# Published totals use the KenPom tempo formula (kp_total_est): in Nov 2026 evaluation the ML total
# model did not beat it on leak-free current-season games. The total model is still trained and
# saved so EvaluateModel.py / the dashboard can keep comparing.
PUBLISH_ML_TOTAL = False

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


def name_key(team):
    """
    Team-name key for joining ESPN names with names stored in Databricks. The insert code
    escapes ' as '' — Spark SQL reads 'Saint Mary''s' as two adjacent literals and drops the
    apostrophe, so stored names lack it ("Saint Marys Gaels").
    """
    return team.replace("'", "")


def db_connect():
    return sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
        _socket_timeout=900,
    )


# =============================================================================
# DATA SYNC + LOADING
# =============================================================================

def load_current_ratings():
    """Load adj_o, adj_d, adj_em, tempo for all teams from the gold rankings table."""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT team_name,
                   adj_offensive_efficiency AS adj_o,
                   adj_defensive_efficiency AS adj_d,
                   adj_efficiency_margin    AS adj_em,
                   adj_tempo                AS tempo
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


def load_point_in_time_ratings(cursor, start, end):
    """
    Ratings as they were when GamePredictions.py predicted each game (prediction_history).
    Returns {(home_team, away_team): [(game_date, ratings_dict), ...]}.
    """
    cursor.execute(f"""
        SELECT game_date, home_team, away_team, home_oe, home_de, home_adj_em, away_oe, away_de, away_adj_em
        FROM workspace.default.prediction_history
        WHERE game_date BETWEEN '{start - timedelta(days=1)}' AND '{end + timedelta(days=1)}'
          AND home_oe IS NOT NULL AND away_oe IS NOT NULL
    """)
    pit = {}
    for r in cursor.fetchall():
        pit.setdefault((name_key(r[1]), name_key(r[2])), []).append((r[0], {
            'home_adj_o': float(r[3]), 'home_adj_d': float(r[4]), 'home_adj_em': float(r[5]),
            'away_adj_o': float(r[6]), 'away_adj_d': float(r[7]), 'away_adj_em': float(r[8]),
        }))
    return pit


def sync_current_season(ratings_df):
    """Append completed current-season games (through yesterday ET) to FEATURE_TABLE."""
    print("Syncing current-season games...")
    yesterday = datetime.now(EASTERN).date() - timedelta(days=1)
    try:
        conn = db_connect()
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT MAX(game_date) FROM {FEATURE_TABLE} WHERE season = {CURRENT_SEASON}")
            last = cursor.fetchone()[0]
        except Exception:
            last = None
        start = (last + timedelta(days=1)) if last else CURRENT_SEASON_START
        if start > yesterday:
            print(f"  Up to date (through {last})")
            cursor.close(); conn.close()
            return
        pit = load_point_in_time_ratings(cursor, start, yesterday)
        cursor.close(); conn.close()
    except Exception as e:
        print(f"  Error preparing sync: {e}")
        return

    records = []
    d = start
    while d <= yesterday:
        for g in fetch_espn_games_for_date(d.strftime('%Y%m%d')):
            h = team_ratings(g['home_team'], ratings_df)
            a = team_ratings(g['away_team'], ratings_df)
            rec = {**g,
                   'home_adj_o': h['adj_o'], 'home_adj_d': h['adj_d'], 'home_adj_em': h['adj_em'], 'home_tempo': h['tempo'],
                   'away_adj_o': a['adj_o'], 'away_adj_d': a['adj_d'], 'away_adj_em': a['adj_em'], 'away_tempo': a['tempo'],
                   'ratings_point_in_time': False}
            gd = date.fromisoformat(g['game_date'])
            for pdate, vals in pit.get((name_key(g['home_team']), name_key(g['away_team'])), []):
                if abs((pdate - gd).days) <= 1:
                    rec.update(vals)
                    rec['ratings_point_in_time'] = True
                    break
            records.append(rec)
        d += timedelta(days=1)

    n_pit = sum(r['ratings_point_in_time'] for r in records)
    print(f"  {len(records)} games {start} → {yesterday} ({n_pit} with point-in-time ratings)")
    upload_features(records, CURRENT_SEASON,
                    delete_where=f"season = {CURRENT_SEASON} AND game_date BETWEEN '{start}' AND '{yesterday}'")


def load_games():
    """All completed games (every season, including non-D1 rows) from FEATURE_TABLE."""
    print("Loading game features...")
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {FEATURE_TABLE} WHERE actual_home_score IS NOT NULL")
        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"  Error loading games: {e}")
        return pd.DataFrame()
    df['game_date'] = pd.to_datetime(df['game_date'])
    for season, cnt in df.groupby('season').size().items():
        print(f"  Season {season}: {cnt} games")
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_schedule_features(df):
    """
    Walk all games chronologically and attach, for each game, values known BEFORE tip-off:
      home/away_rating (online margin rating), home/away_rest, home/away_games_played.
    Returns (df, state) where state holds end-of-data ratings / last game dates / counts
    for predicting upcoming games.
    """
    df = df.sort_values(['game_date', 'home_team', 'away_team']).reset_index(drop=True)
    # Keys are name_key(team) so lookups work whether a name carries its apostrophe or not
    df['home_team'] = df['home_team'].map(name_key)
    df['away_team'] = df['away_team'].map(name_key)
    ratings, last_played, played = {}, {}, {}
    season_of = {}
    cols = {k: np.empty(len(df)) for k in
            ('home_rating', 'away_rating', 'home_rest', 'away_rest', 'home_gp', 'away_gp')}

    for i, g in enumerate(df.itertuples(index=False)):
        sides = (('home', g.home_team, g.home_adj_em), ('away', g.away_team, g.away_adj_em))
        for side, team, em in sides:
            if season_of.get(team) != g.season:            # first game of a new season for this team
                if team in ratings:
                    ratings[team] *= RATING_CARRYOVER
                played[team] = 0
                last_played.pop(team, None)
                season_of[team] = g.season
            if team not in ratings:
                ratings[team] = 0.0 if pd.notna(em) else RATING_NON_D1_START
            cols[f'{side}_rating'][i] = ratings[team]
            prev = last_played.get(team)
            cols[f'{side}_rest'][i] = min((g.game_date - prev).days, REST_CAP) if prev is not None else REST_CAP
            cols[f'{side}_gp'][i] = played[team]

        hca = 0.0 if g.neutral_site else RATING_HCA
        expected = ratings[g.home_team] - ratings[g.away_team] + hca
        err = np.clip((g.actual_home_score - g.actual_away_score) - expected, -RATING_ERR_CAP, RATING_ERR_CAP)
        ratings[g.home_team] += RATING_K * err
        ratings[g.away_team] -= RATING_K * err
        for team in (g.home_team, g.away_team):
            last_played[team] = g.game_date
            played[team] += 1

    for k, v in cols.items():
        df[k] = v
    state = {'ratings': ratings, 'last_played': last_played, 'played': played, 'season_of': season_of}
    return df, state


def add_matchup_features(df, avg_o=None, avg_tempo=None):
    """
    Derived matchup features. avg_o / avg_tempo default to per-season means of the rows;
    pass explicit values when predicting a handful of upcoming games.
    """
    df = df.copy()
    if avg_o is None:
        # Per-season team averages (one value per team, matching the prediction-time
        # average over the gold table's teams)
        teams = pd.concat([
            df[['season', 'home_team', 'home_adj_o', 'home_tempo']].set_axis(['season', 'team', 'o', 't'], axis=1),
            df[['season', 'away_team', 'away_adj_o', 'away_tempo']].set_axis(['season', 'team', 'o', 't'], axis=1),
        ]).drop_duplicates(['season', 'team'])
        means = teams.groupby('season')[['o', 't']].mean()
        avg_o = df['season'].map(means['o'])
        avg_tempo = df['season'].map(means['t'])

    for side in ('home', 'away'):
        df[f'{side}_o_rel'] = df[f'{side}_adj_o'] - avg_o
        df[f'{side}_d_rel'] = df[f'{side}_adj_d'] - avg_o      # avg D == avg O (points per 100 possessions)
        df[f'{side}_tempo_rel'] = df[f'{side}_tempo'] - avg_tempo

    df['em_diff']     = df['home_adj_em'] - df['away_adj_em']
    df['oe_matchup']  = df['home_adj_o'] - df['away_adj_d']
    df['de_matchup']  = df['away_adj_o'] - df['home_adj_d']
    df['home_court']  = (~df['neutral_site'].astype(bool)).astype(float)
    df['rating_diff'] = df['home_rating'] - df['away_rating']
    df['min_games_played'] = np.minimum(df['home_gp'], df['away_gp'])

    # Tempo: KenPom possession estimate and formula total (same method as GamePredictions.py)
    df['poss_est'] = df['home_tempo'] * df['away_tempo'] / avg_tempo
    df['kp_total_est'] = df['poss_est'] / 100 * (df['home_adj_o'] * df['away_adj_d'] / avg_o
                                                 + df['away_adj_o'] * df['home_adj_d'] / avg_o)

    # Mismatch size (blowouts change end-of-game pace: walk-ups, benches, fouling)
    df['abs_em_diff'] = df['em_diff'].abs()

    df['margin'] = df['actual_home_score'] - df['actual_away_score']
    df['total']  = df['actual_home_score'] + df['actual_away_score']
    return df


def build_training_frame(games):
    """Schedule features over ALL games, then keep D1-vs-D1 rows with complete ratings."""
    games, state = add_schedule_features(games)
    need = ['home_adj_em', 'away_adj_em', 'home_tempo', 'away_tempo']
    d1 = games.dropna(subset=need).copy()
    return add_matchup_features(d1), state


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
    """Nightly training. Honest out-of-sample comparisons live in EvaluateModel.py."""
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

def save_model(model, features, model_name, xgb_rmse, ridge_rmse, n_games, weights_used):
    if model is None:
        return
    # Store the feature list with the model so prediction always uses matching columns
    payload = {'model': model, 'features': list(features)}
    model_b64 = base64.b64encode(pickle.dumps(payload)).decode('utf-8')
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
            payload = pickle.loads(base64.b64decode(data_b64))
            if not isinstance(payload, dict):
                print(f"  {name} is a legacy model without a feature list — skipping until retrained")
                continue
            models[name] = {
                'model': payload['model'],
                'features': payload['features'],
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

ATS_CONFIDENCE_HIGH   = 8.0
ATS_CONFIDENCE_MEDIUM = 4.0


def fetch_spreads_from_history(game_date):
    """Pull vegas_spread for today's games from prediction_history (populated by GamePredictions.py)."""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT home_team, away_team, vegas_spread
            FROM workspace.default.prediction_history
            WHERE game_date = '{game_date}' AND vegas_spread IS NOT NULL
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {(name_key(r[0]), name_key(r[1])): float(r[2]) for r in rows}
    except Exception:
        return {}


def fetch_todays_games():
    today = datetime.now(EASTERN).strftime('%Y%m%d')
    today_date = today[:4] + '-' + today[4:6] + '-' + today[6:]
    print(f"\nFetching today's games ({today_date})...")
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        resp = requests.get(url, params={'dates': today, 'limit': 300, 'groups': 50}, timeout=15)
        resp.raise_for_status()
        games, locked = [], 0
        for event in resp.json().get('events', []):
            # Lock: only games that haven't tipped off are (re)predicted. Started and finished
            # games keep the pick made before tip-off (same rule as GamePredictions.py).
            if event['status']['type']['state'] != 'pre':
                locked += 1
                continue
            competition = event['competitions'][0]
            home = away = home_id = away_id = None
            for c in competition['competitors']:
                if c['homeAway'] == 'home':
                    home, home_id = c['team']['displayName'], int(c['team']['id'])
                else:
                    away, away_id = c['team']['displayName'], int(c['team']['id'])
            if home and away:
                games.append({'home_team': home, 'away_team': away, 'game_date': today_date,
                              'home_id': home_id, 'away_id': away_id,   # ESPN ids (= box-score team_id)
                              'neutral_site': bool(competition.get('neutralSite', False))})
        print(f"  {len(games)} upcoming games")
        if locked:
            print(f"  🔒 {locked} games already started or finished — keeping their v2/v3 picks")
        return games
    except Exception as e:
        print(f"  Error fetching games: {e}")
        return []


def upcoming_feature_rows(games, ratings_df, state):
    """Feature rows for upcoming games, using current KenPom ratings and end-of-data schedule state."""
    today = pd.Timestamp(datetime.now(EASTERN).date())
    rows = []
    for game in games:
        h = team_ratings(game['home_team'], ratings_df)
        a = team_ratings(game['away_team'], ratings_df)
        if h['adj_em'] is None or a['adj_em'] is None:
            missing = game['home_team'] if h['adj_em'] is None else game['away_team']
            print(f"  SKIP (no KenPom match): {game['away_team']} @ {game['home_team']}  — unmatched: {missing}")
            continue
        row = {**game, 'season': CURRENT_SEASON}
        for side, team, r in (('home', game['home_team'], h), ('away', game['away_team'], a)):
            row.update({f'{side}_adj_o': r['adj_o'], f'{side}_adj_d': r['adj_d'],
                        f'{side}_adj_em': r['adj_em'], f'{side}_tempo': r['tempo']})
            # Schedule state is keyed by names as stored in Databricks (apostrophes dropped), while
            # today's games come from ESPN with apostrophes — look up by the shared key.
            key = name_key(team)
            in_season = state['season_of'].get(key) == CURRENT_SEASON
            rating = state['ratings'].get(key, 0.0)
            row[f'{side}_rating'] = rating if in_season else rating * RATING_CARRYOVER
            prev = state['last_played'].get(key) if in_season else None
            row[f'{side}_rest'] = min((today - prev).days, REST_CAP) if prev is not None else REST_CAP
            row[f'{side}_gp'] = state['played'].get(key, 0) if in_season else 0
        row['actual_home_score'] = row['actual_away_score'] = np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return add_matchup_features(pd.DataFrame(rows),
                                avg_o=ratings_df['adj_o'].mean(), avg_tempo=ratings_df['tempo'].mean())


def generate_predictions(models, games, ratings_df, state, spreads=None, bet_params=Betting.SCATTER_DEFAULT):
    if not models or not games:
        return []
    margin = models.get('margin_model')
    total = models.get('total_model')
    if not margin or (PUBLISH_ML_TOTAL and not total):
        return []

    X = upcoming_feature_rows(games, ratings_df, state)
    if X.empty:
        return []
    pred_margins = margin['model'].predict(X[margin['features']])
    pred_totals = (total['model'].predict(X[total['features']]) if PUBLISH_ML_TOTAL
                   else X['kp_total_est'].values)
    return build_records(X, pred_margins, pred_totals, spreads, margin.get('version', 'unknown'),
                         bet_params=bet_params)


def build_records(X, pred_margins, pred_totals, spreads, version, extras=None, bet_params=Betting.SCATTER_DEFAULT):
    """
    Prediction rows for upload. extras: optional list of dicts merged into each record.
    Each row also carries the model's bet (Betting.make_bet on the stored, rounded margin).
    """
    spreads = spreads or {}
    extras = extras or [{}] * len(X)
    records = []
    for (_, game), pred_margin, pred_total, extra in zip(X.iterrows(), pred_margins, pred_totals, extras):
        pred_margin, pred_total = float(pred_margin), float(pred_total)
        pred_home = (pred_total + pred_margin) / 2
        pred_away = (pred_total - pred_margin) / 2
        win_prob = float(1 / (1 + np.exp(-pred_margin / 10)))

        spread = spreads.get((name_key(game['home_team']), name_key(game['away_team'])))  # None = no line, 0 = pick 'em
        ats_edge = abs(pred_margin + spread) if spread is not None else None
        if ats_edge is None:
            ats_conf = 'Low'
        elif ats_edge >= ATS_CONFIDENCE_HIGH:
            ats_conf = 'High'
        elif ats_edge >= ATS_CONFIDENCE_MEDIUM:
            ats_conf = 'Medium'
        else:
            ats_conf = 'Low'

        # Bet on the margin as stored (rounded), the same number grading uses
        bet = Betting.make_bet(game['home_team'], game['away_team'], round(pred_margin, 1), spread, bet_params)

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
            'vegas_spread':            spread,
            'ats_confidence':          ats_conf,
            **Betting.bet_fields(bet),
            **extra,
        })
    return records


def run_v3(models, df_v2, games_today, ratings_df, state, spreads, bet_params=Betting.SCATTER_DEFAULT):
    """
    v3 = box-score ratings model blended with the v2 KenPom model (see ModelV3.py).
    Any failure (e.g. box-score data unavailable) skips v3 with an Actions warning; v2 is unaffected.
    """
    import BoxRatings as B
    import ModelV3 as V3
    print("\n" + "=" * 70)
    print("v3 — BOX-SCORE RATINGS + KENPOM BLEND")
    print("=" * 70)
    try:
        print("Building point-in-time box-score ratings...")
        box, bstate = B.build_asof(V3.SEASONS, current_season=CURRENT_SEASON)
        latest = box['game_date'].max().date()
        stale_days = (datetime.now(EASTERN).date() - latest).days
        if stale_days > 2:
            print(f"::warning::v3 skipped — newest box score is {latest} ({stale_days} days old); v2 unaffected")
            return []
        box = B.add_box_features(box)
        df = V3.merge_frames(df_v2, box, name_key)
        df = df[df['season'] >= 2024].reset_index(drop=True)
        weights, weights_used = compute_weights(df)
        box_model = V3.fit_box_model(df, weights, XGBOOST_PARAMS)
        trust, desc = V3.fit_trust(df, CURRENT_SEASON, MARGIN_FEATURES, XGBOOST_PARAMS, weights_used)
        print(f"  Box model trained on {len(df)} games (box scores through {latest})")
        print(f"  Trust weights: {desc}")
        try:   # current box-score team ratings for the website (BuildSite.py reads this)
            os.makedirs(os.path.join('site', 'data'), exist_ok=True)
            with open(os.path.join('site', 'data', 'box_ratings.json'), 'w') as fh:
                json.dump({'through': str(latest), 'trust': desc, 'teams': B.export_team_ratings(bstate)}, fh)
        except Exception as e:
            print(f"  (could not write box ratings for the site: {e})")

        margin = models.get('margin_model')
        X = upcoming_feature_rows(games_today, ratings_df, state)
        if margin is None or X.empty:
            return []
        Xb = B.upcoming_rows(X.to_dict('records'), bstate)
        if Xb.empty:
            return []
        skipped = len(X) - len(Xb)
        if skipped:
            print(f"  {skipped} game(s) skipped — team not in box-score ratings yet")
        kp = margin['model'].predict(Xb[margin['features']])
        bx = box_model.predict(Xb[V3.BOX_FEATURES])
        share = trust(Xb['box_min_gp'].values)
        blended = share * bx + (1 - share) * kp
        extras = [{'box_margin': round(float(b), 1), 'kp_margin': round(float(k), 1), 'box_share': round(float(s), 3)}
                  for b, k, s in zip(bx, kp, share)]
        return build_records(Xb, blended, Xb['kp_total_est'].values, spreads,
                             f"v3_{datetime.now().strftime('%Y%m%d')}", extras, bet_params=bet_params)
    except Exception as e:
        print(f"::warning::v3 skipped ({type(e).__name__}: {e}); v2 predictions unaffected")
        return []


PREDICTIONS_TABLE = 'workspace.default.ml_predictions'


def upload_predictions(records, family=MODEL_FAMILY, table=PREDICTIONS_TABLE):
    """
    Upsert (MERGE) this model's predictions: a game already stored for this date + family is
    updated in place, a new game is inserted, and every other row (e.g. games that have already
    started, which fetch_todays_games leaves out) is untouched.
    """
    if not records:
        print(f"  No {family} predictions to upload")
        return
    game_date = records[0]['game_date']
    print(f"Uploading {len(records)} {family} predictions for {game_date}...")
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                game_date DATE, home_team STRING, away_team STRING,
                ml_predicted_home_score DOUBLE, ml_predicted_away_score DOUBLE,
                ml_predicted_margin DOUBLE, ml_predicted_total DOUBLE,
                ml_win_probability DOUBLE, model_version STRING, generated_at TIMESTAMP,
                vegas_spread DOUBLE, ats_confidence STRING,
                actual_home_score INT, actual_away_score INT, ml_ats_correct INT
            ) USING DELTA
        """)
        for col_def in [
            "vegas_spread DOUBLE", "ats_confidence STRING",
            "actual_home_score INT", "actual_away_score INT", "ml_ats_correct INT",
            "model_family STRING",
            # v3 components: box-score model margin, KenPom (v2) model margin, weight on the box model
            "box_margin DOUBLE", "kp_margin DOUBLE", "box_share DOUBLE",
        ]:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMNS ({col_def})")
            except Exception:
                pass
        Betting.ensure_bet_columns(cursor, table)

        # One source row per game (v2 and v3 are stored side by side, matched by family).
        # Team names are compared with apostrophes stripped: stored names have lost them
        # (see name_key), and the same escaping is used here so new rows match old ones.
        num = lambda v: 'NULL' if v is None else v
        bet_names = [c for c, _ in Betting.BET_COLUMNS]
        bet_casts = ', '.join(f"CAST({c} AS {kind}) AS {c}" for c, kind in Betting.BET_COLUMNS)
        rows, seen = [], set()
        for rec in records:
            key = (rec['game_date'], name_key(rec['home_team']), name_key(rec['away_team']))
            if key in seen:
                continue
            seen.add(key)
            he = rec['home_team'].replace("'", "''")
            ae = rec['away_team'].replace("'", "''")
            rows.append(f"""(
                '{rec['game_date']}', '{he}', '{ae}',
                {rec['ml_predicted_home_score']}, {rec['ml_predicted_away_score']},
                {rec['ml_predicted_margin']}, {rec['ml_predicted_total']},
                {rec['ml_win_probability']}, '{rec['model_version']}',
                {num(rec['vegas_spread'])}, '{rec['ats_confidence']}', '{family}',
                {num(rec.get('box_margin'))}, {num(rec.get('kp_margin'))}, {num(rec.get('box_share'))},
                {', '.join(Betting.sql_value(rec.get(c)) for c in bet_names)})""")
        value_cols = ('game_date, home_team, away_team, ml_predicted_home_score, ml_predicted_away_score, '
                      'ml_predicted_margin, ml_predicted_total, ml_win_probability, model_version, '
                      'vegas_spread, ats_confidence, model_family, box_margin, kp_margin, box_share, '
                      + ', '.join(bet_names))
        cursor.execute(f"""
            MERGE INTO {table} AS t
            USING (
                SELECT CAST(game_date AS DATE) AS game_date, home_team, away_team,
                       CAST(ml_predicted_home_score AS DOUBLE) AS ml_predicted_home_score,
                       CAST(ml_predicted_away_score AS DOUBLE) AS ml_predicted_away_score,
                       CAST(ml_predicted_margin AS DOUBLE) AS ml_predicted_margin,
                       CAST(ml_predicted_total AS DOUBLE) AS ml_predicted_total,
                       CAST(ml_win_probability AS DOUBLE) AS ml_win_probability,
                       model_version, CAST(vegas_spread AS DOUBLE) AS vegas_spread,
                       ats_confidence, model_family,
                       CAST(box_margin AS DOUBLE) AS box_margin, CAST(kp_margin AS DOUBLE) AS kp_margin,
                       CAST(box_share AS DOUBLE) AS box_share, {bet_casts}
                FROM VALUES {', '.join(rows)} AS v({value_cols})
            ) AS s
            ON  t.game_date = s.game_date
            AND REPLACE(t.home_team, CHR(39), '') = REPLACE(s.home_team, CHR(39), '')
            AND REPLACE(t.away_team, CHR(39), '') = REPLACE(s.away_team, CHR(39), '')
            AND COALESCE(t.model_family, 'v1') = s.model_family
            WHEN MATCHED THEN UPDATE SET
                ml_predicted_home_score = s.ml_predicted_home_score,
                ml_predicted_away_score = s.ml_predicted_away_score,
                ml_predicted_margin = s.ml_predicted_margin,
                ml_predicted_total = s.ml_predicted_total,
                ml_win_probability = s.ml_win_probability,
                model_version = s.model_version,
                generated_at = current_timestamp(),
                vegas_spread = s.vegas_spread,
                ats_confidence = s.ats_confidence,
                box_margin = s.box_margin, kp_margin = s.kp_margin, box_share = s.box_share,
                {', '.join(f'{c} = s.{c}' for c in bet_names)}
            WHEN NOT MATCHED THEN INSERT
                (game_date, home_team, away_team,
                 ml_predicted_home_score, ml_predicted_away_score,
                 ml_predicted_margin, ml_predicted_total,
                 ml_win_probability, model_version, generated_at,
                 vegas_spread, ats_confidence,
                 actual_home_score, actual_away_score, ml_ats_correct, model_family,
                 box_margin, kp_margin, box_share, {', '.join(bet_names)})
            VALUES
                (s.game_date, s.home_team, s.away_team,
                 s.ml_predicted_home_score, s.ml_predicted_away_score,
                 s.ml_predicted_margin, s.ml_predicted_total,
                 s.ml_win_probability, s.model_version, current_timestamp(),
                 s.vegas_spread, s.ats_confidence,
                 NULL, NULL, NULL, s.model_family,
                 s.box_margin, s.kp_margin, s.box_share, {', '.join(f's.{c}' for c in bet_names)})
        """)
        cursor.close()
        conn.close()
        print(f"  Uploaded {len(rows)} predictions (updated or added; started games untouched)")
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

    ratings_df = load_current_ratings()
    if ratings_df.empty:
        print("Could not load ratings. Exiting.")
        return

    # 1. Sync + load training data
    sync_current_season(ratings_df)
    games = load_games()
    if games.empty:
        print("\nNo training data yet. Exiting.")
        return

    # 2. Build features (schedule state is also needed to predict today's games)
    df, state = build_training_frame(games)
    print(f"  {len(df)} D1-vs-D1 games with complete features")
    weights, weights_used = compute_weights(df)
    df = df.reset_index(drop=True)

    # 3-4. Train margin + total models
    margin_model, margin_xgb, margin_ridge = train_model(
        df[MARGIN_FEATURES], df['margin'], weights, "Margin (home - away)")
    total_model, total_xgb, total_ridge = train_model(
        df[TOTAL_FEATURES], df['total'], weights, "Total (home + away)")

    # 5. Save to Databricks
    print("\nSaving models...")
    n_games = len(df)
    save_model(margin_model, MARGIN_FEATURES, 'margin_model', margin_xgb, margin_ridge, n_games, weights_used)
    save_model(total_model, TOTAL_FEATURES, 'total_model', total_xgb, total_ridge, n_games, weights_used)

    # 6. Generate today's predictions from saved models
    models = load_models()
    if not models:
        print("\nNo trained models found.")
        return

    games_today = fetch_todays_games()
    today_date = games_today[0]['game_date'] if games_today else datetime.now(EASTERN).strftime('%Y-%m-%d')
    spreads = fetch_spreads_from_history(today_date)
    print(f"  Loaded spreads for {len(spreads)} games from prediction_history")
    try:
        conn = db_connect(); cursor = conn.cursor()
        bet_params = Betting.fit_for_run(cursor, today_date)
        cursor.close(); conn.close()
    except Exception as e:
        bet_params = Betting.SCATTER_DEFAULT
        print(f"  Could not fit the bet scatter ({e}); using the default")
    prediction_records = generate_predictions(models, games_today, ratings_df, state, spreads=spreads,
                                              bet_params=bet_params)
    upload_predictions(prediction_records)

    # 6b. v3 (box-score ratings + KenPom blend), published alongside v2
    v3_records = run_v3(models, df, games_today, ratings_df, state, spreads, bet_params=bet_params)
    import ModelV3
    upload_predictions(v3_records, family=ModelV3.MODEL_FAMILY)
    if v3_records:
        print(f"\n  v3 picks ({len(v3_records)} games):")
        for rec in v3_records:
            print(f"    {rec['away_team']} @ {rec['home_team']}:  v3 {rec['ml_predicted_margin']:+.1f}  "
                  f"(box {rec['box_margin']:+.1f}, KenPom {rec['kp_margin']:+.1f}, box share {rec['box_share']:.2f})")

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
