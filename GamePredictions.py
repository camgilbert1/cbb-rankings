"""
College Basketball Game Predictions
Fetches today's games and generates predictions using KenPom efficiency metrics
"""

import pandas as pd
import requests
from databricks import sql
import os
import math
from collections import Counter
from datetime import datetime, timedelta
from statistics import NormalDist, median
from dotenv import load_dotenv
import pytz

# Load environment variables from .env file for local development
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DAY_START_HOUR_ET = 4   # betting day boundary for Vegas lines (see get_vegas_spreads)

from team_names import find_team_match, odds_name_for, unmapped_odds_teams, validate_d1_table
import Betting


# =============================================================================
# DATA FETCHING
# =============================================================================

def consensus_line(lines):
    """Most common line across bookmakers; on a tie, the median rounded to the nearest half point."""
    counts = Counter(lines)
    top = max(counts.values())
    modes = [v for v, c in counts.items() if c == top]
    line = modes[0] if len(modes) == 1 else round(median(lines) * 2) / 2
    return line + 0.0  # turn -0.0 into 0.0 (pick 'em)


def get_vegas_spreads(game_date=None):
    """
    Fetch Vegas spreads from The Odds API for games tipping off on game_date.

    The feed covers several days of games. Only games tipping off between 4:00 AM ET on game_date
    and 3:59 AM ET the next morning are kept, so a team's next game can't overwrite today's line. Each game's line is the
    consensus (most common) home spread across bookmakers.

    Args:
        game_date (str): YYYYMMDD, Eastern time. Defaults to today.

    Returns:
        dict: Odds API team name -> that team's spread (negative = favorite)
    """
    if not ODDS_API_KEY:
        print("⚠️  No Odds API key found, spreads will not be available")
        return {}

    eastern = pytz.timezone('US/Eastern')
    if game_date is None:
        game_date = datetime.now(eastern).strftime('%Y%m%d')

    try:
        url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/"
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us',
            'markets': 'spreads',
            'oddsFormat': 'american'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        # Log API usage
        remaining = response.headers.get('x-requests-remaining', 'unknown')
        print(f"  Odds API requests remaining: {remaining}")

        data = response.json()

        # Build mapping of team names to spreads (today's games only)
        spreads = {}
        other_days = 0
        for game in data:
            tip = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00')).astimezone(eastern)
            # The betting day runs 4:00 AM ET to 3:59 AM ET the next morning, so a late tip
            # (10:30 PM Mountain = 12:30 AM ET) counts toward the evening ESPN lists it on.
            if (tip - timedelta(hours=DAY_START_HOUR_ET)).strftime('%Y%m%d') != game_date:
                other_days += 1
                continue
            home_team = game['home_team']
            away_team = game['away_team']

            # Home spread from every bookmaker that posts one
            lines = []
            for bookmaker in game.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market['key'] != 'spreads':
                        continue
                    point = next((o.get('point') for o in market['outcomes'] if o['name'] == home_team), None)
                    if point is not None:
                        lines.append(float(point))
            if not lines:
                continue

            line = consensus_line(lines)
            if home_team in spreads or away_team in spreads:
                print(f"  ⚠️  {away_team} @ {home_team}: a team appears twice on {game_date}; keeping the first line")
                continue
            spreads[home_team] = line          # negative means home favorite
            spreads[away_team] = -line + 0.0
            spread_detail = ', '.join(f"{v:+g} ×{c}" for v, c in Counter(lines).most_common())
            print(f"  {away_team} @ {home_team}: {home_team} {line:+.1f} ({len(lines)} books: {spread_detail})")

        print(f"✓ Fetched spreads for {len(spreads) // 2} games on {game_date} "
              f"({other_days} games on other days ignored)")
        print(f"  Odds API teams: {sorted(spreads.keys())}")
        return spreads

    except Exception as e:
        print(f"⚠️  Error fetching spreads: {e}")
        return {}




def get_todays_games(date=None):
    """
    Fetch scheduled college basketball games from ESPN API.

    Args:
        date (str): Date string in YYYYMMDD format. Defaults to today (Eastern time).

    Returns:
        pd.DataFrame: Scheduled games with neutral_site flag
    """
    if date is None:
        eastern = pytz.timezone('US/Eastern')
        date = datetime.now(eastern).strftime('%Y%m%d')
    print(f"Fetching games from ESPN for {date}...")

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {'dates': date, 'limit': 300, 'groups': 50}
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        games = data.get('events', [])

        game_list = []
        for game in games:
            competition = game['competitions'][0]
            competitors = competition['competitors']

            # Find home and away teams
            home_team = next(c for c in competitors if c['homeAway'] == 'home')
            away_team = next(c for c in competitors if c['homeAway'] == 'away')

            # Detect neutral-site games (NCAA tournament, conference tournaments, bowls)
            neutral_site = competition.get('neutralSite', False)

            game_list.append({
                'game_id': game['id'],
                'game_time': game['date'],
                'home_team': home_team['team']['displayName'],
                'away_team': away_team['team']['displayName'],
                'status': game['status']['type']['name'],
                'neutral_site': neutral_site,
            })

        df = pd.DataFrame(game_list)
        neutral_count = df['neutral_site'].sum() if not df.empty else 0
        print(f"✓ Found {len(df)} games ({neutral_count} neutral-site)")
        return df

    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return pd.DataFrame()


def get_team_stats_from_databricks():
    """
    Load team statistics from Databricks gold table

    Returns:
        pd.DataFrame: Team statistics
    """
    print("\nLoading team stats from Databricks...")

    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN,
            _socket_timeout=900
        )

        cursor = connection.cursor()
        cursor.execute("""
            SELECT
                team_name,
                adj_efficiency_margin,
                adj_offensive_efficiency,
                adj_defensive_efficiency,
                adj_tempo,
                overall_rank
            FROM workspace.default_gold.gold_team_rankings
        """)

        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        connection.close()

        print(f"✓ Loaded stats for {len(df)} teams")
        return df

    except Exception as e:
        print(f"❌ Error loading team stats: {e}")
        return pd.DataFrame()


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

def calculate_matchup_features(home_team, away_team, team_stats, neutral_site=False):
    """
    Calculate prediction features for a matchup

    Args:
        home_team (str): Home team name
        away_team (str): Away team name
        team_stats (pd.DataFrame): Team statistics
        neutral_site (bool): True for neutral-site games (tournament, etc.) — zeroes home court advantage

    Returns:
        dict: Calculated features
    """
    # Match ESPN team names to KenPom team names
    home_match = find_team_match(home_team, team_stats)
    away_match = find_team_match(away_team, team_stats)

    if home_match and home_match != home_team:
        print(f"  📎 {home_team} → KenPom: '{home_match}'")
    if away_match and away_match != away_team:
        print(f"  📎 {away_team} → KenPom: '{away_match}'")

    if not home_match or not away_match:
        print(f"  ⚠️  Could not find stats for {home_team if not home_match else away_team}")
        return None

    # Find team stats
    home_stats = team_stats[team_stats['team_name'] == home_match]
    away_stats = team_stats[team_stats['team_name'] == away_match]

    if home_stats.empty or away_stats.empty:
        return None

    home_stats = home_stats.iloc[0]
    away_stats = away_stats.iloc[0]

    # No home court advantage at neutral sites (NCAA tournament, conference tournaments)
    hca = 0.0 if neutral_site else 3.5

    # Calculate differences
    features = {
        'efficiency_margin_diff': home_stats['adj_efficiency_margin'] - away_stats['adj_efficiency_margin'],
        'off_efficiency_diff': home_stats['adj_offensive_efficiency'] - away_stats['adj_offensive_efficiency'],
        'def_efficiency_diff': home_stats['adj_defensive_efficiency'] - away_stats['adj_defensive_efficiency'],
        'tempo_diff': home_stats['adj_tempo'] - away_stats['adj_tempo'],
        'rank_diff': away_stats['overall_rank'] - home_stats['overall_rank'],
        'home_court_advantage': hca,
        'neutral_site': neutral_site,
        'home_oe': float(home_stats['adj_offensive_efficiency']),
        'home_de': float(home_stats['adj_defensive_efficiency']),
        'away_oe': float(away_stats['adj_offensive_efficiency']),
        'away_de': float(away_stats['adj_defensive_efficiency']),
        'home_adj_em': float(home_stats['adj_efficiency_margin']),
        'away_adj_em': float(away_stats['adj_efficiency_margin']),
        'home_rank': int(home_stats['overall_rank']),
        'away_rank': int(away_stats['overall_rank']),
        # KenPom possession estimate: home_tempo × away_tempo / national avg tempo
        'expected_tempo': float(home_stats['adj_tempo']) * float(away_stats['adj_tempo']) / float(team_stats['adj_tempo'].mean()),
    }

    return features


# Win probability = Phi(predicted margin / width), where width is the typical size of the Formula's
# miss in points. The width is re-fit every run on recently graded games (see fit_win_prob_width).
WIN_PROB_WIDTH_DEFAULT = 15.5   # fit on Nov 2026 games; used until enough games are graded
WIN_PROB_WIDTH_MIN, WIN_PROB_WIDTH_MAX = 11.0, 18.0
WIN_PROB_WINDOW_DAYS = 30
WIN_PROB_MIN_GAMES = 150


def fit_win_prob_width(graded, day):
    """
    Width (points) for Phi(margin / width), fit on Formula games graded before `day`.

    graded: DataFrame with game_date (datetime64), margin (predicted home - away), home_won (bool)
            for the current season.
    Uses the last WIN_PROB_WINDOW_DAYS days if they hold WIN_PROB_MIN_GAMES games, else the whole
    season, else the default. The fit is clamped to [WIN_PROB_WIDTH_MIN, WIN_PROB_WIDTH_MAX].
    Returns (width, description).
    """
    day = pd.Timestamp(day)
    hist = graded[graded['game_date'] < day]
    recent = hist[hist['game_date'] >= day - pd.Timedelta(days=WIN_PROB_WINDOW_DAYS)]
    if len(recent) >= WIN_PROB_MIN_GAMES:
        sample, source = recent, f"{len(recent)} games, last {WIN_PROB_WINDOW_DAYS} days"
    elif len(hist) >= WIN_PROB_MIN_GAMES:
        sample, source = hist, f"{len(hist)} games, season to date"
    else:
        return WIN_PROB_WIDTH_DEFAULT, f"default ({len(hist)} graded games < {WIN_PROB_MIN_GAMES})"

    # Maximum-likelihood width on a 0.1-point grid
    phi = NormalDist().cdf
    margins, won = sample['margin'].tolist(), sample['home_won'].tolist()
    best_width, best_ll = None, None
    for tenths in range(int(WIN_PROB_WIDTH_MIN * 10) - 30, int(WIN_PROB_WIDTH_MAX * 10) + 71):
        w = tenths / 10
        ll = sum(math.log(max(1e-9, phi(m / w) if y else 1 - phi(m / w))) for m, y in zip(margins, won))
        if best_ll is None or ll > best_ll:
            best_width, best_ll = w, ll
    width = min(WIN_PROB_WIDTH_MAX, max(WIN_PROB_WIDTH_MIN, best_width))
    note = '' if width == best_width else f", clamped from {best_width}"
    return width, f"{source}{note}"


def load_graded_formula_games(season_start):
    """Current-season graded Formula games (predicted margin + result) for fit_win_prob_width."""
    try:
        connection = sql.connect(server_hostname=DATABRICKS_HOST, http_path=DATABRICKS_HTTP_PATH,
                                 access_token=DATABRICKS_TOKEN)
        cursor = connection.cursor()
        cursor.execute(f"""
            SELECT game_date,
                   predicted_home_score - predicted_away_score AS margin,
                   actual_home_score > actual_away_score AS home_won
            FROM workspace.default.prediction_outcomes
            WHERE game_date >= '{season_start}'
              AND predicted_home_score > 0
              AND actual_home_score IS NOT NULL AND actual_home_score <> actual_away_score
        """)
        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        connection.close()
        df['game_date'] = pd.to_datetime(df['game_date'])
        return df
    except Exception as e:
        print(f"⚠️  Could not load graded games for the win-probability fit ({e}); using the default width")
        return pd.DataFrame(columns=['game_date', 'margin', 'home_won'])


def predict_game(home_team, away_team, features, national_avg=106.0, win_prob_width=WIN_PROB_WIDTH_DEFAULT):
    """
    Predict game outcome using efficiency-based model

    Args:
        home_team (str): Home team name
        away_team (str): Away team name
        features (dict): Matchup features

    Returns:
        dict: Prediction results
    """
    if features is None:
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': 'Unknown',
            'win_probability': 0.5,
            'predicted_home_score': 0,
            'predicted_away_score': 0,
            'confidence': 'Low',
            'key_factors': ['Insufficient data']
        }

    # Calculate net advantage (efficiency margin diff + home court)
    net_advantage = features['efficiency_margin_diff'] + features['home_court_advantage']

    # KenPom methodology: expected score = (team_OE * opp_DE / national_avg) * (possessions / 100)
    expected_tempo = features['expected_tempo']
    predicted_home_score = (features['home_oe'] / national_avg) * (features['away_de'] / national_avg) * national_avg * (expected_tempo / 100) + features['home_court_advantage']
    predicted_away_score = (features['away_oe'] / national_avg) * (features['home_de'] / national_avg) * national_avg * (expected_tempo / 100)

    # Win probability from the predicted score (as rounded for display, so the listed winner always
    # matches the shown score): share of a normal curve around the predicted margin that lands above 0
    predicted_margin = round(predicted_home_score, 1) - round(predicted_away_score, 1)
    win_prob = NormalDist().cdf(predicted_margin / win_prob_width)

    # Determine confidence
    if abs(net_advantage) > 15:
        confidence = 'High'
    elif abs(net_advantage) > 8:
        confidence = 'Medium'
    else:
        confidence = 'Low'

    # Key factors
    key_factors = []
    if features['efficiency_margin_diff'] > 5:
        key_factors.append(f"✓ Better overall efficiency (+{features['efficiency_margin_diff']:.1f})")
    elif features['efficiency_margin_diff'] < -5:
        key_factors.append(f"✗ Worse overall efficiency ({features['efficiency_margin_diff']:.1f})")

    if features['off_efficiency_diff'] > 3:
        key_factors.append(f"✓ Superior offense (+{features['off_efficiency_diff']:.1f})")
    elif features['off_efficiency_diff'] < -3:
        key_factors.append(f"✗ Weaker offense ({features['off_efficiency_diff']:.1f})")

    if features['def_efficiency_diff'] < -3:  # Lower defensive efficiency is better
        key_factors.append(f"✓ Better defense ({features['def_efficiency_diff']:.1f})")
    elif features['def_efficiency_diff'] > 3:
        key_factors.append(f"✗ Weaker defense (+{features['def_efficiency_diff']:.1f})")

    if features['home_court_advantage'] > 0:
        key_factors.append(f"✓ Home court advantage (+{features['home_court_advantage']:.1f})")

    return {
        'home_team': home_team,
        'away_team': away_team,
        'predicted_winner': home_team if win_prob > 0.5 else away_team,
        'win_probability': win_prob if win_prob > 0.5 else 1 - win_prob,
        'predicted_home_score': round(predicted_home_score, 1),
        'predicted_away_score': round(predicted_away_score, 1),
        'confidence': confidence,
        'key_factors': key_factors
    }


PREDICTION_HISTORY_TABLE = 'workspace.default.prediction_history'

# Insert order for prediction_history. Columns are named explicitly so adding columns to the table
# (e.g. the bet columns) can never shift values into the wrong column.
HISTORY_COLUMNS = ['game_date', 'game_time', 'home_team', 'away_team', 'predicted_winner', 'win_probability',
                   'predicted_home_score', 'predicted_away_score', 'confidence', 'vegas_spread', 'cover_pick',
                   'cover_confidence', 'prediction_date', 'home_adj_em', 'away_adj_em', 'home_oe', 'away_oe',
                   'home_de', 'away_de', 'home_rank', 'away_rank'] + [c for c, _ in Betting.BET_COLUMNS]


def upload_predictions_to_databricks(predictions_df, table=PREDICTION_HISTORY_TABLE):
    """
    Upload predictions to Databricks

    Args:
        predictions_df (pd.DataFrame): Predictions dataframe
        table (str): target table (default prediction_history; tests pass a scratch copy)
    """
    print("\nUploading predictions to Databricks...")

    if predictions_df.empty:
        print("⚠️  No predictions to upload")
        return

    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN,
            _socket_timeout=900
        )

        cursor = connection.cursor()

        # Create table if it doesn't exist (preserve historical predictions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace.default.prediction_history (
                game_date DATE,
                game_time STRING,
                home_team STRING,
                away_team STRING,
                predicted_winner STRING,
                win_probability DOUBLE,
                predicted_home_score DOUBLE,
                predicted_away_score DOUBLE,
                confidence STRING,
                vegas_spread DOUBLE,
                cover_pick STRING,
                cover_confidence STRING,
                prediction_date TIMESTAMP,
                home_adj_em DOUBLE,
                away_adj_em DOUBLE,
                home_oe DOUBLE,
                away_oe DOUBLE,
                home_de DOUBLE,
                away_de DOUBLE,
                home_rank INT,
                away_rank INT
            )
            USING DELTA
        """)

        # Add KenPom columns to existing table if they don't exist yet
        try:
            cursor.execute("""
                ALTER TABLE workspace.default.prediction_history
                ADD COLUMNS (
                    home_adj_em DOUBLE,
                    away_adj_em DOUBLE,
                    home_oe DOUBLE,
                    away_oe DOUBLE,
                    home_de DOUBLE,
                    away_de DOUBLE,
                    home_rank INT,
                    away_rank INT
                )
            """)
        except Exception:
            pass  # Columns already exist

        # Delete ONLY predictions for games we're about to regenerate
        # This preserves predictions for games that have already started (lock mechanism)
        Betting.ensure_bet_columns(cursor, table)

        print("  Removing old predictions for unstarted games...")
        deleted_count = 0
        for _, row in predictions_df.iterrows():
            # Escape single quotes in team names to prevent SQL injection
            home_team_escaped = row['home_team'].replace("'", "''")
            away_team_escaped = row['away_team'].replace("'", "''")
            cursor.execute(f"""
                DELETE FROM {table}
                WHERE home_team = '{home_team_escaped}'
                  AND away_team = '{away_team_escaped}'
                  AND game_date = '{row['game_date']}'
            """)
            deleted_count += 1
        print(f"  ✓ Removed {deleted_count} old predictions")

        # Insert predictions
        for _, row in predictions_df.iterrows():
            # Escape single quotes in team names and text fields to prevent SQL injection
            home_team_escaped = row['home_team'].replace("'", "''")
            away_team_escaped = row['away_team'].replace("'", "''")
            predicted_winner_escaped = row['predicted_winner'].replace("'", "''")
            cover_pick_escaped = row['cover_pick'].replace("'", "''")
            spread_sql = 'NULL' if pd.isna(row['vegas_spread']) else row['vegas_spread']

            bet_sql = ', '.join(Betting.sql_value(row[c]) for c, _ in Betting.BET_COLUMNS)
            cursor.execute(f"""
                INSERT INTO {table} ({', '.join(HISTORY_COLUMNS)}) VALUES (
                    '{row['game_date']}',
                    '{row['game_time']}',
                    '{home_team_escaped}',
                    '{away_team_escaped}',
                    '{predicted_winner_escaped}',
                    {row['win_probability']},
                    {row['predicted_home_score']},
                    {row['predicted_away_score']},
                    '{row['confidence']}',
                    {spread_sql},
                    '{cover_pick_escaped}',
                    '{row['cover_confidence']}',
                    current_timestamp(),
                    {row['home_adj_em']},
                    {row['away_adj_em']},
                    {row['home_oe']},
                    {row['away_oe']},
                    {row['home_de']},
                    {row['away_de']},
                    {row['home_rank']},
                    {row['away_rank']},
                    {bet_sql}
                )
            """)

        cursor.close()
        connection.close()

        print(f"✓ Successfully uploaded {len(predictions_df)} predictions to Databricks")

    except Exception as e:
        print(f"❌ Error uploading predictions: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(date=None):
    """Main execution function

    Args:
        date (str): Optional date in YYYYMMDD format. Defaults to today.
    """
    print("="*80)
    print("COLLEGE BASKETBALL GAME PREDICTIONS")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%B %d, %Y')}\n")

    # Get games for specified date (or today)
    games = get_todays_games(date=date)

    if games.empty:
        print("\nNo games scheduled for today.")
        return

    # Filter to only games that haven't started yet (lock mechanism)
    unstarted_games = games[games['status'] == 'STATUS_SCHEDULED'].copy()
    started_games = games[games['status'] != 'STATUS_SCHEDULED']

    if not started_games.empty:
        print(f"\n🔒 LOCKED PREDICTIONS: {len(started_games)} games already started/finished")
        print("   (Keeping existing predictions for these games)")
        for _, game in started_games.iterrows():
            print(f"   - {game['away_team']} @ {game['home_team']} ({game['status']})")

    if unstarted_games.empty:
        print("\n⚠️  All games have already started. No new predictions to generate.")
        return

    print(f"\n🔓 GENERATING NEW PREDICTIONS: {len(unstarted_games)} games not yet started")

    # Get Vegas spreads
    vegas_spreads = get_vegas_spreads(date)   # date None = today (Eastern)
    unmapped_odds_teams(vegas_spreads)        # warns about Odds API names missing from d1_teams.ODDS_NAMES

    # Get team statistics
    team_stats = get_team_stats_from_databricks()

    if team_stats.empty:
        print("\nCould not load team statistics.")
        return

    national_avg = team_stats['adj_offensive_efficiency'].mean()
    print(f"  National avg OE: {national_avg:.2f}")
    validate_d1_table(team_stats)   # warns if a KenPom team has no ESPN school in d1_teams.py

    # Win-probability width, re-fit on graded games before the prediction date
    eastern = pytz.timezone('US/Eastern')
    run_day = datetime.strptime(date, '%Y%m%d').date() if date else datetime.now(eastern).date()
    season_start = f"{run_day.year if run_day.month >= 7 else run_day.year - 1}-11-01"
    win_prob_width, width_source = fit_win_prob_width(load_graded_formula_games(season_start), run_day)
    print(f"  Win-prob width: {win_prob_width:.1f} pts ({width_source})")

    # Bet scatter (how far results land from the line), re-fit on graded games before the prediction date
    try:
        connection = sql.connect(server_hostname=DATABRICKS_HOST, http_path=DATABRICKS_HTTP_PATH,
                                 access_token=DATABRICKS_TOKEN)
        cursor = connection.cursor()
        bet_params = Betting.fit_for_run(cursor, run_day)
        cursor.close()
        connection.close()
    except Exception as e:
        bet_params = Betting.SCATTER_DEFAULT
        print(f"⚠️  Could not fit the bet scatter ({e}); using the default")

    # Generate predictions for each game
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80 + "\n")

    predictions = []

    for _, game in unstarted_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']

        print(f"\n{home_team} vs {away_team}")
        print("-" * 60)

        # Get Vegas spread (normalize team names for Odds API lookup)
        home_team_normalized = odds_name_for(home_team, vegas_spreads)
        away_team_normalized = odds_name_for(away_team, vegas_spreads)
        home_spread = vegas_spreads.get(home_team_normalized)
        away_spread = vegas_spreads.get(away_team_normalized)

        if home_spread is None and vegas_spreads:
            print(f"  ⚠️  No spread found for: '{home_team}' (tried: '{home_team_normalized}')")
        if away_spread is None and vegas_spreads:
            print(f"  ⚠️  No spread found for: '{away_team}' (tried: '{away_team_normalized}')")

        # Calculate features (neutral site = no home court advantage)
        neutral_site = bool(game.get('neutral_site', False))
        if neutral_site:
            print(f"  🏟️  Neutral site game")
        features = calculate_matchup_features(home_team, away_team, team_stats, neutral_site=neutral_site)

        # Predict
        prediction = predict_game(home_team, away_team, features, national_avg, win_prob_width)

        # Calculate cover pick (Against The Spread)
        predicted_margin = prediction['predicted_home_score'] - prediction['predicted_away_score']

        # Fall back to the away line if only the away team's name matched the Odds API
        if home_spread is None and away_spread is not None:
            home_spread = -away_spread

        # home_spread None = no line found; 0.0 = genuine pick 'em
        if home_spread is not None:
            # Adjusted margin = predicted margin accounting for spread
            # Positive = home covers, negative = away covers
            adjusted_margin = predicted_margin + home_spread

            # Determine cover pick
            if home_spread == 0:
                cover_pick = f"{home_team if adjusted_margin > 0 else away_team} PK"
            elif adjusted_margin > 0:
                cover_pick = f"{home_team} {home_spread:+.1f}"
            else:
                cover_pick = f"{away_team} {-home_spread:+.1f}"

            # Determine cover confidence based on adjusted margin
            abs_diff = abs(adjusted_margin)
            if abs_diff > 5:
                cover_confidence = 'High'
            elif abs_diff > 2:
                cover_confidence = 'Medium'
            else:
                cover_confidence = 'Low'
        else:
            cover_pick = 'N/A'
            cover_confidence = 'N/A'

        # Bet only on a real prediction (an unmatched team leaves 0-0 scores, which would look like a huge edge)
        bet = (Betting.make_bet(home_team, away_team, predicted_margin, home_spread, bet_params)
               if features is not None else None)

        # Display prediction
        print(f"Predicted Winner: {prediction['predicted_winner']} ({prediction['win_probability']:.0%} confidence)")
        print(f"Predicted Score: {home_team} {prediction['predicted_home_score']}, {away_team} {prediction['predicted_away_score']}")
        if home_spread is not None:
            # Show the favored team with the spread (home_spread < 0 means home favored)
            if home_spread == 0:
                spread_text = "Pick 'em"
            else:
                spread_text = f"{home_team} {home_spread:+.1f}" if home_spread < 0 else f"{away_team} {-home_spread:+.1f}"
            print(f"Vegas Spread: {spread_text}")
            print(f"Cover Pick: {cover_pick} (Confidence: {cover_confidence})")
        if bet:
            print(f"BET: {bet['label']}  ({bet['prob']:.0%} to cover, edge {bet['edge']:.1f}, {Betting.BET_UNITS:g}u)")
        print(f"Confidence Level: {prediction['confidence']}")
        print(f"\nKey Factors:")
        for factor in prediction['key_factors']:
            print(f"  {factor}")

        # Store prediction
        # Extract game date from game_time, converting from UTC to Eastern Time
        if 'T' in game['game_time']:
            # Parse UTC timestamp and convert to Eastern Time
            utc_time = datetime.strptime(game['game_time'], '%Y-%m-%dT%H:%MZ')
            utc_time = pytz.utc.localize(utc_time)
            eastern = pytz.timezone('US/Eastern')
            eastern_time = utc_time.astimezone(eastern)
            game_date = eastern_time.strftime('%Y-%m-%d')
        else:
            game_date = datetime.now().strftime('%Y-%m-%d')

        predictions.append({
            'game_date': game_date,
            'game_time': game['game_time'],
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': prediction['predicted_winner'],
            'win_probability': prediction['win_probability'],
            'predicted_home_score': prediction['predicted_home_score'],
            'predicted_away_score': prediction['predicted_away_score'],
            'confidence': prediction['confidence'],
            'vegas_spread': home_spread,  # None = no line (stored NULL); 0.0 = pick 'em
            'cover_pick': cover_pick,
            'cover_confidence': cover_confidence,
            'home_adj_em': features['home_adj_em'] if features else 0.0,
            'away_adj_em': features['away_adj_em'] if features else 0.0,
            'home_oe': features['home_oe'] if features else 0.0,
            'away_oe': features['away_oe'] if features else 0.0,
            'home_de': features['home_de'] if features else 0.0,
            'away_de': features['away_de'] if features else 0.0,
            'home_rank': features['home_rank'] if features else 0,
            'away_rank': features['away_rank'] if features else 0,
            **Betting.bet_fields(bet),
        })

    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Upload to Databricks
    if DATABRICKS_HOST and DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN:
        upload_predictions_to_databricks(predictions_df)
    else:
        print("\n⚠️  Databricks credentials not found. Predictions not uploaded.")

    # Save locally as backup
    predictions_df.to_csv("todays_predictions.csv", index=False)
    print(f"\n✓ Predictions saved to: todays_predictions.csv")

    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total Games: {len(predictions_df)}")
    print(f"Bets: {int(predictions_df['is_bet'].sum())} at {Betting.BET_THRESHOLD:.0%}+ cover probability")
    print(f"High Confidence: {len(predictions_df[predictions_df['confidence'] == 'High'])}")
    print(f"Medium Confidence: {len(predictions_df[predictions_df['confidence'] == 'Medium'])}")
    print(f"Low Confidence: {len(predictions_df[predictions_df['confidence'] == 'Low'])}")

    return predictions_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None,
                        help='Date in YYYYMMDD format (default: today)')
    args = parser.parse_args()
    main(date=args.date)
