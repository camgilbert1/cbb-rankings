"""
College Basketball Game Predictions
Fetches today's games and generates predictions using KenPom efficiency metrics
"""

import pandas as pd
import requests
from databricks import sql
import os
from datetime import datetime
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

# ESPN to Odds API team name mappings
# Used when ESPN team names don't match Odds API team names
# Most cases are handled automatically by the normalization function
ESPN_TO_ODDS_MAPPING = {
    # Special cases that need explicit mapping
    'IU Indianapolis Jaguars': 'IUPUI Jaguars',  # Completely different school name
    'Purdue Fort Wayne Mastodons': 'Fort Wayne Mastodons',  # Drops "Purdue" prefix
    'Gardner-Webb Runnin Bulldogs': 'Gardner-Webb Bulldogs',  # Different mascot spelling
    'Florida International Panthers': "Florida Int'l Golden Panthers",  # Different abbreviation and mascot
    'Northern Colorado Bears': 'N Colorado Bears',  # "Northern" vs "N"
    'SIU Edwardsville Cougars': 'SIU-Edwardsville Cougars',  # Dash vs space
    'Cal State Bakersfield Roadrunners': 'CSU Bakersfield Roadrunners',  # "Cal State" vs "CSU"
    'Long Beach State Beach': 'Long Beach St 49ers',  # Different abbreviation and mascot
    'Southeast Missouri State Redhawks': 'SE Missouri St Redhawks',  # "Southeast" vs "SE" + State→St
    'Sacramento State Hornets': 'Sacramento St Hornets',  # State→St (fallback didn't work)
    'Hawaii Rainbow Warriors': "Hawai'i Rainbow Warriors",  # Different spelling with apostrophe
    'Cal State Fullerton Titans': 'CSU Fullerton Titans',  # "Cal State" vs "CSU"
    'Loyola Chicago Ramblers': 'Loyola (Chi) Ramblers',  # City name vs abbreviation
    'UT Martin Skyhawks': 'Tenn-Martin Skyhawks',  # Completely different abbreviation
    'Seattle U Redhawks': 'Seattle Redhawks',  # Drops "U"
    'Cal State Northridge Matadors': 'CSU Northridge Matadors',  # "Cal State" vs "CSU"
    'Mississippi Valley State Delta Devils': 'Miss Valley St Delta Devils',  # Abbreviated
    'Prairie View A&M Panthers': 'Prairie View Panthers',  # Drops "A&M"
    'UT Arlington Mavericks': 'UT-Arlington Mavericks',  # Space vs hyphen
    'California Baptist Lancers': 'Cal Baptist Lancers',  # "California" vs "Cal"
    'Little Rock Trojans': 'Arkansas-Little Rock Trojans',  # Missing "Arkansas-" prefix
    'Kansas City Roos': 'UMKC Kangaroos',  # Completely different name and mascot
    'Grand Canyon Lopes': 'Grand Canyon Antelopes',  # Different mascot name
    'Texas A&M-Corpus Christi Islanders': 'Texas A&M-CC Islanders',  # Abbreviated
    # Add more mappings as needed
}


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_vegas_spreads():
    """
    Fetch Vegas spreads from The Odds API

    Returns:
        dict: Team name to spread mapping
    """
    if not ODDS_API_KEY:
        print("⚠️  No Odds API key found, spreads will not be available")
        return {}

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

        # Build mapping of team names to spreads
        spreads = {}
        for game in data:
            home_team = game['home_team']
            away_team = game['away_team']

            # Get the first bookmaker's spread (usually consensus)
            if game.get('bookmakers') and len(game['bookmakers']) > 0:
                bookmaker = game['bookmakers'][0]
                markets = bookmaker.get('markets', [])

                for market in markets:
                    if market['key'] == 'spreads':
                        outcomes = market['outcomes']
                        for outcome in outcomes:
                            team = outcome['name']
                            spread = outcome['point']

                            # Store spread (negative means favorite)
                            spreads[team] = spread

        print(f"✓ Fetched spreads for {len(spreads) // 2} games")
        print(f"  Odds API teams: {sorted(spreads.keys())}")
        return spreads

    except Exception as e:
        print(f"⚠️  Error fetching spreads: {e}")
        return {}


def normalize_team_name_for_odds(espn_team_name, vegas_spreads=None):
    """
    Normalize ESPN team name to match Odds API team name

    Args:
        espn_team_name (str): Team name from ESPN API
        vegas_spreads (dict): Optional dict of spreads to check against

    Returns:
        str: Normalized team name for Odds API lookup
    """
    # Check if there's a direct mapping (for special cases like IUPUI)
    if espn_team_name in ESPN_TO_ODDS_MAPPING:
        return ESPN_TO_ODDS_MAPPING[espn_team_name]

    # If we have spreads dict, try intelligent fallback
    if vegas_spreads is not None:
        # First try the original name
        if espn_team_name in vegas_spreads:
            return espn_team_name

        # Try replacing "State" with "St" (common Odds API abbreviation)
        # e.g., "Cleveland State Vikings" -> "Cleveland St Vikings"
        if ' State ' in espn_team_name:
            state_abbreviated = espn_team_name.replace(' State ', ' St ')
            if state_abbreviated in vegas_spreads:
                return state_abbreviated

        # Try stripping the last word (mascot name)
        # e.g., "Cleveland State Vikings" -> "Cleveland State"
        words = espn_team_name.split()
        if len(words) > 1:
            team_without_mascot = ' '.join(words[:-1])
            if team_without_mascot in vegas_spreads:
                return team_without_mascot

            # Also try with "State" -> "St" and then stripping mascot
            if ' State ' in team_without_mascot:
                state_abbreviated_no_mascot = team_without_mascot.replace(' State ', ' St ')
                if state_abbreviated_no_mascot in vegas_spreads:
                    return state_abbreviated_no_mascot

    # Return original name as fallback
    return espn_team_name


def get_todays_games():
    """
    Fetch today's scheduled college basketball games from ESPN API

    Returns:
        pd.DataFrame: Today's scheduled games
    """
    # Use Eastern Time for date to match college basketball schedule
    eastern = pytz.timezone('US/Eastern')
    today_eastern = datetime.now(eastern).strftime('%Y%m%d')
    print(f"Fetching games from ESPN for {today_eastern}...")

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {'dates': today_eastern, 'limit': 300, 'groups': 50}
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

            game_list.append({
                'game_id': game['id'],
                'game_time': game['date'],
                'home_team': home_team['team']['displayName'],
                'away_team': away_team['team']['displayName'],
                'status': game['status']['type']['name']
            })

        df = pd.DataFrame(game_list)
        print(f"✓ Found {len(df)} games scheduled for today")
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
            access_token=DATABRICKS_TOKEN
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

def find_team_match(espn_name, team_stats):
    """
    Match ESPN team name to KenPom team name

    Args:
        espn_name (str): Full team name from ESPN (e.g., "Michigan Wolverines")
        team_stats (pd.DataFrame): Team statistics with KenPom names

    Returns:
        str: Matched team name or None
    """
    # Full school name mappings (ESPN school name without mascot -> KenPom name)
    # Used for names that can't be resolved by first-word or substring matching
    full_name_mappings = {
        'UT Martin': 'Tennessee Martin',
        'Southeast Missouri State': 'Southeast Missouri St',
        'SIU Edwardsville': 'SIU Edwardsville',
        'UL Monroe': 'Louisiana Monroe',
        'Ole Miss': 'Mississippi',
        'Utah State': 'Utah St',
        'Washington State': 'Washington St',
        'Mississippi State': 'Mississippi St',
        'Illinois State': 'Illinois St',
        'Missouri State': 'Missouri St',
        'Jacksonville State': 'Jacksonville St',
        'Loyola Maryland': 'Loyola MD',
        'Colorado State': 'Colorado St',
        'Indiana State': 'Indiana St',
        'Oklahoma State': 'Oklahoma St',
        'South Dakota State': 'South Dakota St',
        'North Dakota State': 'North Dakota St',
        'Kansas State': 'Kansas St',
        'Michigan State': 'Michigan St',
        'Oregon State': 'Oregon St',
        'Penn State': 'Penn St',
        'Iowa State': 'Iowa St',
        'Ohio State': 'Ohio St',
        'Arizona State': 'Arizona St',
        'Fresno State': 'Fresno St',
        'Boise State': 'Boise St',
        'Wichita State': 'Wichita St',
        'San Diego State': 'San Diego St',
        'San Jose State': 'San Jose St',
        'Appalachian State': 'Appalachian St',
        'Georgia State': 'Georgia St',
        'Ball State': 'Ball St',
        'Kent State': 'Kent St',
        'Wright State': 'Wright St',
        'Youngstown State': 'Youngstown St',
        'Norfolk State': 'Norfolk St',
        'Coppin State': 'Coppin St',
        'Morgan State': 'Morgan St',
        'Alcorn State': 'Alcorn St',
        'Grambling State': 'Grambling St',
        'Texas State': 'Texas St',
        'Arkansas State': 'Arkansas St',
        'Cleveland State': 'Cleveland St',
        'Portland State': 'Portland St',
        'Weber State': 'Weber St',
        'Idaho State': 'Idaho St',
        'Montana State': 'Montana St',
        'Kennesaw State': 'Kennesaw St',
        'Murray State': 'Murray St',
        'Towson State': 'Towson St',
        'Valdosta State': 'Valdosta St',
        'Nicholls State': 'Nicholls St',
        'McNeese State': 'McNeese St',
        'Northwestern State': 'Northwestern St',
        "Saint Joseph's": "Saint Joseph",
        "Saint Mary's": "Saint Mary",
        "St. John's": "St. John",
        "Hawai'i": "Hawaii",
        'South Carolina Upstate': 'USC Upstate',
        'Florida International': 'FIU',
        'Alabama State': 'Alabama St',
        'Mississippi Valley State': 'Mississippi Val',
        'Tennessee State': 'Tennessee St',
        'California Baptist': 'Cal Baptist',
        'Cal State Fullerton': 'Cal St. Fullerton',
        'Cal State Bakersfield': 'Cal St. Bakersfield',
        'Cal State Northridge': 'Cal St. Northridge',
        'Florida State': 'Florida St',
        'Chicago State': 'Chicago St',
        'South Carolina State': 'South Carolina St',
        'San José State': 'San Jose St',
        'SE Louisiana': 'Southeastern Louisiana',
        'New Mexico State': 'New Mexico St',
        'Texas A&M-Corpus Christi': 'Texas A&M Corpus Chris',
    }

    # Check full school name mapping first
    words = espn_name.split()
    school_name = ' '.join(words[:-1]) if len(words) > 1 else espn_name

    # Try exact match, then prefix match for multi-word mascots (e.g., "Red Storm")
    mapped = None
    if school_name in full_name_mappings:
        mapped = full_name_mappings[school_name]
    else:
        # Prefix match — check longest keys first to prefer "Mississippi State" over "Mississippi"
        for key in sorted(full_name_mappings.keys(), key=len, reverse=True):
            if school_name.startswith(key):
                mapped = full_name_mappings[key]
                break

    if mapped:
        if mapped in team_stats['team_name'].values:
            return mapped
        # Also try with trailing period (KenPom sometimes uses "St." vs "St")
        if mapped + '.' in team_stats['team_name'].values:
            return mapped + '.'
        # Try contains match for apostrophe/encoding differences (e.g., ' vs ')
        contains = team_stats[team_stats['team_name'].str.contains(mapped, case=False, na=False)]
        if len(contains) == 1:
            return contains.iloc[0]['team_name']

    # Common abbreviation mappings (ESPN first word -> KenPom name)
    name_mappings = {
        'UConn': 'Connecticut',
        'UNLV': 'Nevada-Las Vegas',
        'USC': 'Southern California',
        'SMU': 'Southern Methodist',
        'TCU': 'Texas Christian',
        'BYU': 'Brigham Young',
        'UCF': 'Central Florida',
        'VCU': 'Virginia Commonwealth',
        'LSU': 'Louisiana State',
        'UTEP': 'Texas-El Paso',
        'UTSA': 'Texas-San Antonio',
        'UAB': 'Alabama-Birmingham',
        'UIC': 'Illinois Chicago',
    }

    # First try exact match
    if espn_name in team_stats['team_name'].values:
        return espn_name

    # Try school name (without mascot) as exact match
    if school_name in team_stats['team_name'].values:
        return school_name

    # Try with hyphens replaced by spaces (ESPN uses hyphens, KenPom often uses spaces)
    # e.g., "Gardner-Webb" -> "Gardner Webb"
    school_name_no_hyphen = school_name.replace('-', ' ')
    if school_name_no_hyphen != school_name:
        if school_name_no_hyphen in team_stats['team_name'].values:
            return school_name_no_hyphen

    # Try mapping common abbreviations
    first_word = words[0]
    search_term = first_word

    if first_word in name_mappings:
        search_term = name_mappings[first_word]

    # Try matching the search term (either original or mapped)
    matches = team_stats[team_stats['team_name'].str.contains(search_term, case=False, na=False)]

    if len(matches) == 1:
        return matches.iloc[0]['team_name']
    elif len(matches) > 1:
        # Multiple matches — try to find the best one using school name
        # e.g., for "Charleston Southern", prefer "Charleston Southern" over "Charleston"
        for team_name in matches['team_name']:
            if team_name.lower() == school_name.lower() or team_name.lower() == school_name_no_hyphen.lower():
                return team_name
        # Try if school name is contained in KenPom name or vice versa
        # Sort by longest match first to prefer specific matches
        # e.g., "Indiana St" over "Indiana" when searching for "Indiana State"
        containment_matches = []
        for team_name in matches['team_name']:
            if school_name.lower() in team_name.lower() or team_name.lower() in school_name.lower():
                containment_matches.append(team_name)
        if containment_matches:
            # Return the longest (most specific) match
            return max(containment_matches, key=len)
        # Try exact word boundary match on first word
        for team_name in matches['team_name']:
            if search_term.lower() in team_name.lower().split():
                return team_name
        # If no exact word match, return first match
        return matches.iloc[0]['team_name']

    # Try with hyphen replaced by space in search
    if '-' in search_term:
        search_no_hyphen = search_term.replace('-', ' ')
        matches = team_stats[team_stats['team_name'].str.contains(search_no_hyphen, case=False, na=False)]
        if len(matches) == 1:
            return matches.iloc[0]['team_name']
        elif len(matches) > 1:
            return matches.iloc[0]['team_name']

    # Try checking if KenPom name is contained in ESPN name (as fallback)
    for team_name in team_stats['team_name']:
        if team_name.lower() in espn_name.lower():
            return team_name

    return None


def calculate_matchup_features(home_team, away_team, team_stats):
    """
    Calculate prediction features for a matchup

    Args:
        home_team (str): Home team name
        away_team (str): Away team name
        team_stats (pd.DataFrame): Team statistics

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

    # Calculate differences
    features = {
        'efficiency_margin_diff': home_stats['adj_efficiency_margin'] - away_stats['adj_efficiency_margin'],
        'off_efficiency_diff': home_stats['adj_offensive_efficiency'] - away_stats['adj_offensive_efficiency'],
        'def_efficiency_diff': home_stats['adj_defensive_efficiency'] - away_stats['adj_defensive_efficiency'],
        'tempo_diff': home_stats['adj_tempo'] - away_stats['adj_tempo'],
        'rank_diff': away_stats['overall_rank'] - home_stats['overall_rank'],  # Positive if home team is better
        'home_court_advantage': 3.5,  # Standard home court advantage
    }

    return features


def predict_game(home_team, away_team, features):
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

    # Estimate win probability (logistic function)
    # Roughly: 1 point of efficiency margin ≈ 3% win probability
    win_prob = 1 / (1 + 2.718 ** (-0.03 * net_advantage))

    # Predict scores (average is ~75 points, adjust by efficiency)
    baseline_score = 75
    predicted_home_score = baseline_score + (features['efficiency_margin_diff'] / 2) + features['home_court_advantage']
    predicted_away_score = baseline_score - (features['efficiency_margin_diff'] / 2)

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


def upload_predictions_to_databricks(predictions_df):
    """
    Upload predictions to Databricks

    Args:
        predictions_df (pd.DataFrame): Predictions dataframe
    """
    print("\nUploading predictions to Databricks...")

    if predictions_df.empty:
        print("⚠️  No predictions to upload")
        return

    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN
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
                prediction_date TIMESTAMP
            )
            USING DELTA
        """)

        # Delete ONLY predictions for games we're about to regenerate
        # This preserves predictions for games that have already started (lock mechanism)
        print("  Removing old predictions for unstarted games...")
        deleted_count = 0
        for _, row in predictions_df.iterrows():
            # Escape single quotes in team names to prevent SQL injection
            home_team_escaped = row['home_team'].replace("'", "''")
            away_team_escaped = row['away_team'].replace("'", "''")
            cursor.execute(f"""
                DELETE FROM workspace.default.prediction_history
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

            cursor.execute(f"""
                INSERT INTO workspace.default.prediction_history VALUES (
                    '{row['game_date']}',
                    '{row['game_time']}',
                    '{home_team_escaped}',
                    '{away_team_escaped}',
                    '{predicted_winner_escaped}',
                    {row['win_probability']},
                    {row['predicted_home_score']},
                    {row['predicted_away_score']},
                    '{row['confidence']}',
                    {row['vegas_spread']},
                    '{cover_pick_escaped}',
                    '{row['cover_confidence']}',
                    current_timestamp()
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

def main():
    """Main execution function"""

    print("="*80)
    print("COLLEGE BASKETBALL GAME PREDICTIONS")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%B %d, %Y')}\n")

    # Get today's games
    games = get_todays_games()

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
    vegas_spreads = get_vegas_spreads()

    # Get team statistics
    team_stats = get_team_stats_from_databricks()

    if team_stats.empty:
        print("\nCould not load team statistics.")
        return

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
        home_team_normalized = normalize_team_name_for_odds(home_team, vegas_spreads)
        away_team_normalized = normalize_team_name_for_odds(away_team, vegas_spreads)
        home_spread = vegas_spreads.get(home_team_normalized)
        away_spread = vegas_spreads.get(away_team_normalized)

        if home_spread is None and vegas_spreads:
            print(f"  ⚠️  No spread found for: '{home_team}' (tried: '{home_team_normalized}')")
        if away_spread is None and vegas_spreads:
            print(f"  ⚠️  No spread found for: '{away_team}' (tried: '{away_team_normalized}')")

        # Calculate features
        features = calculate_matchup_features(home_team, away_team, team_stats)

        # Predict
        prediction = predict_game(home_team, away_team, features)

        # Calculate cover pick (Against The Spread)
        predicted_margin = prediction['predicted_home_score'] - prediction['predicted_away_score']

        if home_spread is not None and home_spread != 0:
            # Adjusted margin = predicted margin accounting for spread
            # Positive = home covers, negative = away covers
            adjusted_margin = predicted_margin + home_spread

            # Determine cover pick
            if adjusted_margin > 0:
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

        # Display prediction
        print(f"Predicted Winner: {prediction['predicted_winner']} ({prediction['win_probability']:.0%} confidence)")
        print(f"Predicted Score: {home_team} {prediction['predicted_home_score']}, {away_team} {prediction['predicted_away_score']}")
        if home_spread is not None:
            # Show the favored team with the spread (home_spread < 0 means home favored)
            spread_text = f"{home_team} {home_spread:+.1f}" if home_spread < 0 else f"{away_team} {-home_spread:+.1f}"
            print(f"Vegas Spread: {spread_text}")
            print(f"Cover Pick: {cover_pick} (Confidence: {cover_confidence})")
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
            'vegas_spread': home_spread if home_spread is not None else 0.0,
            'cover_pick': cover_pick,
            'cover_confidence': cover_confidence
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
    print(f"High Confidence: {len(predictions_df[predictions_df['confidence'] == 'High'])}")
    print(f"Medium Confidence: {len(predictions_df[predictions_df['confidence'] == 'Medium'])}")
    print(f"Low Confidence: {len(predictions_df[predictions_df['confidence'] == 'Low'])}")

    return predictions_df


if __name__ == "__main__":
    main()
