"""
College Basketball Game Predictions
Fetches today's games and generates predictions using KenPom efficiency metrics
"""

import pandas as pd
import requests
from databricks import sql
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_todays_games():
    """
    Fetch today's scheduled college basketball games from ESPN API

    Returns:
        pd.DataFrame: Today's scheduled games
    """
    print("Fetching today's games from ESPN...")

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        response = requests.get(url)
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
    # Find team stats
    home_stats = team_stats[team_stats['team_name'] == home_team]
    away_stats = team_stats[team_stats['team_name'] == away_team]

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

        # Drop existing predictions table
        cursor.execute("DROP TABLE IF EXISTS workspace.default.game_predictions")

        # Create table schema
        cursor.execute("""
            CREATE TABLE workspace.default.game_predictions (
                game_time STRING,
                home_team STRING,
                away_team STRING,
                predicted_winner STRING,
                win_probability DOUBLE,
                predicted_home_score DOUBLE,
                predicted_away_score DOUBLE,
                confidence STRING,
                prediction_date TIMESTAMP
            )
            USING DELTA
        """)

        # Insert predictions
        for _, row in predictions_df.iterrows():
            cursor.execute(f"""
                INSERT INTO workspace.default.game_predictions VALUES (
                    '{row['game_time']}',
                    '{row['home_team']}',
                    '{row['away_team']}',
                    '{row['predicted_winner']}',
                    {row['win_probability']},
                    {row['predicted_home_score']},
                    {row['predicted_away_score']},
                    '{row['confidence']}',
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

    for _, game in games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']

        print(f"\n{home_team} vs {away_team}")
        print("-" * 60)

        # Calculate features
        features = calculate_matchup_features(home_team, away_team, team_stats)

        # Predict
        prediction = predict_game(home_team, away_team, features)

        # Display prediction
        print(f"Predicted Winner: {prediction['predicted_winner']} ({prediction['win_probability']:.0%} confidence)")
        print(f"Predicted Score: {home_team} {prediction['predicted_home_score']}, {away_team} {prediction['predicted_away_score']}")
        print(f"Confidence Level: {prediction['confidence']}")
        print(f"\nKey Factors:")
        for factor in prediction['key_factors']:
            print(f"  {factor}")

        # Store prediction
        predictions.append({
            'game_time': game['game_time'],
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': prediction['predicted_winner'],
            'win_probability': prediction['win_probability'],
            'predicted_home_score': prediction['predicted_home_score'],
            'predicted_away_score': prediction['predicted_away_score'],
            'confidence': prediction['confidence']
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
