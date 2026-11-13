"""
College Basketball Game Results Updater
Fetches completed games and stores actual results for performance tracking
"""

import pandas as pd
import requests
from databricks import sql
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


def get_completed_games(date_str):
    """
    Fetch completed college basketball games from ESPN API for a specific date

    Args:
        date_str (str): Date in YYYYMMDD format

    Returns:
        pd.DataFrame: Completed games with results
    """
    print(f"Fetching completed games for {date_str}...")

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {'dates': date_str, 'limit': 300, 'groups': 50}
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        games = []

        if 'events' not in data:
            print(f"  No events found for {date_str}")
            return pd.DataFrame()

        for event in data['events']:
            # Only include completed games
            if event['status']['type']['state'] != 'post':
                continue

            competitors = event['competitions'][0]['competitors']

            # Identify home and away teams
            home_team = None
            away_team = None
            home_score = None
            away_score = None

            for team in competitors:
                if team['homeAway'] == 'home':
                    home_team = team['team']['displayName']
                    home_score = int(team['score'])
                else:
                    away_team = team['team']['displayName']
                    away_score = int(team['score'])

            if home_team and away_team:
                games.append({
                    'game_date': datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score
                })

        print(f"✓ Found {len(games)} completed games")
        return pd.DataFrame(games)

    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return pd.DataFrame()


def store_results_to_databricks(results_df):
    """
    Store game results in Databricks

    Args:
        results_df (pd.DataFrame): Game results to store
    """
    if results_df.empty:
        print("⚠️  No results to upload")
        return

    print(f"\nUploading {len(results_df)} game results to Databricks...")

    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN,
            _socket_timeout=900
        )

        cursor = connection.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace.default.game_results (
                game_date DATE,
                home_team STRING,
                away_team STRING,
                home_score INT,
                away_score INT,
                result_loaded_at TIMESTAMP
            )
            USING DELTA
        """)

        # Delete results for this date (avoid duplicates)
        game_date = results_df.iloc[0]['game_date']
        cursor.execute(f"""
            DELETE FROM workspace.default.game_results
            WHERE game_date = '{game_date}'
        """)

        # Insert results
        for _, row in results_df.iterrows():
            home_team_esc = row['home_team'].replace("'", "''")
            away_team_esc = row['away_team'].replace("'", "''")
            cursor.execute(f"""
                INSERT INTO workspace.default.game_results VALUES (
                    '{row['game_date']}',
                    '{home_team_esc}',
                    '{away_team_esc}',
                    {row['home_score']},
                    {row['away_score']},
                    current_timestamp()
                )
            """)

        cursor.close()
        connection.close()

        print(f"✓ Successfully uploaded {len(results_df)} results to Databricks")

    except Exception as e:
        print(f"❌ Error uploading results: {e}")


def update_prediction_outcomes():
    """
    Materialize prediction outcomes by joining prediction_history with game_results.
    Writes one row per completed game with KenPom features + actual result + W/L flags.
    Used as training data for the neural network model.
    """
    print("\nUpdating prediction outcomes...")

    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN,
            _socket_timeout=900
        )
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace.default.prediction_outcomes (
                game_date DATE,
                home_team STRING,
                away_team STRING,
                home_adj_em DOUBLE,
                away_adj_em DOUBLE,
                home_oe DOUBLE,
                away_oe DOUBLE,
                home_de DOUBLE,
                away_de DOUBLE,
                home_rank INT,
                away_rank INT,
                predicted_winner STRING,
                predicted_home_score DOUBLE,
                predicted_away_score DOUBLE,
                predicted_margin DOUBLE,
                win_probability DOUBLE,
                confidence STRING,
                vegas_spread DOUBLE,
                cover_pick STRING,
                actual_home_score INT,
                actual_away_score INT,
                actual_margin INT,
                actual_winner STRING,
                su_correct INT,
                ats_correct INT,
                margin_error DOUBLE,
                recorded_at TIMESTAMP
            )
            USING DELTA
        """)

        # Insert only newly completed games not already in prediction_outcomes
        cursor.execute("""
            INSERT INTO workspace.default.prediction_outcomes
            SELECT
                p.game_date,
                p.home_team,
                p.away_team,
                p.home_adj_em,
                p.away_adj_em,
                p.home_oe,
                p.away_oe,
                p.home_de,
                p.away_de,
                p.home_rank,
                p.away_rank,
                p.predicted_winner,
                p.predicted_home_score,
                p.predicted_away_score,
                (p.predicted_home_score - p.predicted_away_score) AS predicted_margin,
                p.win_probability,
                p.confidence,
                p.vegas_spread,
                p.cover_pick,
                r.home_score AS actual_home_score,
                r.away_score AS actual_away_score,
                (r.home_score - r.away_score) AS actual_margin,
                CASE WHEN r.home_score > r.away_score THEN p.home_team ELSE p.away_team END AS actual_winner,
                CASE WHEN p.predicted_winner =
                     CASE WHEN r.home_score > r.away_score THEN p.home_team ELSE p.away_team END
                     THEN 1 ELSE 0 END AS su_correct,
                CASE
                    WHEN p.vegas_spread = 0 THEN NULL
                    WHEN (p.predicted_home_score - p.predicted_away_score + p.vegas_spread) > 0
                         AND (r.home_score - r.away_score + p.vegas_spread) > 0 THEN 1
                    WHEN (p.predicted_home_score - p.predicted_away_score + p.vegas_spread) <= 0
                         AND (r.home_score - r.away_score + p.vegas_spread) < 0 THEN 1
                    ELSE 0
                END AS ats_correct,
                ((p.predicted_home_score - p.predicted_away_score) - (r.home_score - r.away_score)) AS margin_error,
                current_timestamp() AS recorded_at
            FROM workspace.default.prediction_history p
            INNER JOIN workspace.default.game_results r
                ON p.home_team = r.home_team
                AND p.away_team = r.away_team
                AND ABS(DATEDIFF(p.game_date, r.game_date)) <= 1
            WHERE NOT EXISTS (
                SELECT 1 FROM workspace.default.prediction_outcomes o
                WHERE o.home_team = p.home_team
                  AND o.away_team = p.away_team
                  AND o.game_date = p.game_date
            )
        """)

        cursor.execute("SELECT COUNT(*) FROM workspace.default.prediction_outcomes")
        total = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        print(f"✓ prediction_outcomes updated ({total} total rows)")

    except Exception as e:
        print(f"❌ Error updating prediction outcomes: {e}")


def update_ml_ats_results():
    """Fill in actual scores and ATS result for completed games in ml_predictions."""
    print("\nUpdating ML ATS results...")
    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN,
            _socket_timeout=900
        )
        cursor = connection.cursor()

        cursor.execute("""
            SELECT m.game_date, m.home_team, m.away_team,
                   m.ml_predicted_margin, m.vegas_spread,
                   r.home_score, r.away_score
            FROM workspace.default.ml_predictions m
            JOIN workspace.default.game_results r
                ON m.home_team = r.home_team
               AND m.away_team = r.away_team
               AND m.game_date = r.game_date
            WHERE m.actual_home_score IS NULL
              AND r.home_score IS NOT NULL
        """)
        rows = cursor.fetchall()

        updated = 0
        for row in rows:
            game_date, home, away, pred_margin, spread, actual_home, actual_away = row
            actual_margin = actual_home - actual_away

            if spread and spread != 0:
                ml_pick_home     = (pred_margin + spread) > 0
                actual_home_covers = (actual_margin + spread) > 0
                if (actual_margin + spread) == 0:
                    ats_correct = None  # push
                else:
                    ats_correct = 1 if ml_pick_home == actual_home_covers else 0
            else:
                ats_correct = None

            home_esc = home.replace("'", "''")
            away_esc = away.replace("'", "''")
            ats_val  = str(ats_correct) if ats_correct is not None else 'NULL'

            cursor.execute(f"""
                UPDATE workspace.default.ml_predictions
                SET actual_home_score = {actual_home},
                    actual_away_score = {actual_away},
                    ml_ats_correct = {ats_val}
                WHERE home_team = '{home_esc}'
                  AND away_team = '{away_esc}'
                  AND game_date = '{game_date}'
            """)
            updated += 1

        cursor.close()
        connection.close()
        print(f"  Updated ML ATS results for {updated} games")

    except Exception as e:
        print(f"  Error updating ML ATS results: {e}")


def main():
    """Main execution function"""

    print("="*80)
    print("COLLEGE BASKETBALL GAME RESULTS UPDATE")
    print("="*80)

    # Check the last 3 days for completed games
    # This ensures we catch results even if a prior run was missed or incomplete
    today = datetime.now()

    dates_to_check = [
        (today, "Today"),
        (today - timedelta(days=1), "Yesterday"),
        (today - timedelta(days=2), "2 Days Ago"),
        (today - timedelta(days=3), "3 Days Ago"),
    ]

    all_results = []

    for date_obj, label in dates_to_check:
        date_str = date_obj.strftime('%Y%m%d')
        print(f"\n{label}: {date_obj.strftime('%B %d, %Y')}")

        # Fetch completed games for this date
        results_df = get_completed_games(date_str)

        if not results_df.empty:
            all_results.append(results_df)

            # Display results
            print("\n" + "-"*80)
            print(f"COMPLETED GAMES - {label.upper()}")
            print("-"*80)
            for _, game in results_df.iterrows():
                print(f"{game['away_team']} {game['away_score']} @ {game['home_team']} {game['home_score']}")

            # Upload to Databricks
            if DATABRICKS_HOST and DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN:
                store_results_to_databricks(results_df)
        else:
            print(f"  No completed games found for {label.lower()}")

    # Combine all results for backup
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv("recent_results.csv", index=False)
        print(f"\n✓ All results saved to: recent_results.csv")
    else:
        print("\n⚠️  No completed games found for today or yesterday.")

    if not (DATABRICKS_HOST and DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN):
        print("\n⚠️  Databricks credentials not found. Results not uploaded.")
    else:
        update_prediction_outcomes()
        update_ml_ats_results()

    print("\n" + "="*80)
    print(f"✅ Results update completed successfully!")
    print("="*80)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


if __name__ == "__main__":
    main()
