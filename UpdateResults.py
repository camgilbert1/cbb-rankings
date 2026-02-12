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
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={date_str}"
        response = requests.get(url)
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
            access_token=DATABRICKS_TOKEN
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
            cursor.execute(f"""
                INSERT INTO workspace.default.game_results VALUES (
                    '{row['game_date']}',
                    '{row['home_team']}',
                    '{row['away_team']}',
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


def main():
    """Main execution function"""

    print("="*80)
    print("COLLEGE BASKETBALL GAME RESULTS UPDATE")
    print("="*80)

    # Get yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%Y%m%d')

    print(f"Date: {yesterday.strftime('%B %d, %Y')}\n")

    # Fetch completed games
    results_df = get_completed_games(date_str)

    if results_df.empty:
        print("\nNo completed games found for yesterday.")
        return

    # Display results
    print("\n" + "="*80)
    print("COMPLETED GAMES")
    print("="*80)
    for _, game in results_df.iterrows():
        print(f"{game['away_team']} {game['away_score']} @ {game['home_team']} {game['home_score']}")

    # Upload to Databricks
    if DATABRICKS_HOST and DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN:
        store_results_to_databricks(results_df)
    else:
        print("\n⚠️  Databricks credentials not found. Results not uploaded.")

    # Save locally as backup
    results_df.to_csv("yesterdays_results.csv", index=False)
    print(f"\n✓ Results saved to: yesterdays_results.csv")

    print("\n" + "="*80)
    print(f"✅ Results update completed successfully!")
    print("="*80)

    return results_df


if __name__ == "__main__":
    main()
