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

import Betting

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


def matched_results_sql(pred_table, results_table, partition_extra='', where=''):
    """
    Subquery: each prediction row (alias columns p.*) joined to its game result, plus res_home / res_away =
    the final score of the PREDICTION's home / away team.

    ESPN sometimes relabels home/away after we predict (neutral-site games), so a result matches the
    prediction's two teams in either order, within a day, comparing names without apostrophes. Each team's
    score is read by name, so the prediction's own labels, line and bet are never changed. A same-orientation
    match always wins (rank 1), then the closest date; filter on rn = 1. Games that match exactly today
    match exactly the same way.
    """
    nk = lambda col: f"REPLACE({col}, CHR(39), '')"
    same = f"({nk('r.home_team')} = {nk('p.home_team')} AND {nk('r.away_team')} = {nk('p.away_team')})"
    swapped = f"({nk('r.home_team')} = {nk('p.away_team')} AND {nk('r.away_team')} = {nk('p.home_team')})"
    return f"""(
        SELECT p.*,
               CASE WHEN {same} THEN r.home_score ELSE r.away_score END AS res_home,
               CASE WHEN {same} THEN r.away_score ELSE r.home_score END AS res_away,
               ROW_NUMBER() OVER (PARTITION BY p.game_date, p.home_team, p.away_team{partition_extra}
                                  ORDER BY CASE WHEN {same} THEN 0 ELSE 1 END,
                                           ABS(DATEDIFF(p.game_date, r.game_date))) AS rn
        FROM {pred_table} p
        JOIN {results_table} r
          ON ABS(DATEDIFF(p.game_date, r.game_date)) <= 1 AND ({same} OR {swapped})
        WHERE r.home_score IS NOT NULL {where}
    )"""


OUTCOME_COLUMNS = ['game_date', 'home_team', 'away_team', 'home_adj_em', 'away_adj_em', 'home_oe', 'away_oe',
                   'home_de', 'away_de', 'home_rank', 'away_rank', 'predicted_winner', 'predicted_home_score',
                   'predicted_away_score', 'predicted_margin', 'win_probability', 'confidence', 'vegas_spread',
                   'cover_pick', 'actual_home_score', 'actual_away_score', 'actual_margin', 'actual_winner',
                   'su_correct', 'ats_correct', 'margin_error', 'recorded_at'] + [c for c, _ in Betting.BET_COLUMNS] \
                  + ['vegas_total', 'ou_correct'] + [c for c, _ in Betting.OU_COLUMNS]

# O/U grade of the model's lean on one game: NULL = no line or push; 1 = the lean (over if predicted total is
# above the line) was right. A model's O/U bet record is this grade on rows where ou_is_bet is true.
OU_GRADE_SQL = """CASE
                    WHEN p.vegas_total IS NULL THEN NULL
                    WHEN (p.res_home + p.res_away) = p.vegas_total THEN NULL
                    WHEN ({pred_total} > p.vegas_total) = ((p.res_home + p.res_away) > p.vegas_total) THEN 1
                    ELSE 0
                END"""


def update_prediction_outcomes(outcomes_table='workspace.default.prediction_outcomes',
                               history_table='workspace.default.prediction_history',
                               results_table='workspace.default.game_results'):
    """
    Materialize prediction outcomes by joining prediction_history with game_results.
    Writes one row per completed game with KenPom features + actual result + W/L flags,
    plus the Formula's bet (a bet's result is ats_correct on rows where is_bet is true).
    Table names are parameters so tests can run against scratch copies.
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

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {outcomes_table} (
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

        Betting.ensure_bet_columns(cursor, outcomes_table)
        Betting.ensure_bet_columns(cursor, history_table)   # no-op once GamePredictions has run
        Betting.ensure_ou_columns(cursor, outcomes_table)
        Betting.ensure_ou_columns(cursor, history_table)
        try:
            cursor.execute(f"ALTER TABLE {outcomes_table} ADD COLUMNS (ou_correct INT)")
        except Exception:
            pass

        # Insert only newly completed games not already in prediction_outcomes.
        # Columns are named so the insert can't shift values when the table gains columns.
        bet_select = ',\n                '.join(f"p.{c}" for c, _ in Betting.BET_COLUMNS)
        ou_select = ',\n                '.join(
            ['p.vegas_total', OU_GRADE_SQL.format(pred_total='(p.predicted_home_score + p.predicted_away_score)') + ' AS ou_correct']
            + [f"p.{c}" for c, _ in Betting.OU_COLUMNS])
        cursor.execute(f"""
            INSERT INTO {outcomes_table} ({', '.join(OUTCOME_COLUMNS)})
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
                p.res_home AS actual_home_score,
                p.res_away AS actual_away_score,
                (p.res_home - p.res_away) AS actual_margin,
                CASE WHEN p.res_home > p.res_away THEN p.home_team ELSE p.away_team END AS actual_winner,
                CASE WHEN p.predicted_winner =
                     CASE WHEN p.res_home > p.res_away THEN p.home_team ELSE p.away_team END
                     THEN 1 ELSE 0 END AS su_correct,
                CASE
                    WHEN p.vegas_spread IS NULL THEN NULL  -- no line; 0 = pick 'em is graded
                    WHEN (p.res_home - p.res_away + p.vegas_spread) = 0 THEN NULL  -- push (bet refunded)
                    WHEN (p.predicted_home_score - p.predicted_away_score + p.vegas_spread) > 0
                         AND (p.res_home - p.res_away + p.vegas_spread) > 0 THEN 1
                    WHEN (p.predicted_home_score - p.predicted_away_score + p.vegas_spread) <= 0
                         AND (p.res_home - p.res_away + p.vegas_spread) < 0 THEN 1
                    ELSE 0
                END AS ats_correct,
                ((p.predicted_home_score - p.predicted_away_score) - (p.res_home - p.res_away)) AS margin_error,
                current_timestamp() AS recorded_at,
                {bet_select},
                {ou_select}
            FROM {matched_results_sql(history_table, results_table, where=f'''
                AND NOT EXISTS (SELECT 1 FROM {outcomes_table} o
                                WHERE o.home_team = p.home_team AND o.away_team = p.away_team
                                  AND o.game_date = p.game_date)''')} p
            WHERE p.rn = 1
        """)

        cursor.execute(f"SELECT COUNT(*) FROM {outcomes_table}")
        total = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        print(f"✓ prediction_outcomes updated ({total} total rows)")

    except Exception as e:
        print(f"❌ Error updating prediction outcomes: {e}")


def update_ml_ats_results(ml_table='workspace.default.ml_predictions',
                          results_table='workspace.default.game_results'):
    """
    Fill in actual scores and ATS result for completed games in ml_predictions (v2/v3 graded separately).
    Results are matched like the Formula's (matched_results_sql): either home/away order, within a day.
    Table names are parameters so tests can run against scratch copies.
    """
    print("\nUpdating ML ATS results...")
    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN,
            _socket_timeout=900
        )
        cursor = connection.cursor()

        Betting.ensure_ou_columns(cursor, ml_table)
        try:
            cursor.execute(f"ALTER TABLE {ml_table} ADD COLUMNS (ml_ou_correct INT)")
        except Exception:
            pass
        cursor.execute(f"""
            SELECT p.game_date, p.home_team, p.away_team,
                   p.ml_predicted_margin, p.vegas_spread,
                   p.res_home, p.res_away, COALESCE(p.model_family, 'v1') AS family,
                   p.ml_predicted_total, p.vegas_total
            FROM {matched_results_sql(ml_table, results_table, partition_extra=", COALESCE(p.model_family, 'v1')",
                                      where='AND p.actual_home_score IS NULL')} p
            WHERE p.rn = 1
        """)
        rows = cursor.fetchall()

        updated = 0
        for row in rows:
            game_date, home, away, pred_margin, spread, actual_home, actual_away, family, pred_total, total_line = row
            actual_margin = actual_home - actual_away
            # O/U grade of the model's lean (same rule as the Formula's ou_correct): NULL = no line or push
            if total_line is None or pred_total is None or (actual_home + actual_away) == total_line:
                ou_correct = None
            else:
                ou_correct = 1 if (pred_total > total_line) == ((actual_home + actual_away) > total_line) else 0

            if spread is not None:  # 0 = pick 'em (graded); NULL = no line
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
                UPDATE {ml_table}
                SET actual_home_score = {actual_home},
                    actual_away_score = {actual_away},
                    ml_ats_correct = {ats_val},
                    ml_ou_correct = {ou_correct if ou_correct is not None else 'NULL'}
                WHERE home_team = '{home_esc}'
                  AND away_team = '{away_esc}'
                  AND game_date = '{game_date}'
                  AND COALESCE(model_family, 'v1') = '{family}'
            """)
            updated += 1

        cursor.close()
        connection.close()
        print(f"  Updated ML ATS results for {updated} prediction rows (each model graded separately)")

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
