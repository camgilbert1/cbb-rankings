"""
Backfill Historical Game Features for ML Training
Fetches ESPN game results + Torvik ratings for seasons 2024-2026 and stores
pre-joined features in workspace.default.historical_game_features.

Run this ONCE to seed the ML training table before using TrainModel.py.
Re-running will overwrite existing data for each season processed.
"""

import requests
import pandas as pd
from databricks import sql
import os
import time
from datetime import date, timedelta
from dotenv import load_dotenv
from team_names import find_team_match

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Seasons to backfill: year → (start_date, end_date)
SEASONS = {
    2024: (date(2023, 11, 1), date(2024, 4, 15)),
    2025: (date(2024, 11, 1), date(2025, 4, 15)),
    2026: (date(2025, 11, 1), date(2026, 4, 15)),
}

SLEEP_BETWEEN_DATES = 0.2  # seconds between ESPN API calls
TORVIK_COLS = ['barthag', 'adj_o', 'adj_d', 'adj_t', 'efg_o', 'efg_d', 'tov_o', 'tov_d', 'orb']


def fetch_torvik_for_season(year):
    """Fetch Torvik T-Rank ratings for a historical season from the free JSON API."""
    print(f"  Fetching Torvik ratings for {year}...")
    try:
        url = f"https://barttorvik.com/trank.php?year={year}&json=1"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            print(f"    No data returned for {year}")
            return pd.DataFrame()

        first = data[0]
        if isinstance(first, dict):
            df = pd.DataFrame(data)
            rename_map = {
                'team': 'team_name', 'conf': 'conference', 'g': 'games',
                'barthag': 'barthag', 'adjoe': 'adj_o', 'adjde': 'adj_d', 'adjte': 'adj_t',
                'efg_o': 'efg_o', 'efg_d': 'efg_d',
                'tor': 'tov_o', 'tord': 'tov_d',
                'orb': 'orb', 'drb': 'drb', 'ftr': 'ftr', 'ftrd': 'ftrd',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        else:
            # list-of-lists positional mapping
            torvik_cols = [
                'rk', 'team_name', 'conference', 'games', 'record',
                'barthag', 'adj_o', 'adj_d', 'adj_t',
                'efg_o', 'efg_d', 'tov_o', 'tov_d',
                'orb', 'drb', 'ftr', 'ftrd',
                'two_p_o', 'two_p_d', 'three_p_o', 'three_p_d', 'wab',
            ]
            n = len(first)
            cols = torvik_cols[:n] if n <= len(torvik_cols) else torvik_cols + [f'x{i}' for i in range(n - len(torvik_cols))]
            df = pd.DataFrame(data, columns=cols)

        for col in df.columns:
            if col not in ('team_name', 'conference', 'record'):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[df['team_name'].notna() & ~df['team_name'].str.upper().eq('TEAM')].copy()
        df = df.reset_index(drop=True)
        print(f"    Got {len(df)} teams")
        return df

    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


def fetch_espn_games_for_date(date_str):
    """
    Fetch completed college basketball games from ESPN API for a YYYYMMDD date.
    Returns a list of dicts with home_team, away_team, scores.
    """
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {'dates': date_str, 'limit': 300, 'groups': 50}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        games = []
        for event in data.get('events', []):
            if event['status']['type']['state'] != 'post':
                continue
            competitors = event['competitions'][0]['competitors']
            home_team = away_team = None
            home_score = away_score = None
            for c in competitors:
                if c['homeAway'] == 'home':
                    home_team = c['team']['displayName']
                    home_score = int(c['score'])
                else:
                    away_team = c['team']['displayName']
                    away_score = int(c['score'])
            if home_team and away_team:
                games.append({
                    'game_date': date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:],
                    'home_team': home_team,
                    'away_team': away_team,
                    'actual_home_score': home_score,
                    'actual_away_score': away_score,
                })
        return games

    except Exception:
        return []


def attach_torvik_features(games, torvik_df):
    """
    Join game records with Torvik ratings by ESPN→Torvik team name matching.
    Returns only games where both teams could be matched.
    """
    available_torvik_cols = [c for c in TORVIK_COLS if c in torvik_df.columns]
    result = []

    for game in games:
        home_match = find_team_match(game['home_team'], torvik_df)
        away_match = find_team_match(game['away_team'], torvik_df)
        if not home_match or not away_match:
            continue

        home_row = torvik_df[torvik_df['team_name'] == home_match]
        away_row = torvik_df[torvik_df['team_name'] == away_match]
        if home_row.empty or away_row.empty:
            continue

        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]

        rec = dict(game)
        for col in available_torvik_cols:
            rec[f'home_{col}'] = home_row.get(col)
            rec[f'away_{col}'] = away_row.get(col)

        result.append(rec)

    return result


def db_connect():
    return sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
        _socket_timeout=900
    )


def upload_season_features(records, season):
    """
    Upload a season's pre-joined game features to workspace.default.historical_game_features.
    Replaces any existing rows for that season.
    """
    if not records:
        print(f"  No matched records for {season}")
        return 0

    df = pd.DataFrame(records)
    df['season'] = season
    print(f"  Uploading {len(df)} records for season {season}...")

    def safe_val(v):
        if v is None:
            return 'NULL'
        try:
            if pd.isna(v):
                return 'NULL'
        except (TypeError, ValueError):
            pass
        if isinstance(v, str):
            return f"'{v.replace(chr(39), chr(39)*2)}'"
        return str(v)

    try:
        conn = db_connect()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace.default.historical_game_features (
                season INT,
                game_date DATE,
                home_team STRING,
                away_team STRING,
                actual_home_score INT,
                actual_away_score INT,
                home_barthag DOUBLE,
                home_adj_o DOUBLE,
                home_adj_d DOUBLE,
                home_adj_t DOUBLE,
                home_efg_o DOUBLE,
                home_efg_d DOUBLE,
                home_tov_o DOUBLE,
                home_tov_d DOUBLE,
                home_orb DOUBLE,
                away_barthag DOUBLE,
                away_adj_o DOUBLE,
                away_adj_d DOUBLE,
                away_adj_t DOUBLE,
                away_efg_o DOUBLE,
                away_efg_d DOUBLE,
                away_tov_o DOUBLE,
                away_tov_d DOUBLE,
                away_orb DOUBLE
            ) USING DELTA
        """)

        # Replace entire season
        cursor.execute(f"DELETE FROM workspace.default.historical_game_features WHERE season = {season}")

        col_order = [
            'season', 'game_date', 'home_team', 'away_team',
            'actual_home_score', 'actual_away_score',
            'home_barthag', 'home_adj_o', 'home_adj_d', 'home_adj_t',
            'home_efg_o', 'home_efg_d', 'home_tov_o', 'home_tov_d', 'home_orb',
            'away_barthag', 'away_adj_o', 'away_adj_d', 'away_adj_t',
            'away_efg_o', 'away_efg_d', 'away_tov_o', 'away_tov_d', 'away_orb',
        ]
        # Fill any missing columns with None
        for col in col_order:
            if col not in df.columns:
                df[col] = None

        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df[col_order].iloc[i:i+batch_size]
            values_list = []
            for _, row in batch.iterrows():
                parts = [safe_val(row[c]) for c in col_order]
                values_list.append(f"({', '.join(parts)})")
            cursor.execute(f"INSERT INTO workspace.default.historical_game_features VALUES {', '.join(values_list)}")

        cursor.close()
        conn.close()
        print(f"  Uploaded {len(df)} records for {season}")
        return len(df)

    except Exception as e:
        print(f"  Error uploading season {season}: {e}")
        return 0


def main():
    print("=" * 70)
    print("HISTORICAL GAME FEATURES BACKFILL")
    print("=" * 70)
    print("Populates workspace.default.historical_game_features for ML training.")
    print("This is a one-time setup script — re-running overwrites existing seasons.\n")

    if not all([DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN]):
        print("Databricks credentials not found. Set them in .env and retry.")
        return

    total_uploaded = 0

    for season, (start_dt, end_dt) in sorted(SEASONS.items()):
        print(f"\n{'=' * 50}")
        print(f"Season {season}  ({start_dt}  →  {end_dt})")
        print("=" * 50)

        torvik_df = fetch_torvik_for_season(season)
        if torvik_df.empty:
            print(f"  No Torvik data for {season} — skipping")
            continue

        all_records = []
        current = start_dt
        total_dates = (end_dt - start_dt).days + 1
        dates_checked = 0
        games_found = 0

        while current <= end_dt:
            date_str = current.strftime('%Y%m%d')
            games = fetch_espn_games_for_date(date_str)

            if games:
                matched = attach_torvik_features(games, torvik_df)
                all_records.extend(matched)
                games_found += len(games)

            current += timedelta(days=1)
            dates_checked += 1

            if dates_checked % 30 == 0:
                print(f"  {dates_checked}/{total_dates} dates  |  {games_found} games found  |  {len(all_records)} matched")

            time.sleep(SLEEP_BETWEEN_DATES)

        match_rate = len(all_records) / games_found * 100 if games_found else 0
        print(f"  Done: {games_found} games, {len(all_records)} matched ({match_rate:.0f}% match rate)")

        uploaded = upload_season_features(all_records, season)
        total_uploaded += uploaded

    print(f"\n{'=' * 70}")
    print(f"Backfill complete — {total_uploaded} total records uploaded")
    print("=" * 70)


if __name__ == "__main__":
    main()
