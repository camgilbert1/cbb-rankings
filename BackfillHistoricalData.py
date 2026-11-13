"""
Backfill Historical Game Features for ML Training
Fetches ESPN game results + KenPom ratings for seasons 2024-2026 and stores
pre-joined features in workspace.default.historical_game_features.

Run once to seed the ML training table, then TrainModel.py nightly handles the rest.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import requests
import pandas as pd
from databricks import sql
import os
import time
import cloudscraper
from datetime import date, timedelta
from dotenv import load_dotenv
from kenpompy.summary import get_efficiency

from team_names import find_team_match
from APIPull import login_kenpom

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
KENPOM_EMAIL    = os.getenv("KENPOM_EMAIL")
KENPOM_PASSWORD = os.getenv("KENPOM_PASSWORD")

SEASONS = {
    2024: (date(2023, 11, 1), date(2024, 4, 15)),
    2025: (date(2024, 11, 1), date(2025, 4, 15)),
    2026: (date(2025, 11, 1), date(2026, 4, 15)),
}

SLEEP_BETWEEN_DATES = 0.2


def fetch_kenpom_for_season(year):
    """
    Fetch KenPom season-end ratings via kenpompy.
    Applies the same column mapping as rebuild_gold_table in APIPull.py:
      adj_o = Tempo-Adj column
      adj_d = Off. Efficiency-Adj.Rank column
      adj_em = adj_o - adj_d
    """
    print(f"  Fetching KenPom {year}...")
    try:
        browser = login_kenpom(KENPOM_EMAIL, KENPOM_PASSWORD)
        df = get_efficiency(browser, season=str(year))

        # Apply the same mapping that rebuild_gold_table uses
        # (kenpompy columns are positionally shifted after the KenPom site redesign)
        df = df.rename(columns={'Team': 'team_name'})

        adj_o_col  = 'Tempo-Adj'
        adj_d_col  = 'Off. Efficiency-Adj.Rank'

        if adj_o_col not in df.columns or adj_d_col not in df.columns:
            print(f"    Missing columns. Available: {df.columns.tolist()}")
            return pd.DataFrame()

        df['adj_o']  = pd.to_numeric(df[adj_o_col], errors='coerce')
        df['adj_d']  = pd.to_numeric(df[adj_d_col], errors='coerce')
        df['adj_em'] = df['adj_o'] - df['adj_d']

        # Keep only teams with plausible values
        df = df[df['adj_o'].between(70, 150) & df['adj_d'].between(70, 150)].copy()
        df = df[['team_name', 'adj_o', 'adj_d', 'adj_em']].dropna()

        print(f"    Got {len(df)} teams  (adj_o range {df['adj_o'].min():.1f}–{df['adj_o'].max():.1f}, "
              f"adj_em range {df['adj_em'].min():.1f}–{df['adj_em'].max():.1f})")
        return df

    except Exception as e:
        print(f"    Error fetching KenPom {year}: {e}")
        return pd.DataFrame()


def fetch_espn_games_for_date(date_str, retries=3):
    """Fetch completed games from ESPN for YYYYMMDD. Retries on 429/5xx with backoff."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params={'dates': date_str, 'limit': 300, 'groups': 50}, timeout=15)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited — sleeping {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            games = []
            for event in resp.json().get('events', []):
                if event['status']['type']['state'] != 'post':
                    continue
                competitors = event['competitions'][0]['competitors']
                home = away = None
                home_score = away_score = None
                for c in competitors:
                    if c['homeAway'] == 'home':
                        home = c['team']['displayName']
                        home_score = int(c['score'])
                    else:
                        away = c['team']['displayName']
                        away_score = int(c['score'])
                if home and away:
                    games.append({
                        'game_date':         date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:],
                        'home_team':         home,
                        'away_team':         away,
                        'actual_home_score': home_score,
                        'actual_away_score': away_score,
                    })
            return games
        except Exception:
            if attempt < retries - 1:
                time.sleep(5)
    return []


def attach_kenpom_features(games, kenpom_df):
    """Join game records with KenPom ratings via team name matching."""
    result = []
    for game in games:
        home_match = find_team_match(game['home_team'], kenpom_df)
        away_match = find_team_match(game['away_team'], kenpom_df)
        if not home_match or not away_match:
            continue

        h = kenpom_df[kenpom_df['team_name'] == home_match]
        a = kenpom_df[kenpom_df['team_name'] == away_match]
        if h.empty or a.empty:
            continue
        h, a = h.iloc[0], a.iloc[0]

        result.append({
            **game,
            'home_adj_o':  float(h['adj_o']),
            'home_adj_d':  float(h['adj_d']),
            'home_adj_em': float(h['adj_em']),
            'away_adj_o':  float(a['adj_o']),
            'away_adj_d':  float(a['adj_d']),
            'away_adj_em': float(a['adj_em']),
        })
    return result


def db_connect():
    return sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
        _socket_timeout=900,
    )


def upload_season_features(records, season):
    """Upload pre-joined game features to workspace.default.historical_game_features."""
    if not records:
        print(f"  No matched records for {season}")
        return 0

    df = pd.DataFrame(records)
    df['season'] = season
    print(f"  Uploading {len(df)} records for {season}...")

    def safe(v):
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
                home_adj_o  DOUBLE,
                home_adj_d  DOUBLE,
                home_adj_em DOUBLE,
                away_adj_o  DOUBLE,
                away_adj_d  DOUBLE,
                away_adj_em DOUBLE
            ) USING DELTA
        """)

        cursor.execute(f"DELETE FROM workspace.default.historical_game_features WHERE season = {season}")

        col_order = ['season', 'game_date', 'home_team', 'away_team',
                     'actual_home_score', 'actual_away_score',
                     'home_adj_o', 'home_adj_d', 'home_adj_em',
                     'away_adj_o', 'away_adj_d', 'away_adj_em']
        for col in col_order:
            if col not in df.columns:
                df[col] = None

        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df[col_order].iloc[i:i+batch_size]
            vals = []
            for _, row in batch.iterrows():
                parts = [safe(row[c]) for c in col_order]
                vals.append(f"({', '.join(parts)})")
            cursor.execute(f"INSERT INTO workspace.default.historical_game_features VALUES {', '.join(vals)}")

        cursor.close()
        conn.close()
        print(f"  Uploaded {len(df)} records")
        return len(df)

    except Exception as e:
        print(f"  Error uploading {season}: {e}")
        return 0


def main():
    print("=" * 70)
    print("HISTORICAL GAME FEATURES BACKFILL  (KenPom-based)")
    print("=" * 70)

    if not all([DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN]):
        print("Databricks credentials not found.")
        return
    if not all([KENPOM_EMAIL, KENPOM_PASSWORD]):
        print("KenPom credentials not found.")
        return

    total = 0

    for season, (start_dt, end_dt) in sorted(SEASONS.items()):
        print(f"\n{'=' * 50}")
        print(f"Season {season}  ({start_dt} to {end_dt})")
        print("=" * 50)

        kenpom_df = fetch_kenpom_for_season(season)
        if kenpom_df.empty:
            print(f"  No KenPom data for {season} — skipping")
            continue

        all_records = []
        current      = start_dt
        total_dates  = (end_dt - start_dt).days + 1
        dates_done   = 0
        games_found  = 0

        while current <= end_dt:
            date_str = current.strftime('%Y%m%d')
            games = fetch_espn_games_for_date(date_str)

            if games:
                matched = attach_kenpom_features(games, kenpom_df)
                all_records.extend(matched)
                games_found += len(games)

            current    += timedelta(days=1)
            dates_done += 1

            if dates_done % 30 == 0:
                print(f"  {dates_done}/{total_dates} dates  |  "
                      f"{games_found} games  |  {len(all_records)} matched")

            time.sleep(SLEEP_BETWEEN_DATES)

        rate = len(all_records) / games_found * 100 if games_found else 0
        print(f"  Done: {games_found} games, {len(all_records)} matched ({rate:.0f}%)")

        total += upload_season_features(all_records, season)

    print(f"\n{'=' * 70}")
    print(f"Backfill complete — {total} total records")
    print("=" * 70)


if __name__ == "__main__":
    main()
