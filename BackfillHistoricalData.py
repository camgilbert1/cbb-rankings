"""
Backfill Historical Game Features for ML Training
Fetches ESPN game results + KenPom ratings for seasons 2024-2026 and stores
pre-joined features in workspace.default.ml_game_features.

Every completed ESPN game is stored (including non-D1 opponents, with NULL KenPom
features) so per-team schedules — rest days, pregame ratings — are complete.
Historical KenPom ratings are season-end (ratings_point_in_time = FALSE).

Run once to seed the ML training table; TrainModel.py syncs the current season nightly.
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

FEATURE_TABLE = 'workspace.default.ml_game_features'
FEATURE_COLUMNS = [
    ('season', 'INT'), ('game_date', 'DATE'), ('home_team', 'STRING'), ('away_team', 'STRING'),
    ('actual_home_score', 'INT'), ('actual_away_score', 'INT'), ('neutral_site', 'BOOLEAN'),
    ('home_adj_o', 'DOUBLE'), ('home_adj_d', 'DOUBLE'), ('home_adj_em', 'DOUBLE'), ('home_tempo', 'DOUBLE'),
    ('away_adj_o', 'DOUBLE'), ('away_adj_d', 'DOUBLE'), ('away_adj_em', 'DOUBLE'), ('away_tempo', 'DOUBLE'),
    ('ratings_point_in_time', 'BOOLEAN'),
]


def fetch_kenpom_for_season(year):
    """
    Fetch KenPom season-end ratings via kenpompy.
    Applies the same column mapping as rebuild_gold_table in APIPull.py:
      adj_o = Tempo-Adj column
      adj_d = Off. Efficiency-Adj.Rank column
      adj_em = adj_o - adj_d
      tempo = Def. Efficiency-Raw column (verified vs kenpom.com: Alabama 75.9)
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
        tempo_col  = 'Def. Efficiency-Raw'

        if any(c not in df.columns for c in (adj_o_col, adj_d_col, tempo_col)):
            print(f"    Missing columns. Available: {df.columns.tolist()}")
            return pd.DataFrame()

        df['adj_o']  = pd.to_numeric(df[adj_o_col], errors='coerce')
        df['adj_d']  = pd.to_numeric(df[adj_d_col], errors='coerce')
        df['adj_em'] = df['adj_o'] - df['adj_d']
        df['tempo']  = pd.to_numeric(df[tempo_col], errors='coerce')

        # Keep only teams with plausible values
        df = df[df['adj_o'].between(70, 150) & df['adj_d'].between(70, 150)
                & df['tempo'].between(55, 85)].copy()
        df = df[['team_name', 'adj_o', 'adj_d', 'adj_em', 'tempo']].dropna()

        print(f"    Got {len(df)} teams  (adj_o range {df['adj_o'].min():.1f}–{df['adj_o'].max():.1f}, "
              f"adj_em range {df['adj_em'].min():.1f}–{df['adj_em'].max():.1f}, "
              f"tempo range {df['tempo'].min():.1f}–{df['tempo'].max():.1f})")
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
                        'neutral_site':      bool(event['competitions'][0].get('neutralSite', False)),
                    })
            return games
        except Exception:
            if attempt < retries - 1:
                time.sleep(5)
    return []


def team_ratings(espn_name, kenpom_df):
    """KenPom ratings dict for an ESPN team name, or all-None if no D1 match."""
    match = find_team_match(espn_name, kenpom_df)
    row = kenpom_df[kenpom_df['team_name'] == match] if match else kenpom_df.iloc[0:0]
    if row.empty:
        return {'adj_o': None, 'adj_d': None, 'adj_em': None, 'tempo': None}
    row = row.iloc[0]
    return {k: float(row[k]) for k in ('adj_o', 'adj_d', 'adj_em', 'tempo')}


def attach_kenpom_features(games, kenpom_df, point_in_time=False):
    """
    Join game records with KenPom ratings via team name matching.
    Keeps every game (NULL features for non-D1 sides) so team schedules stay complete.
    """
    result = []
    for game in games:
        h = team_ratings(game['home_team'], kenpom_df)
        a = team_ratings(game['away_team'], kenpom_df)
        result.append({
            **game,
            'home_adj_o': h['adj_o'], 'home_adj_d': h['adj_d'], 'home_adj_em': h['adj_em'], 'home_tempo': h['tempo'],
            'away_adj_o': a['adj_o'], 'away_adj_d': a['adj_d'], 'away_adj_em': a['adj_em'], 'away_tempo': a['tempo'],
            'ratings_point_in_time': point_in_time,
        })
    return result


def db_connect():
    return sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
        _socket_timeout=900,
    )


def upload_features(records, season, delete_where):
    """
    Write game feature rows to FEATURE_TABLE, first deleting rows matching `delete_where`
    (a SQL predicate) so re-runs are idempotent. Used by the backfill and TrainModel's nightly sync.
    """
    if not records:
        print(f"  No records for {season}")
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

        cols_sql = ',\n                '.join(f'{c} {t}' for c, t in FEATURE_COLUMNS)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {FEATURE_TABLE} (
                {cols_sql}
            ) USING DELTA
        """)

        cursor.execute(f"DELETE FROM {FEATURE_TABLE} WHERE {delete_where}")

        col_order = [c for c, _ in FEATURE_COLUMNS]
        for col in col_order:
            if col not in df.columns:
                df[col] = None

        batch_size = 200
        for i in range(0, len(df), batch_size):
            batch = df[col_order].iloc[i:i+batch_size]
            vals = []
            for _, row in batch.iterrows():
                parts = [safe(row[c]) for c in col_order]
                vals.append(f"({', '.join(parts)})")
            cursor.execute(f"INSERT INTO {FEATURE_TABLE} ({', '.join(col_order)}) VALUES {', '.join(vals)}")

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
                records = attach_kenpom_features(games, kenpom_df)
                all_records.extend(records)
                games_found += len(games)

            current    += timedelta(days=1)
            dates_done += 1

            if dates_done % 30 == 0:
                print(f"  {dates_done}/{total_dates} dates  |  {games_found} games")

            time.sleep(SLEEP_BETWEEN_DATES)

        both_d1 = sum(1 for r in all_records if r['home_adj_em'] is not None and r['away_adj_em'] is not None)
        neutral = sum(1 for r in all_records if r['neutral_site'])
        rate = both_d1 / games_found * 100 if games_found else 0
        print(f"  Done: {games_found} games, {both_d1} D1-vs-D1 with features ({rate:.0f}%), {neutral} neutral-site")

        total += upload_features(all_records, season, delete_where=f"season = {season}")

    print(f"\n{'=' * 70}")
    print(f"Backfill complete — {total} total records")
    print("=" * 70)


if __name__ == "__main__":
    main()
