"""
NCAA Tournament Bracket Predictions
Scrapes BracketMatrix consensus bracket and runs game predictions for each matchup.

Run this script on Tuesdays and Fridays when bracketology updates.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from databricks import sql
import os
from datetime import datetime
from dotenv import load_dotenv

# Import prediction functions from GamePredictions
from GamePredictions import (
    get_team_stats_from_databricks,
    calculate_matchup_features,
    predict_game
)

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Standard first-round matchup pairings by seed
FIRST_ROUND_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15)
]

REGIONS = ['East', 'West', 'South', 'Midwest']


# =============================================================================
# SCRAPING
# =============================================================================

def scrape_bracketmatrix():
    """
    Scrape BracketMatrix consensus bracket.

    Table row format: cell[0]=seed, cell[1]=team, cell[2]=conference, cell[3]=avg_seed
    BracketMatrix has no region column — regions are assigned by avg_seed rank within
    each seed line (best avg_seed → East, then West, South, Midwest).

    Seeds 11 and 16 have 6 teams each: top 2 get direct region assignments (East/West),
    bottom 4 are First Four play-in candidates (no region assigned).
    All other seed lines have exactly 4 teams.

    Returns:
        pd.DataFrame: Teams with columns: team, seed, region, avg_seed, conference
    """
    print("Scraping BracketMatrix consensus bracket...")

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        url = "https://www.bracketmatrix.com/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')

        if not table:
            print("❌ No table found on BracketMatrix page")
            return pd.DataFrame()

        raw_teams = []
        rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 4:
                continue

            seed_text = cells[0].get_text(strip=True)
            team_text = cells[1].get_text(strip=True)
            conf_text = cells[2].get_text(strip=True)
            avg_seed_text = cells[3].get_text(strip=True)

            # Seed must be integer 1-16
            try:
                seed = int(seed_text)
                if not (1 <= seed <= 16):
                    continue
            except ValueError:
                continue

            # Team name must start with uppercase (filters header rows)
            if not team_text or not team_text[0].isupper():
                continue

            # avg_seed must be a valid float
            try:
                avg_seed = float(avg_seed_text)
            except ValueError:
                continue

            raw_teams.append({
                'team': team_text,
                'seed': seed,
                'conference': conf_text,
                'avg_seed': avg_seed
            })

        if not raw_teams:
            print("❌ Could not parse any teams from BracketMatrix table")
            return pd.DataFrame()

        raw_df = pd.DataFrame(raw_teams).drop_duplicates(subset=['team'])
        print(f"  Parsed {len(raw_df)} unique teams")

        # Assign regions within each seed line sorted by avg_seed (best first).
        # Seeds 11 and 16: only top 2 get direct region slots (East, West).
        #   Their bottom 4 are play-in candidates — included with region=None so
        #   build_bracket() can detect them via the > 4 team count per seed.
        # All other seeds: top 4 get East/West/South/Midwest.
        region_order = ['East', 'West', 'South', 'Midwest']
        result_rows = []

        for seed_num in range(1, 17):
            seed_teams = raw_df[raw_df['seed'] == seed_num].sort_values('avg_seed')
            direct_count = 2 if seed_num in [11, 16] else 4

            for i, (_, team_row) in enumerate(seed_teams.iterrows()):
                region = region_order[i] if i < direct_count else None
                result_rows.append({
                    'team': team_row['team'],
                    'seed': seed_num,
                    'region': region,
                    'avg_seed': team_row['avg_seed'],
                    'conference': team_row.get('conference', '')
                })

        result_df = pd.DataFrame(result_rows)
        print(f"✓ Scraped {len(result_df)} teams from BracketMatrix ({len(result_df[result_df['region'].notna()])} in field, {len(result_df[result_df['region'].isna()])} play-in)")
        return result_df

    except Exception as e:
        print(f"❌ Error scraping BracketMatrix: {e}")
        return pd.DataFrame()


# =============================================================================
# BRACKET CONSTRUCTION
# =============================================================================

def build_bracket(bracket_df):
    """
    Build First Four + First Round matchups from scraped bracket data.

    Play-in rule:
      - Bottom 4 teams among #16 seeds (by avg_seed desc) → First Four
      - Bottom 4 teams among #11 seeds (by avg_seed desc) → First Four

    Returns:
        tuple: (first_four_matchups, first_round_matchups)
            Each is a list of dicts with keys: round, region, top_seed, top_team, low_seed, low_team
    """
    print("\nBuilding bracket matchups...")

    first_four = []
    first_round = []

    # Identify play-in teams for seeds 16 and 11
    playin_teams = {}  # seed -> list of 4 play-in teams (sorted worst to best)
    main_teams = {}    # (seed, region) -> team name for non-play-in slots

    for playin_seed in [16, 11]:
        seed_teams = bracket_df[bracket_df['seed'] == playin_seed].copy()
        seed_teams = seed_teams.sort_values('avg_seed', ascending=False)  # worst first

        if len(seed_teams) >= 4:
            # Bottom 4 go to play-in
            playin = seed_teams.head(4)['team'].tolist()
            playin_teams[playin_seed] = playin
            print(f"  #{playin_seed} play-in teams: {playin}")
        else:
            playin_teams[playin_seed] = []

    # Build First Four games (2 games per play-in seed)
    for playin_seed in [16, 11]:
        if len(playin_teams.get(playin_seed, [])) >= 4:
            pi = playin_teams[playin_seed]
            # Find which regions these feed into — use the region of the corresponding main seed
            # For simplicity, pair them: worst vs 2nd worst, 3rd worst vs 4th worst
            first_four.append({
                'round': 'First Four',
                'region': f'#{playin_seed} Play-In A',
                'top_seed': playin_seed,
                'top_team': pi[2],  # 3rd worst
                'low_seed': playin_seed,
                'low_team': pi[3],  # 4th worst (best of the 4)
                'playin_feeds_region': None  # Will be filled when we know regions
            })
            first_four.append({
                'round': 'First Four',
                'region': f'#{playin_seed} Play-In B',
                'top_seed': playin_seed,
                'top_team': pi[0],  # worst
                'low_seed': playin_seed,
                'low_team': pi[1],  # 2nd worst
                'playin_feeds_region': None
            })

    # Build First Round matchups per region
    # Only use teams with a direct region assignment (excludes play-in candidates)
    for region in REGIONS:
        region_teams = bracket_df[bracket_df['region'] == region].copy()

        for high_seed, low_seed in FIRST_ROUND_MATCHUPS:
            high_team_row = region_teams[region_teams['seed'] == high_seed]
            low_team_row = region_teams[region_teams['seed'] == low_seed]

            # Skip if teams not found (e.g., play-in seeds don't have a direct team)
            high_team = high_team_row.iloc[0]['team'] if not high_team_row.empty else None
            low_team = low_team_row.iloc[0]['team'] if not low_team_row.empty else None

            # For play-in seeds with no direct team in this region, use placeholder
            if low_team is None and low_seed in [16, 11] and len(playin_teams.get(low_seed, [])) >= 4:
                low_team = f"#{low_seed} Play-In Winner"

            if high_team:
                first_round.append({
                    'round': 'First Round',
                    'region': region,
                    'top_seed': high_seed,
                    'top_team': high_team,
                    'low_seed': low_seed,
                    'low_team': low_team or f'#{low_seed} TBD'
                })

    print(f"  ✓ {len(first_four)} First Four games")
    print(f"  ✓ {len(first_round)} First Round games")
    return first_four, first_round


# =============================================================================
# PREDICTIONS
# =============================================================================

def run_bracket_predictions(first_four, first_round, team_stats):
    """
    Run predictions for all First Four and First Round matchups.

    Returns:
        list: Prediction records ready for Databricks upload
    """
    print("\nRunning bracket predictions...")
    all_matchups = first_four + first_round
    records = []

    for matchup in all_matchups:
        home_team = matchup['top_team']   # Higher seed treated as "home" (neutral court)
        away_team = matchup['low_team']

        # Skip play-in placeholders in first round (can't predict TBD)
        if 'Play-In Winner' in away_team or 'TBD' in away_team:
            print(f"  Skipping {home_team} vs {away_team} (play-in winner TBD)")
            records.append({
                'round': matchup['round'],
                'region': matchup['region'],
                'top_seed': matchup['top_seed'],
                'top_team': home_team,
                'low_seed': matchup['low_seed'],
                'low_team': away_team,
                'predicted_winner': 'TBD',
                'win_probability': None,
                'predicted_top_score': None,
                'predicted_low_score': None,
                'confidence': 'N/A',
                'updated_at': datetime.now().isoformat()
            })
            continue

        print(f"  {home_team} vs {away_team}")

        features = calculate_matchup_features(home_team, away_team, team_stats)
        prediction = predict_game(home_team, away_team, features)

        records.append({
            'round': matchup['round'],
            'region': matchup['region'],
            'top_seed': matchup['top_seed'],
            'top_team': home_team,
            'low_seed': matchup['low_seed'],
            'low_team': away_team,
            'predicted_winner': prediction['predicted_winner'],
            'win_probability': prediction['win_probability'],
            'predicted_top_score': prediction['predicted_home_score'],
            'predicted_low_score': prediction['predicted_away_score'],
            'confidence': prediction['confidence'],
            'updated_at': datetime.now().isoformat()
        })

        print(f"    → {prediction['predicted_winner']} ({prediction['win_probability']:.0%}, {prediction['confidence']})")

    return records


# =============================================================================
# DATABRICKS UPLOAD
# =============================================================================

def upload_bracket_predictions(records):
    """Upload bracket predictions to Databricks."""
    print("\nUploading bracket predictions to Databricks...")

    if not records:
        print("⚠️  No predictions to upload")
        return

    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN
        )
        cursor = connection.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace.default.bracket_predictions (
                round STRING,
                region STRING,
                top_seed INT,
                top_team STRING,
                low_seed INT,
                low_team STRING,
                predicted_winner STRING,
                win_probability DOUBLE,
                predicted_top_score DOUBLE,
                predicted_low_score DOUBLE,
                confidence STRING,
                updated_at STRING
            )
            USING DELTA
        """)

        # Replace all bracket predictions on each run
        cursor.execute("DELETE FROM workspace.default.bracket_predictions")
        print("  ✓ Cleared old bracket predictions")

        for rec in records:
            top_team_esc = rec['top_team'].replace("'", "''")
            low_team_esc = rec['low_team'].replace("'", "''")
            winner_esc = rec['predicted_winner'].replace("'", "''")

            win_prob = f"{rec['win_probability']}" if rec['win_probability'] is not None else "NULL"
            top_score = f"{rec['predicted_top_score']}" if rec['predicted_top_score'] is not None else "NULL"
            low_score = f"{rec['predicted_low_score']}" if rec['predicted_low_score'] is not None else "NULL"

            cursor.execute(f"""
                INSERT INTO workspace.default.bracket_predictions VALUES (
                    '{rec['round']}',
                    '{rec['region']}',
                    {rec['top_seed']},
                    '{top_team_esc}',
                    {rec['low_seed']},
                    '{low_team_esc}',
                    '{winner_esc}',
                    {win_prob},
                    {top_score},
                    {low_score},
                    '{rec['confidence']}',
                    '{rec['updated_at']}'
                )
            """)

        cursor.close()
        connection.close()
        print(f"✓ Uploaded {len(records)} bracket predictions to Databricks")

    except Exception as e:
        print(f"❌ Error uploading bracket predictions: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("NCAA TOURNAMENT BRACKET PREDICTIONS")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%B %d, %Y')}\n")

    # Step 1: Scrape bracket
    bracket_df = scrape_bracketmatrix()
    if bracket_df.empty:
        print("❌ Could not load bracket data. Exiting.")
        return

    print(f"\nBracket snapshot:")
    print(bracket_df.groupby(['seed', 'region'])['team'].apply(list).to_string())

    # Step 2: Build matchups
    first_four, first_round = build_bracket(bracket_df)

    # Step 3: Load team stats
    team_stats = get_team_stats_from_databricks()
    if team_stats.empty:
        print("❌ Could not load team stats. Exiting.")
        return

    # Step 4: Run predictions
    records = run_bracket_predictions(first_four, first_round, team_stats)

    # Step 5: Upload to Databricks
    if DATABRICKS_HOST and DATABRICKS_HTTP_PATH and DATABRICKS_TOKEN:
        upload_bracket_predictions(records)
    else:
        print("⚠️  Databricks credentials not found. Predictions not uploaded.")

    # Step 6: Print summary
    print("\n" + "=" * 80)
    print("BRACKET PREDICTION SUMMARY")
    print("=" * 80)
    for rec in records:
        if rec['predicted_winner'] != 'TBD':
            print(f"  #{rec['top_seed']} {rec['top_team']} vs #{rec['low_seed']} {rec['low_team']}")
            print(f"    → {rec['predicted_winner']} ({rec['win_probability']:.0%}) [{rec['region']}]")

    print(f"\nTotal predictions: {len([r for r in records if r['predicted_winner'] != 'TBD'])}")


if __name__ == "__main__":
    main()
