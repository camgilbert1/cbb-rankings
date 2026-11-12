"""
College Basketball Rankings Data Pull Script
Fetches current college basketball rankings from multiple sources:
- KenPom (ACTIVE)
- Bart Torvik (T-Rank) - TODO: needs Selenium
- Haslametrics - TODO: complex table structure, needs more work
"""

import pandas as pd
import cloudscraper
import kenpompy as kenpompy
from kenpompy.summary import get_efficiency
from databricks import sql
import os
from dotenv import load_dotenv

load_dotenv()


def login_kenpom(email: str, password: str):
    """
    Custom KenPom login that doesn't rely on kenpompy's 'Logged in as' text check,
    which broke when KenPom updated their UI.
    """
    browser = cloudscraper.create_scraper()
    index_page = browser.get('https://kenpom.com/index.php')

    # Print the login form to see current field names and any hidden fields
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(index_page.text, 'html.parser')
    form = soup.find('form')
    print(f"  DEBUG - Login form found: {form is not None}")
    if form:
        print(f"  DEBUG - Form action: {form.get('action')}")
        for inp in form.find_all('input'):
            print(f"  DEBUG - Form field: name='{inp.get('name')}' type='{inp.get('type')}' value='{inp.get('value')}'")

    form_data = {
        'email': email,
        'password': password,
        'submit': 'Login!',
    }

    login_response = browser.post(
        'https://kenpom.com/handlers/login_handler.php',
        data=form_data,
        allow_redirects=True
    )
    print(f"  DEBUG - Login POST status: {login_response.status_code}")
    print(f"  DEBUG - Login POST response (first 300): {login_response.text[:300]}")

    home_page = browser.get('https://kenpom.com/')
    page_text = home_page.text.lower()

    print(f"  DEBUG - Home page status: {home_page.status_code}")
    print(f"  DEBUG - Page contains 'logout': {'logout' in page_text}")
    print(f"  DEBUG - Page contains 'logged in': {'logged in' in page_text}")
    print(f"  DEBUG - Page contains 'login' form: {'name=\"password\"' in home_page.text}")

    authenticated = any(s in page_text for s in ['logout', 'log out', 'sign out', 'logged in'])
    if not authenticated and 'name="password"' in home_page.text:
        raise Exception('Logging in failed - check your KenPom credentials in .env')

    return browser

# =============================================================================
# CONFIGURATION - KenPom credentials from environment variables
# =============================================================================
KENPOM_EMAIL = os.getenv("KENPOM_EMAIL")
KENPOM_PASSWORD = os.getenv("KENPOM_PASSWORD")

# =============================================================================
# FETCH DATA
# =============================================================================

def fetch_kenpom_ratings(year=2026):
    """
    Fetch KenPom ratings for specified year

    Args:
        year (int): Season year (e.g., 2026 for 2025-26 season)

    Returns:
        pd.DataFrame: KenPom ratings data
    """
    print(f"Logging into KenPom...")

    # Create browser session and login
    browser = login_kenpom(KENPOM_EMAIL, KENPOM_PASSWORD)

    print(f"Fetching {year} ratings...")

    # Get current efficiency ratings (main KenPom table)
    ratings_df = get_efficiency(browser, season=str(year))

    print(f"Successfully fetched {len(ratings_df)} teams!")

    return ratings_df


def upload_to_databricks(df, table_name, year):
    """
    Upload dataframe to Databricks, replacing existing table

    Args:
        df (pd.DataFrame): Data to upload
        table_name (str): Name of the table (e.g., 'kenpom_ratings')
        year (int): Season year for table naming
    """
    # Get Databricks credentials from environment variables
    host = os.getenv("DATABRICKS_HOST")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")

    if not all([host, http_path, token]):
        print("⚠️  Databricks credentials not found in environment variables.")
        print("   Skipping upload to Databricks. CSV file saved locally.")
        return

    print(f"\nUploading {table_name}_{year} to Databricks...")

    try:
        # Connect to Databricks
        connection = sql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=token,
            _socket_timeout=900
        )

        cursor = connection.cursor()

        # Drop existing table
        full_table_name = f"workspace.default.{table_name}_{year}"
        cursor.execute(f"DROP TABLE IF EXISTS {full_table_name}")
        print(f"✓ Dropped existing table {full_table_name}")

        # Clean column names (replace invalid characters for Delta tables)
        df_clean = df.copy()
        df_clean.columns = [col.replace('-', '_').replace('.', '_').replace(' ', '_') for col in df_clean.columns]

        # Convert numeric columns to proper types
        for col in df_clean.columns:
            # Try to convert to numeric, leave as-is if fails
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except:
                pass

        # Build column definitions from dataframe
        columns = []
        for col in df_clean.columns:
            dtype = df_clean[col].dtype
            if dtype == 'object':
                sql_type = 'STRING'
            elif dtype == 'int64':
                sql_type = 'BIGINT'
            elif dtype == 'float64':
                sql_type = 'DOUBLE'
            else:
                sql_type = 'STRING'
            columns.append(f"`{col}` {sql_type}")

        # Create empty table with proper schema
        cursor.execute(f"DROP TABLE IF EXISTS {full_table_name}")
        cursor.execute(f"""
            CREATE TABLE {full_table_name} (
                {', '.join(columns)}
            )
            USING DELTA
        """)

        # Insert data in batches
        batch_size = 100
        total_rows = len(df_clean)

        for i in range(0, total_rows, batch_size):
            batch = df_clean.iloc[i:i+batch_size]

            # Build VALUES clause
            values_list = []
            for _, row in batch.iterrows():
                value_parts = []
                for val in row:
                    if pd.isna(val):
                        value_parts.append('NULL')
                    elif isinstance(val, str):
                        # Escape single quotes
                        escaped_val = val.replace("'", "''")
                        value_parts.append(f"'{escaped_val}'")
                    else:
                        value_parts.append(str(val))
                values_list.append(f"({', '.join(value_parts)})")

            values_clause = ', '.join(values_list)

            insert_sql = f"""
                INSERT INTO {full_table_name}
                VALUES {values_clause}
            """

            cursor.execute(insert_sql)

            if (i + batch_size) % 500 == 0:
                print(f"  Inserted {min(i + batch_size, total_rows)}/{total_rows} rows...")

        print(f"✓ Successfully uploaded {total_rows} rows to {full_table_name}")

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"❌ Error uploading to Databricks: {e}")
        print(f"   CSV file saved locally as backup.")


def rebuild_gold_table(year=2027):
    """
    Rebuild the gold_team_rankings table directly from raw KenPom data.
    Replaces the dbt run step so dbt doesn't need to be installed.
    """
    host = os.getenv("DATABRICKS_HOST")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")

    if not all([host, http_path, token]):
        print("⚠️  Databricks credentials not found. Skipping gold table rebuild.")
        return

    print("\nChecking gold_team_rankings table...")

    try:
        from datetime import date
        connection = sql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=token,
            _socket_timeout=900
        )
        cursor = connection.cursor()

        # Skip rebuild if already done this week (KenPom data doesn't change daily)
        try:
            cursor.execute("SELECT MAX(loaded_at) FROM workspace.default_gold.gold_team_rankings")
            row = cursor.fetchone()
            if row and row[0] and hasattr(row[0], 'date'):
                days_since_rebuild = (date.today() - row[0].date()).days
                if days_since_rebuild < 2:
                    print(f"✓ gold_team_rankings rebuilt {days_since_rebuild} day(s) ago, skipping")
                    cursor.close()
                    connection.close()
                    return
        except Exception:
            pass  # Table doesn't exist yet, proceed with rebuild

        print("  Rebuilding gold_team_rankings...")
        cursor.execute(f"""
            CREATE OR REPLACE TABLE workspace.default_gold.gold_team_rankings
            USING DELTA AS
            SELECT
                Team                                                                                                        AS team_name,
                Conference                                                                                                  AS conference,
                (TRY_CAST(Tempo_Adj AS DOUBLE) - TRY_CAST(Off__Efficiency_Adj_Rank AS DOUBLE))                            AS adj_efficiency_margin,
                TRY_CAST(Tempo_Adj AS DOUBLE)                                                                              AS adj_offensive_efficiency,
                TRY_CAST(Off__Efficiency_Adj_Rank AS DOUBLE)                                                               AS adj_defensive_efficiency,
                68.0                                                                                                        AS adj_tempo,
                ROW_NUMBER() OVER (ORDER BY (TRY_CAST(Tempo_Adj AS DOUBLE) - TRY_CAST(Off__Efficiency_Adj_Rank AS DOUBLE)) DESC) AS overall_rank,
                TRY_CAST(Off__Efficiency_Adj AS INT)                                                                       AS offensive_rank,
                TRY_CAST(Def__Efficiency_Adj_Rank AS INT)                                                                  AS defensive_rank,
                current_timestamp()                                                                                        AS loaded_at
            FROM workspace.default.kenpom_ratings_{year}
            WHERE Team IS NOT NULL
            ORDER BY overall_rank
        """)

        cursor.close()
        connection.close()
        print("✓ gold_team_rankings rebuilt successfully")

    except Exception as e:
        print(f"❌ Error rebuilding gold table: {e}")


def main():
    """Main execution function"""

    # =============================================================================
    # FETCH ALL DATA SOURCES
    # =============================================================================

    # 1. KenPom (PRIMARY DATA SOURCE)
    kenpom_data = fetch_kenpom_ratings(year=2027)

    # =============================================================================
    # DISPLAY KENPOM DATA
    # =============================================================================

    print("\n" + "="*80)
    print("KENPOM RATINGS - First 10 Teams:")
    print("="*80)
    print(kenpom_data.head(10))

    print("\n" + "="*80)
    print("KENPOM - AVAILABLE COLUMNS:")
    print("="*80)
    print(kenpom_data.columns.tolist())

    # =============================================================================
    # SAVE DATA
    # =============================================================================

    kenpom_file = "kenpom_ratings_2027.csv"
    kenpom_data.to_csv(kenpom_file, index=False)
    print(f"\n✓ KenPom data saved to: {kenpom_file}")

    upload_to_databricks(kenpom_data, "kenpom_ratings", 2027)
    rebuild_gold_table(year=2027)

    # =============================================================================
    # SUMMARY
    # =============================================================================

    print("\n" + "="*80)
    print("DATA COLLECTION SUMMARY:")
    print("="*80)
    print(f"KenPom Teams: {len(kenpom_data)}")
    print("\nData sources fetched successfully!")

    return kenpom_data


if __name__ == "__main__":
    # Run the script
    data = main()
