"""
College Basketball Rankings Data Pull Script
Fetches current college basketball rankings from multiple sources:
- KenPom (ACTIVE)
- Bart Torvik (T-Rank) - TODO: needs Selenium
- Haslametrics - TODO: complex table structure, needs more work
"""

import pandas as pd
import requests
import kenpompy as kenpompy
from kenpompy.utils import login
from kenpompy.summary import get_efficiency

# =============================================================================
# CONFIGURATION - Add your KenPom credentials here
# =============================================================================
KENPOM_EMAIL = "cameron.gilbert52@gmail.com"  # Replace with your KenPom email
KENPOM_PASSWORD = "Asurules_32"         # Replace with your KenPom password

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
    browser = login(KENPOM_EMAIL, KENPOM_PASSWORD)

    print(f"Fetching {year} ratings...")

    # Get current efficiency ratings (main KenPom table)
    ratings_df = get_efficiency(browser, season=str(year))

    print(f"Successfully fetched {len(ratings_df)} teams!")

    return ratings_df


def fetch_torvik_ratings(year=2026):
    """
    Fetch Bart Torvik T-Rank ratings by scraping barttorvik.com

    Args:
        year (int): Season year (e.g., 2026 for 2025-26 season)

    Returns:
        pd.DataFrame: Torvik ratings data
    """
    print(f"\nFetching Torvik ratings for {year}...")

    # Torvik T-Rank URL
    url = f"https://barttorvik.com/trank.php?year={year}"

    try:
        # Add headers to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Make the request with headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Read HTML tables from the response content
        tables = pd.read_html(response.content)

        # The main T-Rank table is typically the first table
        torvik_df = tables[0]

        print(f"Successfully fetched {len(torvik_df)} teams from Torvik!")

        return torvik_df

    except Exception as e:
        print(f"Error fetching Torvik data: {e}")
        return None


def fetch_haslametrics_ratings():
    """
    Fetch Haslametrics ratings by scraping haslametrics.com

    Returns:
        pd.DataFrame: Haslametrics ratings data
    """
    print(f"\nFetching Haslametrics ratings...")

    # Haslametrics ratings URL
    url = "https://www.haslametrics.com/ratings.php"

    try:
        # Add headers to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Make the request with headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Read HTML tables from the response content
        tables = pd.read_html(response.content)

        print(f"Found {len(tables)} tables on the page")

        # Try to find the table with team ratings (should have many rows)
        hasla_df = None
        for i, table in enumerate(tables):
            print(f"Table {i}: {len(table)} rows, {len(table.columns)} columns")
            # Ratings table should have 300+ teams
            if len(table) > 100:
                hasla_df = table
                print(f"Using table {i} as the main ratings table")

                # Haslametrics has multi-level headers, flatten them
                if isinstance(hasla_df.columns, pd.MultiIndex):
                    # Flatten multi-level columns to single level
                    hasla_df.columns = ['_'.join(col).strip('_') for col in hasla_df.columns.values]

                # For now, just keep all rows - we'll clean it up later in dbt
                # Reset index
                hasla_df = hasla_df.reset_index(drop=True)

                print(f"Haslametrics table has {len(hasla_df)} rows (includes headers)")

                break

        if hasla_df is not None and len(hasla_df) > 0:
            print(f"Successfully fetched {len(hasla_df)} teams from Haslametrics!")
            return hasla_df
        else:
            print("Could not find valid ratings table in Haslametrics page")
            return None

    except Exception as e:
        print(f"Error fetching Haslametrics data: {e}")
        return None


def main():
    """Main execution function"""

    # =============================================================================
    # FETCH ALL DATA SOURCES
    # =============================================================================

    # 1. KenPom (PRIMARY DATA SOURCE)
    kenpom_data = fetch_kenpom_ratings(year=2026)

    # 2. Haslametrics (commented out - needs more work on table parsing)
    # hasla_data = fetch_haslametrics_ratings()
    hasla_data = None

    # 3. Torvik (commented out - needs Selenium for JavaScript rendering)
    # torvik_data = fetch_torvik_ratings(year=2026)
    torvik_data = None

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
    # DISPLAY HASLAMETRICS DATA
    # =============================================================================

    if hasla_data is not None:
        print("\n" + "="*80)
        print("HASLAMETRICS RATINGS - First 10 Teams:")
        print("="*80)
        print(hasla_data.head(10))

        print("\n" + "="*80)
        print("HASLAMETRICS - AVAILABLE COLUMNS:")
        print("="*80)
        print(hasla_data.columns.tolist())

    # =============================================================================
    # DISPLAY TORVIK DATA (if available)
    # =============================================================================

    if torvik_data is not None:
        print("\n" + "="*80)
        print("TORVIK RATINGS - First 10 Teams:")
        print("="*80)
        print(torvik_data.head(10))

        print("\n" + "="*80)
        print("TORVIK - AVAILABLE COLUMNS:")
        print("="*80)
        print(torvik_data.columns.tolist())

    # =============================================================================
    # SAVE DATA
    # =============================================================================

    # Save KenPom
    kenpom_file = "kenpom_ratings_2026.csv"
    kenpom_data.to_csv(kenpom_file, index=False)
    print(f"\n✓ KenPom data saved to: {kenpom_file}")

    # Save Haslametrics
    if hasla_data is not None:
        hasla_file = "haslametrics_ratings_2026.csv"
        hasla_data.to_csv(hasla_file, index=False)
        print(f"✓ Haslametrics data saved to: {hasla_file}")

    # Save Torvik (if available)
    if torvik_data is not None:
        torvik_file = "torvik_ratings_2026.csv"
        torvik_data.to_csv(torvik_file, index=False)
        print(f"✓ Torvik data saved to: {torvik_file}")

    # =============================================================================
    # SUMMARY
    # =============================================================================

    print("\n" + "="*80)
    print("DATA COLLECTION SUMMARY:")
    print("="*80)
    print(f"KenPom Teams: {len(kenpom_data)}")
    if hasla_data is not None:
        print(f"Haslametrics Teams: {len(hasla_data)}")
    if torvik_data is not None:
        print(f"Torvik Teams: {len(torvik_data)}")
    print("\nData sources fetched successfully!")

    return kenpom_data, hasla_data, torvik_data


if __name__ == "__main__":
    # Run the script
    data = main()
