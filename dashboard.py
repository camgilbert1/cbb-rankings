"""
College Basketball Rankings Dashboard
Built with Streamlit - Powered by KenPom Data
Updated: 2026-02-06 - Fixed timezone for cloud deployment
"""

import streamlit as st
import pandas as pd
from databricks import sql
import os
from datetime import timezone
import pytz

# Page config
st.set_page_config(
    page_title="CBB Rankings",
    page_icon="🏀",
    layout="wide"
)

# Title
st.title("🏀 College Basketball Rankings Dashboard")
st.markdown("*Powered by KenPom efficiency metrics*")

# Database connection
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_data():
    """Load rankings from Databricks Gold table"""
    try:
        # Get credentials from Streamlit secrets or environment variables
        connection = sql.connect(
            server_hostname=st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST")),
            http_path=st.secrets.get("DATABRICKS_HTTP_PATH", os.getenv("DATABRICKS_HTTP_PATH")),
            access_token=st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN"))
        )

        cursor = connection.cursor()
        cursor.execute("""
            SELECT
                overall_rank,
                team_name,
                conference,
                adj_efficiency_margin,
                adj_offensive_efficiency,
                adj_defensive_efficiency,
                offensive_rank,
                defensive_rank,
                adj_tempo,
                loaded_at
            FROM workspace.default_gold.gold_team_rankings
            ORDER BY overall_rank
        """)

        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        connection.close()

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_predictions():
    """Load today's game predictions from Databricks with actual results"""
    try:
        connection = sql.connect(
            server_hostname=st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST")),
            http_path=st.secrets.get("DATABRICKS_HTTP_PATH", os.getenv("DATABRICKS_HTTP_PATH")),
            access_token=st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN"))
        )

        cursor = connection.cursor()
        # Get today's date in Eastern Time (not UTC)
        eastern = pytz.timezone('US/Eastern')
        today = pd.Timestamp.now(tz=eastern).strftime('%Y-%m-%d')

        cursor.execute(f"""
            SELECT
                p.game_time,
                p.home_team,
                p.away_team,
                p.predicted_winner,
                p.win_probability,
                p.predicted_home_score,
                p.predicted_away_score,
                p.confidence,
                p.vegas_spread,
                p.cover_pick,
                p.cover_confidence,
                r.home_score as actual_home_score,
                r.away_score as actual_away_score,
                CASE
                    WHEN r.home_score IS NOT NULL THEN 'Final'
                    ELSE 'Scheduled'
                END as game_status
            FROM workspace.default.prediction_history p
            LEFT JOIN workspace.default.game_results r
                ON p.home_team = r.home_team
                AND p.away_team = r.away_team
                AND ABS(DATEDIFF(p.game_date, r.game_date)) <= 1
            WHERE p.game_date = '{today}'
            ORDER BY p.game_time
        """)

        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        connection.close()

        return df
    except Exception as e:
        # Return empty dataframe if no predictions exist yet
        return pd.DataFrame()


@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_performance(time_period):
    """Load prediction performance metrics from Databricks

    Args:
        time_period (str): "Today", "Yesterday", "Last 7 Days", "Last 30 Days", or "All Time"

    Returns:
        dict: Performance metrics by confidence level
    """
    try:
        connection = sql.connect(
            server_hostname=st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST")),
            http_path=st.secrets.get("DATABRICKS_HTTP_PATH", os.getenv("DATABRICKS_HTTP_PATH")),
            access_token=st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN"))
        )

        cursor = connection.cursor()

        # Calculate date filter based on time period (use Eastern Time)
        eastern = pytz.timezone('US/Eastern')
        today = pd.Timestamp.now(tz=eastern)
        if time_period == "Today":
            date_filter = f"AND p.game_date = '{today.strftime('%Y-%m-%d')}'"
        elif time_period == "Yesterday":
            date_filter = f"AND p.game_date = '{(today - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}'"
        elif time_period == "Last 7 Days":
            date_filter = f"AND p.game_date >= '{(today - pd.Timedelta(days=7)).strftime('%Y-%m-%d')}'"
        elif time_period == "Last 30 Days":
            date_filter = f"AND p.game_date >= '{(today - pd.Timedelta(days=30)).strftime('%Y-%m-%d')}'"
        else:  # All Time
            date_filter = ""

        # Query to join predictions with results and calculate performance
        # Join on team matchup within ±1 day to handle ESPN API date inconsistencies
        cursor.execute(f"""
            WITH matched_games AS (
                SELECT
                    p.game_date,
                    p.home_team,
                    p.away_team,
                    p.predicted_winner,
                    p.predicted_home_score,
                    p.predicted_away_score,
                    p.confidence,
                    p.vegas_spread,
                    p.cover_pick,
                    r.home_score AS actual_home_score,
                    r.away_score AS actual_away_score,
                    CASE
                        WHEN r.home_score > r.away_score THEN p.home_team
                        ELSE p.away_team
                    END AS actual_winner,
                    (r.home_score - r.away_score) AS actual_margin
                FROM workspace.default.prediction_history p
                INNER JOIN workspace.default.game_results r
                    ON p.home_team = r.home_team
                    AND p.away_team = r.away_team
                    AND ABS(DATEDIFF(p.game_date, r.game_date)) <= 1
                WHERE 1=1 {date_filter}
            )
            SELECT
                confidence,
                COUNT(*) as total_games,
                SUM(CASE WHEN predicted_winner = actual_winner THEN 1 ELSE 0 END) as su_wins,
                SUM(CASE WHEN predicted_winner != actual_winner THEN 1 ELSE 0 END) as su_losses,
                SUM(CASE
                    -- We picked home (adjusted > 0) and home covered (actual adjusted > 0)
                    WHEN (predicted_home_score - predicted_away_score + vegas_spread) > 0
                         AND (actual_margin + vegas_spread) > 0 THEN 1
                    -- We picked away (adjusted <= 0) and away covered (actual adjusted < 0)
                    WHEN (predicted_home_score - predicted_away_score + vegas_spread) <= 0
                         AND (actual_margin + vegas_spread) < 0 THEN 1
                    ELSE 0
                END) as ats_wins,
                SUM(CASE
                    -- We picked home (adjusted > 0) but away covered (actual adjusted < 0)
                    WHEN (predicted_home_score - predicted_away_score + vegas_spread) > 0
                         AND (actual_margin + vegas_spread) < 0 THEN 1
                    -- We picked away (adjusted <= 0) but home covered (actual adjusted > 0)
                    WHEN (predicted_home_score - predicted_away_score + vegas_spread) <= 0
                         AND (actual_margin + vegas_spread) > 0 THEN 1
                    ELSE 0
                END) as ats_losses
            FROM matched_games
            WHERE confidence IN ('High', 'Medium', 'Low')
            GROUP BY confidence
        """)

        results = cursor.fetchall()
        cursor.close()
        connection.close()

        # Convert to dictionary
        performance = {}
        for row in results:
            confidence = row[0]
            performance[confidence] = {
                'total': row[1],
                'su_wins': row[2],
                'su_losses': row[3],
                'ats_wins': row[4],
                'ats_losses': row[5]
            }

        return performance

    except Exception as e:
        # Return empty dict if no data
        return {}


# Load data
with st.spinner("Loading rankings..."):
    df = load_data()
    predictions_df = load_predictions()

if df is not None:
    # Show last updated time
    if 'loaded_at' in df.columns and len(df) > 0:
        last_updated_utc = df['loaded_at'].iloc[0]
        # Convert from UTC to Eastern Time
        eastern = pytz.timezone('US/Eastern')
        if last_updated_utc.tzinfo is None:
            last_updated_utc = pytz.utc.localize(last_updated_utc)
        last_updated_et = last_updated_utc.astimezone(eastern)
        st.caption(f"📅 Last updated: {last_updated_et.strftime('%B %d, %Y at %I:%M %p ET')}")

    # Sidebar filters
    st.sidebar.header("⚙️ Settings")

    # Dark mode toggle
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

    if dark_mode:
        st.markdown("""
            <style>
                .stApp {
                    background-color: #0e1117;
                    color: #fafafa;
                }
                .stMarkdown, .stText {
                    color: #fafafa;
                }
            </style>
        """, unsafe_allow_html=True)

    st.sidebar.header("🔍 Filters")

    # Conference filter
    conferences = ["All"] + sorted(df['conference'].unique().tolist())
    selected_conference = st.sidebar.selectbox("Conference", conferences)

    # Rank range filter
    min_rank, max_rank = st.sidebar.slider(
        "Rank Range",
        min_value=1,
        max_value=len(df),
        value=(1, 50)
    )

    # Search by team name
    search_term = st.sidebar.text_input("Search Team", "")

    # Apply filters
    filtered_df = df.copy()

    if selected_conference != "All":
        filtered_df = filtered_df[filtered_df['conference'] == selected_conference]

    filtered_df = filtered_df[
        (filtered_df['overall_rank'] >= min_rank) &
        (filtered_df['overall_rank'] <= max_rank)
    ]

    if search_term:
        filtered_df = filtered_df[
            filtered_df['team_name'].str.contains(search_term, case=False, na=False)
        ]

    # Stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Teams", len(df))
    with col2:
        st.metric("#1 Team", df.iloc[0]['team_name'])
    with col3:
        st.metric("Conferences", df['conference'].nunique())
    with col4:
        st.metric("Showing", len(filtered_df))

    st.markdown("---")

    # Game Predictions Section
    if not predictions_df.empty:
        st.subheader(f"🎯 Today's Game Predictions ({len(predictions_df)} games)")

        # Create compact display dataframe
        display_predictions = predictions_df.copy()

        # Format matchup column
        display_predictions['Matchup'] = display_predictions.apply(
            lambda row: f"{row['away_team']} @ {row['home_team']}", axis=1
        )

        # Format game status
        display_predictions['Status'] = display_predictions['game_status']

        # Format predicted score
        display_predictions['Pred Score'] = display_predictions.apply(
            lambda row: f"{row['predicted_away_score']:.0f} - {row['predicted_home_score']:.0f}", axis=1
        )

        # Format actual score (if game is final)
        display_predictions['Actual Score'] = display_predictions.apply(
            lambda row: (
                f"{int(row['actual_away_score'])} - {int(row['actual_home_score'])}"
                if pd.notna(row['actual_home_score']) and pd.notna(row['actual_away_score'])
                else "-"
            ),
            axis=1
        )

        # Format winner with emoji
        display_predictions['Prediction'] = display_predictions.apply(
            lambda row: f"🏆 {row['predicted_winner']}", axis=1
        )

        # Calculate and show actual winner (if game is final)
        def get_actual_winner(row):
            if pd.notna(row['actual_home_score']) and pd.notna(row['actual_away_score']):
                if row['actual_home_score'] > row['actual_away_score']:
                    return f"✓ {row['home_team']}"
                else:
                    return f"✓ {row['away_team']}"
            return "-"

        display_predictions['Actual Winner'] = display_predictions.apply(get_actual_winner, axis=1)

        # Format probability
        display_predictions['Win %'] = display_predictions['win_probability'].apply(
            lambda x: f"{x:.0%}"
        )

        # Format Vegas spread - show favored team with full name
        display_predictions['Spread'] = display_predictions.apply(
            lambda row: (
                f"{row['home_team']} {row['vegas_spread']:.1f}" if row['vegas_spread'] < 0
                else f"{row['away_team']} -{row['vegas_spread']:.1f}" if row['vegas_spread'] > 0
                else "N/A"
            ),
            axis=1
        )

        # Add confidence emoji
        confidence_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴", "N/A": "⚪"}
        display_predictions['Win Conf.'] = display_predictions['confidence'].apply(
            lambda x: f"{confidence_emoji.get(x, '⚪')} {x}"
        )

        # Add ATS pick with confidence
        display_predictions['ATS Pick'] = display_predictions['cover_pick']

        display_predictions['ATS Conf.'] = display_predictions['cover_confidence'].apply(
            lambda x: f"{confidence_emoji.get(x, '⚪')} {x}"
        )

        # Calculate actual ATS result (if game is final)
        def get_ats_result(row):
            # Check if game is finished
            if pd.notna(row['actual_home_score']) and pd.notna(row['actual_away_score']):
                # Check if spread was available
                if row['vegas_spread'] == 0:
                    return "No spread"

                actual_margin = row['actual_home_score'] - row['actual_away_score']
                actual_adjusted = actual_margin + row['vegas_spread']

                if actual_adjusted > 0:
                    # Home team covered
                    return f"✓ {row['home_team']} covered"
                elif actual_adjusted < 0:
                    # Away team covered
                    return f"✓ {row['away_team']} covered"
                else:
                    return "Push"
            return "-"

        display_predictions['Actual ATS'] = display_predictions.apply(get_ats_result, axis=1)

        # Select and display columns
        compact_df = display_predictions[[
            'Matchup', 'Status', 'Pred Score', 'Actual Score',
            'Prediction', 'Actual Winner', 'Win %', 'Win Conf.',
            'Spread', 'ATS Pick', 'ATS Conf.', 'Actual ATS'
        ]]

        st.dataframe(
            compact_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, len(compact_df) * 35 + 38)  # Dynamic height based on number of games
        )

    # Main rankings table
    st.subheader("📊 Team Rankings")

    # Format the dataframe for display
    display_df = filtered_df.copy()
    # Drop loaded_at column before display
    if 'loaded_at' in display_df.columns:
        display_df = display_df.drop('loaded_at', axis=1)

    display_df.columns = [
        "Rank", "Team", "Conference", "Eff. Margin",
        "Off. Eff.", "Def. Eff.", "Off. Rank", "Def. Rank", "Tempo"
    ]

    # Round numeric columns
    numeric_cols = ["Eff. Margin", "Off. Eff.", "Def. Eff.", "Tempo"]
    for col in numeric_cols:
        display_df[col] = display_df[col].round(1)

    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600
    )

    # Download button
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Rankings (CSV)",
        data=csv,
        file_name="cbb_rankings.csv",
        mime="text/csv"
    )

    # Prediction Performance Tracker
    st.markdown("---")
    st.subheader("📈 Prediction Performance")

    # Time period selector
    time_period = st.selectbox(
        "Time Period",
        ["Today", "Yesterday", "Last 7 Days", "Last 30 Days", "All Time"],
        index=4  # Default to "All Time"
    )

    # Load performance data
    performance = load_performance(time_period)

    if performance:
        # Display performance by confidence level
        col1, col2, col3 = st.columns(3)

        # High Confidence
        with col1:
            if 'High' in performance:
                p = performance['High']
                su_pct = (p['su_wins'] / p['total'] * 100) if p['total'] > 0 else 0
                ats_pct = (p['ats_wins'] / p['total'] * 100) if p['total'] > 0 else 0
                st.metric(
                    "🟢 High Confidence",
                    f"{p['total']} games",
                    delta=None
                )
                st.write(f"**SU:** {p['su_wins']}-{p['su_losses']} ({su_pct:.0f}%)")
                st.write(f"**ATS:** {p['ats_wins']}-{p['ats_losses']} ({ats_pct:.0f}%)")
            else:
                st.metric("🟢 High Confidence", "0-0")
                st.write("**SU:** 0-0 (0%)")
                st.write("**ATS:** 0-0 (0%)")

        # Medium Confidence
        with col2:
            if 'Medium' in performance:
                p = performance['Medium']
                su_pct = (p['su_wins'] / p['total'] * 100) if p['total'] > 0 else 0
                ats_pct = (p['ats_wins'] / p['total'] * 100) if p['total'] > 0 else 0
                st.metric(
                    "🟡 Medium Confidence",
                    f"{p['total']} games",
                    delta=None
                )
                st.write(f"**SU:** {p['su_wins']}-{p['su_losses']} ({su_pct:.0f}%)")
                st.write(f"**ATS:** {p['ats_wins']}-{p['ats_losses']} ({ats_pct:.0f}%)")
            else:
                st.metric("🟡 Medium Confidence", "0-0")
                st.write("**SU:** 0-0 (0%)")
                st.write("**ATS:** 0-0 (0%)")

        # Low Confidence
        with col3:
            if 'Low' in performance:
                p = performance['Low']
                su_pct = (p['su_wins'] / p['total'] * 100) if p['total'] > 0 else 0
                ats_pct = (p['ats_wins'] / p['total'] * 100) if p['total'] > 0 else 0
                st.metric(
                    "🔴 Low Confidence",
                    f"{p['total']} games",
                    delta=None
                )
                st.write(f"**SU:** {p['su_wins']}-{p['su_losses']} ({su_pct:.0f}%)")
                st.write(f"**ATS:** {p['ats_wins']}-{p['ats_losses']} ({ats_pct:.0f}%)")
            else:
                st.metric("🔴 Low Confidence", "0-0")
                st.write("**SU:** 0-0 (0%)")
                st.write("**ATS:** 0-0 (0%)")

        # Overall stats
        if performance:
            total_games = sum(p['total'] for p in performance.values())
            total_su_wins = sum(p['su_wins'] for p in performance.values())
            total_su_losses = sum(p['su_losses'] for p in performance.values())
            total_ats_wins = sum(p['ats_wins'] for p in performance.values())
            total_ats_losses = sum(p['ats_losses'] for p in performance.values())

            if total_games > 0:
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    su_pct = (total_su_wins / total_games * 100) if total_games > 0 else 0
                    st.metric("Overall Straight Up", f"{total_su_wins}-{total_su_losses} ({su_pct:.1f}%)")
                with col2:
                    ats_pct = (total_ats_wins / total_games * 100) if total_games > 0 else 0
                    st.metric("Overall Against The Spread", f"{total_ats_wins}-{total_ats_losses} ({ats_pct:.1f}%)")
    else:
        st.info("No completed games with predictions yet. Performance tracking will begin after games are played!")

    # Footer
    st.markdown("---")
    st.markdown("*Data updated daily from KenPom • Built with Streamlit*")

else:
    st.error("Unable to load data. Please check your Databricks connection.")
