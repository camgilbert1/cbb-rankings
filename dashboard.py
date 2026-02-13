"""
College Basketball Rankings Dashboard
Built with Streamlit - Powered by KenPom Data
Updated: 2026-02-06 - Added Bracketology tab
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
              AND r.home_score IS NULL
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


@st.cache_data(ttl=3600)  # Cache for 1 hour (bracket updates Tues/Fri only)
def load_bracket_predictions():
    """Load bracket predictions from Databricks"""
    try:
        connection = sql.connect(
            server_hostname=st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST")),
            http_path=st.secrets.get("DATABRICKS_HTTP_PATH", os.getenv("DATABRICKS_HTTP_PATH")),
            access_token=st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN"))
        )

        cursor = connection.cursor()
        cursor.execute("""
            SELECT
                round,
                region,
                top_seed,
                top_team,
                low_seed,
                low_team,
                predicted_winner,
                win_probability,
                predicted_top_score,
                predicted_low_score,
                confidence,
                updated_at
            FROM workspace.default.bracket_predictions
            ORDER BY round, region, top_seed
        """)

        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        connection.close()

        return df
    except Exception as e:
        return pd.DataFrame()


def render_matchup_card(row):
    """Render a styled matchup card as HTML."""
    top_seed = row['top_seed']
    top_team = row['top_team']
    low_seed = row['low_seed']
    low_team = row['low_team']
    winner = row.get('predicted_winner', 'TBD')
    win_prob = row.get('win_probability')
    confidence = row.get('confidence', 'N/A')

    top_winner = (winner == top_team)
    low_winner = (winner == low_team)

    top_bg = "#28a745" if top_winner else "#f8f9fa"
    top_color = "white" if top_winner else "#212529"
    low_bg = "#28a745" if low_winner else "#f8f9fa"
    low_color = "white" if low_winner else "#212529"

    top_score = row.get('predicted_top_score')
    low_score = row.get('predicted_low_score')

    prob_text = f"{win_prob:.0%}" if win_prob is not None else "TBD"
    conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "⚪")

    top_score_str = f"{top_score:.0f}" if top_score is not None else ""
    low_score_str = f"{low_score:.0f}" if low_score is not None else ""

    return f"""
    <div style="border:1px solid #dee2e6; border-radius:8px; padding:8px; margin:4px 0; font-size:13px;">
        <div style="background:{top_bg}; color:{top_color}; padding:4px 8px; border-radius:4px; margin-bottom:2px; display:flex; justify-content:space-between; align-items:center;">
            <span><b>#{top_seed}</b> {top_team}</span>
            <span style="font-weight:700; margin-left:6px;">{top_score_str}</span>
        </div>
        <div style="text-align:center; color:#6c757d; font-size:11px; line-height:1.4;">vs</div>
        <div style="background:{low_bg}; color:{low_color}; padding:4px 8px; border-radius:4px; margin-top:2px; display:flex; justify-content:space-between; align-items:center;">
            <span><b>#{low_seed}</b> {low_team}</span>
            <span style="font-weight:700; margin-left:6px;">{low_score_str}</span>
        </div>
        <div style="font-size:11px; color:#6c757d; margin-top:4px; text-align:right;">
            {conf_emoji} {prob_text}
        </div>
    </div>
    """


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

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Rankings", "🎯 Today's Picks", "🏆 Bracketology", "📈 Performance"])

    # =========================================================================
    # TAB 1: RANKINGS
    # =========================================================================
    with tab1:
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

        st.subheader("📊 Team Rankings")

        # Format the dataframe for display
        display_df = filtered_df.copy()
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

    # =========================================================================
    # TAB 2: TODAY'S PICKS
    # =========================================================================
    with tab2:
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
                if pd.notna(row['actual_home_score']) and pd.notna(row['actual_away_score']):
                    if row['vegas_spread'] == 0:
                        return "No spread"

                    actual_margin = row['actual_home_score'] - row['actual_away_score']
                    actual_adjusted = actual_margin + row['vegas_spread']

                    if actual_adjusted > 0:
                        return f"✓ {row['home_team']} covered"
                    elif actual_adjusted < 0:
                        return f"✓ {row['away_team']} covered"
                    else:
                        return "Push"
                return "-"

            display_predictions['Actual ATS'] = display_predictions.apply(get_ats_result, axis=1)

            # Sort by ATS confidence (High -> Medium -> Low)
            confidence_order = {'High': 0, 'Medium': 1, 'Low': 2}
            display_predictions['conf_sort'] = display_predictions['cover_confidence'].map(confidence_order)
            display_predictions = display_predictions.sort_values('conf_sort')

            # Select and display columns
            compact_df = display_predictions[[
                'Matchup', 'Status', 'Spread', 'Prediction', 'Pred Score', 'Win %', 'Win Conf.',
                'ATS Pick', 'ATS Conf.', 'Actual Score', 'Actual Winner', 'Actual ATS'
            ]]

            st.dataframe(
                compact_df,
                use_container_width=True,
                hide_index=True,
                height=min(400, len(compact_df) * 35 + 38)
            )
        else:
            st.info("No predictions available for today yet. Check back after the daily predictions run.")

    # =========================================================================
    # TAB 3: BRACKETOLOGY
    # =========================================================================
    with tab3:
        st.subheader("🏀 NCAA Tournament Bracket Predictions")
        st.markdown("*Predictions generated using KenPom efficiency metrics. Bracket updated every Tuesday & Friday.*")

        with st.spinner("Loading bracket predictions..."):
            bracket_df = load_bracket_predictions()

        if bracket_df.empty:
            st.info("No bracket predictions available yet. Run BracketPredictions.py to populate.")
        else:
            # Show last updated
            if 'updated_at' in bracket_df.columns and len(bracket_df) > 0:
                last_updated = str(bracket_df['updated_at'].iloc[0])[:10]
                st.caption(f"Bracket last updated: {last_updated}")

            # ---- First Four ----
            first_four = bracket_df[bracket_df['round'] == 'First Four']
            if not first_four.empty:
                st.markdown("### First Four")
                ff_cols = st.columns(4)
                for i, (_, matchup) in enumerate(first_four.iterrows()):
                    with ff_cols[i % 4]:
                        st.markdown(render_matchup_card(matchup), unsafe_allow_html=True)
                        st.caption(matchup['region'])

                st.markdown("---")

            # ---- First Round by Region ----
            first_round = bracket_df[bracket_df['round'] == 'First Round']

            if not first_round.empty:
                st.markdown("### First Round")
                regions = ['East', 'West', 'South', 'Midwest']
                region_cols = st.columns(4)

                for i, region in enumerate(regions):
                    with region_cols[i]:
                        st.markdown(f"**{region}**")
                        region_matchups = first_round[first_round['region'] == region].sort_values('top_seed')
                        for _, matchup in region_matchups.iterrows():
                            st.markdown(render_matchup_card(matchup), unsafe_allow_html=True)

    # =========================================================================
    # TAB 4: PERFORMANCE
    # =========================================================================
    with tab4:
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
                    st.metric("🟢 High Confidence", f"{p['total']} games")
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
                    st.metric("🟡 Medium Confidence", f"{p['total']} games")
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
                    st.metric("🔴 Low Confidence", f"{p['total']} games")
                    st.write(f"**SU:** {p['su_wins']}-{p['su_losses']} ({su_pct:.0f}%)")
                    st.write(f"**ATS:** {p['ats_wins']}-{p['ats_losses']} ({ats_pct:.0f}%)")
                else:
                    st.metric("🔴 Low Confidence", "0-0")
                    st.write("**SU:** 0-0 (0%)")
                    st.write("**ATS:** 0-0 (0%)")

            # Overall stats
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
