"""
College Basketball Rankings Dashboard
Built with Streamlit - Powered by KenPom Data
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
    """Load today's game predictions from Databricks"""
    try:
        connection = sql.connect(
            server_hostname=st.secrets.get("DATABRICKS_HOST", os.getenv("DATABRICKS_HOST")),
            http_path=st.secrets.get("DATABRICKS_HTTP_PATH", os.getenv("DATABRICKS_HTTP_PATH")),
            access_token=st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN"))
        )

        cursor = connection.cursor()
        cursor.execute("""
            SELECT
                game_time,
                home_team,
                away_team,
                predicted_winner,
                win_probability,
                predicted_home_score,
                predicted_away_score,
                confidence,
                vegas_spread
            FROM workspace.default.game_predictions
            ORDER BY game_time
        """)

        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()
        connection.close()

        return df
    except Exception as e:
        # Return empty dataframe if no predictions exist yet
        return pd.DataFrame()

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

        # Format predicted score
        display_predictions['Score'] = display_predictions.apply(
            lambda row: f"{row['predicted_away_score']:.0f} - {row['predicted_home_score']:.0f}", axis=1
        )

        # Format winner with emoji
        display_predictions['Prediction'] = display_predictions.apply(
            lambda row: f"🏆 {row['predicted_winner']}", axis=1
        )

        # Format probability
        display_predictions['Win %'] = display_predictions['win_probability'].apply(
            lambda x: f"{x:.0%}"
        )

        # Format Vegas spread
        display_predictions['Spread'] = display_predictions.apply(
            lambda row: f"{row['home_team'].split()[-1][:3]} {row['vegas_spread']:+.1f}" if row['vegas_spread'] != 0 else "N/A",
            axis=1
        )

        # Add confidence emoji
        confidence_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        display_predictions['Conf.'] = display_predictions['confidence'].apply(
            lambda x: f"{confidence_emoji.get(x, '⚪')} {x}"
        )

        # Select and display columns
        compact_df = display_predictions[['Matchup', 'Score', 'Spread', 'Prediction', 'Win %', 'Conf.']]

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

    # Footer
    st.markdown("---")
    st.markdown("*Data updated daily from KenPom • Built with Streamlit*")

else:
    st.error("Unable to load data. Please check your Databricks connection.")
