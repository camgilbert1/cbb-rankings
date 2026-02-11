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

# Load data
with st.spinner("Loading rankings..."):
    df = load_data()

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
    st.sidebar.header("Filters")

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

    # Main rankings table
    st.subheader("Rankings")

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
