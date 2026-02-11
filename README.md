# College Basketball Rankings Dashboard

Live college basketball rankings powered by KenPom efficiency metrics.

## Features
- 🏀 Real-time rankings from KenPom data
- 🔍 Filter by conference and rank range
- 📊 Interactive table with search
- 📥 Download rankings as CSV
- ☁️ Hosted on Streamlit Cloud (free)

## Tech Stack
- **Data Ingestion**: Python (kenpompy)
- **Storage**: Databricks
- **Transformation**: dbt (Medallion Architecture)
- **Visualization**: Streamlit

## Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run dashboard.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repo and `dashboard.py`
5. Add secrets in Advanced Settings:
   - Copy contents from `.streamlit/secrets.toml`
6. Deploy!

Your dashboard will be live at: `https://your-app-name.streamlit.app`

## Data Pipeline

```
Python (APIPull.py) → Databricks → dbt → Streamlit
```

1. **Bronze**: Raw KenPom data
2. **Silver**: Cleaned & standardized
3. **Gold**: Analytics-ready rankings
4. **Dashboard**: Interactive web app

## License
Personal project for learning data engineering
