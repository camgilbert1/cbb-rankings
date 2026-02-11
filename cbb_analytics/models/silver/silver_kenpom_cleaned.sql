/*
    Silver Layer - Cleaned KenPom Data

    Transformations:
    - Standardize column names (lowercase, underscores)
    - Add data quality metadata
    - Keep only relevant columns for analytics
*/

{{ config(
    materialized='table',
    tags=['silver', 'kenpom']
) }}

SELECT
    -- Team Info
    Team as team_name,
    Conference as conference,

    -- Overall Metrics (calculate efficiency margin)
    (Off__Efficiency_Adj - Def__Efficiency_Adj) as adj_efficiency_margin,
    Off__Efficiency_Adj as adj_offensive_efficiency,
    Def__Efficiency_Adj as adj_defensive_efficiency,

    -- Tempo
    Tempo_Adj as adj_tempo,

    -- Rankings (calculate overall rank from efficiency margin)
    ROW_NUMBER() OVER (ORDER BY (Off__Efficiency_Adj - Def__Efficiency_Adj) DESC) as overall_rank,

    -- Offensive Rankings
    Off__Efficiency_Adj_Rank as offensive_rank,

    -- Defensive Rankings
    Def__Efficiency_Adj_Rank as defensive_rank,

    -- Metadata
    current_timestamp() as loaded_at

FROM {{ ref('bronze_kenpom_raw') }}
WHERE Team IS NOT NULL
