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
    (`Off. Efficiency-Adj` - `Def. Efficiency-Adj`) as adj_efficiency_margin,
    `Off. Efficiency-Adj` as adj_offensive_efficiency,
    `Def. Efficiency-Adj` as adj_defensive_efficiency,

    -- Tempo
    `Tempo-Adj` as adj_tempo,

    -- Rankings (use offensive rank as proxy for overall rank since we don't have Rk)
    ROW_NUMBER() OVER (ORDER BY (`Off. Efficiency-Adj` - `Def. Efficiency-Adj`) DESC) as overall_rank,

    -- Offensive Rankings
    `Off. Efficiency-Adj.Rank` as offensive_rank,

    -- Defensive Rankings
    `Def. Efficiency-Adj.Rank` as defensive_rank,

    -- Metadata
    current_timestamp() as loaded_at

FROM {{ ref('bronze_kenpom_raw') }}
WHERE Team IS NOT NULL
