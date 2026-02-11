/*
    Gold Layer - Team Rankings Summary

    Analytics-ready table combining all rankings into a single view.
    This is what you'd query for dashboards, reports, or predictions.
*/

{{ config(
    materialized='table',
    tags=['gold', 'rankings']
) }}

SELECT
    team_name,
    conference,

    -- Overall Metrics
    overall_rank,
    adj_efficiency_margin,

    -- Offense
    offensive_rank,
    adj_offensive_efficiency,

    -- Defense
    defensive_rank,
    adj_defensive_efficiency,

    -- Tempo
    adj_tempo,

    -- Metadata
    loaded_at

FROM {{ ref('silver_kenpom_cleaned') }}

ORDER BY overall_rank
