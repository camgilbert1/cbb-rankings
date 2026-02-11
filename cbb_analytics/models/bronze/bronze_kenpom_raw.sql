/*
    Bronze Layer - Raw KenPom Data

    This model simply references the raw KenPom data uploaded to Databricks.
    No transformations - just creating a dbt-managed reference to the source.
*/

{{ config(
    materialized='view',
    tags=['bronze', 'kenpom']
) }}

SELECT *
FROM workspace.default.kenpom_ratings_2026
