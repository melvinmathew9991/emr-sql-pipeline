-- Average and median ICU length of stay (days), grouped by care unit.
SELECT
    first_careunit,
    COUNT(*) AS n_stays,
    ROUND(AVG(los), 2) AS avg_los_days,
    ROUND(MIN(los), 2) AS min_los_days,
    ROUND(MAX(los), 2) AS max_los_days
FROM icustays
WHERE los IS NOT NULL
GROUP BY first_careunit
ORDER BY avg_los_days DESC
