-- In-hospital mortality rate, grouped by first ICU care unit.
SELECT
    i.first_careunit,
    COUNT(DISTINCT a.hadm_id) AS n_admissions,
    SUM(a.hospital_expire_flag) AS n_deaths,
    ROUND(100.0 * SUM(a.hospital_expire_flag) / COUNT(DISTINCT a.hadm_id), 1) AS mortality_rate_pct
FROM admissions a
JOIN icustays i ON a.hadm_id = i.hadm_id
GROUP BY i.first_careunit
ORDER BY mortality_rate_pct DESC
