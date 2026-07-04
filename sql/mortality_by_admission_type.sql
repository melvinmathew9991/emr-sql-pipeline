-- In-hospital mortality rate, grouped by admission type (EMERGENCY, ELECTIVE, URGENT).
SELECT
    admission_type,
    COUNT(*) AS n_admissions,
    SUM(hospital_expire_flag) AS n_deaths,
    ROUND(100.0 * SUM(hospital_expire_flag) / COUNT(*), 1) AS mortality_rate_pct
FROM admissions
GROUP BY admission_type
ORDER BY mortality_rate_pct DESC
