-- Most frequently occurring diagnoses in the cohort, with human-readable titles.
-- Param: ? = number of rows to return (LIMIT)
SELECT
    d.icd9_code,
    di.short_title,
    COUNT(*) AS n_occurrences
FROM diagnoses_icd d
LEFT JOIN d_icd_diagnoses di ON d.icd9_code = di.icd9_code
GROUP BY d.icd9_code, di.short_title
ORDER BY n_occurrences DESC
LIMIT ?
