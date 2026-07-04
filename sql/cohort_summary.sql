-- Basic cohort characterization: patient count, admission count, gender split.
SELECT
    COUNT(DISTINCT p.subject_id) AS n_patients,
    COUNT(DISTINCT a.hadm_id) AS n_admissions,
    SUM(CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END) AS n_male,
    SUM(CASE WHEN p.gender = 'F' THEN 1 ELSE 0 END) AS n_female
FROM patients p
LEFT JOIN admissions a ON p.subject_id = a.subject_id
