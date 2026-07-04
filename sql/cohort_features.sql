-- One row per hospital admission, joined entirely in SQL: admissions and
-- patients joined directly; ICU stays, diagnoses, and labs first aggregated
-- to admission grain in CTEs (to avoid fan-out from their one-to-many
-- relationship to hadm_id), then LEFT JOINed onto the base in this query.
--
-- lab_agg is deliberately windowed to the first 24h after admittime, not the
-- whole stay: labs are drawn throughout an admission, so an unwindowed count
-- mixes in information from days the model wouldn't have at prediction time.
-- icu_agg and diag_agg are still whole-encounter (ICU stay count/duration and
-- diagnosis count are only fully known at discharge) — kept here for
-- descriptive analysis, but the model in mortality_model.py deliberately
-- excludes them from its predictive feature set for the same reason.
WITH icu_agg AS (
    SELECT
        hadm_id,
        COUNT(*) AS n_icu_stays,
        SUM(los) AS total_icu_los_days,
        first_careunit AS first_icu_careunit
    FROM icustays
    GROUP BY hadm_id
),
diag_agg AS (
    SELECT
        hadm_id,
        COUNT(DISTINCT icd9_code) AS n_diagnoses
    FROM diagnoses_icd
    GROUP BY hadm_id
),
lab_agg AS (
    SELECT
        l.hadm_id,
        COUNT(*) AS n_lab_results_24h,
        SUM(CASE WHEN l.flag = 'abnormal' THEN 1 ELSE 0 END) AS n_abnormal_labs_24h
    FROM labevents l
    JOIN admissions adm ON l.hadm_id = adm.hadm_id
    WHERE l.hadm_id IS NOT NULL
      AND julianday(l.charttime) <= julianday(adm.admittime) + 1.0
    GROUP BY l.hadm_id
)
SELECT
    a.subject_id,
    a.hadm_id,
    a.admission_type,
    a.admission_location,
    a.insurance,
    a.ethnicity,
    a.hospital_expire_flag,
    p.gender,
    julianday(a.admittime) - julianday(p.dob) AS age_days,
    julianday(a.dischtime) - julianday(a.admittime) AS hospital_los_days,
    icu.n_icu_stays,
    icu.total_icu_los_days,
    icu.first_icu_careunit,
    diag.n_diagnoses,
    lab.n_lab_results_24h,
    lab.n_abnormal_labs_24h
FROM admissions a
JOIN patients p ON a.subject_id = p.subject_id
LEFT JOIN icu_agg icu ON a.hadm_id = icu.hadm_id
LEFT JOIN diag_agg diag ON a.hadm_id = diag.hadm_id
LEFT JOIN lab_agg lab ON a.hadm_id = lab.hadm_id
