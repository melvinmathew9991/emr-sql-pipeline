-- For each admission, computes whether the patient was readmitted within 30
-- days of discharge, using LEAD to look at the patient's *next* admission —
-- the opposite direction from readmission_intervals.sql's LAG-based
-- days_since_last_discharge. This is a genuinely forward-looking label:
-- it's only knowable in hindsight, once the next admission (or lack of one)
-- has already happened, which is why it's a training target, not a feature.
--
-- No next admission exists for two different reasons that must not be
-- conflated, so both are left NULL here rather than labeled a confirmed 0:
--   - the patient died this admission (hospital_expire_flag = 1) — readmission
--     was never possible, not merely unobserved
--   - the patient simply has no later admission in this dataset — right-censored,
--     unknown whether they were readmitted after the data's cutoff
-- readmission_model.py drops the death case from the modeling cohort entirely
-- and drops the censored case from the training target.
SELECT
    subject_id,
    hadm_id,
    hospital_expire_flag,
    dischtime,
    julianday(
        LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime)
    ) - julianday(dischtime) AS days_to_next_admission,
    CASE
        WHEN LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) IS NULL THEN NULL
        WHEN julianday(
            LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime)
        ) - julianday(dischtime) <= 30 THEN 1
        ELSE 0
    END AS readmit_30d
FROM admissions
ORDER BY subject_id, admittime
