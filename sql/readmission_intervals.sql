-- For each admission, computes the patient's admission sequence number and
-- the time elapsed since their previous discharge, so readmission timing can
-- be analyzed per patient rather than per admission in isolation.
--
-- days_since_last_discharge is NULL for a patient's first admission (no
-- prior discharge exists) and is derived only from that same patient's own
-- earlier admissions via LAG, so it never leaks information from a later
-- encounter into an earlier one.
SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime) AS admission_seq,
    julianday(admittime) - julianday(
        LAG(dischtime) OVER (PARTITION BY subject_id ORDER BY admittime)
    ) AS days_since_last_discharge
FROM admissions
ORDER BY subject_id, admittime
