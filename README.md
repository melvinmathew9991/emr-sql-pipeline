# Clinical EMR Data Pipeline & In-Hospital Mortality Prediction (MIMIC-III)

[![Tests](https://github.com/melvinmathew9991/emr-sql-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/melvinmathew9991/emr-sql-pipeline/actions/workflows/tests.yml)

A SQL-first analytics pipeline on real electronic medical record (EMR) data,
covering descriptive analytics (cohort characterization, length-of-stay,
diagnosis patterns) and predictive analytics (in-hospital mortality prediction)
using real de-identified ICU patient records.

## Dataset

**Source:** [MIMIC-III Clinical Database Demo v1.4](https://physionet.org/content/mimiciii-demo/1.4/)
— Medical Information Mart for Intensive Care, MIT Lab for Computational Physiology.

**Access:** Fully open — no credentialing or registration required for the demo
(unlike the full MIMIC-III database, which requires a data use agreement).
Direct download: https://physionet.org/content/mimiciii-demo/get-zip/1.4/ (13.4 MB)

**What it is:** Real de-identified electronic health records for 100 ICU patients
at Beth Israel Deaconess Medical Center — a genuine 26-table relational EMR
schema (admissions, ICU stays, diagnoses, lab results, prescriptions, procedures),
not a synthetic or toy dataset.

**Tables used in this project:**
- `PATIENTS` — demographics, date of birth/death
- `ADMISSIONS` — hospital admission records, admission/discharge times, in-hospital death flag
- `ICUSTAYS` — ICU stay records, length of stay, care unit
- `DIAGNOSES_ICD` + `D_ICD_DIAGNOSES` — diagnosis codes and descriptions per admission
- `LABEVENTS` — laboratory test results

## To get the real data

1. No account needed — go to https://physionet.org/content/mimiciii-demo/1.4/
2. Click "Download the ZIP file"
3. Unzip and place all CSVs into `data/mimic_demo/`

## What this project does

1. **`build_database.py`** — loads the raw EMR CSVs into a local SQLite relational
   database, exactly as a real clinical data warehouse would be structured.

2. **`queries.py`** — loads and runs a library of SQL queries from `sql/` answering
   real descriptive-analytics questions directly against the relational schema:
   - Patient demographics and admission volume
   - Length-of-stay distribution by care unit
   - Most common diagnoses in the cohort
   - In-hospital mortality rate by admission type and care unit
   - Per-patient admission sequencing and readmission intervals (window
     functions — see "SQL" below), available via `readmission_intervals()`

3. **`cohort_features.py`** — builds a patient-level, ML-ready feature table by
   joining across admissions, ICU stays, diagnoses, and labs entirely in SQL,
   then finishing feature engineering in pandas — the core "EMR to
   analytics-ready dataset" skill.

4. **`mortality_model.py`** — trains a classification model to predict
   in-hospital mortality from admission-time clinical features only (see
   "Avoiding leakage" below), evaluated with a patient-grouped, stratified
   cross-validation scheme against a logistic-regression baseline, and saves
   the fitted model to `outputs/mortality_model.joblib` — a real predictive
   analytics task on real clinical outcome data.

5. **`predict.py`** — loads that saved model and scores a new admission's
   in-hospital mortality risk from its admission-time features, so the
   pipeline can actually be used for inference, not just report metrics.
   Run with `python predict.py` after `main.py` has trained a model.

## Avoiding leakage

Two methodological issues are easy to miss in EMR modeling and are handled
deliberately here:

- **Admission-time-only features.** `hospital_los_days`, `total_icu_los_days`,
  and `n_diagnoses` are computed in `cohort_features.sql` for descriptive use,
  but the model excludes them: length of stay is only known once a stay ends
  (for a patient who dies, discharge time *is* death time), and MIMIC's ICD-9
  codes are assigned at discharge for the whole encounter. Using them to
  predict an admission-time outcome would leak information the model wouldn't
  actually have yet. Lab features are windowed to the first 24h after
  admission for the same reason, rather than aggregated over the whole stay.
- **Patient-grouped splits.** 14 patients in this cohort have more than one
  admission (one has 15). Model evaluation uses `StratifiedGroupKFold`,
  grouped by `subject_id`, so no patient's admissions appear in both the
  training and evaluation data.

## SQL

Every query the pipeline runs lives as a standalone `.sql` file in `sql/`,
loaded and executed by the Python modules above rather than embedded as
inline strings:

- `cohort_summary.sql`, `length_of_stay_by_careunit.sql`, `top_diagnoses.sql`,
  `mortality_by_admission_type.sql`, `mortality_by_careunit.sql` — the 5
  descriptive-analytics queries
- `cohort_features.sql` — the feature-table query; aggregates ICU stays,
  diagnoses, and labs (labs windowed to the first 24h post-admission — see
  "Avoiding leakage" below) to one row per admission in CTEs, then `LEFT
  JOIN`s all three onto `admissions ⋈ patients` in a single query. Every
  table join in this project happens in SQL — pandas is only used afterward
  for feature engineering (age capping, percentage calculations, filling
  nulls left by the joins), never to combine tables.
- `readmission_intervals.sql` — the pipeline's window-function query.
  `ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime)` numbers
  each patient's admissions in sequence, and `LAG(dischtime) OVER (PARTITION
  BY subject_id ORDER BY admittime)` recovers their previous discharge time
  so `days_since_last_discharge` can be computed per admission. A patient's
  first admission gets `NULL` (no prior discharge), and because `LAG` only
  looks at that same patient's strictly earlier rows, the value can't leak
  information from a later encounter — verified against the real data: the
  cohort's most-readmitted patient (15 admissions) sequences 1–15 correctly,
  and 11 of 29 admissions with a prior discharge fall within 30 days.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Results (SQL query outputs, model metrics, plots) are written to `outputs/`.

## Results

From the most recent run against the real MIMIC-III Demo v1.4 data (not the
synthetic test fixtures):

**Cohort:** 100 patients, 129 admissions, 136 ICU stays, 76,074 lab results.
In-hospital mortality: 31.0% (40 deaths / 89 survived).

**Descriptive analytics:**
- Mortality by admission type: URGENT 50.0%, EMERGENCY 32.8%, ELECTIVE 0.0%
- Mortality by care unit: TSICU 63.6%, MICU 34.2%, SICU 31.8%, CCU 31.6%, CSRU 16.7%
- Top diagnoses: hypertension NOS, atrial fibrillation, acute kidney failure NOS,
  CHF NOS, and type II diabetes led the cohort's ICD-9 codes

Both patterns are directionally sensible — sicker admission types and
higher-acuity ICU units show higher mortality — but a chi-square test on each
grouping (printed by `queries.py`) is not significant at this sample size
(admission type: p=0.128; care unit: p=0.282), and several groups have fewer
than 5 expected deaths, the threshold below which the test isn't considered
reliable. Read these as descriptive patterns worth investigating further, not
confirmed findings.

**Predictive model:** class-weighted Random Forest, tuned via `GridSearchCV`
and evaluated with 5-fold `StratifiedGroupKFold` (grouped by patient, so no
one's admissions split across train/test):
- Random Forest ROC-AUC: **0.606 ± 0.107** (cross-validated)
- Logistic regression baseline: 0.510 ± 0.154 (cross-validated)
- Held-out fold: accuracy 0.48, recall on the death class 0.38, ROC-AUC 0.441

An earlier version of this pipeline reported ROC-AUC 0.770 from a single
75/25 split using whole-encounter features (length of stay, total diagnosis
count) that aren't actually known at admission time. Once those leakage
sources were removed and evaluation switched to a patient-grouped,
cross-validated scheme, the honest number dropped to a modest 0.606 — barely
above the logistic regression baseline. That's the correct and expected
outcome of fixing leakage, not a regression: on a 100-patient sample, a model
this close to its own linear baseline is a realistic result, and a real
demonstration of catching and correcting the leakage is a stronger portfolio
signal than the higher, invalid number. The full MIMIC-III database (thousands
of admissions) would be needed to know whether the Random Forest's edge over
the baseline is real or noise.

## Tests & CI

`tests/` has a small pytest suite (19 tests) run automatically on every push
via GitHub Actions (`.github/workflows/tests.yml`) — the badge at the top of
this README reflects the latest run. Tests run entirely against the
synthetic fixture in `data/synthetic_test/` (the real MIMIC data is
gitignored, so CI never needs it), covering all four pipeline stages, the
leakage guard (`hospital_los_days` etc. can never silently reappear in the
model's feature set), a regression test for a real bug this project hit:
`StratifiedGroupKFold` and the confusion matrix plot both broke on the
synthetic fixture's tiny minority class (2 deaths) until the code was made to
adapt `n_splits` accordingly — and three tests for the `readmission_intervals`
window-function query (first-admission nulls, per-patient sequence
numbering, non-negative intervals).

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Testing before you have the real data

`data/synthetic_test/` contains small synthetic CSVs with the identical MIMIC-III
schema (column names/types), used only to verify the pipeline runs correctly.
This is NOT real patient data and must not be used for any actual findings.
Run with `USE_SYNTHETIC = True` in `main.py` to test before your real download
finishes; switch to `False` and point at `data/mimic_demo/` for the real analysis.

## Why SQL-first

Real EMR data lives in normalized relational tables, not flat CSVs — clinically
meaningful features (e.g., "did this patient have a prior admission with a
diabetes diagnosis") require multi-table joins. This project intentionally does
that joining and aggregation in SQL against a real relational schema, rather than
pre-flattening the data, since that's the skill EMR/RWD analytics roles expect.

## Future extensions

- Extend to lab-value trajectories (LABEVENTS is time-series; currently
  aggregated to summary statistics per admission)
- Feed `readmission_intervals.sql`'s `days_since_last_discharge` into
  `mortality_model.py` as a real predictor (the query exists; it isn't wired
  into the model yet) and/or build a standalone 30-day-readmission classifier
- Compare against the full MIMIC-III database (requires PhysioNet credentialing)
