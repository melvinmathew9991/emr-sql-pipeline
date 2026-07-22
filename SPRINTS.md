# Sprint Plan

This project is organized into sprints — scoped units of work, each shipped
through its own feature branch and pull request into `main`, with CI
(`.github/workflows/tests.yml`) required to pass before merge.

Sprints 1–4 were completed and merged before this branch-per-sprint workflow
was formalized, so they're documented here retroactively, mapped to the
phases already recorded in `CHANGELOG.md`. Sprint 5 onward follow the real
workflow: one branch, one PR, one sprint.

Status legend: **Done** (merged to `main`) · **In review** (PR open) ·
**Planned** (not started).

---

## Sprint 1 — Data Foundation
**Status:** Done · corresponds to CHANGELOG Phases 0–3

Goal: get the real MIMIC-III Demo data loading cleanly end to end.

- Verified the pre-existing 4-stage pipeline against `data/synthetic_test/`
- Audited the repo, found the real MIMIC-III data downloaded but sitting in
  the wrong folder (`main.py` never looked there), fixed the mismatch
- First real-data run: 100 patients, 129 admissions, 31.0% mortality

## Sprint 2 — SQL-First Descriptive Layer
**Status:** Done · corresponds to CHANGELOG Phase 4

Goal: make "every join happens in SQL" literally true, not aspirational.

- Rewrote `cohort_features.py` from pandas `.merge()` stitching into a
  single SQL query with CTEs (`icu_agg`, `diag_agg`, `lab_agg`)
- Extracted all 6 queries from inline Python strings into standalone
  `.sql` files under `sql/`

## Sprint 3 — Model Integrity Audit & Fix
**Status:** Done · corresponds to CHANGELOG Phases 5–6

Goal: verify the model's headline result was real, not an artifact.

- Senior-reviewer-style audit found target leakage (whole-stay features
  used as admission-time predictors) and patient-level leakage (repeat
  patients split across train/test)
- Fixed both: admission-time-only features, `StratifiedGroupKFold` grouped
  by `subject_id`, added a logistic-regression baseline, `GridSearchCV`
  tuning, chi-square significance testing on descriptive breakdowns
- ROC-AUC corrected from an invalid 0.770 to an honest 0.606 ± 0.107

## Sprint 4 — Testing, CI, and Inference
**Status:** Done · corresponds to CHANGELOG Phases 7–8

Goal: close the "no tests, no CI, no way to score a new patient" gaps.

- Building the test suite surfaced a real crash bug (`StratifiedGroupKFold`
  on a 2-member minority class); fixed
- Added 16 pytest tests (later 19, see Sprint 5) and GitHub Actions CI
- Added model persistence (`outputs/mortality_model.joblib`) and
  `predict.py` for scoring a new admission from raw fields

## Sprint 5 — Readmission-Interval Window Functions
**Status:** Done · merged via PR #1, corresponds to CHANGELOG Phase 9

Goal: extend the SQL layer past joins/aggregation into window functions.

- Added `sql/readmission_intervals.sql`: `ROW_NUMBER()` sequences each
  patient's admissions, `LAG()` recovers their previous discharge time to
  compute `days_since_last_discharge` without leaking a later encounter
  backward
- Verified against real data: the cohort's most-readmitted patient (15
  admissions) sequences 1–15 correctly; 11 of 29 repeat admissions fall
  within a 30-day interval
- 3 new tests (16 → 19 total)

## SQL query indexing (small, no sprint number)
**Status:** Done

Not a full sprint — a targeted fix folded in ahead of Sprint 6, since Sprint
6's `LEAD(admittime)` readmission target will hit `admissions`/`labevents`
harder and benefits from having this in place first.

- `build_database.py` now indexes every join/filter column `sql/*.sql`
  actually uses (`subject_id`, `hadm_id`, `icd9_code`, a composite
  `labevents(hadm_id, charttime)`) after loading each table
- Verified against the real `cohort_features.sql` query: `EXPLAIN QUERY PLAN`
  shows `icustays` and `diagnoses_icd` switching from a full scan + temp
  B-tree sort (for their `GROUP BY hadm_id`) to an index seek with no sort
  needed; `labevents` switches from a full scan to an index search
- 3 new tests (19 → 22 total): indexes exist, and the two heaviest joins
  (`admissions ⋈ patients`, `labevents ⋈ admissions`) use `USING INDEX`
  rather than a plain nested-loop scan

## Sprint 6 — Readmission Risk Model
**Status:** Done · corresponds to CHANGELOG Phase 10

Goal: turn the readmission-interval feature into a second real predictive
model, not just a queryable column.

- Added `sql/readmission_target.sql`: a genuine 30-day-readmission target
  with `LEAD(admittime)` (a patient's *next* admission, not their last) so
  the label reflects a future event knowable only in hindsight — a
  different leakage direction than Sprint 5's backward-looking `LAG`
- Two "no next admission" cases left `NULL` rather than conflated into a
  confirmed 0: in-hospital death (readmission never possible — dropped
  from the modeling cohort entirely, not just NULL-labeled) vs.
  right-censored (patient's most recent admission — dropped from the
  training target only)
- `readmission_model.py` reuses `mortality_model.py`'s exact admission-time
  feature set and patient-grouped `StratifiedGroupKFold`/baseline/
  `GridSearchCV` scheme; wired into `main.py` as Step 5
- Verified against real data: 129 admissions → 40 deaths excluded, 29 of
  the remaining 89 survivors have a non-censored label, 11 readmitted
  within 30 days. Random Forest ROC-AUC 0.588 ± 0.378 vs. 0.483 ± 0.033
  logistic baseline
- On the synthetic fixture, excluding deaths/censored rows leaves zero
  positive examples — `train_and_evaluate()` guards this single-class case
  explicitly instead of letting `GridSearchCV` crash
- 7 new tests (22 → 29 total): 3 for target censoring/death behavior, 4 for
  the model (cohort filtering, shape alignment, the single-class guard, a
  full train/plot/save run on a handcrafted two-class sample)

## Sprint 7 — Fairness & Explainability
**Status:** Planned

Goal: close two gaps flagged in `CHANGELOG.md`'s "still open" list —
unused subgroup columns, and no per-prediction explanation.

- Mortality/readmission model performance broken out by `ethnicity` and
  `insurance` (already pulled by `cohort_features.sql`, never analyzed)
- SHAP values for the Random Forest, alongside the existing global
  `feature_importances_` plot

## Sprint 8 — Interactive Demo Layer
**Status:** Planned

Goal: make results explorable instead of print-only.

- Streamlit dashboard over `queries.py` + `predict.py`: cohort stats,
  mortality/readmission breakdowns, a form to score a new admission
- FastAPI wrapper around `predict.py` with Pydantic input validation —
  the direct implementation of "how would you deploy this"

## Sprint 9 — Production Hardening
**Status:** Planned

Goal: close the remaining gap between "portfolio project" and
"production-adjacent" tooling.

- `ruff`/`black` + `mypy` as separate CI jobs; `pytest-cov` with a
  coverage badge
- `pyproject.toml` packaging (removes the `sys.path.insert` hacks in
  `main.py` / `mortality_model.py`)
- `Dockerfile` for reproducible runs
- `CONTRIBUTING.md`, PR/issue templates, and ADRs for the two big
  decisions already made and justified in `CHANGELOG.md` (SQL-first joins,
  patient-grouped CV)
