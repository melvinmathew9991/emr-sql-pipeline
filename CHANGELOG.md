# Project History

A single chronological account of this project — what it is, how it was
built, what changed and why, and the exact numbers at each stage. Written to
be re-read quickly before an interview, not just as a change log.

---

## 1. What this project is and why it exists

This is a resume/portfolio project, built specifically to target a **Data
Scientist (Real World Data / Oncology) job description** that asks for:
EMR/genomics/clinical data analysis, descriptive analytics through predictive
analytics, SQL + Python, biostatistics/ML familiarity, and "cutting-edge
models used in production."

**MIMIC-III Clinical Database Demo v1.4** was picked deliberately: it's real,
de-identified ICU patient data (not synthetic), it's a genuine 26-table
relational EMR schema, and — unlike the full MIMIC-III database — it requires
no PhysioNet credentialing, so it was usable immediately.

**Known, honest gaps against that JD** (worth having an answer ready for):
- The JD wants **Oncology** RWD specifically; MIMIC-III Demo is general ICU
  data, not oncology. The transferable claim is "EMR/RWD analytics and
  predictive modeling skill," not oncology domain experience.
- The JD mentions **genomics** — out of scope here entirely.
- The JD wants **biostatistics** familiarity — this project uses a static
  classifier, not survival/time-to-event analysis (mortality is technically a
  censored time-to-event problem). If a separate survival-analysis project
  exists, that's likely where this gap gets covered — don't claim it here.
- The JD wants **production-grade models** — this is intentionally a
  transparent, evaluable baseline, not a deployed system (no model
  persistence, no inference API, no CLI/config, no logging framework).

---

## 2. The pipeline, end to end

```
python main.py
   │
   ├─ 1. build_database.py
   │     reads 6 CSVs (patients, admissions, icustays, diagnoses_icd,
   │     d_icd_diagnoses, labevents) from data/mimic_demo/ (or
   │     data/synthetic_test/ if USE_SYNTHETIC=True)
   │     → writes them into a fresh SQLite file at outputs/mimic_demo.db
   │
   ├─ 2. queries.py
   │     loads and runs 5 SQL queries from sql/*.sql: cohort summary, LOS by
   │     care unit, top diagnoses, mortality by admission type, mortality by
   │     care unit — plus a chi-square significance test on the two
   │     mortality breakdowns
   │
   ├─ 3. cohort_features.py
   │     runs ONE SQL query (sql/cohort_features.sql): a CTE that aggregates
   │     icustays/diagnoses_icd/labevents to one row per hadm_id each
   │     (labs windowed to the first 24h post-admission), then LEFT JOINs
   │     all three onto admissions ⋈ patients
   │     → pandas then only does feature engineering (age capping, % abnormal
   │       labs, filling nulls left by the joins) — no table combination
   │       happens outside SQL
   │
   └─ 4. mortality_model.py
         one-hot encodes categoricals, evaluates a class-weighted Random
         Forest (tuned via GridSearchCV) against a logistic-regression
         baseline, using 5-fold StratifiedGroupKFold grouped by subject_id
         → prints CV ROC-AUC for both models + a held-out-fold classification
           report, saves roc_curve.png / confusion_matrix.png /
           feature_importance.png to outputs/
```

---

## 3. Timeline

### Phase 0 — Initial build (before this session)
The 4-stage pipeline, README, `requirements.txt`, and `data/synthetic_test/`
fixtures already existed. `main.py` had a `USE_SYNTHETIC` flag defaulting to
`True`, pointing at the fake fixtures — the real MIMIC-III demo hadn't been
loaded yet, and all SQL lived as inline Python strings.

### Phase 1 — Verify the pipeline runs (synthetic data)
Ran `python main.py` on `data/synthetic_test/`. Confirmed all 4 stages
execute cleanly:
- 40 patients, 47 admissions, 41 ICU stays, 781 lab results
- In-hospital mortality: 4.3% (2 deaths)
- ROC-AUC: 0.455 (worse than chance — expected and meaningless on a
  47-row synthetic fixture with 1 death in the test fold; this run only
  proved the code executes, nothing about mortality risk)

### Phase 2 — Full folder audit; real data discovered
Audited every file in the repo. Found:
- `sql/` was empty and unreferenced by any code
- The **real MIMIC-III Demo data was already downloaded**, sitting in a
  folder named `data/mimic-iii-clinical-database-demo-1.4/` (full 26-table
  PhysioNet release, 95MB, mostly `CHARTEVENTS.csv` which this pipeline
  never uses) — but `main.py` looked for `data/mimic_demo/`, so it wouldn't
  have been found
- Leftover junk: `src/__pycache__/`, and three `.b64` files accidentally
  left in `outputs/` from building an earlier HTML report

Fixed: renamed the real-data folder to `data/mimic_demo/` to match the code,
deleted the junk, deleted the empty `sql/` folder (temporarily — recreated
in Phase 4), removed an unrelated `.claude/` settings folder.

### Phase 3 — First real-data run
Flipped `USE_SYNTHETIC = False` and re-ran. First genuine result:
- 100 patients, 129 admissions, 136 ICU stays, 76,074 lab results
- In-hospital mortality: 31.0% (40 deaths / 89 survived)
- Mortality by admission type: URGENT 50.0%, EMERGENCY 32.8%, ELECTIVE 0.0%
- Mortality by care unit: TSICU 63.6%, MICU 34.2%, SICU 31.8%, CCU 31.6%,
  CSRU 16.7%
- Top diagnoses: hypertension NOS, atrial fibrillation, acute kidney failure
  NOS, CHF NOS, type II diabetes
- Model (at this point, later found to be flawed — see Phase 5):
  **ROC-AUC 0.770** on a single 75/25 split, accuracy 0.70, recall on the
  death class 0.60

### Phase 4 — SQL-first hardening
Two changes to make the "SQL-first" claim literally true, both verified
behavior-preserving by re-running the pipeline (identical output each time):
1. `cohort_features.py` previously did one real SQL join (`admissions ⋈
   patients`) plus three separate `GROUP BY` queries stitched together with
   pandas `.merge()`. Rewrote it as a single SQL query with CTEs
   (`icu_agg`, `diag_agg`, `lab_agg`), `LEFT JOIN`ed onto the base in one
   query — pandas now only does feature engineering, never table combination.
2. Extracted all 6 SQL queries (5 descriptive + the cohort-features CTE
   query) from inline Python strings into standalone files under `sql/`,
   loaded from disk via a small `_load_sql()` helper — motivated by
   portfolio signal (a reviewer skimming the repo should see real `.sql`
   files, not just Python).

Also: removed an unused `sqlite3` import in `main.py` and an unused
`seaborn` dependency in `requirements.txt`; added a `.gitignore`; added a
README "SQL" section and a "Results" section with the Phase 3 numbers.

### Phase 5 — Senior-reviewer audit, and fixing what it found
Deliberately reviewed the project as an experienced data scientist would,
and verified each concern against the actual code/data rather than
guessing. Found, ranked by severity:

**Critical:**
1. **Target leakage.** `hospital_los_days`, `total_icu_los_days`, and
   `n_diagnoses` were used as "admission-time" predictors, but none of them
   are actually known at admission — LOS ends when a patient dies (so it's
   entangled with the outcome, not predictive of it), and MIMIC's ICD-9
   codes are assigned at discharge for the whole encounter. This likely
   inflated the Phase 3 AUC.
2. **Patient-level leakage.** Verified 14 patients have more than one
   admission (one has 15). The plain `train_test_split` had no grouping, so
   the same patient could appear in both train and test.

**Significant:**
3. No cross-validation — a single split on 129 admissions is a noisy point
   estimate, not a stable result.
4. No baseline model to prove the Random Forest's complexity was earning
   its keep.
5. Hyperparameters (`n_estimators=200, max_depth=5, min_samples_leaf=3`)
   were hand-picked, not tuned.
6. Mortality-by-subgroup rates (e.g., CSRU 16.7% on n=6) were reported with
   no significance testing, despite very small group sizes.

**Fixes applied** (all four `src/` modules plus `sql/cohort_features.sql`):
- Lab features windowed to the **first 24h post-admission**
  (`n_lab_results_24h`, `pct_abnormal_labs_24h`) instead of whole-stay
  aggregation, directly in the SQL query.
- `hospital_los_days`, `total_icu_los_days`, `n_diagnoses` excluded from
  `mortality_model.py`'s feature set (kept in the feature table for
  descriptive use only).
- Evaluation switched to `StratifiedGroupKFold` (5 folds, grouped by
  `subject_id`) throughout — no patient crosses train/test in either the
  cross-validation or the held-out fold used for the plots.
- Added a logistic-regression baseline, evaluated on the same CV scheme.
- Added `GridSearchCV` tuning for the Random Forest
  (`n_estimators` × `max_depth` × `min_samples_leaf`).
- Added a chi-square test of independence on both mortality breakdowns in
  `queries.py`, printing a caveat whenever an expected cell count is below 5.

**The honest result:**

| | Phase 3 (before fix) | Phase 5 (after fix) |
|---|---|---|
| Evaluation | single 75/25 split | 5-fold `StratifiedGroupKFold`, grouped by patient |
| Features | included LOS + diagnosis count (leaky) | admission-time only; labs windowed to 24h |
| Random Forest ROC-AUC | 0.770 | **0.606 ± 0.107** (cross-validated) |
| Baseline (logistic regression) | not measured | 0.510 ± 0.154 (cross-validated) |
| Held-out fold | acc 0.70, recall 0.60, AUC 0.770 | acc 0.48, recall 0.38, AUC 0.441 |

The drop from 0.770 to 0.606 is the **correct, expected outcome** of removing
leakage, not a regression — the original number reflected information the
model wouldn't actually have at prediction time. Only modestly above the
logistic baseline is a realistic result for a 100-patient sample, and
finding + fixing this yourself is a stronger interview story than the
higher, invalid number would have been.

### Phase 6 — Documentation
Updated `README.md` with an explicit "Avoiding leakage" section and a
"Results" section carrying the real, post-fix numbers and the before/after
comparison; wrote this file; rounded out `.gitignore` with standard entries
(`.venv/`, `.pytest_cache/`, `.vscode/`, `.idea/`, etc.).

### Phase 7 — Git, then tests + CI + model persistence

Initialized git, pushed to GitHub
(https://github.com/melvinmathew9991/emr-sql-pipeline).

Then closed the two remaining gaps flagged in Phase 5 ("no automated tests,
no CI" and "no model persistence or inference path"):

**A real bug surfaced while doing this.** Building the test suite meant
actually exercising `train_and_evaluate()` against the synthetic fixture for
the first time since the Phase 5 rewrite — and it crashed.
`StratifiedGroupKFold(n_splits=5)` is invalid when the minority class has
only 2 members (the synthetic fixture has 2 deaths in 47 admissions), and
once that produced `NaN` CV scores, the confusion matrix display then raised
outright when a held-out fold ended up containing only one class. Fixed
both: `n_splits` is now capped at `min(N_CV_SPLITS, minority_class_count)`
with a printed note when it's reduced, and `confusion_matrix(..., labels=[0,
1])` forces a consistent 2×2 matrix regardless of which classes actually
appear in a given fold. This means the synthetic-data smoke-test path the
README has always documented as supported now actually is again.

**Tests added** (`tests/`, 16 tests, `pytest`): one file per pipeline stage,
covering table loading, descriptive query correctness, the feature table's
null-handling, a **regression guard asserting the known-leaky columns
(`hospital_los_days`, `total_icu_los_days`, `n_diagnoses`, `n_icu_stays`)
can never silently reappear** in the model's feature set, a regression test
for the crash above, and tests for the new inference path. All run against
`data/synthetic_test/` only — the real MIMIC data is gitignored, so CI never
needs it.

**CI added**: `.github/workflows/tests.yml` runs the suite on every push;
the README now carries a live status badge.

**Model persistence + inference added**: `train_and_evaluate()` now saves
the fitted pipeline plus its exact training column schema to
`outputs/mortality_model.joblib` via `joblib`. New `src/predict.py` loads
that bundle and scores a new admission's mortality risk from its raw
admission-time fields (age, 24h labs, gender, admission type, care unit),
reindexing one-hot columns so an unseen category degrades to all-zero
dummies instead of raising. This was the direct answer to "how would you
deploy this" — previously there was no way to score a new patient at all.

### Phase 8 — Readmission-interval analysis (window functions)

The descriptive-analytics side of the pipeline had five `GROUP BY` queries
but nothing that reasoned across a single patient's admissions in sequence —
a gap, since 14 patients in this cohort have more than one admission (one
has 15) and readmission timing is a standard clinical quality metric.

Added `sql/readmission_intervals.sql`: for every admission, `ROW_NUMBER()
OVER (PARTITION BY subject_id ORDER BY admittime)` gives that patient's
admission sequence number, and `LAG(dischtime) OVER (PARTITION BY
subject_id ORDER BY admittime)` recovers their previous discharge time so
`days_since_last_discharge` can be computed directly in SQL. First
admissions get `NULL` by construction (no prior discharge exists), and
because `LAG` only ever looks at that same patient's strictly earlier rows,
the feature can't leak information from a later encounter into an earlier
one — the same leakage discipline already applied to `lab_agg` in
`cohort_features.sql`.

Wired in as `queries.readmission_intervals()`, with three new tests
(first-admission nulls, per-patient sequence numbering, non-negative
intervals) — the suite is now 19 tests, up from 16.

**Verified against the real MIMIC-III data**, not just the synthetic
fixture: the cohort's most-readmitted patient (15 admissions) sequences
correctly from 1–15, and across the full cohort, 11 of the 29 admissions
that have a prior discharge fall within a 30-day interval — a real signal,
consistent with the "Add readmission prediction" item previously listed
under Future Extensions in the README.

**Not yet done**: this only computes the interval feature; it isn't fed
into `mortality_model.py` yet. Doing so would need the same scrutiny as
the Phase 5 audit — deciding whether `days_since_last_discharge` at the
time of a given admission is legitimately known then (it is, since it only
depends on strictly prior admissions) before treating it as a predictor.

### Phase 9 — SQL query indexing

A targeted fix folded in ahead of Sprint 6, since its `LEAD(admittime)`
readmission target hits `admissions`/`labevents` harder than the earlier
descriptive queries and benefits from having this in place first.
`build_database.py` now indexes every join/filter column `sql/*.sql`
actually uses. Verified with `EXPLAIN QUERY PLAN` against the real data:
`icustays` and `diagnoses_icd` switched from a full scan + temp B-tree sort
to an index seek with no sort, and `labevents` switched from a full scan to
an index search. 3 new tests (19 → 22 total).

### Phase 10 — Readmission risk model (Sprint 6)

Turned the readmission-interval *feature* from Phase 8 into a genuine
second predictive model, with its own target rather than reusing
`days_since_last_discharge`.

Added `sql/readmission_target.sql`: for each admission, `LEAD(admittime)
OVER (PARTITION BY subject_id ORDER BY admittime)` looks at that same
patient's *next* admission — the opposite direction from Phase 8's
backward-looking `LAG` — so `readmit_30d` is a genuinely forward-looking
label, only knowable in hindsight. Two distinct "no next admission" cases
are both left `NULL` rather than collapsed into a confirmed 0, since
conflating them would teach a model the wrong thing:
- **death this admission** (`hospital_expire_flag=1`) — readmission was
  never possible, not merely unobserved. These admissions are dropped from
  `readmission_model.py`'s cohort entirely, not just given a NULL label.
- **right-censored** — the patient's most recent admission in this
  dataset, with no later encounter to check against. Dropped from the
  training target only.

`readmission_model.py` reuses `mortality_model.py`'s exact admission-time
feature set (`FEATURE_COLS_NUMERIC`/`FEATURE_COLS_CATEGORICAL`) and its
patient-grouped `StratifiedGroupKFold` scheme, logistic-regression
baseline, and `GridSearchCV`-tuned Random Forest — same rigor, different
target. Wired into `main.py` as Step 5, running after the mortality model
against the same feature table and a second `readmission_target()` query.

**Verified against the real MIMIC-III data**: of 129 admissions, 40 end in
death (excluded from the cohort) and 89 survive; of those, 29 have a known
(non-censored) label and 11 fall within a 30-day readmission window — the
same 29/11 split Phase 8 found from the other direction, as expected since
both are counting the same consecutive-admission pairs. Random Forest
ROC-AUC: 0.588 ± 0.378 (5-fold CV, grouped by patient) vs. a 0.483 ± 0.033
logistic baseline — high variance given only 11 positive examples, and
reported honestly rather than smoothed over.

On the small synthetic fixture, excluding deaths and censored rows leaves
*zero* admissions readmitted within 30 days — a single-class target no
classifier can fit. `train_and_evaluate()` guards this explicitly (checks
`y.nunique() < 2` up front, prints why, returns without raising or writing
outputs) rather than letting `GridSearchCV` crash. Since the real fixture
can't exercise the actual training path, a handcrafted two-class sample
covers that instead. 7 new tests (22 → 29 total): 3 for the target's
censoring/death behavior in `queries.py`, 4 for the model (cohort
filtering, shape alignment, the single-class guard, and a full train/plot/
save run on synthetic two-class data).

---

## 4. Still open (not fixed, know these going in)

From the Phase 5 audit, addressed in Phase 7: model persistence/inference,
tests, and CI. Still not addressed:
- **No survival/time-to-event framing** — mortality is technically censored;
  a static classifier sidesteps that. May belong in a separate project.
- **`days_since_last_discharge` still isn't a model feature** — Phase 10
  built a standalone readmission classifier (closing the second half of
  this gap), but it predicts a separate `readmit_30d` target from the same
  admission-time features as the mortality model; the Phase 8 interval
  itself isn't used as a predictor anywhere.
- **Unused SQL-fetched columns** — `admission_location`, `insurance`,
  `ethnicity` are pulled by `cohort_features.sql` and then never analyzed or
  checked for subgroup/fairness performance.
- **No CLI/config, no logging framework** — `main.py` still hardcodes
  `USE_SYNTHETIC` and paths; fine for a demo, not "production" in the JD sense.
- **`sql/` folder was briefly empty/deleted, then reintroduced** — if
  reviewing git history later, don't be confused by that back-and-forth;
  the current state is the final one, with real `.sql` files.

---

## 5. One-line answers, if asked

- **"Walk me through this project."** SQL-first EMR pipeline on real
  MIMIC-III ICU data — SQLite build, 5 descriptive SQL queries, a
  SQL-CTE-joined feature table, and two classifiers (in-hospital mortality,
  30-day readmission) evaluated with patient-grouped cross-validation.
- **"What's your best result?"** Mortality: ROC-AUC 0.606 ± 0.107,
  cross-validated, vs. a 0.510 ± 0.154 logistic baseline. Readmission:
  0.588 ± 0.378 vs. a 0.483 ± 0.033 baseline — both modest, and the
  readmission variance is reported honestly rather than smoothed over
  (only 11 positive examples in the real cohort).
- **"What was the hardest part?"** Catching that my own first version had
  target leakage (whole-stay LOS/diagnosis features) and patient-level
  leakage (repeat patients split across train/test) — the initial 0.770 AUC
  was inflated by both.
- **"What would you do with more data/time?"** Full MIMIC-III database for
  a stable AUC estimate; survival analysis instead of a static classifier;
  subgroup fairness checks on ethnicity/insurance.
- **"How would you deploy this?"** It's already at a first step:
  `train_and_evaluate()` saves the fitted pipeline to
  `outputs/mortality_model.joblib`, and `predict.py` loads it to score a new
  admission from raw fields. Next step for real deployment would be wrapping
  that in an API, adding input validation, and monitoring for feature drift.
- **"How do you know the pipeline actually works?"** 29 pytest tests, run on
  every push via GitHub Actions — including a regression guard that fails
  the build if the leaky columns are ever reintroduced as model features.
