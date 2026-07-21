# PROGRESS.md — Implementation Log

Running log of implementation work. One section per completed notebook/file: purpose,
what each major step does, key printed numbers, design decisions + why, limitations +
how handled. Raw factual notes, not polished prose. Newest sections at the bottom.

---

## notebooks/01_features.ipynb (2026-07-14)

### (a) Purpose
Turn `data/jobstreet_cleaned_final.csv` (69,024 × 29) into the modelling table for
`02_models.ipynb`: filter to usable-salary rows, extract `experience_years` from
description text, add 75 binary skill flags, save the table + skill lists.
TF-IDF / one-hot deliberately NOT done here — they are fitted inside the sklearn
Pipeline in 02_models (identical transform at inference, no leakage).

### (b) Steps
1. Load + check: reads the cleaned CSV, verifies the 7 required columns exist.
2. Filter: `salary_usable_flag == 1`, then `salary_monthly_final` in [500, 30000].
3. Experience extraction: two regexes (range "2-4 years" → min of pair; single
   "3 years"/"5+ yrs" → number), context check (see decisions), min across mentions,
   cap at 20, `has_experience_req` flag. Demo cell on hand-made strings + 5 real snippets.
4. Skill flags: 50 hard + 25 soft skills, one word-boundary regex per skill applied to
   lowercased `descriptions_clean`; one binary column per skill + `skill_count`.
5. Assemble model table: 4 raw columns for the Pipeline + engineered features + target.
6. Save `data/model_table.csv` + `models/skill_lists.joblib`, reload both as round-trip check.

### (c) Key printed results
- Filter: 69,024 → 31,468 (usable flag) → **31,406** rows (62 outside RM500–30,000).
- Target: median RM3,750, IQR RM3,000–5,000 (matches dataset facts). Skewness raw 2.76 → 0.02 after log (computed separately, used to justify log target).
- Experience: 51.9% of ads state a requirement (16,307/31,406); mean 1.47, median 1, 75th pct 2; 126 rows at the 20-year cap.
- Skills: no skill matched zero ads; top = communication 51.3%, independent 26.7%, interpersonal 23.3%, accounting 18.8%, excel 18.3%. Mean 3.67 skills/ad.
- Model table: 31,406 × 83, 0 missing values. Files: model_table.csv 7,259 KB, skill_lists.joblib 3 KB.

### (d) Design decisions
- Min across all experience mentions = conservative reading of the entry requirement.
- Keyword dictionary for skills (not embeddings): every flag traces to one keyword,
  viva-defensible; the list is saved to models/ so the dashboard multiselect and the
  model can never disagree on the skill vocabulary.
- Regex details: spaces in skill names also match hyphens ("problem-solving"), optional
  trailing "s", hand-written patterns for c++/c#/.net/node.js (punctuation breaks `\b`)
  and audit/independent/multitasking/organizational/work-under-pressure (word forms).
- Column-name mapping for special chars: c++ → skill_c_plus_plus etc.
- CSV (not parquet) for the model table — no new dependencies allowed.

### (e) Iteration: experience false positives (v1 → v2)
- v1 matched any "N years". Found in printed examples: company-age text
  ("bathroom and kitchen industry more than 40 years") extracted as 20 (capped from 40);
  324 rows sat at the cap, many suspected company-age matches.
- Fix (approved by student, CLAUDE.md locked decision 3 updated): a years match only
  counts if "experience" or "pengalaman" appears within ±60 characters. Substring match
  also catches "experienced" / "berpengalaman".
- Before → after: stated-requirement share 55.0% → 51.9%; mean years 1.70 → 1.47;
  rows at 20-cap 324 → 126. Median/75th pct unchanged (1/2) — noise removed, signal kept.
  The company-age example now extracts (0 years, flag 0).

### (f) Limitations (current)
- 126 rows remain at the 20-year cap; some are genuine "20+ years" senior roles.
- Malay-only requirements ("3 tahun pengalaman") are not matched — the year patterns
  require "years/yrs". Accepted scope decision; "pengalaman" only serves as context word.
- Min-across-mentions can under-read ads quoting several figures ("at least 7 years"
  ad extracted 2 because a smaller mention existed) — conservative by design.
- Feature imbalance for later: type_clean 93.4% Full time; tiny groups exist
  (Government & Defence 9 rows, Self Employment 1, Perlis 4, one literal "Malaysia"
  state row — unmapped leftover from cleaning, harmless with handle_unknown='ignore').

---

## notebooks/02_models.ipynb (2026-07-14)

### (a) Purpose
Train and compare the 4 locked models on the 01_features model table, pick the winner
by test MAE, add P25/P75 quantile models for the displayed range, SHAP explainability,
subgroup error analysis, and save every artifact the dashboard needs to models/.

### (b) Steps
1. Load model_table.csv (31,406 × 83); target = np.log(salary); one 80/20 split
   (random_state=42) reused by all models; metrics always converted back to RM.
2. Preprocessing inside the Pipeline: TF-IDF(800, 1–2grams, min_df=5) on job_title,
   OHE(handle_unknown='ignore') on category/state/type, passthrough for 78 numerics.
   sparse_threshold=0 → dense (HistGBR rejects sparse). 930 features total.
   clone(preprocessor) per model so pipelines share no state.
3. DecisionTree min_samples_leaf tuned on a validation split from TRAIN (5/10/20/50).
4. Train + compare LinearRegression / DecisionTree / RandomForest / HistGBR.
5. Quantile HistGBR (0.25 / 0.75) pipelines → displayed range + honesty checks.
6. Subgroup MAE by category and state (tables + bar charts).
7. Save model artifacts BEFORE SHAP (see iteration notes), then SHAP TreeExplainer
   (interventional, 100-row background), adaptive sample, summary plot, save explainer.

### (c) Key printed results
- Split: train 25,124 / test 6,282. Skew: raw 2.76 → log 0.02. 930 features.
- DT tuning: min_samples_leaf=20 best (val MAE RM1,170).
- Comparison (test, RM): **RandomForest MAE 976, RMSE 1,776, R² 0.598 — WINNER**;
  HistGBR 996 / 1,791 / 0.591; Linear 1,067 / 1,876 / 0.551; DT 1,130 / 2,006 / 0.487.
  Fit times: RF 280s, HistGBR 34s, DT 4s, LR 2s.
- Quantile range: 47.4% of test salaries inside [P25,P75] (ideal ~50%); median range
  RM3,349–4,386 (width RM1,054); lower>upper crossings 2/6,282; 86.6% of point
  predictions inside their own range (dashboard clips the rest).
- SHAP additivity check passes exactly (base+shap == model output). 0.22 s/row →
  full 1,500-row sample in 279 s. Top mean |SHAP|: title word "manager" 0.126,
  experience_years 0.089, "assistant" 0.044, state KL 0.042, "internship" 0.042,
  "executive", "intern", "senior", "engineer", category ICT.
- Subgroup MAE: best categories Admin & Office Support RM548, Advertising/Arts RM685;
  worst (large-n) ICT RM1,568 (n=550), Banking RM1,462 (n=150); tiny-n outliers
  Sport & Recreation RM2,521 (n=3), CEO & General Mgmt RM2,371 (n=15).
  States: best Perak RM706; worst KL RM1,179 (n=1,882) — high-salary market is harder.
- Artifacts: salary_pipeline 39 MB, shap_explainer 45 MB, quantile pipelines ~410 KB
  each, plus feature_names / input_options / model_comparison.csv. Reload check OK
  (one test row: predicted RM2,955 vs actual RM2,300).

### (d) Design decisions
- Winner by test MAE per locked decision → RandomForest (beats HistGBR by RM20).
- sparse_threshold=0 (dense matrix, ~230 MB) so one preprocessor serves all 4 models.
- DT tuned on a train-carved validation split — test set never touched until the
  final table.
- SHAP: TreeExplainer with feature_perturbation='interventional' + 100-row training
  background; notebook prints its own additivity check as evidence.
- Model artifacts saved BEFORE the SHAP step so explainability failures can't cost
  training time.

### (e) Iterations (3 runs to green)
1. **Run 1 — SHAP timeout.** Explaining 1,500 rows with the default explainer blew the
   3,600 s cell timeout; nbconvert discards outputs on failure → hour of training lost.
   Fix: adaptive sample sizing (10-row probe → ~10 min budget) + save-before-SHAP
   reordering.
2. **Run 2 — training timeout.** The 4-model cell itself crossed 3,600 s (RF is right
   at the edge; machine was evidently throttled that run). Fix: per-cell timeout raised
   to 10,800 s at the nbconvert call (no notebook change).
3. **Run 3 — SHAP additivity explosion.** Default path-dependent TreeExplainer returned
   corrupt values for this RF under shap 0.52 + sklearn 1.8 (sum of SHAP values ≈
   −1.9e23 vs model output 8.29). Reproduced standalone against the saved pipeline;
   fix = interventional mode with background data (exact additivity, 0.22 s/row).
   Run 4 completed end-to-end in ~13 min.

### (f) Limitations
- RF artifacts are heavy (39 MB pipeline + 45 MB explainer) and per-row SHAP is
  ~0.2–0.3 s — fine for single dashboard predictions.
- Interventional SHAP answers "vs the average of a 100-ad background sample";
  wording in the dashboard reflects that framing.
- Subgroup MAE for tiny groups (n<20) is noise, flagged via n_test_rows column.
- KL has the worst state MAE (RM1,179) — high-salary segment has more variance;
  quantile range communicates this uncertainty.
- tqdm progress-bar noise is embedded in the SHAP cell's output stream (cosmetic).

---

## dashboards/app.py + dashboards/config.yaml (2026-07-14)

### (a) Purpose
Streamlit dashboard per CLAUDE.md spec: login-gated salary prediction with range,
verdict vs entered salary, SHAP waterfall + plain-language sentences, and skill-based
improvement tips. Loads ONLY models/ artifacts — never the CSV.

### (b) Structure
1. Login gate (streamlit-authenticator 0.4.2): config.yaml holds 2 demo accounts
   (demo_user/demo123, supervisor/super123) with bcrypt-hashed passwords; failure/None
   states handled; logout button + welcome name in sidebar; nothing renders pre-login.
2. Artifact loading: missing-file check runs on every rerun (NOT cached) → friendly
   "models not trained yet" info + st.stop(); heavy joblib loads are @st.cache_resource.
3. st.form inputs: job title, category/state/type selectboxes (from input_options),
   experience slider 0–20, skills multiselect (75 display names from skill_lists),
   optional salary number_input (0 = not provided).
4. build_input_row(): one-row DataFrame with all 82 training columns;
   has_experience_req = 1 if years > 0.
5. Results: P25/point/P75 metrics (np.exp back to RM, lower≤point≤upper enforced);
   verdict badge below/within/above; SHAP waterfall via the saved explainer;
   top-3 plain-language sentences (% effect = exp(shap)−1); improvement tips by
   batch-re-predicting all absent skills in ONE predict() call, top 3 RM uplifts shown.
6. All prediction/SHAP/tips blocks wrapped in try/except → st.error/st.warning, no crash.

### (c) Verification (streamlit AppTest harness)
- Unauthenticated: login form + hint, app stops. demo_user/demo123 → status True
  (bcrypt hashes valid). Wrong password → status False + error message.
- Logged in, models missing → "not trained yet" info + missing-file list (tested
  before training finished).
- Full path with real artifacts: Senior Software Engineer / ICT / KL / 5 yrs /
  python+sql / RM4,000 → range RM6,628–11,229, estimate RM9,482, verdict BELOW;
  sentences: ICT +19%, experience +18%, "senior" +17%; tips: linux +RM897,
  vue +RM717, kubernetes +RM503. Zero uncaught exceptions.

### (d) Design decisions
- Missing-artifact check kept outside the cache so the app recovers as soon as
  training finishes, without restart.
- Improvement tips evaluate ALL absent skills in one batched predict (simpler and
  cheaper than picking "top 10" first; uplift measured against the unclipped point).
- Plain-language effects reported as % (log-space SHAP → exp(v)−1) so sentences are
  honest about the multiplicative model.

### (e) Limitations
- Login-in-browser cookie reuse untestable in AppTest (cookie decode noise is
  expected there); works in a real browser session.
- Per-prediction SHAP ≈ 0.3 s + RF batch predicts — dashboard feels responsive but
  not instant; acceptable for a single-user FYP demo.

---

## Dashboard v2: register / guest / history / forgot-password / autocomplete (2026-07-15)

Files: dashboards/app.py (restructured), dashboards/db.py (new), dashboards/emailer.py
(new), notebooks/01_features.ipynb (new Step 6), .streamlit/secrets.toml (new template),
.gitignore (new), scripts/test_dashboard.py (new).

### (a) Purpose
Upgrade the dashboard from 2 hard-coded demo accounts to a small real system:
Login / Register / Continue-as-guest entry screen, per-user saved prediction history
(guests must log in to save), email+password registration, forgot-password with real
email delivery, job-title autocomplete ("data" → Data Analyst, Data Entry, ...), and
category auto-fill from the chosen title.

### (b) Steps / structure
1. 01_features Step 6: builds `models/job_title_suggestions.joblib` — titles grouped
   case-insensitively, kept if ≥3 ads, ordered by frequency, displayed with the most
   common casing; `title_to_category` = modal category per title, keyed by LOWERCASED
   title. Dashboard still loads only models/ artifacts (never the CSV).
2. db.py: SQLite (stdlib sqlite3, no new dependency) at data/dashboard.db.
   Tables: users (username PK, name, email UNIQUE, bcrypt password_hash, created_at)
   and predictions (inputs + range + verdict per saved row). Demo accounts seeded from
   config.yaml ONCE (empty-table check); after that the DB is the source of truth.
   Short-lived connection per operation (no cross-rerun sharing issues).
3. emailer.py: forgot-password email via stdlib smtplib + EmailMessage, STARTTLS.
   SMTP settings read from .streamlit/secrets.toml; returns False when unconfigured
   or on failure → app shows the new password on screen instead (demo never breaks).
4. app.py: entry hub (tabs Log in / Register / Forgot password + guest button) using
   streamlit-authenticator 0.4.2 widgets; register_user/forgot_password/reset_password
   update the in-memory credentials dict → app persists the bcrypt hash to SQLite.
   Prediction results stored in st.session_state["last_result"] and rendered from
   there, so results survive reruns AND a guest logging in to save. Predict/History
   tabs; history = st.dataframe with multi-row selection → Delete selected / Clear all
   (confirm checkbox). Job title = st.selectbox(accept_new_options=True) → typing
   filters 1,257 suggestions, free text allowed; on_change callback auto-fills the
   category. All prediction/SHAP/tips math unchanged from v1.

### (c) Key printed results
- Title artifact: 17,134 unique titles (case-insensitive) → 1,257 suggestions covering
  44.4% of the 31,406 ads; case-duplicate "ACCOUNT EXECUTIVE" merged into "Account
  Executive" (592 ads); 10 "data" titles with sensible categories (Data Analyst → ICT,
  Data Entry → Admin & Office Support). Artifact 96 KB.
- scripts/test_dashboard.py: 42/42 AppTest checks pass on a temp DB — hub gating,
  seeding, wrong/right login, guest predict (Data Analyst / KL / 5 yrs / python+sql →
  RM4,756–10,991, estimate RM7,713, verdict BELOW for RM4,000), category auto-fill,
  guest result surviving login, save/double-save guard, registration (valid, duplicate
  rejected, weak password rejected), forgot-password fallback (new hash verified with
  bcrypt, old password dead), per-user delete isolation. Zero uncaught exceptions.
- Real server boot check: streamlit run headless → /healthz 200, no startup errors.

### (d) Design decisions
- SQLite over YAML/JSON for accounts + history (student's choice via question):
  stdlib module = no new dependency; bcrypt hashes only; one file, git-ignored.
- Student chose REAL email for forgot-password. Discovery: the library's
  send_email=True does NOT do SMTP — it calls the author's paid cloud API (api_key).
  So the widget only generates the password and smtplib sends it; on-screen fallback
  when SMTP is unconfigured so the viva demo cannot dead-end.
- Replaced st.form with plain widgets + button: widgets inside a form cannot trigger
  the rerun needed for category auto-fill.
- captcha=False on registration (fewer live-demo surprises; one flag to re-enable).
- Password policy comes from the library validator (8–20 chars, upper/lower/digit/
  special) — registration rejects weak passwords for free.

### (e) Iterations / debugging
- AppTest ghost-widget crash: mid-script st.rerun() (guest button, save button) left
  the aborted partial run's widgets in AppTest's element tree with cleaned session
  state → next at.run() crashed harvesting them (KeyError "$$ID-...-None"). Fix:
  all state-flipping buttons now use on_click callbacks (state updates BEFORE the
  natural rerun), and a successful login clears the hub via st.empty() and falls
  through to the app in the same run. Zero st.rerun() calls remain — also one less
  redraw per click in the browser.
- AppTest session_state proxy has no .get() → helper in the test script.

### (f) Limitations / student actions
- REAL email needs the student to create a Gmail App Password (Google Account →
  Security → 2-Step Verification → App passwords) and fill .streamlit/secrets.toml;
  until then forgot-password shows the new password on screen with a warning.
- Demo passwords (demo123/super123) predate the registration password policy — they
  are pre-hashed seeds and bypass validation; acceptable for the viva demo.
- History row-selection deletion and login cookies cannot be driven by AppTest —
  verified manually in a browser; SMTP delivery still to be tested by the student
  after the app password exists.
- The registration email is stored but unverified (no confirmation-link flow) —
  out of scope for the FYP, noted as future work.

---

## Dashboard v3: XAI clarity, charts, wording, state cleanup, auto-save (2026-07-15)

Files: dashboards/app.py (charts + labels + auto-save), scripts/test_dashboard.py
(new checks; suite 42 → 49). No retraining, no artifact or dependency changes.

### (a) Purpose
Student feedback on the running v2 dashboard: (1) "the skill 'count' raised the
estimate by 13%" is meaningless; (2) expected friendlier charts (pie/line) than the
SHAP waterfall; (3) "Adding analytical could raise..." reads oddly; (4) states
"Malaysia" and "Others" look wrong for a Malaysia-only dataset; (5) after a guest
logs in via "Log in / Register to save", the result was NOT saved automatically.

### (b) What changed
1. **Label bug (root cause of "skill 'count'")**: in v2's human_label() the generic
   `startswith("skill_")` branch ran before the exact `skill_count` check, so the
   engineered skill-count feature was mislabelled as a skill named "count" — the
   correct branch was dead code. Rewrote human_label(name, value): exact names
   checked first, skill columns resolved to their real display names (skill_c_plus_plus
   → "c++") via an inverted skill_columns map, and yes/no features now phrase
   presence vs absence honestly ("not having the skill 'python'", "No 'manager' in
   job title") using the encoded value of THIS prediction.
2. **Charts** (per student's choice; pie rejected — SHAP factors are signed, a pie
   can only show parts of a whole, so it would hide direction): NEW factor bar chart
   (top 8 factors, x = % effect via exp(shap)−1, blue = raises / red = lowers,
   colorblind-safe pair, values at bar tips, legend, recessive hairline grid); NEW
   salary-vs-experience line chart (profile re-predicted at 0–20 years in one batched
   call at predict time, stored as exp_curve; marker + "You now: RM X" label placed
   on the empty side of the curve); waterfall KEPT (locked decision 7) inside a
   "Detailed SHAP breakdown" expander with humanized labels ("5 = Years of
   experience" instead of "experience_years") and a caption warning that SHAP's
   standard colours are reversed (red = up) and its x-axis is log-salary.
3. **Tips wording**: "Skills the market pays for that are missing from your profile:"
   then "Learning **python** ..." for hard skills, "Strengthening your **analytical**
   skills ..." for soft skills (lists from skill_lists.joblib decide which phrasing).
4. **States**: verified in model_table — "Malaysia" (1 row, whole-country ad) and
   "Others" (104 rows, cleaning bucket for unmapped locations) are leftovers, not
   states; dataset is Malaysia-only. Both hidden from the dropdown (16 real
   states/territories remain); model untouched (OHE handle_unknown='ignore').
5. **Auto-save**: "Log in / Register to save" now sets a pending_save session flag
   via its callback; right after any login completes, the flag is popped and the
   surviving last_result is saved automatically (works via the Register→Login path
   too) with a distinct "saved automatically" confirmation. Double-save guarded by
   the existing saved flag; a plain rerun does not save twice (tested).

### (c) Verification
- scripts/test_dashboard.py: **49/49 checks pass.** New: state dropdown has exactly
  16 options without Malaysia/Others; exp_curve = 21 positive RM values; rendered
  markdown contains no "skill 'count'"; tips contain Learning/Strengthening; no
  "unavailable" warnings; exactly ≥3 images rendered (element type "image" in this
  Streamlit version, found via probe); auto-save happens with NO Save click and a
  plain rerun stays at one row.
- Rendered PNGs extracted from the AppTest media storage (MemoryMediaFileStorage
  captured by subclassing, since the mock runtime is torn down per run) and visually
  checked: bar labels/legend collision-free; line-chart label initially sat on the
  rising curve → moved below the marker when the curve ahead rises.
- Headless boot: /healthz 200, no startup errors.

### (d) Test-safety change (important)
The student put REAL Gmail SMTP credentials into .streamlit/secrets.toml. The test
suite now monkeypatches emailer.send_password_email to return False so tests can
NEVER send real emails (and deterministically exercise the on-screen fallback).
The app's fallback warning was reworded ("could not be sent") since it now also
covers transient SMTP failures, not just missing configuration.

### (e) Limitations / observations
- The experience curve for the sample profile peaks around 6 years and then falls —
  honest model behaviour, not a bug: JobStreet ads demanding 10+ years rarely
  advertise salaries (and 01_features takes the minimum of mentioned ranges), so
  the training data under-represents senior pay. Worth one line in the report.
- SHAP waterfall keeps its library-standard colours (red = up), the reverse of the
  new bar chart — mitigated with the expander caption rather than monkeypatching
  shap internals.
- The "You now" label placement is a two-case heuristic (rising/falling curve);
  extreme profiles could still brush the line — cosmetic only.

---

## Dashboard v4: education feature + retrain, scenario comparison, profile autofill, 30-day login (2026-07-15)

Files: notebooks/01_features.ipynb (new Step 3, re-run), notebooks/02_models.ipynb
(one markdown count fixed, full re-run), dashboards/config.yaml (cookie), dashboards/db.py
(migration + profiles), dashboards/app.py (education input, profile, comparison),
scripts/test_dashboard.py (suite 49 → 66), CLAUDE.md (locked decision 3 extended —
approved by student via scope question).

### (a) Purpose
Student asked four improvement questions; approved scope: (1) education as a REAL model
feature (extract from descriptions + retrain — a cosmetic input the model ignores was
rejected), (2) compare 2–3 saved predictions side by side at full re-predict depth,
(3) saved profile that prefills the form on login, (4) stay logged in 30 days instead of 1.

### (b) What was built
1. **01_features Step 3 — education extraction.** Ordinal `edu_level` 0–4 (not stated /
   SPM–secondary / diploma / bachelor's / postgraduate) + `has_edu_req`. Minimum level
   mentioned = entry requirement (same conservative rule as experience). Keyword regexes;
   the word "degree" gets two exclusions (digit right before = angle "360 degree
   feedback"; followed by " of" = "a high degree of accuracy"); bare "master" never
   matched (hits "Scrum Master"); "ijazah" counts as bachelor's. No context window needed
   (unlike bare year numbers, "diploma"/"spm" are unambiguous). 8/8 hand-made demo cases
   correct, incl. both non-education "degree" uses.
2. **Full retrain** (model table 31,406 × 85, features 930 → 932). Same protocol,
   winner by test MAE.
3. **db.py**: `PRAGMA table_info` migration guard adds `edu_level` to an existing
   predictions table in place (old rows NULL → shown as "Not specified"); new `profiles`
   table (one row per user, INSERT OR REPLACE) + save_profile/load_profile.
4. **app.py**: "Highest education" selectbox (index = ordinal level, default "Not
   specified"); education flows through build_input_row, SHAP labels, history table and
   saved rows. "Save these inputs as my profile" button + one-shot prefill right after
   login (skipped when a guest's un-saved result is in the session, so their form
   survives login); logout wipes the form keys + prefill flag so the next visitor on
   the same tab starts clean. History tab: select 2–3 rows → "Compare selected
   scenarios" → per-scenario columns (range metrics + top-3 factors in plain language),
   a range-interval chart (P25–P75 bar + point marker per scenario) and overlaid
   salary-vs-experience curves. Scenario colors = first 3 slots of a validated
   colorblind-safe categorical palette (slot 1 = the app's existing blue); A/B/C
   letters carry identity in text so color is never the only channel.

### (c) Key printed results
- Education extraction: 69.9% of ads state a level (21,949/31,406). Distribution:
  not stated 30.1%, SPM/secondary 9.2%, diploma 33.2%, bachelor's 27.4%, postgrad 0.1%
  (25 ads). Median salary by level: SPM RM3,000 → diploma RM3,750 → bachelor's RM4,750 —
  real, monotonic signal.
- Retrain (test, RM): **RandomForest MAE 963 (was 976), RMSE 1,763, R² 0.604 — still
  the winner**; HistGBR 983 (was 996); Linear 1,061; DT 1,118 (min_samples_leaf=20
  again, val MAE RM1,184). Every model improved with the education features.
- Quantile honesty: 47.3% of test salaries inside [P25,P75] (ideal ~50%); median range
  RM3,348–4,370 (width RM1,033); lower>upper 9/6,282; 86.9% of points inside their own
  range. SHAP: additivity exact, 0.17 s/row, 1,500 rows in 330 s; **edu_level is the
  #6 feature globally** (mean |SHAP| 0.0335, just under state KL at 0.0368).
- scripts/test_dashboard.py: **66/66 checks pass** (was 49). New: education selectbox
  (5 levels, default "Not specified"); estimate moves when education changes; saved row
  + history table include education; profile persisted and form prefilled on a fresh
  login (checked via widget values, not just session state); full comparison path runs
  end to end (2 scenarios, A/B labels, 2 metrics, factor lines, 2 charts, no warnings);
  v3-schema database gains edu_level + profiles on init_db with the old row still
  listable. Headless boot: /healthz 200.
- Sample: guest Data Analyst / KL / 5 yrs / python+sql / bachelor's → RM4,806–10,146,
  estimate RM7,823, verdict BELOW for RM4,000 (new model's numbers).

### (d) Design decisions
- **Ordinal edu_level, not one-hot**: the levels are genuinely ordered, one column keeps
  the table lean, and "higher number = higher requirement" is easy to defend at the viva.
- **Input semantics**: the model learned the level an AD REQUIRES; the user enters their
  own highest level and is matched against ads requiring it — the same framing already
  accepted for years of experience (help text says so).
- **Comparison re-predicts with the current model** instead of charting the stored
  numbers — the stored rows may predate a retrain; a caption states this honestly. The
  compute helper takes plain dicts (no Streamlit calls) so tests can drive it.
- **AppTest cannot select dataframe rows** (known limit) — the test monkeypatches
  st.dataframe to fake a 2-row selection, so the whole compare path (re-predict + SHAP
  + both charts) still runs end to end in the suite; real click-selection stays a manual
  browser check.
- Range chart labels: one combined "RM low – high" label ABOVE each interval + muted
  estimate below the marker — first render (checked visually from AppTest's media
  storage) had the left end-label colliding with the y-axis names for a far-left
  interval, and a duplicated label when the clipped point sits on the range edge.
- Profile prefill runs before any widget is created (Streamlit's only legal moment for
  programmatic widget-state writes); a profile job title not in the suggestion list is
  prepended to the options for that run so the selectbox accepts it.

### (e) Limitations / student actions
- Postgraduate has only 25 ads (0.1%) — level 4's effect is essentially bachelor's;
  salaried postgrad roles rarely advertise pay on JobStreet. One line for the report.
- Education extraction misses Malay "sarjana (muda)" forms except "ijazah"; "degree"
  exclusions are heuristic. Accepted scope, consistent with the experience extractor.
- Education is a proxy: ad-requirement, not the applicant's attainment.
- Pre-v4 history rows have no education (shown/compared as "Not specified").
- The per-profile education effect can be small (Data Analyst minimal profile:
  RM3,850 → RM3,842) even though the global effect is rank #6 — title dominates.
- Prefilling the experience slider logs a harmless Streamlit warning ("default value
  but also set via Session State API") in the console — cosmetic, not user-visible.
- STUDENT: re-screenshot the report figures that show model numbers (comparison table,
  SHAP summary, subgroup MAE) — they all changed with the retrain; and click through
  the comparison in a real browser (2 and 3 rows) once.

### (f) Robustness check: 5-seed paired ablation of the education feature (2026-07-15)

Question: is the seed-42 RandomForest improvement (MAE RM976 → RM963) real, or split
noise? Script: `scripts/ablation_edu_level.py` (standalone on purpose — nbconvert cannot
run a single added cell, and a notebook placement would force a full retrain of all
models + SHAP). Design: 5 seeds (0–4); per seed ONE 80/20 split shared by both arms;
production RF hyperparameters (n_estimators=200, min_samples_leaf=3, random_state=42) in
both arms; the "without" arm drops BOTH education columns (edu_level + has_edu_req —
the pair the 976→963 delta came from, choice confirmed by student) upstream of the
ColumnTransformer, so encoders stay valid. Runtime ~38 min (10 RF fits).

- MAE with education    : [959.1, 993.0, 1004.9, 958.6, 943.6] — mean RM971.8, range RM943.6–1,004.9
- MAE without education : [959.9, 995.2, 1013.6, 967.2, 953.9] — mean RM978.0, range RM953.9–1,013.6
- Paired differences    : [−0.9, −2.2, −8.7, −8.6, −10.3] — mean **−RM6.1** (negative = education helps)
- Ranges overlap: YES. Paired diffs all negative: 5/5.

**Conclusion: education did not measurably reduce MAE.** The two MAE ranges overlap
heavily — split-to-split variance (~RM60) dwarfs the effect, and the seed-42 delta of
RM13 overstated it (mean paired effect −RM6.1, <1% of MAE). For transparency: all 5
paired differences are negative, so a tiny consistent effect likely exists, but it is
not a meaningful accuracy gain. The feature is retained because SHAP shows the model
genuinely uses it (#6 feature globally) and because a user-facing education input must
feed the model to be honest — its value is interpretability and completeness, not MAE.

---

## Dashboard v5: layout redesign + natural-language XAI + three-lever advice (2026-07-16)

Files: dashboards/app.py (restructured), .streamlit/config.toml (new), scripts/
test_dashboard.py (suite 66 → 76), docs/PROGRESS.md. No retraining, no artifact or
dependency changes; db.py/emailer.py/notebooks untouched.

### (a) Purpose
Student feedback on v4: (1) the XAI copy ("Why this prediction?", "How to increase
your market value") reads unnaturally — the "X raised the estimate by about N%"
template is hard to catch; (2) the overall design is too plain for the finished FYP
system. Directions chosen by the student via questions: profile form + history access
in the sidebar with results centered; "for you / against you" explanation anchored on
an RM baseline; market-value section expanded to three levers (skills, experience,
education).

### (b) What changed
1. **Native theme** (.streamlit/config.toml): `[theme]` with primaryColor #2a78d6 —
   the same blue every chart uses — so buttons/slider/focus rings match the charts.
   One config file, zero custom HTML/CSS (keeps the CLAUDE.md "no HTML hacks" rule).
2. **Layout**: top-level Predict/History tabs replaced by a sidebar radio (key `nav`);
   the whole profile form + Predict + Save-profile buttons moved into the sidebar
   (stacked, full-width buttons). The form renders on EVERY run regardless of page —
   Streamlit drops widget state for non-rendered widgets, so hiding it on the history
   page would wipe the inputs. All widget keys unchanged → prefill/logout-wipe/
   profile-save callbacks untouched. The Predict button now works via an on_click
   callback (`request_prediction`) that sets a `do_predict` flag AND jumps `nav` back
   to Predict, so predicting from the history page lands on the result. Heavy compute
   stays in the script body (flag consumed via pop), not in the callback.
3. **Hero result card** (bordered container): one big st.metric estimate (was 3 equal
   metrics) + NEW `plot_salary_range()` — horizontal P25–P75 band, white-ringed marker
   at the estimate, and a black vertical tick "Your salary: RM X" when a salary was
   entered. Tick (not a filled diamond) because the first render showed the diamond
   fully hiding the estimate marker when offer ≈ estimate. Endpoint labels below the
   band, offer label above → the two text levels can never collide. Verdict copy
   reworded ("Your salary of RM 4,000 is **below the market range** — comparable jobs
   typically pay RM 4,806 – 10,146.").
4. **"Why this estimate?"**: opens with the baseline — "A typical Malaysian job ad
   pays around RM {exp(SHAP base)}." — then two columns "▲ Working in your favour" /
   "▼ Working against you" (≤4 factors each, |shap|>0.005). Each row: neutral
   `friendly_phrase()` noun phrase ("'Senior' in your job title", "Being based in
   Kuala Lumpur") + caption "≈ +RM 1,280 (+19%)". RM effect = point·(1−exp(−v)) =
   what the estimate loses if the factor's effect is removed; caption states the
   figures interact and don't sum exactly, and that "typical ad" = the explainer's
   background sample. Factor bar chart kept below the columns; SHAP waterfall expander
   kept unchanged (locked decision 7). `human_label()` retained for the comparison view.
5. **"How to raise your market value" — three levers**: 📚 skills (same batched
   computation; NEW small uplift bar chart + "Adding **linux** to your skill set is
   worth about +RM 897/month" / "Strengthening your **X** skills..." wording);
   ⏳ experience outlook read off the already-stored exp_curve (value at +2 years,
   zero extra model calls); 🎓 education — one extra single-row predict at
   edu_level+1 stored as `edu_uplift` at predict time. Every lever has an honest
   fallback below MIN_MEANINGFUL_RM = RM50 ("adds little for this profile...") —
   RM50 chosen because the model's MAE is ~RM960, so smaller uplifts are noise.
6. **Transparency caption** under the title: winner model + test MAE read from
   models/model_comparison.csv at runtime (added to ARTIFACT_FILES; load_artifacts
   now handles .csv via pd.read_csv) — never hard-coded, so it survives retrains.
7. **Empty state**: a welcome card explains the four outputs before the first
   prediction (the main column used to be blank except the form).

### (c) Verification
- scripts/test_dashboard.py: **76/76 checks pass** (was 66). New: nav radio options;
  welcome empty state; transparency caption; hero metric (now exactly 1 on Predict);
  baseline sentence; both for/against headers; "≈ +RM" factor rows; three lever
  headers; Adding/Strengthening wording; edu_uplift stored (Bachelor's → next level
  Master's/PhD); ≥5 images on the results page; guest history page prompts login;
  form + result survive a nav round-trip; history/comparison checks now switch the
  nav radio first. Suite re-run after the tick fix: still 76/76.
- Rendered PNGs re-extracted from AppTest media storage (v3 technique, script kept in
  the session scratchpad): checked hero band with offer BELOW range and offer ≈
  estimate (tick fix), uplift bars — no label collisions.
- Headless boot: streamlit run → /healthz 200.

### (d) Design decisions
- Sidebar radio instead of tabs: satisfies "profile and history at the side, graphs
  centered"; the history TABLE still renders in the main column (too wide for a
  sidebar) — the sidebar is its entry point.
- friendly_phrase() phrases are deliberately NEUTRAL noun phrases: direction lives in
  the column header and the signed RM figure, so no sentence template has to bend
  around positive/negative ("raised/lowered by N%" is gone from the main view).
- RM attribution formula point·(1−exp(−v)) is multiplicative-exact for removing ONE
  factor; the interaction caveat is stated in-app rather than pretending additivity.
- use_container_width on the sidebar buttons still works on Streamlit 1.59.2.

### (e) Limitations / student actions
- The for/against RM figures use the CLIPPED point (the displayed number) — for the
  ~13% of predictions where the point is clipped into the quantile range, the RM
  effects are scaled to the displayed estimate, not the raw model output. Consistent
  for users; worth one line in the report.
- Prefilling the experience slider still logs the harmless "default value + Session
  State API" console warning (pre-existing, cosmetic).
- STUDENT: click through in a real browser once — sidebar nav, form surviving page
  switches, history row selection + comparison, and re-screenshot the report figures
  that show the dashboard UI (layout changed substantially in v5).

---

## Dashboard v6: top nav + user menu, onboarding, concept-level XAI, evidence-based skill advice (2026-07-16)

Files: notebooks/01_features.ipynb (new Step 8, re-run — NO model retrain),
dashboards/db.py (onboarded flag + profile scope), dashboards/app.py (restructured),
scripts/test_dashboard.py (rewritten for v6; suite 76 → 96), docs/PROGRESS.md.
02_models.ipynb and all model artifacts untouched.

### (a) Purpose
Student's 10-point improvement list + a scenario-comparison mock-up: (1) optional
onboarding after registration; (2) profile = personal facts only (education, location,
experience, skills — job title/category/employment type stay prediction-specific);
(3) proper top navigation (Predict | What-if Analysis | History | Profile | About the
model) instead of the sidebar radio; (4) Logout/Change-password in a top-right user
menu; (5) no prediction form on History/comparison views; (6) sidebar inputs grouped
into sections; (7) waterfall relabelled "Advanced model explanation"; (8) explanations
as concept sentences ("Job seniority increased the estimate.") instead of per-feature
phrases ("'HR manager' in your job title"); (9) "Career improvement opportunities"
that never recommends role-irrelevant skills; (10) per-skill evidence: % of
higher-paying ads for the role + supporting-ad count + model RM difference. Mock-up:
comparison = what-changed diff table, one-line takeaway, ONE range chart, experience
curves demoted to optional detail. Decisions confirmed by student: What-if page =
scenario comparison (career advice stays on Predict); title-level skill evidence with
category fallback; onboarding for new registrations only.

### (b) What was built
1. **01_features Step 8 — models/skill_stats.joblib (50 KB).** Per job category (29
   groups; Self Employment n=1 is degenerate and skipped) and per job title with ≥30
   usable ads (81 groups, keyed lowercase): "higher-paying" = ads above the group's own
   median salary; per skill, (count, share) of higher-paying ads mentioning it. Printed
   evidence: higher-paying 'hr manager' ads n=25 — leadership 64% (16 ads), payroll 52%;
   ICT n=1,340 — sql 22.8% (306 ads).
2. **db.py**: `users.onboarded` column (PRAGMA migration, DEFAULT 1 so every existing
   account is grandfathered; add_user inserts 0), needs_onboarding/mark_onboarded;
   save_profile/load_profile now write/read ONLY state/experience/edu/skills (old
   columns kept in the table, written empty — pre-v6 DBs keep working).
3. **app.py layout**: st.segmented_control top nav (key `nav`, 5 pages; None → Predict
   because a segmented control can be de-selected); top-right st.popover user menu
   (signed-in name, Change password expander, Logout) or a guest popover with
   Log-in/Register; the sidebar form renders ONLY on the Predict page, grouped into
   💼 Job details / 🎓 Your background / 💰 Salary check. Form values survive page
   switches via the documented keep-alive idiom (`st.session_state[k] =
   st.session_state[k]` for the 8 form keys, before any widget exists) — replaces v5's
   "render the form on every page" workaround. Employment type now defaults to
   "Full time" (91% of ads) and defaults are pre-seeded via session state, which also
   removed the long-standing "default value + Session State API" console warnings.
4. **Onboarding**: after login, accounts with onboarded=0 get one optional setup card
   (education, state, experience slider, skills) instead of the app; "Save & continue"
   writes the profile + marks onboarded, "Skip for now" only marks; both are on_click
   callbacks (still zero st.rerun in the app). Saved values prefill the form in the
   same run. Login prefill itself now sets only the personal fields.
5. **Pages**: History = table + delete/clear only, with a pointer to What-if.
   Profile = editor for the four personal fields (seeded from DB on first render,
   Save persists + confirmation). About the model = comparison table from
   model_comparison.csv (winner marked 🏆), how-to-read bullets, honest limitations —
   all numbers read from artifacts at runtime.
6. **Concept-level XAI**: feature_group() maps each of the 932 encoded features into
   8 concepts (job seniority = TF-IDF n-grams containing a level word — senior, manager,
   executive, intern...; type of role = other title n-grams; experience; education;
   skills incl. skill_count; location; category; employment type). SHAP values are
   summed per group in log space, converted with the same math as v5 (RM =
   point·(1−exp(−v)), % = exp(v)−1), and rendered as sentences whose VERB comes from
   the RM size: ≥RM250 "increased/reduced the estimate", RM50–250 "had a small effect",
   <RM50 "had limited influence" (▲/▼/• markers + "≈ ±RM (±%)" captions). The factor
   bar chart now shows the same concept groups; the per-feature bars + waterfall moved
   into the renamed "Advanced model explanation" expander (locked decision 7 kept).
   friendly_phrase()/split_factors()/human_label() deleted.
7. **Career improvement opportunities**: skills lever = the same one-batch re-predict
   of all absent skills, then a relevance filter from skill_stats — evidence group =
   exact title if in by_title else category; keep a skill only if uplift > RM50 AND
   share ≥ 15% AND count ≥ 10 (title) / 20 (category); top 3 shown as bordered cards
   ("**leadership** — Common in **64%** of higher-paying HR Manager advertisements. /
   Model-estimated difference: approximately **+RM 1,341**/month. / Based on 16
   higher-paying advertisements...") with an honest fallback when nothing passes.
   Uplift bar chart dropped (the cards carry the numbers). Experience/education levers
   unchanged from v5.
8. **What-if page**: scenario picker = st.multiselect over saved rows (max 3) —
   replaces dataframe row-selection, so AppTest can drive the real path (the
   st.dataframe monkeypatch is gone from the tests). Output per the mock-up:
   what-changed diff table (only differing factors + Salary estimate row, columns
   Scenario A/B/C), takeaway via st.info computed from the concept-group SHAP diffs
   between the highest and lowest scenario ("Scenario B is RM 1,034 higher than
   scenario A, mainly because of experience and education."), ONE range chart,
   experience curves inside an "optional detail" expander.

### (c) Verification
- scripts/test_dashboard.py: **96/96 checks pass** (was 76). New coverage: 5-page nav
  (AppTest reports segmented-control labels with the emoji split into the icon slot);
  user menu; form hidden on History/What-if/Profile/About and values surviving the
  round-trip; onboarding shown once → skip/save paths, grandfathering, profile-subset
  write + immediate prefill; Profile page save/reload; personal-only prefill (job title
  stays empty); concept sentences present + old phrasing absent; "Advanced model
  explanation" expander; evidence cards, and every displayed skill re-verified against
  skill_stats.joblib thresholds inside the test; What-if diff table (only differing
  factors), takeaway wording, both charts; v3-era DB migration now also checks
  users.onboarded + grandfathering. Console clean (no deprecation/session-state
  warnings; use_container_width → width="stretch").
- Rendered PNGs extracted by wrapping st.pyplot in AppTest: hero band with offer tick,
  concept bars (7 groups, labels collision-free), advanced feature bars, waterfall,
  experience curve, comparison range chart + curves. HR Manager sample: estimate
  RM6,027 (range 6,027–9,006 — point clipped to P25, known behaviour), recs =
  leadership +RM1,341/64%/16 ads, payroll, communication.
- Headless boot: streamlit run → /healthz 200.

### (d) Design decisions
- **Keep-alive idiom instead of always-rendered form**: requirement 5 forbids the form
  on other pages; re-assigning the widget keys each run marks them app-owned so
  Streamlit stops garbage-collecting them. Predict-from-anywhere still works because
  the Predict callback jumps nav home before the rerun.
- **segmented_control over st.navigation(position="top")**: keyed widget → the
  existing nav-jump callback pattern keeps working; no multipage refactor of the
  auth/hub logic; AppTest drives it natively.
- **Title stats need ≥30 ads** (81 titles qualify): a percentage over fewer
  higher-paying ads (~15) is noise; anything rarer falls back to the 29 categories.
  Thresholds (15% share; 10/20 ads; RM50 uplift) are named constants next to the
  other honesty thresholds.
- **"Higher-paying" is measured within the group** (above the group's own median), so
  the share answers "is this skill common in the better-paid ads OF THIS ROLE", not
  "is it common in high-paying jobs generally".
- **Onboarding gate sits after artifact loading** (its widgets need state/skill lists)
  and before the pending-save/prefill blocks, so a guest's "log in to save" intent
  survives the onboarding step and auto-saves afterwards.
- Takeaway drivers = concept groups whose SHAP-sum diff pushes the higher scenario up
  (positive diffs only, top 2) — naming a group that pulls the other way would read
  as nonsense.

### (e) Limitations / student actions
- Postgraduate evidence, Malay education forms, clipped-point RM attribution: all
  pre-existing limitations from v4/v5 still apply.
- Title-level evidence covers only titles with ≥30 usable ads (81 titles, but they are
  the most common ones); everyone else gets category-level sentences — the card text
  always names which group was used.
- The diff table shows skills as counts ("5 skills"), not the lists — full lists are
  in the History table; keeps the mock-up's at-a-glance layout.
- An account registered BEFORE v6 never sees onboarding (grandfathered by design);
  the Profile page offers the same fields at any time.
- STUDENT: click through in a real browser once — top nav, user menu popover,
  onboarding with a freshly registered account, Profile page, What-if picker — and
  re-screenshot the report figures showing the dashboard UI (layout changed again) and
  the new career-advice cards; the 01_features figures gain a Step 8 output worth one
  screenshot (skill evidence per role).

---

## Dashboard v6.1: clean form for new accounts, emoji cleanup, disclaimer-style About (2026-07-16)

Files: dashboards/app.py, scripts/test_dashboard.py (suite 96 → 105), docs/PROGRESS.md.
No notebook, db.py or artifact changes.

### (a) Purpose
Student's browser test of v6 found a bug + three polish requests: (1) a freshly
registered account that SKIPPED onboarding still showed pre-filled prediction inputs
("don't know from where"); (2) remove emojis everywhere except the 💼 beside the app
title; (3) remove the "Powered by … typical prediction error ≈ RM 963/month" caption;
(4) rewrite "About the model" as a short disclaimer — no model comparison table, no
limitations list, no 31,406-ads count.

### (b) Bug root cause + fix
The v6 keep-alive re-assigns the 8 form widget keys on every app run, and only logout
wiped them — nothing reset the form when a DIFFERENT user (or a brand-new account)
entered the app in the same browser session. Probing with AppTest showed Streamlit
itself drops widget state after ONE full run without the form, so multi-run flows
(register: 2+ hub runs) self-clean — but the one-run flow (guest fills form → clicks
"Log in / Register" → logs in immediately; the login submit run falls through the hub
into the app) genuinely leaks, and any values re-owned by the keep-alive while logged
in leak too (the student's screenshot showed exactly the profile-prefill field set —
education/state/skills/experience — with job fields clean). Two-layer fix:
1. **Fresh-login reset**: the once-per-login prefill block now FIRST pops all form
   keys, then applies the user's own profile — unless an unsaved guest prediction is
   on screen (`last_result` present), which keeps the v2 "log in to save" contract.
2. **Onboarding wipe**: `complete_onboarding` (both Save & Skip) pops the form keys
   AND `last_result` unless `pending_save` is set — a brand-new account always starts
   clean, even from a guest result it never asked to keep.

### (c) Polish changes
- Emojis removed from: nav labels (segmented control now shows plain
  Predict / What-if Analysis / History / Profile / About the model), page subheaders,
  sidebar section headers, result-section headers, lever bullets, buttons, popover
  labels (now just the user's name / "Guest"), onboarding + welcome cards. Kept: 💼 in
  page_icon and both st.title calls, and the semantic ▲/▼/• direction markers.
- Transparency caption deleted; with the About rewrite this left MODEL_DISPLAY,
  best_model and the model_comparison.csv artifact unused → all removed from app.py
  (ARTIFACT_FILES no longer lists the CSV; load_artifacts is joblib-only again).
  models/model_comparison.csv itself still exists for the report.
- About page = 3 short paragraphs: model trained on real Malaysian job ads (no count),
  what the band/estimate/explanations/skill-evidence mean, and a bolded "guidance,
  not a promise" disclaimer close. No tables, no numbers.

### (d) Verification
- scripts/test_dashboard.py: **105/105 checks pass.** Updated: plain NAV constants
  (AppTest emoji-strip mapping gone); caption check inverted ("typical prediction
  error" and "31,406" must NOT render); About = disclaimer wording present, report
  content absent, zero dataframes; career-lever check uses "**Education**". New 6b
  regression block: (i) the one-run leak is first PROVEN (guest job title still in
  session on the hub), then shown wiped by a demo_user login; (i-b) register →
  onboarding → Skip lands on a fully clean form; (ii) pending-save contract — the
  guest's prediction is auto-saved to the brand-new account after onboarding and the
  result stays rendered (the form itself legitimately resets during the multi-run
  registration — Streamlit drops widget state after one form-less run); (iii) logout
  from the popover menu still wipes the form.
- AppTest quirk found while testing: streamlit-authenticator's logout applies its
  state changes MID-run (an `if st.button(...)` pattern, not a callback), so the
  click run still renders the logged-in UI and the hub appears one rerun later — the
  logout test settles with one extra at.run(). The earlier "returns to hub" check had
  passed spuriously ("Repeat password" also exists in the change-password widget);
  it now asserts on the hub-only "Continue as guest" button.
- Headless boot: streamlit run → /healthz 200.

### (e) Limitations / student actions
- A guest's typed-but-unpredicted inputs are now intentionally discarded on ANY login;
  only a guest with an unsaved prediction keeps their inputs+result. Worth one line in
  the report's user-flow description.
- The exact browser path that produced the student's screenshot could not be replayed
  under AppTest (widget-state GC differs by run count), but both plausible sources —
  one-run leaks and keep-alive-owned values — are now cleared by the fresh-login
  reset, and the invariant "every login starts from defaults or your own profile"
  holds by construction.
- STUDENT: re-test the register → skip flow in the browser; the form should now show
  only defaults (Full time, 2 years, empty skills). Re-screenshot the About page for
  the report if it was already captured (now a short disclaimer).

---

## Dashboard v6.2: student's About copy + truly blank default form (2026-07-16)

Files: dashboards/app.py, scripts/test_dashboard.py, docs/PROGRESS.md. Small follow-up
to v6.1; no notebook/db/artifact changes.

### (a) What changed and why
1. **About page**: the three paragraphs replaced verbatim with the student's own copy
   (wording deltas: "pay, **and** the single figure is the **trained model's** best
   estimate"; "no model can see **such as** company budgets…"; "**So,** use the
   estimates…"). The bold on "Please treat the numbers as guidance, not a promise."
   kept — only wording was replaced, not styling.
2. **"Onboarding wipe still not working — new user sees 2 years."** Root cause found:
   the wipe DOES fire; v6.1's form pre-seed (`experience_input = 2`, a "sensible
   default" carried from v1's slider) refilled the key the moment the wiped form
   re-rendered — indistinguishable from leftover data to a fresh user. Fix: the
   experience pre-seed is deleted (st.slider naturally defaults to its min, 0) along
   with the redundant salary pre-seed (st.number_input also defaults to min 0). Only
   the `type_input = "Full time"` seed remains (91% of ads; not the natural first
   option). A blank form now reads: no title, Accounting (first category), Full time,
   0 years, Not specified, no skills, RM 0 — nothing that looks like residue.

### (b) Verification
- scripts/test_dashboard.py: **105/105 checks pass** (count unchanged — existing
  checks extended, not added): both clean-form regressions (fresh-login reset and
  onboarding-skip) now also assert `experience == 0`; the About disclaimer check also
  requires the new "trained model's best estimate" phrase.
- Headless boot: streamlit run → /healthz 200.

### (c) Notes
- The default prediction (nothing changed by the user) now assumes 0 years instead of
  2 — estimates for an untouched form shift accordingly; all displayed numbers still
  come from the same model.
- v6.1's student-action note said the clean form shows "2 years" — superseded: it now
  shows 0 years.

---

## Dashboard v6.3: auto-login after register, multi-word usernames, demo/technical copy removed (2026-07-17)

Files: dashboards/app.py, scripts/test_dashboard.py (suite 105 → 114), docs/PROGRESS.md.
No notebook, db.py, emailer.py or artifact changes.

### (a) Purpose
Student's "professional system, not a demo" list: (1) auto-login right after
registration; (2) usernames of 2–3 words ("Lulu Man") must be able to register;
(3) remove the demo-accounts hint from the landing page; (4) remove the blue-band
percentile caption, the "RM effects are approximate / background sample" caption and
the what-if "Re-evaluated with the current model" caption; (5) no user-visible mention
of 25th/75th percentiles anywhere — users should not be shown how the range is computed.

### (b) What changed
1. **Multi-word usernames.** Root cause: streamlit-authenticator 0.4.2's default
   `Validator.validate_username` regex `[a-zA-Z0-9_-]{1,20}` has no space, so any
   two-word username was rejected at registration. Fix: `MultiWordUsernameValidator`
   subclass in app.py (1–3 words of `[a-zA-Z0-9_-]{1,20}` separated by single spaces)
   passed to `stauth.Authenticate(validator=...)`. Verified in library source: the
   controller lowercases + strips usernames on BOTH register and login, so "Lulu Man"
   is stored and logged in as "lulu man" — consistent either way; SQLite PK unaffected.
2. **Auto-login after registration.** After `db.add_user(...)` the register tab now
   calls `authenticator.authentication_controller.login(token={"username": ...})` —
   the library's own cookie-restore path, which fills the same session-state keys a
   form login does — then `cookie_controller.set_cookie()` so a browser refresh stays
   logged in. The existing hub fall-through (`hub.empty()` when authentication_status
   turns True mid-run) shows the app in the same run; a new account lands on the
   onboarding card as before. The old "you can now log in" success was removed (the
   hub it rendered in is wiped in that same run); instead a one-shot `just_registered`
   flag shows "Account created — you are signed in as **Name**." in the app body,
   above onboarding. The guest pending-save contract is untouched: the flag survives
   onboarding and the prediction still auto-saves (re-tested).
3. **Copy removals** (all rendered strings, code comments kept): demo-accounts caption
   on the login tab; "Half of comparable job ads pay inside the blue band … 25th–75th
   percentile" hero caption; "RM effects are approximate … background sample" caption
   in Why-this-estimate; "Re-evaluated with the current model …" caption on What-if;
   comparison range chart x-label "(P25 – P75)" dropped → "Predicted monthly salary
   range". Kept: the "A typical Malaysian job ad pays around RM X" anchor sentence and
   the student's own About copy ("where half of comparable advertisements pay" — no
   percentile language).

### (c) Verification
- scripts/test_dashboard.py: **114/114 checks pass** (was 105). Registration test
  rewritten around a shared `do_register` helper: auto-login asserted
  (authentication_status True + username set + onboarding card + welcome success in
  the SAME session, no manual login); duplicate/weak-password attempts moved to a
  fresh hub (the old session is now logged in). New: "Lulu Man" registers and lands
  logged in as "lulu man"; stored lowercased in SQLite; fresh-session login typing
  "Lulu Man" works; a four-word username is rejected and not persisted. 6b flows
  drop their post-register `do_login` calls (register lands logged-in now) — the
  stale-form wipe and pending-save auto-save still pass unchanged. New absence
  checks: hub has no "Demo accounts"/"demo123"; results page has no "percentile"/
  "blue band"/"P25"/"P75"/"RM effects are approximate"/"background sample"; What-if
  has no "current model" caveat.
- Headless boot: streamlit run → /healthz 200, console clean.
- Grep over app.py rendered strings: zero hits for all removed phrases.

### (d) Design decisions
- **Token login path, not hand-set session keys**: `login(token=...)` is what the
  library itself runs when restoring a cookie, so every key (email/name/roles/
  authentication_status/username) is set exactly as a real login sets them, and the
  logged_in bookkeeping stays consistent. `set_cookie()` is the same call the login
  widget makes on success (already exercised under AppTest, so no new harness risk).
- **Username rule stays conservative**: same character class as the library default,
  just 1–3 space-separated words, ≤20 chars per word — nothing that could break the
  SQLite PK, the credentials dict, or the cookie payload.
- The welcome message lives in the app body behind a one-shot flag because anything
  rendered inside the hub container is wiped by `hub.empty()` in the registration run.

### (e) Limitations / student actions
- Usernames are still lowercased by the library on registration AND login — "Lulu
  Man" displays as username "lulu man" (the display NAME keeps its casing; only the
  login identifier is lowercase). Worth one line in the report's user-management text.
- The removed captions carried honest caveats (quantile meaning of the band, SHAP
  interaction approximation, retrain drift on What-if). The information is gone from
  the UI by explicit product decision; the report can still state these caveats —
  the numbers themselves are unchanged.
- AppTest cookie noise ("Invalid token type. Token must be a <class 'bytes'>") now
  also appears after registration runs (set_cookie in a harness without a browser) —
  same pre-existing, cosmetic noise the login path always produced.
- STUDENT: in a real browser, register a throwaway 2-word account once — you should
  land signed-in on the onboarding card, and a page refresh should keep you signed
  in (cookie). Re-screenshot the landing page (demo hint gone) and the hero card
  (caption gone) for the report figures.

---

## Dashboard v7: SQLite → Supabase migration for live deployment (2026-07-17)

Files: dashboards/db.py (rewritten), dashboards/app.py (startup block + docstring),
dashboards/config.yaml (credentials + cookie key removed), .streamlit/secrets.toml
(new [supabase] + [auth] sections), requirements.txt (+supabase), CLAUDE.md
(environment + dashboard spec updated), scripts/test_dashboard.py (suite 114 → 113),
docs/DEPLOYMENT.md (new), data/dashboard.db (deleted). No notebook, model-artifact,
emailer or prediction/UI logic changes.

### (a) Purpose
The app is being deployed to Streamlit Community Cloud, whose container filesystem
is ephemeral — the local SQLite file would be wiped on every redeploy/restart, losing
all accounts and history. Persistence moved to a managed Postgres database on the
Supabase free tier. Student decisions (via questions): official supabase-py client
(not raw SQL over psycopg2); complete fresh start with the demo accounts
(demo_user/supervisor) REMOVED entirely — no seeding, all accounts come from in-app
registration; deploy target Streamlit Community Cloud. Auth stays with
streamlit-authenticator — Supabase is only the data store, not the auth system.

### (b) What was built
1. **db.py rewritten** (the 11 surviving public functions keep their exact
   signatures, so app.py's ~18 call sites are untouched): the supabase client (REST over HTTPS — no connection
   strings/pooling) is created once behind @st.cache_resource; `_client()` is a
   late-bound accessor so tests can swap in a fake; `DatabaseError` raised when
   [supabase] secrets are missing. `init_db()` DELETED — the schema is created once
   in the Supabase SQL editor (DDL in docs/DEPLOYMENT.md), and there is no seeding.
   SQLite-isms replaced: AUTOINCREMENT → identity column (insert returns the new id
   in the REST response, replacing cursor.lastrowid); INSERT OR REPLACE → upsert;
   PRAGMA migration guards dropped (no legacy DBs to support — profiles also drops
   the v6-era empty legacy columns, and users.onboarded becomes a real boolean).
   `list_predictions` orders by created_at DESC then id DESC (stable within one
   second) and returns an empty DataFrame WITH the 15 expected columns when a user
   has no rows (the REST API returns [] — no column names to infer, and the History
   page needs them). `_now()` stamps rows in Asia/Kuala_Lumpur time minus the offset
   suffix — the deployed server runs in UTC, which would otherwise show history
   times 8 h behind Malaysia. delete_predictions int()-casts ids (numpy int64 from
   the dataframe selection is not JSON-serialisable).
2. **Security model**: RLS enabled on all three tables with NO policies → the
   public/publishable key can read nothing; the app uses the SECRET key, which
   bypasses RLS and lives only in .streamlit/secrets.toml (git-ignored) locally and
   the Secrets panel on Streamlit Cloud. The cookie SIGNING key also moved out of
   the committed config.yaml into st.secrets ([auth] cookie_key) — the old key was
   burned (it is in git history), a fresh random one was generated; config.yaml now
   holds only the cookie name + expiry. app.py wraps the startup
   `db.load_credentials()` and the cookie-key read in friendly st.error + st.stop()
   gates (same pattern as the missing-artifacts check).
3. **Test suite**: the seam moved from `db.DB_PATH = <temp file>` to
   `db._client = lambda: FAKE_DB` — an in-memory FakeSupabaseClient (~100 lines)
   implementing exactly the query-builder slice db.py uses
   (table/select/insert/upsert/update/delete/eq/in_/order/execute, auto-increment
   ids, PK enforcement, stable multi-key ordering). All real db.py code paths still
   run; only HTTP is faked. demo_user/supervisor now exist ONLY as pre-seeded rows
   in the fake store (bcrypt hashes moved from config.yaml into the test script) so
   the login-driven checks kept working unchanged. The cookie key is injected via
   `at.secrets["auth"]` in fresh_app() (32+ chars, else PyJWT logs a key-length
   warning). The v3-era SQLite migration test (4 checks) was removed — schema is no
   longer app-managed — replaced by 3 db-layer checks: empty history carries the
   full column set, unknown user → profile None, upsert keeps one profile row per
   user.
4. **docs/DEPLOYMENT.md** (new): step-by-step Supabase setup (project in the
   Singapore region, the full DDL, which key to copy and why NOT the publishable
   one, local verification) + Streamlit Community Cloud deploy (secrets panel,
   Python version) + gotchas (free tier pauses after ~1 week idle — restore from
   the Supabase dashboard before a demo).

### (c) Verification
- scripts/test_dashboard.py: **113/113 checks pass** (was 114: −4 migration,
  +3 db-layer), console clean (no key-length or session-state warnings; the known
  AppTest cookie noise remains). Register → auto-login → onboarding → predict →
  auto-save → history → what-if → profile → forgot/change password → logout all
  green against the fake client on the first full run.
- Headless boot: streamlit run → /healthz 200.
- Grep sweep: zero functional references to sqlite/DB_PATH/dashboard.db/demo
  accounts left in dashboards/ (test-fake seeds and explanatory comments excepted);
  data/dashboard.db (32 KB local test data) deleted per the fresh-start decision.
- NOT yet verified (needs the student's Supabase project): the real end-to-end
  smoke test — fill [supabase] secrets, register, save, check rows in the Table
  Editor, restart, log in again (steps in DEPLOYMENT.md Part A4).

### (d) Design decisions
- **supabase-py over raw SQL**: no session-pooler/IPv4 concerns on Streamlit Cloud,
  no connection-limit management on the free tier, one dependency instead of two
  (SQLAlchemy + psycopg2), and query calls read almost like the SQL they replace.
- **Same public db.py API**: keeps the migration surface to one module; every
  app.py call site and most tests unchanged.
- **Timestamps stay TEXT ISO strings** (not timestamptz): identical History-table
  display and lexicographic ordering as before; timezone correctness handled at
  write time via zoneinfo. Defensible simplification, noted as such.
- **No Supabase Auth**: the persistence problem needed a database, not a new auth
  system; streamlit-authenticator + bcrypt hashes in the users table carry over 1:1.
- **Demo accounts became test fixtures**: the requirement was "no demo accounts in
  the real system" — the fake store still seeds them so 20+ existing login checks
  did not need rewriting.

### (e) Limitations / student actions
- Every db call is now a network round-trip (~50–150 ms to Singapore) instead of a
  local file read — imperceptible for this single-user-per-session app; the client
  itself is cached, calls are not (auth data must stay fresh across sessions).
- The FakeSupabaseClient mimics, not proves, PostgREST behaviour — the real-network
  path is covered by the manual smoke test in DEPLOYMENT.md Part A4.
- Free-tier pause (~1 week idle) will make the deployed app show the friendly
  "could not reach the database" error until the project is restored — restore it
  before any demo/viva.
- STUDENT: (1) create the Supabase project and run the DDL (DEPLOYMENT.md Part A —
  ~10 min), fill [supabase] in .streamlit/secrets.toml, then run the Part A4 smoke
  test; (2) deploy per Part B and paste the secrets into the Streamlit Cloud panel;
  (3) the old demo accounts are gone — register your own accounts for the viva
  demo; (4) if the repo was ever public, consider the old cookie key compromised
  (already replaced) and never commit secrets.toml.

---

## Dashboard v7.1: change-password form rewritten — working reveal, fields that clear (2026-07-18)

Files: dashboards/app.py (user-menu block + two new callbacks), scripts/test_dashboard.py
(suite 113 → 133), docs/PROGRESS.md. No notebook, db.py, artifact or dependency changes.

### (a) Purpose
Two defects the student found in a browser: (1) the eye/reveal icon in the Change
password fields does nothing when clicked; (2) the Current / New / Repeat inputs stay
filled after a successful password change, and are still filled when the section is
closed and reopened. Student decision: keep the existing layout (user menu popover →
"Change password" section in the 26 % top-right column); fix the behaviour only.

### (b) Root causes
1. **Fields never cleared — confirmed in the library source.**
   `authenticator.reset_password()` (streamlit-authenticator 0.4.2,
   `views/authentication_view.py:527-591`) builds `st.form(key='Reset password',
   clear_on_submit=False)` with a CONSTANT form key, and creates its three
   `st.text_input`s with **no `key=`**. Their values therefore live under
   auto-generated widget IDs that never change and are unreachable from
   `st.session_state` — this app's `st.session_state.pop(key, None)` convention
   (`clear_result_on_logout`) structurally could not touch them. The success path
   (`models/authentication_model.py:585-615`) clears nothing either.
   Compounding it: the popover and the expander had no `key`, so collapsing them
   ran no Python at all — nothing could react to the section being closed.
2. **Eye icon — root cause NOT found.** In Streamlit 1.59.2 the reveal button is
   rendered unconditionally for every `type="password"` input and is pure client-side
   React state (`static/js/TextInput.C78pm77f.js`: `[z,B]=useState(!1)`,
   `ie=()=>B(e=>!e)`, input type `z?'text':...`), so no server-side code can break it
   and toggling it triggers no rerun. Nothing in the authenticator or in Streamlit
   suppresses it. Rather than guess at a frontend cause, the reveal was replaced with
   a control the app owns.

### (c) What changed (dashboards/app.py)
1. **The library's `reset_password()` view is no longer used** — only its
   `authentication_controller.reset_password(...)`, which does ALL the checking
   exactly as before: new password non-empty, both copies match, different from the
   current one, meets the password policy, and the current password verifies against
   the stored bcrypt hash. Security semantics are unchanged; only the presentation
   layer is now ours.
2. **Own widgets, own keys**: `pw_show` (checkbox), `pw_current` / `pw_new` /
   `pw_repeat` (text inputs), `pw_submit` (button), collected in
   `PASSWORD_WIDGET_KEYS`. Plain widgets, NOT `st.form` — a widget inside a form does
   not take effect until submit, and the reveal checkbox has to work immediately (the
   same reason the prediction form dropped `st.form` in v2).
3. **Reveal**: one "Show passwords" checkbox drives `type="default"` vs
   `type="password"` on all three fields. Server-side state, so it works by
   construction.
4. **`submit_password_change(user)`** — a button `on_click` callback (callbacks run
   before the next script run, the only moment Streamlit allows widget keys to be
   removed). Success → persist the hash via `db.update_password`, set
   `pw_message=("success", ...)`, pop all three fields. Failure → set
   `pw_message=("error", str(exc))` and KEEP the typed values, so only the wrong
   field needs retyping. The message is `pop`ped when rendered, so it does not linger
   into the next interaction.
5. **Clearing on close** — two independent mechanisms:
   - the popover and expander now take `key="user_menu"` / `key="pw_expander"` plus
     `on_change=clear_password_fields`; a keyed container with an `on_change` is what
     makes opening/closing it rerun the app and expose its state (Streamlit 1.59.2,
     `elements/layouts.py:988` and `:1319`);
   - the fields are rendered only when `password_section.open` is true, so a closed
     section stops rendering them and Streamlit drops their state — reopening always
     starts blank.
6. **`PASSWORD_HELP`** is read from `stauth.params.PASSWORD_INSTRUCTIONS` rather than
   retyped, so the on-screen rules can never drift from the enforced rule.
7. **Logout deliberately does NOT wipe the password keys** (comment in
   `clear_result_on_logout` says why): the library applies its logout mid-run, so the
   user menu is still on screen during that run and popping those keys would leave
   widgets whose state no longer exists — the v2 "ghost widget" crash. It is also
   unnecessary: once the entry hub takes over the form is not rendered, and Streamlit
   drops any widget state after one run without the widget. Verified by test.

### (d) Verification
- scripts/test_dashboard.py: **133/133 checks pass** (was 113), console clean (only
  the known AppTest cookie noise). New section 13 covers: section closed by default;
  opening renders three empty masked fields (proto type 1); "Show passwords" flips all
  three to plain text (type 0); wrong current password → error, typed values kept,
  stored hash untouched; mismatch → error; weak new password → the library's policy
  error; success → confirmation, all three fields empty, new bcrypt hash persisted,
  old password dead; the confirmation does not persist to the next run; typing then
  closing the section stops rendering the fields AND drops their session state, and
  reopening shows empty fields; logout after using the form is exception-free; the new
  password logs the account in. The section-2 check was split in two (the section is
  now collapsed by default, so the fields are legitimately absent until it is opened).
- Headless boot: `streamlit run` → /healthz 200, / 200, no startup errors.

### (e) Design decisions
- **Keep the library controller, replace only the view**: the widget's markup was the
  problem, not its validation. This keeps the bcrypt check and password policy
  identical to before while making the three inputs addressable.
- **Errors keep the typed values, success clears them**: the student asked for
  clearing on success; wiping all three fields because one character of the current
  password was wrong would be worse UX than the bug being fixed.
- **Two clearing mechanisms, not one**: the `on_change` callback is what fires in a
  browser, but AppTest cannot click an expander header, so the `.open` gating (which
  AppTest *can* drive through session state) is what makes the behaviour testable —
  and each mechanism is independently sufficient.

### (f) Limitations / student actions
- AppTest does not carry an expander's open state between runs (a browser does), so
  the tests re-assert `pw_expander` before each run via
  `run_with_password_section_open()`. The real click-to-collapse path is still a
  manual browser check.
- **Streamlit's own eye icon still renders while the fields are masked and may still
  do nothing** — its failure could not be reproduced or explained from the source, and
  hiding it would need the custom CSS that CLAUDE.md rules out. The "Show passwords"
  checkbox is the supported way to reveal them.
- STUDENT: in a browser — tick "Show passwords" (all three fields must unmask); enter
  a wrong current password (error, text kept); change it properly (confirmation, all
  three fields empty); type something, collapse the section, reopen (empty); log out
  and back in with the new password. If the built-in eye still does nothing, note the
  browser (Edge/Chrome/Firefox) and whether it was the local or the deployed app.
  Re-screenshot the user-menu figure for the report — the form has a new field label
  ("Repeat new password") and a "Show passwords" checkbox.

---

## Dashboard v7.2: email reset-link for Forgot password + change-password polish (2026-07-18)

Files: dashboards/app.py, dashboards/db.py (3 new functions), dashboards/emailer.py
(function replaced), scripts/test_dashboard.py (suite 133 → 148), docs/DEPLOYMENT.md,
.streamlit/secrets.toml (+[app] url), docs/PROGRESS.md. No notebook, artifact or
dependency changes (secrets/hashlib/time are stdlib).

### (a) Purpose
Student's four requests after browser-testing v7.1: (1) forgot password should take an
EMAIL and send a reset LINK the user opens to set their own password (old flow: enter
username → a random new password is generated and emailed); (2) remove the built-in
reveal eye from the Change-password fields ("Show passwords" is the working reveal);
(3) move "Show passwords" below the inputs; (4) the change-password fields STILL
showed old input in the browser after a successful change + collapse + reopen, despite
v7.1's fix passing all AppTest checks. Student decisions (asked): a successful reset
signs the user straight in; an unknown email gets an explicit
"No account uses this email address." error.

### (b) Root cause of (4) — pop vs assign
v7.1 cleared the fields by POPPING their session keys. Popping deletes only the
server-side copy: the browser's widget manager still remembers the typed text for
those widget IDs and re-reports it on the next sync, resurrecting the old input.
AppTest rebuilds widget state from the element tree on every run — after a pop the
rebuilt tree carries the default "" — which is exactly why the suite passed while the
browser kept the text. Fix: every clearing path now ASSIGNS empty values
(`st.session_state["pw_current"] = ""` …), which marks the keys app-owned and pushes
the empty value down to the frontend — Streamlit's documented clear-an-input recipe.
Belt-and-braces: all password inputs now pass `autocomplete="off"` (what the library's
own widgets do) so a browser password manager cannot refill them either. Bonus of
assign-semantics: `clear_result_on_logout` can now clear the password form too
(assignment cannot cause the v2 ghost-widget crash that popping rendered widgets
could), so a half-typed password no longer survives logout.

### (c) Change-password polish
- Eye icons removed via the app's ONE style override (student-approved exception to
  the "no custom HTML/CSS" rule, noted in the module docstring): a scoped
  `st.html("<style>…")` hides the reveal button inside the two keyed containers
  `st-key-pw_fields` / `st-key-reset_fields` only — Streamlit offers no parameter to
  disable the eye, and the checkbox is the reveal that actually works.
- "Show passwords" moved BELOW the fields; its value is read from session state
  before the inputs render (clicking it reruns the app first), so the ordering is
  purely visual.

### (d) Forgot password = single-use emailed reset link
1. **Hub tab**: email input + "Send reset link". The username is looked up in the
   already-loaded credentials dict (emails are UNIQUE; case-insensitive compare; the
   dict reloads every run, so no extra network call). Invalid format / unknown email →
   explicit errors. Known → `secrets.token_urlsafe(32)` token; the users row stores
   its SHA-256 hash + `now + 1800s` expiry (columns reset_token_hash /
   reset_token_expires); link = `{APP_URL}/?reset_token={token}`.
   `emailer.send_reset_link_email` (replaces send_password_email) mails it; on SMTP
   failure the link is shown on screen — same never-dead-end fallback as before.
2. **Landing page** (new block BEFORE the entry hub, first `st.query_params` use in
   the app): unknown-or-expired tokens share one message ("invalid or has expired" —
   probing tokens reveals nothing) + a "Back to log in" button that clears the URL.
   Valid → New/Repeat password + Show passwords (same pattern/CSS as the change form),
   policy enforced by the SAME shared validator instance the registration widget uses;
   `stauth.Hasher.hash` → `db.update_password`, token cleared (single use), then
   **auto sign-in** via `authentication_controller.login(token=…)` + `set_cookie()` —
   the exact v6.3 post-registration pattern — and `st.query_params.clear()`; the next
   run falls into the app with a one-shot "Password changed — you are signed in"
   message (`just_reset` flag, rendered beside `just_registered`).
3. **db.py**: set_reset_token / get_reset_request / clear_reset_token (plain
   update/select on users, same style as update_password). Token at rest is a hash:
   a leaked database row cannot be turned into a working link.
4. **Config**: `APP_URL` from new `[app] url` secret (localhost fallback) — the
   emailed link must point at the deployed address, which the server cannot discover
   itself. DEPLOYMENT.md: users DDL gains the two columns, a boxed ALTER TABLE note
   for the student's EXISTING project, [app] url documented for local + Cloud, gotcha
   reworded. The library's forgot_password widget is no longer used anywhere (its
   Helpers.generate_random_string is random-module-based; the token uses the
   cryptographic `secrets` module instead).

### (e) Verification
- scripts/test_dashboard.py: **148/148 checks pass** (was 133), console clean.
  Section 7 rewritten: link fallback on screen; link carries reset_token; only the
  hash is stored (expiry in future, password hash untouched by the request); unknown
  email + bad format errors; landing page renders instead of the hub (driven via
  AppTest's query_params support); weak/mismatched passwords rejected on the landing
  page; success → signed in as the user, confirmation shown, new bcrypt hash
  persisted, old password dead, token nulled; REUSED link rejected; EXPIRED link
  rejected (expiry forced into the past); the new password logs in through the normal
  hub. Section 13 additions: autocomplete=off on all three fields; the scoped
  eye-hiding style block renders; "Show passwords" verified BELOW the inputs by
  walking the element tree in render order. The monkeypatched emailer function is now
  send_reset_link_email (tests can never send real email).
- Headless boot: /healthz 200, / 200, no startup errors.

### (f) Limitations / student actions
- The pop-vs-assign browser behaviour cannot be replayed under AppTest (the harness
  has no persistent frontend widget manager) — the fix follows Streamlit's documented
  callback-assignment pattern; final confirmation is the student's browser test below.
- One reset request per account (a newer link replaces the older one) — fine for this
  system, noted for the report.
- The on-screen link fallback reveals account existence when SMTP is down; with SMTP
  configured the flow only reveals it via the explicit unknown-email error, which the
  student chose deliberately (report can note enumeration as a known trade-off).
- STUDENT ACTIONS: (1) run the ALTER TABLE from DEPLOYMENT.md A2 in the Supabase SQL
  editor (the two reset columns do not exist in your live project yet — forgot
  password will error until then); (2) after deploying, set `[app] url` in the Cloud
  secrets panel to the public app address; (3) browser test: request a link with your
  real Gmail SMTP, open it, set a weak then a valid password, confirm you land signed
  in, then try the same link again (must be rejected); (4) change-password: confirm no
  eye icons, checkbox under the fields, and that after a successful change +
  collapse/reopen the fields are finally EMPTY in the browser; (5) re-screenshot the
  user-menu and forgot-password figures for the report.

---

## Dashboard v7.3: logout crash fix (StreamlitAPIException on the deployed app) (2026-07-18)

Files: dashboards/app.py (clear_result_on_logout only), scripts/test_dashboard.py
(suite 148 → 150), docs/PROGRESS.md. Nothing else changed.

### (a) Symptom
On the deployed app (v7.2), clicking **Logout** while the "Change password" section
was open crashed the whole page with `streamlit.errors.StreamlitAPIException`
(student's screenshot: traceback through `authenticator.logout(...)` → library
callback → our `clear_result_on_logout`).

### (b) Root cause
Three facts combined:
1. streamlit-authenticator applies logout **mid-run** (an `if st.button` check, not
   an on_click callback — documented here since v6.1). So `clear_result_on_logout`
   executes in the MIDDLE of a script run, unlike every other callback in the app.
2. At that point the change-password widgets are already instantiated in the same run
   whenever the section is open — they render above the Logout button in the popover.
3. v7.2 made `clear_result_on_logout` call `clear_password_fields()`, which ASSIGNS
   `st.session_state["pw_current"] = ""` etc. Streamlit's session state forbids
   assigning to a key whose widget was already instantiated this run
   (`session_state.py` `__setitem__`: "`st.session_state.<key>` cannot be modified
   after the widget with key `<key>` is instantiated."). Reproduced verbatim.
   Crucially, `__delitem__` has NO such guard — which is why the POPs in the same
   callback (form/onboard/profile keys) have always been safe, and why logout never
   crashed before v7.2 added the one assignment path.

The crash needs the section OPEN at logout time (otherwise the widgets are not
instantiated and assignment is legal) — exactly the student's screenshot state.

### (c) Why the 148-check suite missed it
AppTest does not persist an expander's open state between runs (the very reason the
suite's `run_with_password_section_open()` helper re-asserts it before every run).
In the v7.2 logout test the section had auto-collapsed by the click run, so the
fields were never instantiated and the assignments were legal. A real browser keeps
the expander open — fields instantiated — crash. The new regression test closes this
gap by re-asserting `pw_expander = True` in the same run batch as the Logout click;
against the v7.2 code it reproduces the exact StreamlitAPIException (verified before
fixing), against the fixed code it passes.

### (d) Fix
`clear_result_on_logout` no longer calls `clear_password_fields()`; `pw_message`
returned to its popped-keys list; docstring now states the rule (only pops in this
callback, never assignments, because it runs mid-run). No privacy regression: the
password fields cannot leak to the next visitor because `clear_password_fields`
still fires as the on_change of BOTH the user-menu popover and the expander — real
pre-run callbacks — so opening the menu or the section always clears the fields
before they render; widget-owned leftovers are additionally dropped by Streamlit's
own GC during the hub runs. `clear_password_fields` keeps exactly three callers,
all pre-run: popover on_change, expander on_change, submit_password_change.

### (e) Verification
- Reproduction first: standalone probe (section open during the logout click run)
  raised `StreamlitAPIException: st.session_state.pw_current cannot be modified
  after the widget with key pw_current is instantiated` against the v7.2 code;
  zero exceptions after the fix.
- scripts/test_dashboard.py: **150/150 checks pass** (was 148). New: logout with the
  password section open raises no exception; password fields are empty on the next
  login (proves the logout wipe was unnecessary).
- Headless boot: /healthz 200, / 200, boot log clean.
- STUDENT: commit + push, let Streamlit Cloud redeploy, then on the live app: log
  in, open Change password, click Logout — must land on the entry hub with no error
  page. (This is the same push that carries v7.2, so run the DEPLOYMENT.md ALTER
  TABLE and the [app] url Cloud secret first if not done yet.)

### (f) Lesson recorded for the report
Streamlit session-state writes are only legal for widgets not yet instantiated in
the current run; the one library callback that runs mid-run (logout) therefore must
never assign widget state. The set/del asymmetry (assign guarded, pop exempt) is
why the bug hid behind otherwise-identical cleanup code, and the AppTest/browser
difference in expander-state persistence is why the suite could not see it.

---

## docs/ARCHITECTURE.md: text-based system architecture (2026-07-19)

Files: docs/ARCHITECTURE.md (new), docs/PROGRESS.md. Documentation only — no code,
notebook, artifact or dependency changes.

### (a) Purpose
The student needs a system architecture figure for the FYP report (reference samples:
layered diagrams with labelled arrows — data serving vs offline processing bands).
They will draw the picture themselves; this document supplies the content and flow:
which boxes exist, what goes in each layer, and what label each connector carries.

### (b) Structure of the document
Written so each section maps 1:1 to a diagram element:
1. **Offline training pipeline** (band): CSV → 01_features (experience/education/
   skill extraction, suggestion + skill-evidence artifacts) → 02_models (4-model
   comparison, RF winner MAE ≈ RM963, quantile P25/P75 pipelines, SHAP explainer)
   → models/ artifact store, with the "app loads only artifacts, never the CSV" arrow.
2. **User interface layer**: entry hub (login/register/forgot/guest), onboarding
   card, 5-page top nav, sidebar form groups, result views.
3. **Application layer** (app.py, one Streamlit process) as 6 sub-boxes:
   authentication manager, prediction engine, XAI engine (8-concept grouping),
   career advice engine (3 levers), what-if comparison, session/state manager —
   plus the db.py and emailer.py helper modules.
4. **Data & external services**: Supabase Postgres (users / predictions / profiles,
   RLS locked, secret key), Gmail SMTP with on-screen fallback, secrets/config split.
5. **Deployment view**: browser → Streamlit Community Cloud → Supabase (Singapore)
   + Gmail; offline band stays on the developer machine.
6. **Five numbered end-to-end flows** (login, predict, save/history/what-if,
   forgot-password, profile/account maintenance) — the cross-diagram arrows.

### (c) Accuracy basis
Content cross-checked against the v7.3 state recorded in this log and the real file
listing (dashboards/ has exactly app.py, db.py, emailer.py, config.yaml; the 10
models/ artifacts are named as they exist on disk). Deliberately absent because they
no longer exist: SQLite/init_db, demo accounts, the model-transparency caption.
Numbers quoted (31,406 rows, MAE ≈ RM963, 84 input columns, 75 skills) match the
v4 retrain results.

### (d) Notes for the student
- Each "Box"/"Arrows" entry is meant to become one shape/connector in Whimsical or
  draw.io; the two-band offline/online split mirrors the sample figures.
- Edit freely — nothing in the app reads this file.

---

## Dashboard v7.4: profile autofill fix (assign-not-pop), copy updates, SHAP note (2026-07-20)

Files: dashboards/app.py, scripts/test_dashboard.py (suite 150 → 154),
docs/PROGRESS.md. No notebook, db.py, emailer.py, artifact or dependency changes.

### (a) Purpose
Student's browser test of the deployed app found: (1) profile autofill broken —
after login the form showed the PREVIOUS session's prediction inputs (even from a
different account on the same browser tab) instead of the logged-in user's saved
profile, and saving the profile did not fill the form in the same session;
(2) hero metric should read "Estimated advertised monthly salary"; (3) welcome
card copy replaced with the student's own wording; (4) the RM figures in "Why this
estimate?" do not add up to the displayed estimate. Student decisions (asked):
for (4) keep the per-factor math and add back an explanatory note (a similar
caption was removed in v6.3); for (1) the repro was both "saved profile, no
re-login" and the cross-account leak on the same tab.

### (b) Root cause of (1) — the v7.2 pop-vs-assign lesson, again
The v6.1 fresh-login reset POPPED the 8 form widget keys before applying the
profile. Popping deletes only the server-side copy: the browser's widget manager
still remembers values for those widget IDs and re-reports them on the next sync,
resurrecting the previous visitor's inputs and overriding the prefill — exactly
the mechanism documented for the change-password fields in v7.2. The logout run
makes the stale frontend state inevitable: streamlit-authenticator applies logout
mid-run, so the form renders once more in that run. AppTest rebuilds widget state
from the element tree each run (no persistent frontend), which is why all 150
checks passed while the browser failed — the same AppTest/browser gap as v7.2.
The onboarding wipe (complete_onboarding) had the same popping pattern.

### (c) What changed (dashboards/app.py)
1. `form_defaults()` + `reset_form_to_defaults()`: one blank-form value per widget
   key (title None, category first option, Full time, state first option, 0 years,
   Not specified, no skills, RM 0 — matches the v6.2 blank form). Every pre-run
   clearing path now ASSIGNS these values instead of popping: the fresh-login
   reset (then overlays the profile's personal fields, same logic as before) and
   complete_onboarding (pending-save guest contract kept; last_result is server
   state and stays a pop). Logout keeps popping only — v7.3 rule: it runs mid-run,
   where assigning to instantiated widget keys raises StreamlitAPIException; safe
   because the next login now assigns everything.
2. `save_profile_from_page` also writes the four personal values straight into the
   form keys, so a saved profile shows up on the Predict page immediately, not
   only after the next login (legal: the form never renders on the Profile page).
   Success message reworded accordingly.
3. Copy: hero metric label → "Estimated advertised monthly salary"; welcome card
   bullets replaced with the student's wording ("a verdict on a salary you were
   offered, whether it is below, within or above the market range", "the factors
   behind your estimate" — "in plain language" dropped).
4. "Why this estimate?": one new caption under the factor rows — each figure shows
   how much of the estimate rests on that factor by itself; the factors strengthen
   and offset one another, so adding them up will not reproduce the final estimate
   exactly. No math changes (concept grouping and point·(1−exp(−v)) untouched);
   wording deliberately avoids the v6.3-removed phrases the suite bans.

### (d) Verification
- scripts/test_dashboard.py: **154/154 checks pass** (was 150). New: welcome card
  v7.4 wording (and "in plain language" absent); metric label asserted exactly;
  the don't-add-up note renders while the banned v6.3 phrases stay absent; after a
  fresh login ALL 8 form keys are present in session state (assignment semantics —
  nothing left popped for a browser to resurrect); Profile save applies to the
  form in the SAME session (Profile page → Save → Predict page shows the values).
  All v6.1/6b regressions (guest-leak reset, onboarding-skip clean form,
  pending-save contract, logout wipe) still green on the assign-based reset.
- Headless boot: streamlit run → /healthz 200, / 200, no startup errors.

### (e) Limitations / student actions
- The browser-side resurrection itself cannot be replayed under AppTest (no
  persistent frontend widget manager) — the fix follows the same documented
  assignment pattern that fixed the v7.2 password-field bug in the browser.
- STUDENT browser test on the deployed app: (1) log in as account A, type some
  inputs / predict, log out; log in as account B on the same tab — the form must
  show B's saved profile (or a blank form), none of A's inputs; (2) Profile page →
  change values → Save — the Predict form must show them immediately; (3) check
  the new metric label, welcome text and the note under "Why this estimate?".

---

## Dashboard v7.5: chart how-to-read captions (2026-07-22)

Files: dashboards/app.py (two captions), scripts/test_dashboard.py (suite 154 → 156),
docs/PROGRESS.md. No model, artifact, dependency or logic changes — microcopy only.

### (a) Purpose
The student asked how the SHAP "Why this estimate?" figures and the "Salary vs
experience" curve are produced (multiplicative log-salary SHAP; RF re-prediction of
the same profile at 0–20 years), then chose two pieces of explanatory microcopy so
users can read the charts correctly.

### (b) What changed
1. **"Why this estimate?" how-to-read legend** — a new `st.caption` in the `if shown:`
   block, placed BEFORE the v7.4 don't-add-up note (order: factor rows → legend →
   don't-add-up note → bar chart): "**Blue** raises your estimate, **red** lowers it /
   **%** — how much a factor multiplies the salary up or down / **RM** — how much of
   your estimate depends on that single factor". The student's chosen option had a
   fourth "figures don't add up" bullet; it was dropped because the existing note
   already states that in full. Placement rationale: blue/red is the bar chart's key
   (two lines below), while the ▲/▼ row markers already carry direction, so the legend
   works as the section's colour/number key.
2. **"Salary vs experience" disclaimer** — the thin one-line caption ("Each point
   re-predicts your exact profile…") replaced with a fuller one: "This curve shows the
   model's expected salary as experience changes, holding everything else in your
   profile fixed. It mirrors the job ads it learned from, where high-experience
   salaries are under-represented." This answers the student's "why the same
   rise-then-dip shape every time" question in-app (root cause: training ads are
   concentrated at 0–2 years — median 1, 75th pct 2 — and senior roles rarely
   advertise pay, so RF has little/biased data past ~10 years).

### (c) Verification
- scripts/test_dashboard.py: **156/156 checks pass** (was 154). New: the how-to-read
  legend renders ("How to read this" + "depends on that single factor"); the
  experience disclaimer renders ("holding everything else in your profile fixed" +
  "under-represented"). No collision with the v6.3 absence checks (the legend says
  "Blue raises", not "blue band"; no percentile/background-sample phrasing), and the
  v7.4 don't-add-up note check still passes.
- Headless boot: streamlit run → /healthz 200, console clean.

### (d) Notes
- STUDENT: re-screenshot the "Why this estimate?" and "Salary vs experience" figures
  for the report — both now carry an explanatory caption.
