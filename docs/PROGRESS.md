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
