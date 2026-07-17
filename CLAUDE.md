# CLAUDE.md — FYP: Explainable AI Wage Prediction System (Malaysia)

## Project overview
Final Year Project (APU). Goal: an explainable AI wage prediction system using Malaysian
JobStreet job posting data. Users log in, enter their profile (job title, category, state,
employment type, years of experience, skills, optionally a salary they were offered), and
receive: (1) a predicted monthly salary range in RM, (2) a classification of the entered
salary as below / within / above the predicted market range, (3) SHAP-based explanations
of the key factors, (4) improvement suggestions.

Code quality: write simple, professional, explainable code — clear structure, sensible
naming, and robust enough for a real system (input validation, graceful error handling).
Avoid clever or over-optimized code. The student must be able to explain every line during the fianl year project presentation add short comments and markdown cells explaining WHY, not just what.

## Environment
- Windows 11, Python 3.13 in `.venv` (activate: `.venv\Scripts\activate`)
- Installed: pandas, numpy, scikit-learn, shap, streamlit, streamlit-authenticator,
  matplotlib, supabase (approved exception 2026-07-17 — dashboard persistence only)
- Do NOT add new dependencies (no xgboost/lightgbm/spacy). scikit-learn only for models.
- CPU only. Dataset fits in memory (16 GB RAM).
- Deploy target: Streamlit Community Cloud; database: Supabase free tier (Postgres),
  accessed via the official supabase client. Secrets live in `.streamlit/secrets.toml`
  locally (git-ignored) and in the app's Secrets panel when deployed.

## Folder structure
- `data/` — datasets (git-ignored). Main file: `data/jobstreet_cleaned_final.csv`
- `notebooks/` — Jupyter notebooks. Existing: data_diagnosis, data_cleaning, data_understanding
  (cleaning phase, already done — do not modify). New work: `01_features.ipynb`, `02_models.ipynb`
- `models/` — saved artifacts (joblib) produced by 02_models
- `dashboards/` — Streamlit app (`app.py`)
- `scripts/` — helper scripts if needed
- Never commit CSVs or `.venv` (already in .gitignore)

## Dataset facts (data/jobstreet_cleaned_final.csv)
- 69,024 rows × 29 columns. Already cleaned in previous notebooks.
- Target: `salary_monthly_final` (monthly RM). Only rows with `salary_usable_flag == 1`
  are usable for training: 31,468 rows. Median RM3,750, IQR RM3,000–5,000.
- Key feature columns: `job_title` (text), `category` (30 values), `subcategory` (310),
  `state_clean` (18 states, top: Selangor, Kuala Lumpur, Johor, Penang), `type_clean`
  (Full time 91%, Contract/Temp, Part time, Casual/Vacation), `role_clean` (5,483 values —
  too granular, do not one-hot), `descriptions_clean` (text, median ~1,764 chars,
  99.97% present), `description_length`.
- Also present: `salary_min`, `salary_max` (98.3% of usable rows have a true advertised
  min–max range), various salary parsing flag columns.
- NEVER impute the target. Rows without usable salary are excluded from training.

## Locked modelling decisions (do not change without asking the student)
1. Filter: `salary_usable_flag == 1` AND `salary_monthly_final` between 500 and 30,000.
2. Target: `np.log(salary_monthly_final)`. Convert predictions back with `np.exp` before
   computing metrics or displaying (metrics must be in RM, not log units).
3. Features:
   - `job_title` → TF-IDF (max_features≈800, ngram_range=(1,2), min_df=5), fitted inside
     the sklearn Pipeline (not in 01_features).
   - `category`, `state_clean`, `type_clean` → OneHotEncoder(handle_unknown='ignore'),
     inside the Pipeline.
   - `experience_years` → extracted in 01_features by regex from `descriptions_clean`
     (patterns like "3 years", "2-4 years", "5+ years"; a years match only counts if
     "experience" or "pengalaman" appears within ~60 characters of it, to reject
     company-age phrases like "established 40 years ago"; take the minimum of a range;
     cap at 20; 0 + a `has_experience_req` flag when not stated).
   - Skill flags → extracted in 01_features from `descriptions_clean` using a keyword
     dictionary of ~50 hard skills and ~25 soft skills (lowercase substring/word-boundary
     match). One binary column per skill, plus `skill_count`. Keep the skill list in a
     Python list that gets saved to models/ (the dashboard multiselect uses it).
   - `edu_level` + `has_edu_req` → extracted in 01_features from `descriptions_clean`
     (approved extension, 2026-07-15). Ordinal 0–4: not stated / SPM–secondary /
     diploma / bachelor's / postgraduate; take the MINIMUM level mentioned (entry
     requirement, same reading as experience). Keyword patterns with hand-written
     exclusions for the word "degree" ("360 degree", "a high degree of") and never
     bare "master" (fires on "Scrum Master").
4. Split: train_test_split(test_size=0.2, random_state=42). Same split for all models.
5. Exactly these 4 models, compared in one table (MAE, RMSE, R², all in RM):
   LinearRegression, DecisionTreeRegressor(min_samples_leaf tuned lightly),
   RandomForestRegressor(n_estimators≈200, min_samples_leaf≈3, n_jobs=-1),
   HistGradientBoostingRegressor. Winner = best test MAE (expected: HistGBR or RF).
6. Salary range: HistGradientBoostingRegressor(loss="quantile", quantile=0.25) and (0.75)
   trained on the same features → lower/upper bound of displayed range. Ensure
   lower ≤ point prediction ≤ upper when displaying (clip if needed).
7. XAI: shap.TreeExplainer on the winning model. Global summary plot (report figure) +
   per-prediction values for the dashboard waterfall. Precompute the explainer and save it.
8. Subgroup error analysis (final model only): mean absolute error grouped by `category`
   and by `state_clean`, shown as sorted tables/bar charts.
9. Artifacts saved to `models/` with joblib: full sklearn Pipeline (preprocessing + winner),
   both quantile pipelines, SHAP explainer, skill list, category list, state list, type list,
   and `model_comparison.csv`. The dashboard must load ONLY these artifacts — never the CSV.

## Dashboard spec (dashboards/app.py)
- Streamlit, run with `streamlit run dashboards/app.py`. Single page is fine.
- Login/logout via streamlit-authenticator. All accounts come from in-app registration
  (no demo/seed accounts — removed 2026-07-17); bcrypt password hashes, prediction
  history and profiles are stored in Supabase via `dashboards/db.py`. config.yaml holds
  only non-secret cookie settings; the cookie signing key comes from st.secrets.
  App content only renders after login (or guest mode).
- Inputs: job title (text_input), category (selectbox), state (selectbox),
  employment type (selectbox), years of experience (slider 0–20), highest education
  (selectbox, 5 ordinal levels), skills (multiselect from saved skill list), optional
  "salary offered / current salary (RM)" (number_input, 0 = not provided).
- Extras (v4, approved 2026-07-15): per-user saved profile that prefills the form on
  login, and a history scenario comparison (select 2–3 saved rows → re-predict with the
  current model, side-by-side ranges, top factors and experience curves).
- Outputs: predicted monthly range (RM lower – upper, with point estimate),
  verdict badge when a salary was entered (below / within / above predicted range),
  SHAP waterfall for this prediction, 2–3 auto-generated plain-language sentences from the
  top SHAP factors, and improvement tips = re-predict with each of the top ~10 absent
  skills toggled on, show the 3 biggest gainers with the RM uplift.
- Load artifacts with @st.cache_resource. Handle unknown/empty inputs gracefully.
- Keep styling minimal and clean; no custom HTML hacks needed.

## Working style
- One notebook/file at a time. After creating or editing, run it (or its key cells) and
  show the student the output before moving on.
- Every notebook needs markdown cells explaining each step (these become report figures —
  the student screenshots notebook outputs as evidence, like Figures 32–108 in their
  Investigation Report).
- Print shapes, counts, and examples after each transformation so results are verifiable.
- After completing each notebook or file, append a section to `docs/PROGRESS.md` covering:
  (a) purpose of the file, (b) what each major step/cell does in 1–2 sentences, (c) the
  key printed results and numbers, (d) design decisions made and why, (e) limitations
  found and how they were handled. Raw factual notes, not polished prose.
- Ask before deviating from any locked decision above.
