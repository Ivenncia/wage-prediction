# System Architecture — Explainable AI Wage Prediction System (text version)

Text-based architecture for the FYP report figure. Each numbered section below is one
BOX (or group of boxes) in the future diagram; the "Arrows" lines say what connects to
what and what label to write on the connector. The "Key flows" section at the end gives
the numbered end-to-end paths, like the flow arrows in typical architecture figures.

The system has two halves, the same split used in many ML system diagrams:

- **OFFLINE** (top or bottom band of the diagram): the training pipeline that runs once
  on the developer machine and produces model artifacts.
- **ONLINE** (main band): the deployed Streamlit dashboard that serves users. It loads
  ONLY the saved artifacts — it never touches the raw dataset.

---

## 1. OFFLINE TRAINING PIPELINE  (band: "Offline Model Development")

Runs on the developer's machine (Jupyter notebooks). Not part of the live app.

**Box 1a — Data Source**
- `data/jobstreet_cleaned_final.csv` — 69,024 Malaysian JobStreet job ads,
  already cleaned in earlier notebooks (diagnosis → cleaning → understanding).

**Box 1b — Feature Engineering (`notebooks/01_features.ipynb`)**
- Filter to usable-salary ads: 31,406 rows, monthly salary RM500–30,000.
- Regex extraction from job descriptions:
  - years of experience required (+ has_experience_req flag)
  - education level required, ordinal 0–4 (+ has_edu_req flag)
  - 75 binary skill flags (50 hard + 25 soft skills) + skill_count
- Also builds dashboard support artifacts: job-title suggestion list
  (autocomplete + category auto-fill) and per-role skill evidence statistics
  (which skills appear in higher-paying ads of each role).
- Output: `data/model_table.csv` (the modelling table).

**Box 1c — Model Training & Explainability (`notebooks/02_models.ipynb`)**
- Preprocessing inside one sklearn Pipeline: TF-IDF on job title (800 features,
  1–2 grams) + one-hot encoding of category / state / employment type + numeric
  passthrough (experience, education, skills).
- Target = log(monthly salary); one fixed 80/20 train/test split.
- Four models compared on test MAE / RMSE / R² (in RM): Linear Regression,
  Decision Tree, Random Forest, HistGradientBoosting.
  **Winner: Random Forest (test MAE ≈ RM963, R² ≈ 0.60).**
- Two extra quantile HistGBR pipelines (25th / 75th percentile) → the displayed
  salary range.
- SHAP TreeExplainer fitted on the winning model (interventional mode) → global
  summary + the saved per-prediction explainer.
- Subgroup error analysis by category and state (report evidence).

**Box 1d — Model Artifact Store (`models/`, joblib files)**
- salary_pipeline (preprocessing + Random Forest)
- quantile_lower_pipeline, quantile_upper_pipeline (P25 / P75)
- shap_explainer
- skill_lists, skill_stats, job_title_suggestions, input_options, feature_names
- model_comparison.csv (report table)

**Arrows (offline band)**
- CSV → 01_features: "load cleaned data"
- 01_features → 02_models: "model table (31,406 × 85)"
- 01_features / 02_models → Model Artifact Store: "save artifacts (joblib)"
- Model Artifact Store → Application Layer (online band):
  **"loaded once at app startup (@st.cache_resource) — the live app never reads the CSV"**

---

## 2. USER INTERFACE LAYER  (box group: "Frontend — Streamlit in the browser")

What the user sees. All rendered by Streamlit; no separate frontend framework.

**Box 2a — Entry Hub (before login)**
- Tabs: Log in | Register (auto-login after success) | Forgot password
- "Continue as guest" button (guests can predict; must log in to save)
- Password-reset landing page (opened from the emailed link)

**Box 2b — Onboarding Card**
- Shown once to newly registered accounts (optional): education, state,
  experience, skills → saved as the user's profile; "Skip for now" available.

**Box 2c — Main App (after login / as guest)**
- Top navigation (5 pages): Predict | What-if Analysis | History | Profile |
  About the model
- Top-right user menu: signed-in name, Change password, Logout
- Sidebar prediction form (Predict page only), grouped into:
  Job details (title with autocomplete, category, state, employment type) /
  Your background (experience 0–20, education, skills multiselect) /
  Salary check (optional salary in RM)

**Box 2d — Result Views (Predict page output)**
- Salary range band (P25–P75) + point estimate + "your salary" marker
- Verdict badge: entered salary below / within / above the market range
- "Why this estimate?" — plain-language concept sentences + factor bar chart
- Advanced model explanation expander — SHAP waterfall
- Career improvement opportunities — skill cards with real-ad evidence,
  experience outlook, education uplift

**Arrows**
- User ↔ UI layer: "inputs / clicks" and "rendered pages & charts"
- UI layer ↔ Application layer: "widget events / Streamlit rerun"

---

## 3. APPLICATION LAYER  (box group: "Backend logic — dashboards/app.py, one Streamlit process")

Streamlit is both frontend and backend here: one Python process re-runs app.py on
every interaction. Internal components (one sub-box each):

**Box 3a — Authentication Manager** (streamlit-authenticator library)
- bcrypt password verification; signed 30-day session cookie
- Registration with password policy + multi-word username validator
- Auto-login after registration and after a successful password reset
- Change password (app-owned form, library's validation controller)

**Box 3b — Prediction Engine**
- Builds one 84-column input row from the form
- Random Forest pipeline → point estimate (log-RM converted back to RM)
- Quantile pipelines → P25–P75 range (estimate clipped inside the range)
- Verdict: compares the entered salary against the predicted range

**Box 3c — Explainability (XAI) Engine**
- SHAP explainer → per-feature contributions for THIS prediction
- Features grouped into 8 concepts (job seniority, type of role, experience,
  education, skills, location, category, employment type)
- Outputs: baseline sentence, concept sentences with RM/% effects, factor bar
  chart, SHAP waterfall (advanced view)

**Box 3d — Career Advice Engine**
- Skills lever: re-predicts all absent skills in one batch, keeps only skills
  with evidence from skill_stats (common in higher-paying ads of the same role)
- Experience lever: salary-vs-experience curve (profile re-predicted at 0–20 years)
- Education lever: re-predict at the next education level

**Box 3e — What-if Comparison**
- 2–3 saved predictions re-predicted with the current model → what-changed diff
  table, one-line takeaway (driven by concept SHAP differences), range chart,
  optional experience curves

**Box 3f — Session & State Manager**
- Holds the last result across reruns, guest "log in to save" hand-off,
  profile prefill on login, form wipe on logout / new account

**Helper modules (small boxes attached to Layer 3)**
- `dashboards/db.py` — all database access (see Layer 4)
- `dashboards/emailer.py` — SMTP email sending (see Layer 4)

**Arrows**
- Application layer → Model Artifact Store: "load artifacts (startup, cached)"
- 3b/3c/3d/3e → artifacts: "predict() / SHAP calls (in-memory)"
- 3a/3e/3f → db.py: "read/write users, predictions, profiles"
- 3a (forgot password) → emailer.py: "send reset link"

---

## 4. DATA & EXTERNAL SERVICES LAYER  (band: "Persistence & external services")

**Box 4a — Supabase (managed PostgreSQL, free tier, Singapore region)**
- Accessed from db.py via the official supabase client — REST over HTTPS,
  using the secret service key; Row Level Security enabled with no public
  policies (the public key can read nothing).
- Tables:
  - **users** — username (PK), name, email, bcrypt password hash, onboarded
    flag, reset-token hash + expiry
  - **predictions** — saved prediction history (inputs, range, verdict, timestamp)
  - **profiles** — one row per user: personal defaults (state, experience,
    education, skills) used to prefill the form
- Arrows: db.py ↔ Supabase: "DB query (REST/HTTPS)" / "DB response (JSON)"

**Box 4b — Gmail SMTP server**
- emailer.py sends the single-use password-reset link (STARTTLS).
- If SMTP is unavailable, the app shows the link on screen instead (no dead end).
- Arrow: emailer.py → Gmail SMTP: "reset-link email"

**Box 4c — Configuration & Secrets**
- `.streamlit/secrets.toml` (git-ignored) / Streamlit Cloud Secrets panel:
  Supabase URL + secret key, cookie signing key, SMTP credentials, public app URL
- `dashboards/config.yaml`: non-secret cookie name + expiry only
- Arrow: secrets → Application layer: "read at startup (st.secrets)"

---

## 5. DEPLOYMENT VIEW  (optional small diagram or caption)

- **User's browser** —HTTPS→ **Streamlit Community Cloud** (runs app.py; the
  GitHub repo supplies the code and the models/ artifacts; secrets come from the
  Cloud Secrets panel)
- **Streamlit Community Cloud** —HTTPS/REST→ **Supabase** (Singapore)
- **Streamlit Community Cloud** —SMTP/STARTTLS→ **Gmail**
- The offline band (notebooks + CSV) stays on the developer machine; only the
  produced `models/` artifacts are committed and deployed.

---

## 6. KEY FLOWS  (the numbered arrows to draw across the diagram)

**Flow 1 — Register / Log in**
User credentials → Entry Hub → Authentication Manager → users table (bcrypt
check / insert) → session cookie set → Main App (new accounts see the
Onboarding Card first; profile prefills the form).

**Flow 2 — Predict (core flow)**
Sidebar form → Prediction Engine → salary + quantile pipelines (artifacts) →
range + estimate + verdict → XAI Engine (SHAP → concept sentences + charts) →
Career Advice Engine (skill evidence + experience + education levers) →
Result Views.

**Flow 3 — Save, History & What-if**
Result → predictions table (guests: "log in to save" → auto-saved after login) →
History page lists rows → user picks 2–3 → What-if Comparison re-predicts them →
diff table + takeaway + range chart.

**Flow 4 — Forgot password**
Email address → token generated, hash + expiry stored in users → reset link sent
via Gmail SMTP (on-screen fallback) → user opens link → landing page → new
password (policy enforced) → hash updated, token cleared (single use) →
auto sign-in.

**Flow 5 — Profile & account maintenance**
Profile page / Onboarding → profiles table (upsert) → prefill on next login.
User menu → Change password → Authentication Manager → users table.
