# Deployment guide — Supabase database + Streamlit Community Cloud

The dashboard stores accounts, prediction history and profiles in a Supabase
(Postgres) database, so the data survives every redeploy of the app. This guide
covers the one-time Supabase setup and the Streamlit Community Cloud deploy.

---

## Part A — Set up the Supabase database (one time, ~10 minutes)

### A1. Create the project
1. Go to https://supabase.com and sign up (logging in with GitHub is easiest).
2. Click **New project**:
   - Name: e.g. `wage-prediction`
   - Database password: generate a strong one and store it somewhere safe.
     (The app never uses this password — it connects with an API key — but
     Supabase requires it and you may need it for direct database access later.)
   - Region: **Southeast Asia (Singapore)** — closest to Malaysia, lowest latency.
3. Wait ~2 minutes while the project is provisioned.

### A2. Create the tables
1. In the left sidebar open **SQL Editor** → **New query**.
2. Paste ALL of the SQL below and click **Run**:

```sql
create table public.users (
    username            text primary key,
    name                text not null,
    email               text not null unique,
    password_hash       text not null,    -- bcrypt hash only, never plaintext
    created_at          text not null,    -- ISO string written by the app
    onboarded           boolean not null default false,
    reset_token_hash    text,             -- SHA-256 of the emailed reset token
    reset_token_expires bigint            -- unix time the reset link dies
);

create table public.predictions (
    id               bigint generated always as identity primary key,
    username         text not null references public.users (username) on delete cascade,
    created_at       text not null,
    job_title        text,
    category         text,
    state            text,
    emp_type         text,
    experience_years integer,
    edu_level        integer,
    skills           text,                -- comma-joined skill names
    salary_offered   double precision,
    pred_low         double precision,
    pred_point       double precision,
    pred_high        double precision,
    verdict          text
);

create table public.profiles (
    username         text primary key references public.users (username) on delete cascade,
    state            text,
    experience_years integer,
    edu_level        integer,
    skills           text,
    updated_at       text not null
);

-- Lock the public REST API: RLS with no policies means the public
-- (anon/publishable) key can read NOTHING. The app uses the secret key,
-- which bypasses RLS and only ever lives in server-side secrets.
alter table public.users       enable row level security;
alter table public.predictions enable row level security;
alter table public.profiles    enable row level security;
```

3. Open **Table Editor** and confirm the three tables exist (all empty —
   accounts are created through the app's registration form).

> **Already created the tables before v7.2?** The forgot-password reset link
> needs two extra columns on `users`. Run this once in the SQL Editor:
>
> ```sql
> alter table public.users
>     add column reset_token_hash    text,
>     add column reset_token_expires bigint;
> ```

### A3. Copy the credentials
1. Go to **Project Settings → API** (or "API Keys").
2. Copy two values:
   - **Project URL** — looks like `https://abcdefgh.supabase.co`
   - **Secret key** — starts with `sb_secret_...` (on older projects this is
     the key labelled `service_role`). **NOT** the publishable/anon key.
3. Put them into `.streamlit/secrets.toml` locally (this file is git-ignored):

```toml
[supabase]
url = "https://YOUR-PROJECT-REF.supabase.co"
key = "sb_secret_..."
```

Never commit the secret key or paste it into any client-side code — it
bypasses Row Level Security by design.

Also add the app's own address — the forgot-password reset links point at it:

```toml
[app]
url = "https://wage-prediction-dashboard-system.streamlit.app/"
```

(Locally this stays localhost; after deploying, set it to the public
`https://….streamlit.app` address in the Cloud secrets panel, or emailed
reset links will point at localhost.)

### A4. Verify locally
1. `streamlit run dashboards/app.py`
2. Register a throwaway account, make a prediction, save it.
3. In the Supabase **Table Editor**, refresh `users` and `predictions` —
   the rows should be there.
4. Stop and restart streamlit, log back in — the history is still there.
   That persistence across restarts is exactly what the deployed app gets.

---

## Part B — Deploy to Streamlit Community Cloud

1. Push the repository to GitHub. Already handled by `.gitignore`: the `data/`
   folder, `.venv` and `.streamlit/secrets.toml` stay out of git; the trained
   model artifacts in `models/` ARE committed (the app needs them).
2. Go to https://share.streamlit.io → **New app**:
   - Repository: this repo, branch `main`
   - Main file path: `dashboards/app.py`
   - Advanced settings → Python version: match `.python-version` (3.13)
3. After the first deploy, open the app's **Settings → Secrets** and paste the
   full contents of your local `.streamlit/secrets.toml` — the `[supabase]`,
   `[auth]`, `[smtp]` and `[app]` sections. **Change `[app] url` to the app's
   public `https://….streamlit.app` address** so emailed reset links point at
   the deployed app, not localhost. Save; the app reboots with the secrets.
4. Open the app URL: the login hub should appear. Register an account, save a
   prediction, then **Reboot** the app from the Streamlit Cloud menu — logging
   back in must show the saved history (data now lives in Supabase, not in the
   app container).

---

## Ongoing / gotchas

- **Supabase free tier pauses after ~1 week of inactivity.** A paused project
  makes the app show its "could not reach the database" message. Before a demo
  or the viva: open https://supabase.com/dashboard, select the project, click
  **Restore** and wait ~1 minute.
- **Rotating the cookie key** (`[auth] cookie_key`): allowed at any time; every
  browser is simply logged out once. Keep it at 32+ random characters.
- **Forgot-password email** needs the `[smtp]` secrets (Gmail App Password)
  and a correct `[app] url`. The app emails a single-use reset link (30-minute
  expiry); without working SMTP it falls back to showing the link on screen.
- **The database is the single source of truth for accounts.** There are no
  seeded/demo accounts; to create an account, use the registration form. To
  delete one, remove its row in the Supabase Table Editor (its predictions and
  profile are removed automatically via `on delete cascade`).
