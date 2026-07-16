"""SQLite persistence for the wage prediction dashboard.

One git-ignored database file (data/dashboard.db) with three tables:
- users:       registered accounts. Passwords are stored as bcrypt HASHES only —
               the plaintext never touches the database.
- predictions: saved prediction history, one row per prediction a logged-in
               user chose to save.
- profiles:    one saved input profile per user, used to prefill the prediction
               form automatically on login.

sqlite3 is part of the Python standard library, so this adds no dependency.
Every function opens a fresh connection and closes it immediately — the
dashboard is a low-traffic single-server app, and short-lived connections
avoid any sharing problems across Streamlit reruns and threads.
"""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "dashboard.db"


def _connect():
    """Open a connection to the dashboard database (creating data/ if needed)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


# ------------------------------------------------------------------ schema
def init_db(seed_users=None):
    """Create the tables if they do not exist yet.

    seed_users: the `usernames` dict from config.yaml. It is only used ONCE —
    when the users table is empty — so the two demo accounts keep working
    after the move from YAML to SQLite. After that first run, the database
    is the single source of truth for accounts.
    """
    with closing(_connect()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username      TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                onboarded     INTEGER NOT NULL DEFAULT 1
            )""")
        # Migration for databases created before the onboarding flow (v6):
        # DEFAULT 1 grandfathers every existing account — only accounts
        # registered from now on (inserted with onboarded=0) see the flow.
        existing_user_cols = {row[1] for row in
                              conn.execute("PRAGMA table_info(users)").fetchall()}
        if "onboarded" not in existing_user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN "
                         "onboarded INTEGER NOT NULL DEFAULT 1")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                username         TEXT NOT NULL REFERENCES users(username),
                created_at       TEXT NOT NULL,
                job_title        TEXT,
                category         TEXT,
                state            TEXT,
                emp_type         TEXT,
                experience_years INTEGER,
                skills           TEXT,
                salary_offered   REAL,
                pred_low         REAL,
                pred_point       REAL,
                pred_high        REAL,
                verdict          TEXT
            )""")
        # Migration for databases created before the education feature (v3):
        # add the column in place so existing history rows are kept. Old rows
        # get NULL, which the app displays as "Not specified" (level 0).
        existing_cols = {row[1] for row in
                         conn.execute("PRAGMA table_info(predictions)").fetchall()}
        if "edu_level" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN edu_level INTEGER")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                username         TEXT PRIMARY KEY REFERENCES users(username),
                job_title        TEXT,
                category         TEXT,
                state            TEXT,
                emp_type         TEXT,
                experience_years INTEGER,
                edu_level        INTEGER,
                skills           TEXT,
                updated_at       TEXT NOT NULL
            )""")
        if seed_users:
            n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if n_users == 0:
                for username, info in seed_users.items():
                    conn.execute(
                        "INSERT INTO users (username, name, email, password_hash, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (username, info["name"], info["email"], info["password"],
                         datetime.now().isoformat(timespec="seconds")))
        conn.commit()


# ------------------------------------------------------------------- users
def load_credentials():
    """All accounts, in the exact dict format streamlit-authenticator expects."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT username, name, email, password_hash FROM users").fetchall()
    return {"usernames": {username: {"name": name, "email": email, "password": pw_hash}
                          for username, name, email, pw_hash in rows}}


def add_user(username, name, email, password_hash):
    """Insert a newly registered account (hash comes from streamlit-authenticator).

    onboarded=0 marks the account for the one-time onboarding flow on its
    first login; seeded/pre-existing accounts default to 1 and never see it.
    """
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO users (username, name, email, password_hash, created_at, "
            "onboarded) VALUES (?, ?, ?, ?, ?, 0)",
            (username, name, email, password_hash,
             datetime.now().isoformat(timespec="seconds")))
        conn.commit()


def needs_onboarding(username):
    """True only for accounts registered through the app that have not yet
    completed (or skipped) the onboarding flow."""
    with closing(_connect()) as conn:
        row = conn.execute("SELECT onboarded FROM users WHERE username = ?",
                           (username,)).fetchone()
    return row is not None and row[0] == 0


def mark_onboarded(username):
    """Record that the user finished or skipped onboarding — it never re-appears."""
    with closing(_connect()) as conn:
        conn.execute("UPDATE users SET onboarded = 1 WHERE username = ?", (username,))
        conn.commit()


def update_password(username, password_hash):
    """Store a new bcrypt hash (forgot-password and change-password flows)."""
    with closing(_connect()) as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                     (password_hash, username))
        conn.commit()


# -------------------------------------------------------------- predictions
def save_prediction(username, record):
    """Save one prediction for a logged-in user. Returns the new row id."""
    with closing(_connect()) as conn:
        cursor = conn.execute(
            """INSERT INTO predictions
               (username, created_at, job_title, category, state, emp_type,
                experience_years, edu_level, skills, salary_offered,
                pred_low, pred_point, pred_high, verdict)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (username, datetime.now().isoformat(timespec="seconds"),
             record["job_title"], record["category"], record["state"],
             record["emp_type"], record["experience_years"],
             record.get("edu_level", 0),
             ", ".join(record["skills"]), record["salary_offered"],
             record["pred_low"], record["pred_point"], record["pred_high"],
             record["verdict"]))
        conn.commit()
        return cursor.lastrowid


def list_predictions(username):
    """All saved predictions of one user, newest first, as a DataFrame."""
    with closing(_connect()) as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions WHERE username = ? ORDER BY created_at DESC",
            conn, params=(username,))


def delete_predictions(username, ids):
    """Delete selected history rows. The username filter stops one user's ids
    from ever touching another user's rows."""
    if not ids:
        return
    placeholders = ", ".join("?" for _ in ids)
    with closing(_connect()) as conn:
        conn.execute(
            f"DELETE FROM predictions WHERE username = ? AND id IN ({placeholders})",
            (username, *ids))
        conn.commit()


def clear_history(username):
    """Delete every saved prediction of one user."""
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM predictions WHERE username = ?", (username,))
        conn.commit()


# ----------------------------------------------------------------- profiles
# A profile holds only PERSONAL facts — education, location, experience and
# skills. Job title, category and employment type are prediction-specific and
# deliberately not part of the profile (v6 decision). The old columns stay in
# the table so pre-v6 databases keep working; they are simply no longer used.
def save_profile(username, record):
    """Save (or replace) the user's one input profile for form autofill."""
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO profiles
               (username, job_title, category, state, emp_type,
                experience_years, edu_level, skills, updated_at)
               VALUES (?, '', '', ?, '', ?, ?, ?, ?)""",
            (username, record["state"], record["experience_years"],
             record.get("edu_level", 0), ", ".join(record["skills"]),
             datetime.now().isoformat(timespec="seconds")))
        conn.commit()


def load_profile(username):
    """The user's saved profile as a dict (skills back to a list), or None."""
    with closing(_connect()) as conn:
        row = conn.execute(
            """SELECT state, experience_years, edu_level, skills
               FROM profiles WHERE username = ?""", (username,)).fetchone()
    if row is None:
        return None
    state, experience_years, edu_level, skills = row
    return {
        "state": state,
        "experience_years": int(experience_years or 0),
        "edu_level": int(edu_level or 0),
        "skills": [s for s in (skills or "").split(", ") if s],
    }
