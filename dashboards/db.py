"""Supabase (Postgres) persistence for the wage prediction dashboard.

Accounts and history used to live in a local SQLite file, which worked on
localhost but not for a live deployment: Streamlit Community Cloud gives the
app an ephemeral filesystem, so every redeploy or restart would wipe every
user and saved prediction. The data now lives in a managed Postgres database
on Supabase and survives redeploys.

Three tables, created once in the Supabase SQL editor (see docs/DEPLOYMENT.md
for the exact SQL and setup steps):
- users:       registered accounts. Passwords are stored as bcrypt HASHES only —
               the plaintext never touches the database.
- predictions: saved prediction history, one row per prediction a logged-in
               user chose to save.
- profiles:    one saved input profile per user (personal facts only), used to
               prefill the prediction form automatically on login.

Access goes through the official `supabase` client, which talks to the
database over HTTPS — no connection strings and no connection pooling to
manage. Security model: Row Level Security is enabled on every table with no
policies, so the project's public key can read nothing at all; this app
authenticates with the SECRET key, which bypasses RLS. The secret key only
ever lives server-side — in .streamlit/secrets.toml locally (git-ignored) and
in the Secrets panel on Streamlit Community Cloud — never in the browser or
in git.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from supabase import create_client


class DatabaseError(Exception):
    """Raised when the Supabase connection is not configured, so app.py can
    show a friendly setup message instead of a stack trace."""


# Column order of the predictions table. Needed because the API returns plain
# row dicts: with zero rows there are no keys to infer column names from, and
# the History page still expects a DataFrame with these columns.
PREDICTION_COLUMNS = [
    "id", "username", "created_at", "job_title", "category", "state",
    "emp_type", "experience_years", "edu_level", "skills", "salary_offered",
    "pred_low", "pred_point", "pred_high", "verdict",
]


@st.cache_resource(show_spinner=False)
def _create_client():
    """One shared Supabase client per server process. It is just an HTTP
    client, so sharing it across users and reruns is safe — and caching it
    avoids re-doing the setup on every Streamlit rerun."""
    try:
        cfg = st.secrets["supabase"]
        url, key = cfg["url"], cfg["key"]
    except (KeyError, FileNotFoundError) as exc:
        raise DatabaseError(
            "Supabase is not configured. Add a [supabase] section with 'url' "
            "and 'key' to .streamlit/secrets.toml (locally) or to the app's "
            "Secrets panel (Streamlit Community Cloud)."
        ) from exc
    return create_client(url, key)


def _client():
    """Late-bound accessor: every function below calls _client() at call time,
    so the test suite can swap in an in-memory fake by replacing this one
    function — the same seam the SQLite version offered via its DB_PATH."""
    return _create_client()


def _now():
    """Current time in Malaysia as a plain ISO string (no offset suffix).
    The deployed server runs in UTC, so a naive datetime.now() would stamp
    history rows 8 hours behind the user's clock."""
    return (datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
            .replace(tzinfo=None).isoformat(timespec="seconds"))


# ------------------------------------------------------------------- users
def load_credentials():
    """All accounts, in the exact dict format streamlit-authenticator expects."""
    rows = (_client().table("users")
            .select("username, name, email, password_hash")
            .execute().data)
    return {"usernames": {row["username"]: {"name": row["name"],
                                            "email": row["email"],
                                            "password": row["password_hash"]}
                          for row in rows}}


def add_user(username, name, email, password_hash):
    """Insert a newly registered account (hash comes from streamlit-authenticator).

    onboarded=False marks the account for the one-time onboarding flow on its
    first login."""
    _client().table("users").insert({
        "username": username,
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "created_at": _now(),
        "onboarded": False,
    }).execute()


def needs_onboarding(username):
    """True only for accounts that have not yet completed (or skipped) the
    one-time onboarding flow."""
    rows = (_client().table("users").select("onboarded")
            .eq("username", username).execute().data)
    return bool(rows) and not rows[0]["onboarded"]


def mark_onboarded(username):
    """Record that the user finished or skipped onboarding — it never re-appears."""
    (_client().table("users").update({"onboarded": True})
     .eq("username", username).execute())


def update_password(username, password_hash):
    """Store a new bcrypt hash (forgot-password and change-password flows)."""
    (_client().table("users").update({"password_hash": password_hash})
     .eq("username", username).execute())


# -------------------------------------------------------------- predictions
def save_prediction(username, record):
    """Save one prediction for a logged-in user. Returns the new row id
    (assigned by the database's identity column)."""
    res = _client().table("predictions").insert({
        "username": username,
        "created_at": _now(),
        "job_title": record["job_title"],
        "category": record["category"],
        "state": record["state"],
        "emp_type": record["emp_type"],
        "experience_years": int(record["experience_years"]),
        "edu_level": int(record.get("edu_level", 0)),
        "skills": ", ".join(record["skills"]),
        "salary_offered": float(record["salary_offered"]),
        "pred_low": float(record["pred_low"]),
        "pred_point": float(record["pred_point"]),
        "pred_high": float(record["pred_high"]),
        "verdict": record["verdict"],
    }).execute()
    return res.data[0]["id"]


def list_predictions(username):
    """All saved predictions of one user, newest first, as a DataFrame.
    The id is the tie-breaker: several saves within the same second must
    still come back in a stable, latest-first order."""
    rows = (_client().table("predictions").select("*")
            .eq("username", username)
            .order("created_at", desc=True).order("id", desc=True)
            .execute().data)
    if not rows:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.DataFrame(rows)


def delete_predictions(username, ids):
    """Delete selected history rows. The username filter stops one user's ids
    from ever touching another user's rows."""
    if not ids:
        return
    # int() also converts numpy integers from the DataFrame selection into
    # plain Python ints the API client can serialise.
    (_client().table("predictions").delete()
     .eq("username", username).in_("id", [int(i) for i in ids]).execute())


def clear_history(username):
    """Delete every saved prediction of one user."""
    _client().table("predictions").delete().eq("username", username).execute()


# ----------------------------------------------------------------- profiles
# A profile holds only PERSONAL facts — education, location, experience and
# skills. Job title, category and employment type are prediction-specific and
# deliberately not part of the profile (v6 decision).
def save_profile(username, record):
    """Save (or replace) the user's one input profile for form autofill.
    upsert = insert the row, or update it if this user already has one."""
    _client().table("profiles").upsert({
        "username": username,
        "state": record["state"],
        "experience_years": int(record["experience_years"]),
        "edu_level": int(record.get("edu_level", 0)),
        "skills": ", ".join(record["skills"]),
        "updated_at": _now(),
    }).execute()


def load_profile(username):
    """The user's saved profile as a dict (skills back to a list), or None."""
    rows = (_client().table("profiles")
            .select("state, experience_years, edu_level, skills")
            .eq("username", username).execute().data)
    if not rows:
        return None
    row = rows[0]
    return {
        "state": row["state"],
        "experience_years": int(row["experience_years"] or 0),
        "edu_level": int(row["edu_level"] or 0),
        "skills": [s for s in (row["skills"] or "").split(", ") if s],
    }
