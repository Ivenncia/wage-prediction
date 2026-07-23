# Supabase (Postgres) persistence for the wage prediction dashboard

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from supabase import create_client


class DatabaseError(Exception):
    """Raised when the Supabase connection is not configured, so app.py can
    show a friendly setup message instead of a stack trace."""


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


#-------------------------------------------------------------------------------- users
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


def set_reset_token(username, token_hash, expires_epoch):
    """Store a forgot-password reset request: the SHA-256 hash of the emailed
    token (never the token itself) and the unix time the link stops working.
    One request per user — a newer link replaces any older one."""
    (_client().table("users")
     .update({"reset_token_hash": token_hash, "reset_token_expires": expires_epoch})
     .eq("username", username).execute())


def get_reset_request(token_hash):
    """The user a reset-token hash belongs to, as {'username', 'reset_token_expires'},
    or None when no account carries this hash (unknown or already-used token).
    Expiry is checked by the caller, so 'expired' can share the same message."""
    rows = (_client().table("users").select("username, reset_token_expires")
            .eq("reset_token_hash", token_hash).execute().data)
    return rows[0] if rows else None


def clear_reset_token(username):
    """Invalidate the user's reset link (called the moment it is used)."""
    (_client().table("users")
     .update({"reset_token_hash": None, "reset_token_expires": None})
     .eq("username", username).execute())


#------------------------------------------------------------------------- predictions
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
    (_client().table("predictions").delete()
     .eq("username", username).in_("id", [int(i) for i in ids]).execute())


def clear_history(username):
    """Delete every saved prediction of one user."""
    _client().table("predictions").delete().eq("username", username).execute()


# ----------------------------------------------------------------------------- profiles
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
