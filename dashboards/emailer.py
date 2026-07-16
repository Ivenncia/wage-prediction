"""Password-reset email for the dashboard, using only the Python standard library.

streamlit-authenticator 0.4.x can only send emails through the library author's
cloud service (it needs a paid `api_key`), so the email is sent here instead
with smtplib. SMTP settings live in `.streamlit/secrets.toml` (git-ignored) —
credentials never appear in code. When SMTP is not configured or sending
fails, functions return False and the app falls back to showing the new
password on screen, so the demo works without any email setup.
"""

import smtplib
from email.message import EmailMessage

import streamlit as st


def _smtp_settings():
    """The [smtp] block from .streamlit/secrets.toml, or None if incomplete.

    st.secrets raises when no secrets file exists at all, hence the try/except.
    """
    try:
        smtp = st.secrets["smtp"]
        if all(str(smtp.get(key, "")).strip() for key in ("host", "port", "user", "app_password")):
            return smtp
    except Exception:
        pass
    return None


def smtp_configured():
    """True when a complete [smtp] block exists in .streamlit/secrets.toml."""
    return _smtp_settings() is not None


def send_password_email(to_email, username, new_password):
    """Email the newly generated password to the user.

    Returns True on success, False on any failure — the caller then shows
    the password on screen instead, so forgot-password can never dead-end.
    """
    smtp = _smtp_settings()
    if smtp is None:
        return False

    message = EmailMessage()
    message["Subject"] = "Malaysia Wage Predictor — your new password"
    message["From"] = smtp["user"]
    message["To"] = to_email
    message.set_content(
        f"Hello {username},\n"
        f"\n"
        f"A password reset was requested for your Malaysia Wage Predictor account.\n"
        f"Your new password is:\n"
        f"\n"
        f"    {new_password}\n"
        f"\n"
        f"Please log in with it and change it right away via 'Change password'\n"
        f"in the sidebar.\n"
        f"\n"
        f"If you did not request this reset, you can ignore this email.\n"
    )

    try:
        with smtplib.SMTP(str(smtp["host"]), int(smtp["port"]), timeout=15) as server:
            server.starttls()  # upgrade to an encrypted connection before logging in
            server.login(smtp["user"], smtp["app_password"])
            server.send_message(message)
        return True
    except Exception:
        return False
