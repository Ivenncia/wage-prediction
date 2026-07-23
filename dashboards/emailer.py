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


def send_reset_link_email(to_email, username, reset_link):
    """Email a password-reset link to the user.

    Returns True on success, False on any failure — the caller then shows
    the link on screen instead, so forgot-password can never dead-end.
    """
    smtp = _smtp_settings()
    if smtp is None:
        return False

    message = EmailMessage()
    message["Subject"] = "Malaysia Wage Predictor — reset your password"
    message["From"] = smtp["user"]
    message["To"] = to_email
    message.set_content(
        f"Hello {username},\n"
        f"\n"
        f"A password reset was requested for your Malaysia Wage Predictor account.\n"
        f"Open this link to create a new password:\n"
        f"\n"
        f"    {reset_link}\n"
        f"\n"
        f"The link works once and expires in 30 minutes.\n"
        f"\n"
        f"If you did not request this reset, please ignore this email! \n"
    )

    try:
        with smtplib.SMTP(str(smtp["host"]), int(smtp["port"]), timeout=15) as server:
            server.starttls()  
            server.login(smtp["user"], smtp["app_password"])
            server.send_message(message)
        return True
    except Exception:
        return False
