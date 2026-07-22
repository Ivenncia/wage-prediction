"""Explainable AI Wage Prediction Dashboard (FYP, APU).

Run with:  streamlit run dashboards/app.py

The app loads ONLY the artifacts saved in models/ by the notebooks — it never
reads the training CSV. If the models have not been trained yet, it shows a
friendly message instead of crashing.

Accounts and saved predictions live in a Supabase (Postgres) database in the
cloud (see db.py), so data survives redeploys of the app. Visitors can log in,
register a new account, recover a forgotten password, or continue as a guest —
guests can predict but must log in to save results to their history. Newly registered accounts get a one-time optional
onboarding step (education / location / experience / skills) that seeds their
profile.

Layout (v6): a top navigation bar (Predict / Compare Predictions / History /
Profile / About the model) with a user menu in the top-right corner. The
prediction form lives in the sidebar of the Predict page only, grouped into
sections; form values survive page switches through the session-state
keep-alive below. Styling comes from native Streamlit theming
(.streamlit/config.toml) and bordered containers; the single CSS override in
the app hides the built-in reveal eye on the password forms (student-approved
exception — they have their own "Show passwords" checkbox).
"""

import hashlib
import re
import secrets
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from matplotlib.ticker import FuncFormatter
from yaml.loader import SafeLoader

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR.parent / "models"

# Make the sibling modules importable no matter where streamlit was started from
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import db        # Supabase persistence (users + prediction history + profiles)
import emailer   # forgot-password email via smtplib

# Every artifact the dashboard needs, by the exact names the notebooks save them
# under. model_comparison.csv feeds the transparency caption + the About page.
ARTIFACT_FILES = {
    "pipeline": "salary_pipeline.joblib",            # winner incl. preprocessing
    "q_low": "quantile_lower_pipeline.joblib",       # P25 of the displayed range
    "q_high": "quantile_upper_pipeline.joblib",      # P75 of the displayed range
    "explainer": "shap_explainer.joblib",            # SHAP waterfall
    "feature_names": "feature_names.joblib",         # labels for the waterfall
    "options": "input_options.joblib",               # category/state/type lists
    "skills": "skill_lists.joblib",                  # skill multiselect options
    "titles": "job_title_suggestions.joblib",        # autocomplete + category auto-fill
    "skill_stats": "skill_stats.joblib",             # skill evidence per role/category
}

st.set_page_config(page_title="Malaysia Wage Predictor", page_icon="💼", layout="centered")

# The app's ONE style override (student-approved exception to the no-custom-CSS
# rule): Streamlit always draws a reveal eye inside password inputs, but the
# two password forms here have their own "Show passwords" checkbox, so the
# built-in eye is hidden. Scoped to the two keyed containers that hold password
# fields — no other widget is affected.
st.html("<style>"
        ".st-key-pw_fields div[data-testid='stTextInputRootElement'] button,"
        ".st-key-reset_fields div[data-testid='stTextInputRootElement'] button"
        "{display:none;}"
        "</style>")

# Chart styling: an accessible, colorblind-safe diverging pair (blue = raises,
# red = lowers) plus recessive grey chrome, so the data is the loudest thing.
CHART_RAISE = "#2a78d6"     # blue — factors pushing the salary up
CHART_LOWER = "#e34948"     # red — factors pulling it down
CHART_INK = "#0b0b0b"       # primary text (value labels)
CHART_TEXT = "#52514e"      # secondary text (factor names)
CHART_MUTED = "#898781"     # axis ticks and captions
CHART_GRID = "#e1e0d9"      # hairline gridlines
CHART_BASELINE = "#c3c2b7"  # zero line / axis baseline

# Scenario colors for the Compare Predictions view: the first three categorical
# slots of a validated colorblind-safe palette ordering. Slot 1 is the same blue
# the other charts use. Identity is also carried by the A/B/C scenario letters,
# so color is never the only channel.
SCENARIO_COLORS = ["#2a78d6", "#1baf7a", "#eda100"]

# Education levels for the input selectbox. The list INDEX is the model's
# ordinal edu_level value (0-4) — 01_features extracts the same scale from the
# requirement text of the job ads.
EDU_LEVELS = ["Not specified", "SPM / secondary school", "Diploma",
              "Bachelor's degree", "Master's / PhD"]

# Top navigation pages (the strings double as the segmented-control labels)
NAV_PREDICT = "Predict"
NAV_COMPARE = "Compare Predictions"
NAV_HISTORY = "History"
NAV_PROFILE = "Profile"
NAV_ABOUT = "About the model"
NAV_PAGES = [NAV_PREDICT, NAV_COMPARE, NAV_HISTORY, NAV_PROFILE, NAV_ABOUT]

# An RM effect below this is presented as "limited influence" instead of a
# number — the model's typical error is ~RM960, so tiny effects would be false
# precision. Effects between the two thresholds read as "a small effect".
MIN_MEANINGFUL_RM = 50
SMALL_EFFECT_RM = 250

# The salary-vs-experience charts stop at this many years: only 1.33% of the
# training ads ask for 10+ years (0.57% for 15+), so the curve beyond 10 is
# drawn from almost no data. The curves themselves are still computed to 20
# (the experience-outlook advice reads them past the chart's edge).
CHART_MAX_YEARS = 10

# Skill recommendations must be backed by dataset evidence: the skill has to
# appear in at least this share of the higher-paying ads for the user's role,
# supported by at least this many ads. Title groups are smaller than category
# groups (min 30 ads vs hundreds), so their count threshold is lower.
MIN_SHARE_HIGH = 0.15
MIN_SUPPORT_CATEGORY = 20
MIN_SUPPORT_TITLE = 10

# Title words that signal the LEVEL of a job rather than its field. TF-IDF
# features containing one of these are explained as "job seniority"; the rest
# of the title features are "the type of role".
SENIORITY_WORDS = {"senior", "sr", "junior", "jr", "lead", "head", "chief",
                   "principal", "director", "manager", "executive", "intern",
                   "internship", "trainee", "apprentice", "assistant",
                   "graduate", "fresh", "supervisor"}

# Short names of the concept groups, used in the comparison takeaway sentence
# ("... mainly because of job seniority and the job category").
GROUP_TAKEAWAY_NAMES = {"seniority": "job seniority",
                        "role": "the type of role",
                        "category": "the job category",
                        "location": "location",
                        "experience": "experience",
                        "education": "education",
                        "skills": "skills",
                        "emp_type": "the employment type"}


# ------------------------------------------------------------ auth + database
def load_auth_config():
    with open(APP_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoader)


config = load_auth_config()

# The database (Supabase) is the single source of truth for accounts — every
# account comes from in-app registration. If the database is unreachable or
# not configured, stop with a setup message instead of a stack trace (the
# same friendly-gate pattern as the missing-artifacts check further down).
try:
    credentials = db.load_credentials()
except db.DatabaseError as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    st.error("Could not reach the database. Check the [supabase] settings in "
             ".streamlit/secrets.toml (or the deployed app's Secrets panel), "
             "your internet connection, and that the Supabase project is not "
             "paused.")
    st.stop()

# The cookie signing key lives in st.secrets, NOT in config.yaml: config.yaml
# is committed to git, and anyone who knows the signing key could forge a
# login cookie on the live app.
try:
    cookie_key = st.secrets["auth"]["cookie_key"]
except (KeyError, FileNotFoundError):
    st.error("Missing cookie signing key. Add an [auth] section with "
             "'cookie_key' to .streamlit/secrets.toml (or the deployed "
             "app's Secrets panel).")
    st.stop()


class MultiWordUsernameValidator(stauth.Validator):
    """The library's default username rule rejects spaces, so a natural
    username like "Lulu Man" could not register. This relaxes the rule to
    1-3 words separated by single spaces (the library lowercases and strips
    usernames on both registration and login, so casing stays consistent)."""

    def validate_username(self, username: str) -> bool:
        return bool(re.match(r"^[a-zA-Z0-9_-]{1,20}( [a-zA-Z0-9_-]{1,20}){0,2}$",
                             username))


# One shared validator instance: the Authenticate widget uses it for
# registration, and the forgot-password / reset-link forms call its
# validate_email / validate_password / diagnose_password directly, so every
# entry point enforces the identical rules.
validator = MultiWordUsernameValidator()

authenticator = stauth.Authenticate(
    credentials,
    config["cookie"]["name"],
    cookie_key,
    config["cookie"]["expiry_days"],
    validator=validator,
)

# ---------------------------------------------------- password-reset settings
# The emailed reset link must point back at THIS app. The address cannot be
# discovered from inside a request, so it lives in secrets: localhost while
# developing, the *.streamlit.app address once deployed.
try:
    APP_URL = st.secrets["app"]["url"].rstrip("/")
except (KeyError, FileNotFoundError):
    APP_URL = "http://localhost:8501"

RESET_LINK_TTL_SECONDS = 30 * 60   # a reset link dies 30 minutes after it is requested


def hash_reset_token(token):
    """Reset tokens are stored as SHA-256 hashes: someone who can read the
    database still cannot reconstruct a working reset link from a stored row."""
    return hashlib.sha256(token.encode()).hexdigest()

# The password rules shown next to the "New password" field. Read from the
# library rather than retyped here, so the help text can never drift away from
# the rule the library actually enforces.
PASSWORD_HELP = authenticator.attrs.get("password_instructions",
                                        stauth.params.PASSWORD_INSTRUCTIONS)


# Every prediction-form widget key: the keep-alive loop below preserves these
# across page switches, and logout wipes them in one place.
FORM_WIDGET_KEYS = ["job_title_input", "category_input", "state_input", "type_input",
                    "experience_input", "edu_input", "skills_input", "salary_input"]
# The onboarding and Profile-page editors have their own keys — they are
# seeded from the database, so they only need wiping on logout.
ONBOARD_WIDGET_KEYS = ["onboard_edu", "onboard_state", "onboard_exp", "onboard_skills"]
PROFILE_WIDGET_KEYS = ["profile_edu", "profile_state", "profile_exp", "profile_skills"]
def clear_password_fields():
    """Empty the change-password form. Runs whenever the user menu or the
    'Change password' section is opened or closed, so half-typed passwords are
    never left sitting in the form waiting to be reopened.

    The keys are ASSIGNED empty values, never popped. Popping only deletes the
    server-side copy — the browser's widget manager still remembers the typed
    text for those widgets and re-reports it on the next sync, which is exactly
    the "old input reappears" bug v7.1 shipped with. Assigning marks each key
    app-owned and pushes the empty value down to the browser (Streamlit's
    documented way to clear an input from a callback)."""
    st.session_state["pw_current"] = ""
    st.session_state["pw_new"] = ""
    st.session_state["pw_repeat"] = ""
    st.session_state["pw_show"] = False
    st.session_state.pop("pw_message", None)


def clear_result_on_logout(_callback_info=None):
    """Wipe everything personal on logout — the unsaved prediction, the
    profile-prefill flag, the navigation position and all form/editor inputs —
    so the next visitor on this browser tab starts from a clean app, not the
    previous user's data.

    ONLY pops in here, never assignments. Unlike every other callback in this
    app, the library invokes this one MID-run (its logout is an `if st.button`
    check, not an on_click), after the change-password widgets were already
    created in the run whenever that section is open — and Streamlit raises
    StreamlitAPIException on ASSIGNING to an already-instantiated widget key
    (popping is exempt). v7.2 called clear_password_fields() here and crashed
    the deployed app's logout exactly that way. The password fields need no
    logout wipe anyway: opening the user menu or the section fires their
    on_change clearing first, so the next visitor always sees empty fields."""
    keys = (["last_result", "profile_applied", "nav", "whatif_pick", "pw_message"]
            + FORM_WIDGET_KEYS + ONBOARD_WIDGET_KEYS + PROFILE_WIDGET_KEYS)
    for key in keys:
        st.session_state.pop(key, None)


def submit_password_change(user):
    """Verify the current password, set the new one, then clear the form.

    This runs as a button callback, i.e. BEFORE the next script run — the only
    moment at which Streamlit allows a widget's key to be removed from session
    state. It is the same callback-not-st.rerun() pattern used everywhere else
    in this app.

    All of the actual checking is still done by streamlit-authenticator: its
    controller confirms the new password is non-empty, that both copies match,
    that it differs from the current one, that it meets the password policy,
    and that the current password verifies against the stored bcrypt hash. On
    success it updates the in-memory credentials dict, which we then persist.
    """
    try:
        authenticator.authentication_controller.reset_password(
            user,
            st.session_state.get("pw_current", ""),
            st.session_state.get("pw_new", ""),
            st.session_state.get("pw_repeat", ""),
        )
    except Exception as exc:   # wrong current password, weak new one, mismatch, ...
        # Keep whatever was typed: usually only one field needs correcting.
        st.session_state["pw_message"] = ("error", str(exc))
        return
    db.update_password(user, credentials["usernames"][user]["password"])
    clear_password_fields()   # assigns "" — the browser really clears (v7.2 fix)
    st.session_state["pw_message"] = ("success", "Password changed.")


def leave_reset_page():
    """'Back to log in' on a dead reset link: drop the token from the URL so
    the next run shows the normal entry hub."""
    st.query_params.clear()


def submit_password_reset(reset_username):
    """Set a new password from the emailed reset link, then sign the user in.

    The link already proved control of the account's email address, so no
    current password is asked for. The new password passes the same policy as
    registration; the token is single-use (cleared the moment it is spent)."""
    new = st.session_state.get("rp_new", "")
    repeat = st.session_state.get("rp_repeat", "")
    if not new:
        st.session_state["rp_message"] = ("error", "Please enter a new password.")
        return
    if new != repeat:
        st.session_state["rp_message"] = ("error", "Passwords do not match.")
        return
    if not validator.validate_password(new):
        st.session_state["rp_message"] = ("error", validator.diagnose_password(new))
        return
    new_hash = stauth.Hasher.hash(new)
    db.update_password(reset_username, new_hash)
    db.clear_reset_token(reset_username)   # single use — the link dies here
    credentials["usernames"][reset_username]["password"] = new_hash
    # Sign the user straight in: the token path is the library's own
    # cookie-restore mechanism (the same pattern as auto-login after
    # registration, v6.3), and set_cookie keeps the session across a refresh.
    authenticator.authentication_controller.login(token={"username": reset_username})
    authenticator.cookie_controller.set_cookie()
    st.session_state["just_reset"] = True
    st.session_state["rp_new"] = ""
    st.session_state["rp_repeat"] = ""
    st.session_state["rp_show"] = False
    st.query_params.clear()


# Button callbacks run BEFORE the next script run, so the flag is already set
# when the page redraws — no st.rerun() needed anywhere in this app.
def enter_guest_mode():
    st.session_state["guest_mode"] = True


def exit_guest_mode():
    st.session_state["guest_mode"] = False


def request_login_to_save():
    """The guest clicked 'Log in / Register to save': remember the save intent,
    so the prediction is saved automatically as soon as their login completes."""
    st.session_state["guest_mode"] = False
    st.session_state["pending_save"] = True


def request_prediction():
    """Predict-button callback: set the compute flag and jump the navigation
    back to the Predict page, so a prediction made from anywhere lands the
    user in front of their new result."""
    st.session_state["do_predict"] = True
    st.session_state["nav"] = NAV_PREDICT


def complete_onboarding(username_done, save_profile_too):
    """Onboarding-button callback for both 'Save & continue' and 'Skip'.

    Either way the account is marked onboarded, so the flow appears exactly
    once. The profile fields can all stay empty — everything is optional.

    A brand-new account must start with a CLEAN prediction form: the session
    keep-alive preserves form values across pages, so anything typed earlier
    in this browser session (e.g. while browsing as a guest) would otherwise
    leak into the new account. The form is reset by ASSIGNING blank values
    (see form_defaults — a popped key can be resurrected by the browser).
    The one exception is a guest who clicked 'Log in / Register to save' —
    they registered exactly to keep their prediction, so their inputs and
    result survive to be auto-saved."""
    if save_profile_too:
        edu_choice = st.session_state.get("onboard_edu", EDU_LEVELS[0])
        db.save_profile(username_done, {
            "state": st.session_state.get("onboard_state"),
            "experience_years": int(st.session_state.get("onboard_exp", 0)),
            "edu_level": EDU_LEVELS.index(edu_choice) if edu_choice in EDU_LEVELS else 0,
            "skills": list(st.session_state.get("onboard_skills", [])),
        })
    db.mark_onboarded(username_done)
    if not st.session_state.get("pending_save"):
        reset_form_to_defaults()
        st.session_state.pop("last_result", None)


def save_profile_from_page(username_to_save):
    """'Save profile' callback on the Profile page. Only personal facts are
    stored — job title, category and employment type are prediction-specific
    and deliberately not part of the profile.

    The saved values are also written straight into the prediction form, so
    the change is visible on the Predict page immediately — not only after
    the next login. Safe here: the form widgets never render on the Profile
    page, so these keys are free to assign in a callback."""
    edu_choice = st.session_state.get("profile_edu", EDU_LEVELS[0])
    state_choice = st.session_state.get("profile_state")
    skills_choice = list(st.session_state.get("profile_skills", []))
    experience_choice = int(st.session_state.get("profile_exp", 0))
    db.save_profile(username_to_save, {
        "state": state_choice,
        "experience_years": experience_choice,
        "edu_level": EDU_LEVELS.index(edu_choice) if edu_choice in EDU_LEVELS else 0,
        "skills": skills_choice,
    })
    if state_choice in state_options:
        st.session_state["state_input"] = state_choice
    st.session_state["experience_input"] = max(0, min(experience_choice, 20))
    st.session_state["edu_input"] = (edu_choice if edu_choice in EDU_LEVELS
                                     else EDU_LEVELS[0])
    st.session_state["skills_input"] = [s for s in skills_choice if s in all_skills]
    st.session_state["profile_saved"] = True


# ------------------------------------------- password-reset landing page
# Reached from the emailed link (…?reset_token=…). It renders INSTEAD of the
# entry hub for visitors who are not signed in; a successful reset signs the
# user straight in through the submit callback, which also clears the token
# from the URL — so the next run falls through to the app below.
reset_token_param = st.query_params.get("reset_token")
if reset_token_param and not st.session_state.get("authentication_status"):
    st.title("💼 Malaysia Wage Predictor")
    st.subheader("Set a new password")
    reset_request = db.get_reset_request(hash_reset_token(reset_token_param))
    reset_expired = (bool(reset_request)
                     and time.time() > (reset_request.get("reset_token_expires") or 0))
    if reset_request is None or reset_expired:
        # Same message for unknown and expired tokens: an attacker probing
        # random tokens learns nothing about which ones once existed.
        st.error("This reset link is invalid or has expired. You can request "
                 "a new one from the 'Forgot password' tab on the log-in page.")
        st.button("Back to log in", on_click=leave_reset_page)
    else:
        st.write("Choose a new password for your account. You will be signed "
                 "in right after.")
        # Same reveal pattern as the change-password form: the checkbox below
        # the fields drives the masking, read from session state BEFORE the
        # inputs render (its click reruns the app first).
        rp_type = "default" if st.session_state.get("rp_show", False) else "password"
        with st.container(key="reset_fields"):
            st.text_input("New password", type=rp_type, key="rp_new",
                          help=PASSWORD_HELP, autocomplete="off")
            st.text_input("Repeat new password", type=rp_type, key="rp_repeat",
                          autocomplete="off")
        st.checkbox("Show passwords", key="rp_show")
        st.button("Set new password and sign in", type="primary",
                  on_click=submit_password_reset, args=(reset_request["username"],))
        rp_kind, rp_text = st.session_state.pop("rp_message", (None, None))
        if rp_kind == "error":
            st.error(rp_text)
    st.stop()

# ------------------------------------------------------- entry screen (hub)
# Three ways in: log in, register a new account, or continue as a guest.
# Nothing else renders until one of them happened. The hub lives inside an
# st.empty() so a login that succeeds DURING this run (form submit or a
# restored cookie) can wipe it and fall through to the app in the same run.
if not st.session_state.get("guest_mode") and not st.session_state.get("authentication_status"):
    hub = st.empty()
    with hub.container():
        st.title("💼 Malaysia Wage Predictor")
        st.write("Log in or register to save your predictions — or continue as a guest.")

        tab_login, tab_register, tab_forgot = st.tabs(["Log in", "Register", "Forgot password"])

        with tab_login:
            authenticator.login(location="main")   # also restores a valid login cookie
            if st.session_state.get("authentication_status") is False:
                st.error("Username or password is incorrect.")

        with tab_register:
            try:
                reg_email, reg_username, reg_name = authenticator.register_user(
                    location="main", captcha=False, password_hint=False)
                if reg_email:
                    # The widget validated the input, hashed the password with bcrypt
                    # and put the new account into the credentials dict — persist it.
                    # add_user marks the account for the one-time onboarding flow.
                    db.add_user(reg_username, reg_name, reg_email,
                                credentials["usernames"][reg_username]["password"])
                    # Log the new account in right away — the token path is the
                    # library's own cookie-restore mechanism (it fills the same
                    # session-state keys a form login does), and set_cookie keeps
                    # the session alive across a page refresh. The hub is wiped
                    # below in this same run, so the welcome message is shown in
                    # the app body via the one-shot flag.
                    authenticator.authentication_controller.login(
                        token={"username": reg_username})
                    authenticator.cookie_controller.set_cookie()
                    st.session_state["just_registered"] = True
            except Exception as exc:  # RegisterError: duplicate user/email, weak password, ...
                st.error(str(exc))

        with tab_forgot:
            st.caption("Enter your account's email address and we will send "
                       "you a link to set a new password.")
            fp_email = st.text_input("Email", key="fp_email")
            if st.button("Send reset link", key="fp_send"):
                entered_email = fp_email.strip().lower()
                # Emails are unique in the users table, so at most one account
                # matches. The credentials dict is reloaded on every run, so
                # this needs no extra database call.
                fp_username = next(
                    (u for u, info in credentials["usernames"].items()
                     if (info.get("email") or "").strip().lower() == entered_email),
                    None)
                if not validator.validate_email(entered_email):
                    st.error("That does not look like a valid email address.")
                elif fp_username is None:
                    st.error("No account uses this email address.")
                else:
                    # The link carries a random single-use token; only its
                    # SHA-256 hash is stored, with a 30-minute expiry.
                    reset_token = secrets.token_urlsafe(32)
                    db.set_reset_token(fp_username, hash_reset_token(reset_token),
                                       int(time.time()) + RESET_LINK_TTL_SECONDS)
                    reset_link = f"{APP_URL}/?reset_token={reset_token}"
                    if emailer.send_reset_link_email(entered_email, fp_username,
                                                     reset_link):
                        st.success(f"A reset link has been emailed to "
                                   f"**{entered_email}**. It expires in 30 minutes.")
                    else:
                        st.warning("The reset email could not be sent (SMTP not "
                                   "configured or unreachable), so the link is "
                                   "shown below instead. Open it to set a new "
                                   "password.")
                        st.code(reset_link)

        st.divider()
        st.button("Continue as guest", on_click=enter_guest_mode,
                  help="Predict without an account — results are not saved "
                       "unless you log in later.")

    if st.session_state.get("authentication_status"):
        hub.empty()   # logged in during this run — clear the hub, show the app
    else:
        st.stop()

logged_in = bool(st.session_state.get("authentication_status"))
username = st.session_state.get("username")
display_name = st.session_state.get("name", "")
if logged_in:
    st.session_state["guest_mode"] = False   # a real login always ends guest mode

# Keep-alive for the prediction form: Streamlit drops a widget's session state
# when the widget is not rendered during a run, and the form only renders on
# the Predict page. Re-assigning each key marks it as app-owned state, so the
# inputs survive visits to the other pages. Runs before any widget exists.
for _key in FORM_WIDGET_KEYS:
    if _key in st.session_state:
        st.session_state[_key] = st.session_state[_key]


# ------------------------------------------------------ top bar: title + user
col_title, col_user = st.columns([0.74, 0.26], vertical_alignment="center")
with col_title:
    st.title("💼 Malaysia Wage Predictor")
with col_user:
    if logged_in:
        # Giving the menu and the section a key plus an on_change callback is
        # what makes collapsing them run the app again — without it Streamlit
        # opens and closes these containers purely in the browser, and the
        # password fields could never be cleared on close.
        with st.popover(display_name, width="stretch", key="user_menu",
                        on_change=clear_password_fields):
            st.markdown(f"Signed in as **{display_name}**")
            st.caption(f"Username: {username}")
            password_section = st.expander("Change password", key="pw_expander",
                                           on_change=clear_password_fields)
            with password_section:
                # The fields exist only while the section is open. Together
                # with the on_change callback above this is what empties the
                # form when the user closes it: the inputs are not rendered on
                # the next run, so Streamlit drops their state and reopening
                # the section always starts from blank fields.
                if password_section.open:
                    # This form deliberately does NOT use the library's
                    # reset_password() widget. That widget puts its three
                    # inputs in an st.form with no keys, so their values are
                    # unreachable from session state: they survive a
                    # successful change and reappear when the section is
                    # reopened. Owning the keys here fixes that, and lets one
                    # checkbox drive the masking of all three fields.
                    # Plain widgets rather than st.form, because a widget
                    # inside a form does not take effect until the form is
                    # submitted — the same reason the prediction form dropped
                    # st.form in v2.
                    # The "Show passwords" checkbox sits BELOW the fields, so
                    # its value is read from session state before they render
                    # (clicking it reruns the app, and by then the new value
                    # is already there). autocomplete="off" keeps the browser
                    # password manager from refilling the cleared fields. The
                    # container key scopes the CSS that hides the built-in
                    # reveal eye (the checkbox is the working reveal here).
                    show = st.session_state.get("pw_show", False)
                    field_type = "default" if show else "password"
                    with st.container(key="pw_fields"):
                        st.text_input("Current password", type=field_type,
                                      key="pw_current", autocomplete="off")
                        st.text_input("New password", type=field_type, key="pw_new",
                                      help=PASSWORD_HELP, autocomplete="off")
                        st.text_input("Repeat new password", type=field_type,
                                      key="pw_repeat", autocomplete="off")
                    st.checkbox("Show passwords", key="pw_show")
                    st.button("Change password", key="pw_submit",
                              on_click=submit_password_change, args=(username,))
                    # Popped, not read: the message belongs to the click that
                    # produced it and should not linger on the next interaction.
                    kind, message = st.session_state.pop("pw_message", (None, None))
                    if kind == "success":
                        st.success(message)
                    elif kind == "error":
                        st.error(message)
            authenticator.logout("Logout", "main", callback=clear_result_on_logout)
    else:
        with st.popover("Guest", width="stretch"):
            st.write("Browsing as **guest** — predictions are not saved.")
            st.button("Log in / Register", on_click=exit_guest_mode)

# One-shot confirmation after registering: the account was created inside the
# hub (which is wiped once the auto-login completes), so the message renders
# here in the app body instead — above the onboarding card a new account sees.
if logged_in and st.session_state.pop("just_registered", False):
    st.success(f"Account created — you are signed in as **{display_name}**.")

# Same idea after a reset-link password change: the reset page is gone by the
# time this run renders, so the confirmation shows here in the app body.
if logged_in and st.session_state.pop("just_reset", False):
    st.success(f"Password changed — you are signed in as **{display_name}**.")


# ---------------------------------------------------------- artifact loading
def missing_artifacts():
    """Checked on every rerun (NOT cached), so the app recovers as soon as
    the notebooks finish training and the files appear."""
    return [f for f in ARTIFACT_FILES.values() if not (MODELS_DIR / f).exists()]


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    return {key: joblib.load(MODELS_DIR / fname)
            for key, fname in ARTIFACT_FILES.items()}


missing = missing_artifacts()
if missing:
    st.info(
        "**The models are not trained yet.** Run `notebooks/01_features.ipynb` and "
        "`notebooks/02_models.ipynb` first, then refresh this page."
    )
    st.caption("Missing files in models/: " + ", ".join(missing))
    st.stop()

artifacts = load_artifacts()
skills_info = artifacts["skills"]                    # hard_skills / soft_skills / skill_columns
options = artifacts["options"]
skill_columns = skills_info["skill_columns"]         # display name -> model column
all_skills = skills_info["hard_skills"] + skills_info["soft_skills"]
feature_names = artifacts["feature_names"]
title_suggestions = artifacts["titles"]["titles"]                # ordered by frequency
title_to_category = artifacts["titles"]["title_to_category"]     # lowercased title -> category
title_display = {t.lower(): t for t in title_suggestions}        # lowercased -> nice casing
col_to_skill = {col: skill for skill, col in skill_columns.items()}   # skill_c_plus_plus -> "c++"

# "Malaysia" (a single ad posted for the whole country) and "Others" (the
# cleaning phase's bucket for unmapped locations) are leftovers, not real
# states — the dataset is Malaysia-only. They are hidden from the dropdown;
# the model itself is untouched (OneHotEncoder ignores unknown values anyway).
EXCLUDED_STATE_OPTIONS = {"Malaysia", "Others"}
state_options = [s for s in options["states"] if s not in EXCLUDED_STATE_OPTIONS]


def form_defaults():
    """What a truly blank prediction form holds — one value per widget key.

    Resetting the form means ASSIGNING these values, never popping the keys:
    popping only deletes the server-side copy, and the browser's widget
    manager re-reports whatever was typed before on the next sync (the v7.2
    lesson). That is exactly how one visitor's inputs used to reappear for
    the next account on the same browser tab. An assigned value is pushed
    down to the browser and genuinely replaces the old one."""
    return {
        "job_title_input": None,       # selectbox with index=None starts empty
        "category_input": options["categories"][0],
        "state_input": state_options[0],
        "type_input": ("Full time" if "Full time" in options["types"]
                       else options["types"][0]),
        "experience_input": 0,
        "edu_input": EDU_LEVELS[0],
        "skills_input": [],
        "salary_input": 0,
    }


def reset_form_to_defaults():
    """Overwrite every prediction-form key with its blank-form value."""
    for key, value in form_defaults().items():
        st.session_state[key] = value


# --------------------------------------------------- onboarding (first login)
# Newly registered accounts get one optional setup step before the app: the
# personal facts that make up a profile. Everything can be skipped; either
# button marks the account onboarded, so the flow never appears again.
if logged_in and db.needs_onboarding(username):
    with st.container(border=True):
        st.markdown(f"#### Welcome, {display_name}!")
        st.write("Tell us a little about yourself. These details are saved as "
                 "your profile and prefill the prediction form every time you "
                 "log in — everything is optional, and you can edit it later "
                 "on the **Profile** page.")
        st.selectbox("Highest education", EDU_LEVELS, key="onboard_edu")
        st.selectbox("Where are you based?", state_options, index=None,
                     placeholder="Choose a state (optional)", key="onboard_state")
        st.slider("Years of work experience", 0, 20, 0, key="onboard_exp")
        st.multiselect("Your skills", all_skills, key="onboard_skills",
                       help="Skills detected in Malaysian job ads — pick "
                            "everything that applies.")
        col_save, col_skip = st.columns(2)
        col_save.button("Save & continue", type="primary", width="stretch",
                        on_click=complete_onboarding, args=(username, True))
        col_skip.button("Skip for now", width="stretch",
                        on_click=complete_onboarding, args=(username, False))
    st.stop()


# A guest who clicked "Log in / Register to save" gets the prediction saved
# automatically the moment their login completes (works after registering too,
# because the intent flag stays in the session until a login happens — even
# across the onboarding step).
if logged_in and st.session_state.pop("pending_save", False):
    pending = st.session_state.get("last_result")
    if pending and not pending["saved"]:
        db.save_prediction(username, {
            **pending["inputs"],
            "pred_low": pending["low"], "pred_point": pending["point"],
            "pred_high": pending["high"], "verdict": pending["verdict"],
        })
        pending["saved"] = True
        pending["auto_saved"] = True


# Fresh-login form reset + profile prefill, once per login session. This must
# run BEFORE the form widgets are created — that is the moment Streamlit
# allows programmatic writes to widget keys. A login is a fresh start: every
# form key is ASSIGNED its blank-form value first (assigning, not popping, is
# what actually reaches the browser — a popped key lets the browser resurrect
# whatever the previous visitor on this tab typed, which is how one account's
# inputs used to leak into the next), then the user's own saved profile fills
# the PERSONAL fields (education, location, experience, skills) — job title,
# category and employment type always start fresh. The one exception: a guest
# whose unsaved prediction is on screen keeps their result and inputs (that
# is what they logged in for).
if logged_in and not st.session_state.get("profile_applied"):
    st.session_state["profile_applied"] = True
    if not st.session_state.get("last_result"):
        reset_form_to_defaults()
        profile = db.load_profile(username)
        if profile:
            if profile["state"] in state_options:
                st.session_state["state_input"] = profile["state"]
            st.session_state["experience_input"] = max(0, min(profile["experience_years"], 20))
            if 0 <= profile["edu_level"] < len(EDU_LEVELS):
                st.session_state["edu_input"] = EDU_LEVELS[profile["edu_level"]]
            st.session_state["skills_input"] = [s for s in profile["skills"] if s in all_skills]
            st.session_state["profile_prefilled"] = True   # one-off notice in the form


# ------------------------------------------------------------------ helpers
def build_input_row(job_title, category, state, emp_type, experience, edu_level,
                    selected_skills):
    """One-row DataFrame with every column the training pipeline expects."""
    selected = set(selected_skills)
    row = {
        "job_title": job_title.strip(),
        "category": category,
        "state_clean": state,
        "type_clean": emp_type,
        "experience_years": experience,
        "has_experience_req": 1 if experience > 0 else 0,
        "edu_level": edu_level,
        "has_edu_req": 1 if edu_level > 0 else 0,
    }
    for skill, col in skill_columns.items():
        row[col] = 1 if skill in selected else 0
    row["skill_count"] = len(selected)
    return pd.DataFrame([row])


def feature_group(name):
    """Concept group of one encoded feature, for the plain-language explanation.

    Users think in concepts (seniority, location, skills), not in the model's
    932 encoded columns — so the explanation aggregates SHAP values per group.
    """
    if name in ("experience_years", "has_experience_req"):
        return "experience"
    if name in ("edu_level", "has_edu_req"):
        return "education"
    if name == "skill_count" or name in col_to_skill:
        return "skills"
    if name.startswith("category_"):
        return "category"
    if name.startswith("state_clean_"):
        return "location"
    if name.startswith("type_clean_"):
        return "emp_type"
    # Everything else is a TF-IDF n-gram from the job title. Level words
    # ("senior", "manager") read as seniority; the rest is the type of role.
    for word in name.split():
        if word in SENIORITY_WORDS or (word.endswith("s") and word[:-1] in SENIORITY_WORDS):
            return "seniority"
    return "role"


def shap_group_sums(values, job_title):
    """Sum this prediction's SHAP values per concept group (log-salary space).

    With no job title there are no title features worth talking about, so the
    two title groups are dropped rather than explained as an empty string.
    """
    sums = {}
    for i, name in enumerate(feature_names):
        v = float(values[i])
        if v != 0.0:
            group = feature_group(name)
            sums[group] = sums.get(group, 0.0) + v
    if not (job_title or "").strip():
        sums.pop("seniority", None)
        sums.pop("role", None)
    return sums


def describe_group(group, rm, inputs):
    """Chart label + natural sentence for one concept group of this prediction.

    The verb comes from the SIZE of the RM effect, so small factors read as
    small ("had a small effect", "had limited influence") instead of
    over-claiming — direction alone is not the whole story.
    """
    years = int(inputs.get("experience_years", 0))
    edu = int(inputs.get("edu_level", 0))
    n_skills = len(inputs.get("skills", []))
    edu_name = EDU_LEVELS[min(edu, len(EDU_LEVELS) - 1)]

    if group == "seniority":
        label, subject = "Job seniority", "Job seniority"
    elif group == "role":
        label, subject = "Type of role", "The type of role in your job title"
    elif group == "experience":
        if years == 0:
            label, subject = "No previous experience", "No previous experience"
        else:
            label = f"Experience ({years} yr{'s' if years != 1 else ''})"
            subject = f"Your {years} year{'s' if years != 1 else ''} of experience"
    elif group == "education":
        if edu == 0:
            label, subject = "Education not stated", "Not stating an education level"
        else:
            label = f"Education ({edu_name})"
            subject = f"Your education level ({edu_name})"
    elif group == "skills":
        if n_skills == 0:
            label, subject = "No skills selected", "Not selecting any skills"
        else:
            label, subject = "Selected skills", "Selected skills"
    elif group == "location":
        label = f"Location ({inputs['state']})"
        subject = f"{inputs['state']} location"
    elif group == "category":
        label = f"Category ({inputs['category']})"
        subject = f"The {inputs['category']} category"
    else:  # emp_type
        label = f"{inputs['emp_type']} employment"
        subject = f"{inputs['emp_type']} employment"

    if abs(rm) < MIN_MEANINGFUL_RM:
        verb = "had limited influence"
    elif abs(rm) < SMALL_EFFECT_RM:
        verb = "had a small effect"
    elif rm > 0:
        verb = "increased the estimate"
    else:
        verb = "reduced the estimate"
    return label, f"{subject} {verb}."


def concept_rows(values, inputs, point):
    """This prediction's explanation, one row per concept group.

    Each row carries the chart label, the sentence, and the group's RM / %
    effect. SHAP values are summed per group in log-salary space first, so the
    conversion is the same honest math as before: % = exp(v) - 1 and
    RM = point * (1 - exp(-v)) — what the estimate would lose if the whole
    group's effect were removed. Rows are ordered biggest effect first.
    """
    rows = []
    for group, v in shap_group_sums(values, inputs.get("job_title")).items():
        rm = point * (1 - np.exp(-v))
        pct = (np.exp(v) - 1) * 100
        label, sentence = describe_group(group, rm, inputs)
        rows.append({"group": group, "rm": rm, "pct": pct,
                     "label": label, "sentence": sentence})
    rows.sort(key=lambda r: -abs(r["rm"]))
    return rows


def short_label(name, value):
    """Compact label for one encoded feature (advanced bars + SHAP waterfall)."""
    if name == "experience_years":
        return "Years of experience"
    if name == "has_experience_req":
        return "Experience stated" if value else "No experience stated"
    if name == "skill_count":
        return "Number of skills selected"
    if name == "edu_level":
        if value == 0:
            return "No education specified"
        return f"Education: {EDU_LEVELS[min(int(value), len(EDU_LEVELS) - 1)]}"
    if name == "has_edu_req":
        return "Education stated" if value else "No education stated"
    if name in col_to_skill:
        skill = col_to_skill[name]
        return f"Skill: {skill}" if value else f"Missing skill: {skill}"
    if name.startswith("category_"):
        cat = name[len("category_"):]
        return f"Category: {cat}" if value else f"Not category: {cat}"
    if name.startswith("state_clean_"):
        state_name = name[len("state_clean_"):]
        return f"Working in {state_name}" if value else f"Not in {state_name}"
    if name.startswith("type_clean_"):
        emp = name[len("type_clean_"):]
        return f"Job type: {emp}" if value else f"Not {emp}"
    return f"'{name}' in job title" if value else f"No '{name}' in job title"


def find_evidence_group(job_title, category):
    """Most specific skill-evidence group for this profile.

    Title-level stats ("higher-paying HR Manager ads") when the dataset has
    at least 30 ads for the exact title, otherwise the job category. Returns
    (kind, display label, stats) or (None, None, None) when neither exists.
    """
    stats = artifacts["skill_stats"]
    key = (job_title or "").strip().lower()
    if key and key in stats["by_title"]:
        return "title", title_display.get(key, job_title.strip()), stats["by_title"][key]
    if category in stats["by_category"]:
        return "category", category, stats["by_category"][category]
    return None, None, None


# ------------------------------------------------------------------- charts
def plot_salary_range(low, point, high, offered=None):
    """The hero chart: one horizontal P25-P75 band with a white-ringed marker
    at the point estimate, plus (when the user entered one) a dark tick
    showing where their offered salary sits relative to the market."""
    fig, ax = plt.subplots(figsize=(7, 1.7))
    y = 0
    ax.plot([low, high], [y, y], color=CHART_RAISE, linewidth=9,
            solid_capstyle="round", alpha=0.9)
    ax.plot([point], [y], marker="o", markersize=11, markerfacecolor=CHART_RAISE,
            markeredgecolor="white", markeredgewidth=2, linestyle="none")
    # Band endpoints labeled BELOW, the offer labeled ABOVE — two separate
    # text levels, so the labels cannot collide whatever the offer is.
    ax.annotate(f"RM {low:,.0f}", (low, y), textcoords="offset points",
                xytext=(0, -18), ha="center", fontsize=9, color=CHART_TEXT)
    ax.annotate(f"RM {high:,.0f}", (high, y), textcoords="offset points",
                xytext=(0, -18), ha="center", fontsize=9, color=CHART_TEXT)
    if offered is not None:
        # A vertical tick (not a filled marker) so it stays readable even when
        # the offered salary lands exactly on the estimate marker
        ax.vlines(offered, -0.3, 0.3, color=CHART_INK, linewidth=2)
        ax.annotate(f"Your salary: RM {offered:,.0f}", (offered, 0.3),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=9, color=CHART_INK)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylim(-0.8, 0.9)
    ax.margins(x=0.15)
    fig.tight_layout()
    return fig


def plot_signed_bars(pairs):
    """Horizontal bar chart of (label, % effect) pairs, biggest first.

    The % effects live in multiplicative log-salary space (exp(v) - 1). Blue
    bars push the salary up, red bars pull it down; the pair is colorblind-safe.
    """
    pairs = [(label, pct) for label, pct in pairs if abs(pct) > 1e-6]
    if not pairs:
        return None
    pairs = pairs[::-1]   # barh draws the first row at the bottom -> biggest on top
    labels = [label for label, _ in pairs]
    pcts = [pct for _, pct in pairs]
    colors = [CHART_RAISE if p > 0 else CHART_LOWER for p in pcts]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(pairs) + 1.3))
    ax.barh(range(len(pairs)), pcts, height=0.55, color=colors)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels)
    for y, pct in enumerate(pcts):   # value at the tip of each bar, in plain ink
        ax.annotate(f"{pct:+.0f}%", (pct, y), textcoords="offset points",
                    xytext=(4 if pct > 0 else -4, 0), va="center",
                    ha="left" if pct > 0 else "right", fontsize=9, color=CHART_INK)
    ax.axvline(0, color=CHART_BASELINE, linewidth=1)
    ax.xaxis.grid(True, color=CHART_GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=CHART_MUTED, labelsize=9, length=0)
    plt.setp(ax.get_yticklabels(), color=CHART_TEXT)
    ax.set_xlabel("Effect on the salary estimate (%)", fontsize=9, color=CHART_MUTED)
    ax.margins(x=0.18)   # air for the value labels at the bar tips
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=CHART_RAISE),
               plt.Rectangle((0, 0), 1, 1, color=CHART_LOWER)],
              ["Raises your estimate", "Lowers your estimate"],
              frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_feature_bars(values, enc_values):
    """The advanced per-feature view: the top raw model features as signed bars."""
    top = [i for i in np.argsort(-np.abs(values))[:8] if abs(values[i]) > 1e-4]
    if not top:
        return None
    return plot_signed_bars([(short_label(feature_names[i], enc_values[i]),
                              (np.exp(values[i]) - 1) * 100) for i in top])


def plot_experience_curve(curve, current_years):
    """Line chart: this exact profile re-predicted at 0-10 years of experience,
    with a marker + label at the user's current experience. The chart stops at
    CHART_MAX_YEARS (data past that is too sparse); a profile with more years
    than that shows the curve without a marker."""
    shown = curve[:CHART_MAX_YEARS + 1]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(len(shown)), shown, color=CHART_RAISE, linewidth=2,
            solid_capstyle="round", solid_joinstyle="round")
    if current_years <= CHART_MAX_YEARS:
        ax.plot([current_years], [shown[current_years]], marker="o", markersize=9,
                markerfacecolor=CHART_RAISE, markeredgecolor="white",
                markeredgewidth=2, linestyle="none")
        near_right = current_years > CHART_MAX_YEARS - 3
        # Put the label on the empty side of the marker: below when the curve
        # rises ahead of it, above when it falls — so text never sits on the line
        ahead = shown[min(current_years + 2, len(shown) - 1)]
        label_below = ahead > shown[current_years]
        ax.annotate(f"You now: RM {shown[current_years]:,.0f}",
                    (current_years, shown[current_years]),
                    textcoords="offset points",
                    xytext=(-10 if near_right else 10, -16 if label_below else 10),
                    ha="right" if near_right else "left",
                    fontsize=9, color=CHART_INK)
    ax.yaxis.grid(True, color=CHART_GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(CHART_BASELINE)
    ax.set_xticks(range(0, CHART_MAX_YEARS + 1, 2))
    ax.tick_params(colors=CHART_MUTED, labelsize=9, length=0)
    ax.set_xlabel("Years of experience", fontsize=9, color=CHART_MUTED)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"RM {v:,.0f}"))
    ax.margins(y=0.2)    # air for the "You now" annotation
    fig.tight_layout()
    return fig


# ------------------------------------------------- scenario comparison helpers
def scenario_label(index, job_title):
    """'A — Data Analyst': the letter ties the diff table to both charts,
    so identity never relies on color alone."""
    title = (job_title or "").strip() or "(no job title)"
    if len(title) > 26:
        title = title[:25] + "…"
    return f"{chr(65 + index)} — {title}"


def compare_scenarios(rows):
    """Re-evaluate 2-3 saved history rows with the CURRENT model.

    Returns one dict per scenario: the clipped salary range, the raw inputs
    (for the what-changed table), the per-concept SHAP sums (for the takeaway
    sentence), and the salary-vs-experience curve. No Streamlit calls in here,
    so the test suite can drive it directly with fabricated rows.
    """
    scenarios = []
    for i, row in enumerate(rows):
        skills = [s for s in (row.get("skills") or "").split(", ") if s]
        edu_raw = row.get("edu_level")
        edu = int(edu_raw) if pd.notna(edu_raw) else 0   # pre-v4 rows -> Not specified
        experience = max(0, min(int(row.get("experience_years") or 0), 20))
        job_title = (row.get("job_title") or "").strip()
        X_row = build_input_row(job_title, row["category"],
                                row["state"], row["emp_type"], experience, edu, skills)

        point_raw = float(np.exp(artifacts["pipeline"].predict(X_row))[0])
        low = float(np.exp(artifacts["q_low"].predict(X_row))[0])
        high = float(np.exp(artifacts["q_high"].predict(X_row))[0])
        low, high = min(low, high), max(low, high)
        point = min(max(point_raw, low), high)

        # Concept-level SHAP sums, same grouping as the main explanation —
        # the takeaway sentence names the groups that differ most.
        preprocess = artifacts["pipeline"].named_steps["preprocess"]
        x_enc = np.asarray(preprocess.transform(X_row))
        sv = artifacts["explainer"].shap_values(x_enc)[0]
        group_sums = shap_group_sums(sv, job_title)

        # The same profile at every experience level 0-20, one batched predict
        grid = pd.concat([X_row] * 21, ignore_index=True)
        grid["experience_years"] = list(range(21))
        grid["has_experience_req"] = [1 if y > 0 else 0 for y in range(21)]
        curve = [float(v) for v in np.exp(artifacts["pipeline"].predict(grid))]

        scenarios.append({
            "letter": chr(65 + i),
            "label": scenario_label(i, job_title),
            "job_title": job_title,
            "category": row["category"], "state": row["state"],
            "emp_type": row["emp_type"], "experience": experience,
            "edu_name": EDU_LEVELS[edu] if 0 <= edu < len(EDU_LEVELS) else EDU_LEVELS[0],
            "n_skills": len(skills),
            "low": low, "point": point, "high": high,
            "group_sums": group_sums, "curve": curve,
        })
    return scenarios


def build_diff_table(scenarios):
    """The what-changed table: only the inputs that DIFFER between the
    scenarios, plus the salary estimate — so the reader sees at a glance
    what drives the comparison."""
    candidates = [
        ("Job title", [sc["job_title"] or "(none)" for sc in scenarios]),
        ("Category", [sc["category"] for sc in scenarios]),
        ("State", [sc["state"] for sc in scenarios]),
        ("Employment type", [sc["emp_type"] for sc in scenarios]),
        ("Experience", [f"{sc['experience']} yrs" for sc in scenarios]),
        ("Education", [sc["edu_name"] for sc in scenarios]),
        ("Skills", [f"{sc['n_skills']} skill{'s' if sc['n_skills'] != 1 else ''}"
                    for sc in scenarios]),
    ]
    rows = [(factor, vals) for factor, vals in candidates if len(set(vals)) > 1]
    rows.append(("Salary estimate", [f"RM {sc['point']:,.0f}" for sc in scenarios]))
    table = {"Factor": [factor for factor, _ in rows]}
    for i, sc in enumerate(scenarios):
        table[f"Scenario {sc['letter']}"] = [vals[i] for _, vals in rows]
    return pd.DataFrame(table)


def comparison_takeaway(scenarios):
    """One-sentence explanation of the comparison: which scenario pays most,
    by how much, and because of which concept groups (from the SHAP sums)."""
    hi = max(scenarios, key=lambda sc: sc["point"])
    lo = min(scenarios, key=lambda sc: sc["point"])
    delta = hi["point"] - lo["point"]
    if delta < MIN_MEANINGFUL_RM:
        return "All selected scenarios are predicted almost the same salary."
    groups = set(hi["group_sums"]) | set(lo["group_sums"])
    diffs = {g: hi["group_sums"].get(g, 0.0) - lo["group_sums"].get(g, 0.0)
             for g in groups}
    # Only groups that push the higher scenario UP relative to the lower one
    # can be named as reasons; take the two strongest.
    drivers = [GROUP_TAKEAWAY_NAMES[g] for g, v in
               sorted(diffs.items(), key=lambda kv: -kv[1]) if v > 0.01][:2]
    if not drivers:
        return (f"Scenario {hi['letter']} is **RM {delta:,.0f} higher** than "
                f"scenario {lo['letter']}.")
    return (f"Scenario {hi['letter']} is **RM {delta:,.0f} higher** than "
            f"scenario {lo['letter']}, mainly because of {' and '.join(drivers)}.")


def plot_range_comparison(scenarios):
    """One row per scenario: the P25-P75 interval in the scenario's color with
    a white-ringed marker at the point estimate. Position and the A/B/C letter
    carry identity; the colors match the curves chart below."""
    n = len(scenarios)
    fig, ax = plt.subplots(figsize=(7, 0.9 * n + 1.1))
    for i, sc in enumerate(scenarios):
        y = n - 1 - i                       # first scenario drawn on top
        color = SCENARIO_COLORS[i]
        ax.plot([sc["low"], sc["high"]], [y, y], color=color, linewidth=6,
                solid_capstyle="round")
        ax.plot([sc["point"]], [y], marker="o", markersize=9, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=2, linestyle="none")
        # One combined range label ABOVE the interval (labels at the ends collide
        # with the y-axis names when an interval sits far left) and the muted
        # point estimate BELOW the marker.
        ax.annotate(f"RM {sc['low']:,.0f} – {sc['high']:,.0f}",
                    ((sc["low"] + sc["high"]) / 2, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9, color=CHART_INK)
        ax.annotate(f"RM {sc['point']:,.0f}", (sc["point"], y),
                    textcoords="offset points", xytext=(0, -18), ha="center",
                    fontsize=8, color=CHART_MUTED)
    ax.set_yticks(range(n))
    ax.set_yticklabels([sc["label"] for sc in reversed(scenarios)])
    ax.xaxis.grid(True, color=CHART_GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=CHART_MUTED, labelsize=9, length=0)
    plt.setp(ax.get_yticklabels(), color=CHART_TEXT)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"RM {v:,.0f}"))
    ax.set_xlabel("Predicted monthly salary range",
                  fontsize=9, color=CHART_MUTED)
    ax.margins(x=0.10, y=0.40)   # air for the labels above and below each interval
    fig.tight_layout()
    return fig


def plot_curves_comparison(scenarios):
    """Salary-vs-experience curves of all scenarios on one shared axis, each
    with a marker at that scenario's current experience level. Stops at
    CHART_MAX_YEARS like the main experience chart; scenarios with more years
    than that keep their curve but get no marker."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, sc in enumerate(scenarios):
        color = SCENARIO_COLORS[i]
        shown = sc["curve"][:CHART_MAX_YEARS + 1]
        ax.plot(range(len(shown)), shown, color=color, linewidth=2,
                solid_capstyle="round", solid_joinstyle="round", label=sc["label"])
        if sc["experience"] <= CHART_MAX_YEARS:
            ax.plot([sc["experience"]], [shown[sc["experience"]]], marker="o",
                    markersize=8, markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=2, linestyle="none")
    ax.yaxis.grid(True, color=CHART_GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(CHART_BASELINE)
    ax.set_xticks(range(0, CHART_MAX_YEARS + 1, 2))
    ax.tick_params(colors=CHART_MUTED, labelsize=9, length=0)
    ax.set_xlabel("Years of experience", fontsize=9, color=CHART_MUTED)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"RM {v:,.0f}"))
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def autofill_category():
    """When the chosen job title is a known one, pre-select its most common
    category. Free-text titles the data has never seen simply do nothing."""
    title = (st.session_state.get("job_title_input") or "").strip().lower()
    category = title_to_category.get(title)
    if category and category in options["categories"]:
        st.session_state["category_input"] = category


def save_result_to_history(username_to_save):
    """Save-button callback: persist the current prediction before the rerun.
    The saved flag stops the same result from being written twice."""
    result = st.session_state.get("last_result")
    if result and not result["saved"]:
        db.save_prediction(username_to_save, {
            **result["inputs"],
            "pred_low": result["low"], "pred_point": result["point"],
            "pred_high": result["high"], "verdict": result["verdict"],
        })
        result["saved"] = True


# ------------------------------------------------------------ top navigation
if "nav" not in st.session_state:
    st.session_state["nav"] = NAV_PREDICT
page = st.segmented_control("Page", NAV_PAGES, key="nav",
                            label_visibility="collapsed")
if page is None:            # a segmented control can be de-selected by clicking
    page = NAV_PREDICT      # the active item again — fall back to the home page


# ================================================================ predict page
if page == NAV_PREDICT:
    # ------------------------------------------- sidebar: the prediction form
    # The form renders ONLY on this page; the keep-alive loop near the top of
    # the script preserves its values while other pages are open.
    # One sensible starting value, written to session state BEFORE the widget
    # exists (a keyed widget reads its value from there, and this way no
    # "default value + Session State API" warning is logged): 91% of the
    # training ads are full-time jobs, so that is the natural default. Every
    # other input starts genuinely blank — the experience slider and salary
    # box sit at 0 — so a fresh form never looks like someone else's data.
    if "type_input" not in st.session_state and "Full time" in options["types"]:
        st.session_state["type_input"] = "Full time"

    sidebar = st.sidebar
    sidebar.markdown("### Prediction inputs")
    if st.session_state.pop("profile_prefilled", False):
        sidebar.caption("Education, location, experience and skills were "
                        "prefilled from your saved profile.")

    sidebar.markdown("#### Job details")
    # A restored free-text title may not be in the suggestion list — prepend it
    # for this run so the selectbox accepts the preset value.
    title_options = title_suggestions
    preset_title = st.session_state.get("job_title_input")
    if preset_title and preset_title not in title_suggestions:
        title_options = [preset_title] + title_suggestions
    job_title = sidebar.selectbox(
        "Job title", title_options, index=None, accept_new_options=True,
        placeholder="Start typing, e.g. data ...", key="job_title_input",
        on_change=autofill_category,
        help="Suggestions are the most common titles in Malaysian job ads — "
             "picking one auto-fills the category. Any other title can be typed freely.")
    category = sidebar.selectbox("Job category", options["categories"], key="category_input")
    emp_type = sidebar.selectbox("Employment type", options["types"], key="type_input")
    state = sidebar.selectbox("State", state_options, key="state_input")

    sidebar.markdown("#### Your background")
    experience = sidebar.slider("Years of experience", 0, 20, key="experience_input")
    edu_choice = sidebar.selectbox(
        "Highest education", EDU_LEVELS, key="edu_input",
        help="The model matches you against job ads requiring this level. "
             "Leave 'Not specified' to match ads that do not state one.")
    selected_skills = sidebar.multiselect(
        "Your skills", all_skills, key="skills_input",
        help="Skills detected in Malaysian job ads — pick everything that applies.")

    sidebar.markdown("#### Salary check")
    salary_offered = sidebar.number_input(
        "Salary offered / current (RM)", min_value=0, step=100,
        key="salary_input",
        help="Optional — leave 0 if not applicable. Used only to judge whether "
             "that salary is below, within or above the predicted market range.")

    sidebar.button("Predict my market salary", type="primary",
                   on_click=request_prediction, width="stretch")

    # ------------------------------------------------------- run a prediction
    # The predict button's callback set this flag (and jumped nav back here);
    # the heavy work happens in the script body, not in the callback.
    if st.session_state.pop("do_predict", False):
        job_title = job_title or ""
        edu_level = EDU_LEVELS.index(edu_choice) if edu_choice in EDU_LEVELS else 0
        X_row = build_input_row(job_title, category, state, emp_type,
                                experience, edu_level, selected_skills)
        try:
            point_raw = float(np.exp(artifacts["pipeline"].predict(X_row))[0])
            low = float(np.exp(artifacts["q_low"].predict(X_row))[0])
            high = float(np.exp(artifacts["q_high"].predict(X_row))[0])
        except Exception as exc:  # any artifact/input mismatch ends here, not a crash
            st.error(f"Prediction failed — please check your inputs. ({exc})")
            st.stop()

        # Display rule: lower <= point <= upper (quantile models are separate, so clip)
        low, high = min(low, high), max(low, high)
        point = min(max(point_raw, low), high)

        if salary_offered > 0:
            if salary_offered < low:
                verdict = "BELOW"
            elif salary_offered > high:
                verdict = "ABOVE"
            else:
                verdict = "WITHIN"
        else:
            verdict = ""   # no salary entered, nothing to judge

        # SHAP values for this one prediction (plotted from the stored numbers
        # on every rerun, so the result survives Save clicks and logins)
        try:
            preprocess = artifacts["pipeline"].named_steps["preprocess"]
            x_enc = np.asarray(preprocess.transform(X_row))
            sv = artifacts["explainer"].shap_values(x_enc)
            base = float(np.ravel(artifacts["explainer"].expected_value)[0])
            shap_payload = {"values": sv[0], "base": base, "data": x_enc[0]}
        except Exception as exc:
            shap_payload = {"error": str(exc)}

        # Skill recommendations: re-predict once per missing skill (ONE batched
        # call), then keep only skills the dataset shows are genuinely common
        # in higher-paying ads for this role — a positive model effect alone
        # is never enough to recommend a skill.
        try:
            absent = [s for s in all_skills if s not in set(selected_skills)]
            group_kind, group_label, group_stats = find_evidence_group(job_title, category)
            recs = []
            if absent and group_stats:
                candidates = pd.concat([X_row] * len(absent), ignore_index=True)
                for i, skill in enumerate(absent):
                    candidates.loc[i, skill_columns[skill]] = 1
                    candidates.loc[i, "skill_count"] = len(selected_skills) + 1
                preds = np.exp(artifacts["pipeline"].predict(candidates))
                uplifts = (pd.Series(preds, index=absent) - point_raw).sort_values(
                    ascending=False)
                min_support = (MIN_SUPPORT_TITLE if group_kind == "title"
                               else MIN_SUPPORT_CATEGORY)
                for skill, gain in uplifts.items():
                    if gain <= MIN_MEANINGFUL_RM:
                        break   # sorted by gain — everything after is smaller
                    evidence = group_stats["skills"].get(skill)
                    if evidence is None:
                        continue
                    n_with, share = evidence
                    if share >= MIN_SHARE_HIGH and n_with >= min_support:
                        recs.append({"skill": skill, "gain": float(gain),
                                     "share": float(share), "n_ads": int(n_with)})
                    if len(recs) == 3:
                        break
            tips_payload = {"recs": recs, "group_kind": group_kind,
                            "group_label": group_label, "all_selected": not absent}
        except Exception as exc:
            tips_payload = {"error": str(exc)}

        # Salary-vs-experience curve: the same profile re-predicted at every
        # experience level 0-20 (one batched call), for the line chart
        try:
            grid = pd.concat([X_row] * 21, ignore_index=True)
            grid["experience_years"] = list(range(21))
            grid["has_experience_req"] = [1 if y > 0 else 0 for y in range(21)]
            exp_curve = [float(v) for v in np.exp(artifacts["pipeline"].predict(grid))]
        except Exception:
            exp_curve = None   # chart is skipped; everything else still works

        # Education lever: what happens to this exact profile one level up?
        # (Stored now so the advice survives reruns without re-predicting.)
        try:
            if edu_level < len(EDU_LEVELS) - 1:
                X_edu = X_row.copy()
                X_edu["edu_level"] = edu_level + 1
                X_edu["has_edu_req"] = 1
                edu_point = float(np.exp(artifacts["pipeline"].predict(X_edu))[0])
                edu_uplift = {"next_level": EDU_LEVELS[edu_level + 1],
                              "gain": edu_point - point_raw}
            else:
                edu_uplift = None   # already at the highest level in the data
        except Exception:
            edu_uplift = None

        st.session_state["last_result"] = {
            "inputs": {
                "job_title": job_title.strip(),
                "category": category,
                "state": state,
                "emp_type": emp_type,
                "experience_years": experience,
                "edu_level": edu_level,
                "skills": list(selected_skills),
                "salary_offered": float(salary_offered),
            },
            "low": low, "point": point, "high": high,
            "verdict": verdict,
            "shap": shap_payload,
            "tips": tips_payload,
            "exp_curve": exp_curve,
            "edu_uplift": edu_uplift,
            "saved": False,
        }

    # ------------------------------------------------------------ the results
    # Rendered from session_state, NOT from the button click: this keeps the
    # result on screen across reruns — including a guest logging in to save it.
    result = st.session_state.get("last_result")

    if not result:
        # Friendly empty state until the first prediction is made
        with st.container(border=True):
            st.markdown("#### Welcome")
            st.write("Fill in your details in the **sidebar on the left** and click "
                     "**Predict my market salary**. You will get:")
            st.markdown(
                "- your predicted monthly salary range on the Malaysian market\n"
                "- a verdict on a salary you were offered, whether it is "
                "below, within or above the market range\n"
                "- the factors behind your estimate\n"
                "- career improvement opportunities backed by real job-ad data")
    else:
        inputs = result["inputs"]
        low, point, high = result["low"], result["point"], result["high"]
        offered = inputs["salary_offered"]

        if not inputs["job_title"]:
            st.caption("Tip: adding a job title makes the prediction much more accurate.")

        # ------------------------------------------------------ hero result card
        with st.container(border=True):
            st.metric("Estimated advertised monthly salary", f"RM {point:,.0f}")
            fig = plot_salary_range(low, point, high, offered if offered > 0 else None)
            st.pyplot(fig)
            plt.close(fig)

            if result["verdict"] == "BELOW":
                st.error(f"Your salary of RM {offered:,.0f} is **below the market "
                         f"range** — comparable jobs typically pay "
                         f"RM {low:,.0f} – {high:,.0f}.")
            elif result["verdict"] == "ABOVE":
                st.warning(f"Your salary of RM {offered:,.0f} is **above the market "
                           f"range** — comparable jobs typically pay "
                           f"RM {low:,.0f} – {high:,.0f}.")
            elif result["verdict"] == "WITHIN":
                st.success(f"Your salary of RM {offered:,.0f} is **within the market "
                           f"range** of RM {low:,.0f} – {high:,.0f}.")

        # ---------------------------------------------------- save to history
        if logged_in:
            if result["saved"]:
                if result.get("auto_saved"):
                    st.success("Logged in — this prediction was saved to your history "
                               "automatically. See the **History** page.")
                else:
                    st.success("Saved to your history — see the **History** page.")
            else:
                st.button("Save this prediction to my history",
                          on_click=save_result_to_history, args=(username,))
        else:
            st.info("You are predicting as a **guest** — log in or register and this "
                    "result will be saved to your history automatically.")
            st.button("Log in / Register to save", on_click=request_login_to_save)

        # --------------------------------------------------- SHAP explanation
        with st.container(border=True):
            st.markdown("### Why this estimate?")
            shap_payload = result["shap"]
            if "error" in shap_payload:
                st.warning("Explanation unavailable for this prediction. "
                           f"({shap_payload['error']})")
            else:
                try:
                    values, enc_values = shap_payload["values"], shap_payload["data"]
                    base_rm = float(np.exp(shap_payload["base"]))
                    st.markdown(f"A typical Malaysian job ad pays around "
                                f"**RM {base_rm:,.0f}** per month. Here is what "
                                f"moved your estimate to **RM {point:,.0f}**:")

                    rows = concept_rows(values, inputs, point)
                    shown = [r for r in rows if abs(r["rm"]) >= 1][:7]
                    if not shown:
                        st.caption("No factor moved this estimate meaningfully — "
                                   "it sits very close to the typical job ad.")
                    for r in shown:
                        if abs(r["rm"]) < MIN_MEANINGFUL_RM:
                            icon = "•"
                        elif r["rm"] > 0:
                            icon = "▲"
                        else:
                            icon = "▼"
                        st.markdown(f"{icon} {r['sentence']}")
                        sign = "+" if r["rm"] >= 0 else "−"
                        st.caption(f"≈ {sign}RM {abs(r['rm']):,.0f} ({r['pct']:+.0f}%)")
                    if shown:
                        # Short legend for the numbers above and the coloured
                        # bar chart below: blue/red = direction, % = the
                        # multiplier, RM = how much of the estimate leans on
                        # that one factor. The "don't add up" caveat follows.
                        st.caption("**How to read this:**\n\n"
                                   "- **Blue** raises your estimate, "
                                   "**red** lowers it\n"
                                   "- **%** — how much a factor multiplies "
                                   "the salary up or down\n"
                                   "- **RM** — how much of your estimate "
                                   "depends on that single factor")
                        # Honest note about the arithmetic: each RM figure is
                        # "what the estimate would lose without this factor
                        # alone", and the factors influence each other — so
                        # the figures are not meant to sum to the estimate.
                        st.caption("Each figure shows how much of your "
                                   "estimate rests on that one factor by "
                                   "itself. Because the factors also "
                                   "strengthen and offset one another, adding "
                                   "the figures up will not reproduce the "
                                   "final estimate exactly.")

                    # The same groups as a signed bar chart, for visual readers
                    fig = plot_signed_bars([(r["label"], r["pct"]) for r in shown])
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)

                    # The raw per-feature view stays for the detailed/report
                    # look behind the summary (locked decision 7: waterfall)
                    with st.expander("Advanced model explanation"):
                        st.caption("The raw model view behind the summary above: "
                                   "each encoded feature's own contribution, "
                                   "before grouping into concepts.")
                        fig2 = plot_feature_bars(values, enc_values)
                        if fig2 is not None:
                            st.pyplot(fig2)
                            plt.close(fig2)
                        display_names = [short_label(n, v)
                                         for n, v in zip(feature_names, enc_values)]
                        explanation = shap.Explanation(
                            values=values, base_values=shap_payload["base"],
                            data=enc_values, feature_names=display_names)
                        shap.plots.waterfall(explanation, max_display=12, show=False)
                        st.pyplot(plt.gcf())
                        plt.close("all")
                        st.caption("SHAP's standard waterfall — note its colours are "
                                   "the reverse of the bar charts: red pushes the "
                                   "estimate up, blue pushes it down. The x-axis is "
                                   "log-salary (the model predicts log RM); the bar "
                                   "charts show the same numbers as % effects.")
                except Exception as exc:
                    st.warning(f"Explanation unavailable for this prediction. ({exc})")

        # ---------------------------------------------- salary vs experience
        if result.get("exp_curve"):
            with st.container(border=True):
                st.markdown("### Your salary vs experience")
                st.pyplot(plot_experience_curve(result["exp_curve"],
                                                inputs["experience_years"]))
                plt.close("all")
                st.caption("This curve shows the model's expected salary as "
                           "experience changes, holding everything else in "
                           "your profile fixed.")

        # ------------------------------------------------------ career advice
        with st.container(border=True):
            st.markdown("### Career improvement")

            # Lever 1: skills — recommended only with dataset evidence
            st.markdown("**Skills worth learning**")
            tips_payload = result["tips"]
            if "error" in tips_payload:
                st.warning(f"Skill recommendations unavailable. ({tips_payload['error']})")
            elif tips_payload["all_selected"]:
                st.write("You already selected every skill in the list — impressive.")
            elif not tips_payload["recs"]:
                if tips_payload["group_label"]:
                    st.write(f"No missing skill is both common in higher-paying "
                             f"**{tips_payload['group_label']}** advertisements and "
                             f"predicted to raise your salary — for this profile, "
                             f"experience and the role itself matter more.")
                else:
                    st.write("Not enough advertisements match this role to "
                             "recommend skills with confidence.")
            else:
                kind_word = ("job title" if tips_payload["group_kind"] == "title"
                             else "job category")
                for rec in tips_payload["recs"]:
                    with st.container(border=True):
                        st.markdown(f"**{rec['skill']}**")
                        st.write(f"Common in **{rec['share']:.0%}** of higher-paying "
                                 f"{tips_payload['group_label']} advertisements.")
                        st.write(f"Model-estimated difference: approximately "
                                 f"**+RM {rec['gain']:,.0f}**/month.")
                st.caption(f"Only skills that are genuinely common in higher-paying "
                           f"advertisements for your {kind_word} are recommended.")

            # Lever 2: experience outlook, read off the already-computed curve
            st.markdown("**Experience outlook**")
            curve = result.get("exp_curve")
            years = inputs["experience_years"]
            if not curve:
                st.write("The experience outlook is unavailable for this prediction.")
            elif years >= 20:
                st.write("You are already at the top of the experience scale "
                         "in this data.")
            else:
                future = min(years + 2, 20)
                gain = curve[future] - curve[years]
                if gain > MIN_MEANINGFUL_RM:
                    st.write(f"In **{future - years} more year"
                             f"{'s' if future - years != 1 else ''}** of experience, "
                             f"profiles like yours are predicted at about "
                             f"**RM {curve[future]:,.0f}**/month "
                             f"(+RM {gain:,.0f}).")
                else:
                    st.write("More years of experience alone add little for this "
                             "profile — skills and the role itself matter more here.")

            # Lever 3: the next education level, predicted at predict time
            st.markdown("**Education**")
            edu_now = inputs["edu_level"]
            edu_uplift = result.get("edu_uplift")
            if edu_now >= len(EDU_LEVELS) - 1:
                st.write("You already hold the highest education level in the data.")
            elif edu_uplift and edu_uplift["gain"] > MIN_MEANINGFUL_RM:
                now_name = EDU_LEVELS[edu_now] if edu_now > 0 else "no stated level"
                st.write(f"Job ads requiring a **{edu_uplift['next_level']}** "
                         f"(vs {now_name}) pay about "
                         f"**+RM {edu_uplift['gain']:,.0f}**/month for profiles "
                         f"like yours.")
            else:
                st.write("A higher education level adds little for this profile — "
                         "skills and experience matter more here.")

# ===================================================== compare-predictions page
elif page == NAV_COMPARE:
    st.subheader("Compare predictions")
    if not logged_in:
        st.info("Log in or register, save a few predictions, and compare them "
                "here side by side.")
    else:
        history = db.list_predictions(username)
        if len(history) < 2:
            st.info("Save at least two predictions first (Predict page → "
                    "**Save this prediction to my history**), then compare "
                    "them here side by side.")
        else:
            st.write("Pick two or three saved predictions and see exactly what "
                     "changes between them.")
            # One human-readable label per saved row; the #id keeps labels
            # unique even when two rows share a title and timestamp.
            row_labels = {}
            for _, row in history.iterrows():
                when = pd.to_datetime(row["created_at"]).strftime("%d %b %Y, %H:%M")
                title = (row["job_title"] or "").strip() or "(no job title)"
                row_labels[f"#{row['id']} · {when} · {title}, {row['state']}"] = row["id"]
            chosen = st.multiselect("Saved predictions to compare",
                                    list(row_labels), max_selections=3,
                                    key="whatif_pick")
            if len(chosen) < 2:
                st.caption("Tip: choose 2–3 saved predictions to unlock the comparison.")
            elif st.button("Compare scenarios", type="primary"):
                scenarios = None
                try:
                    with st.spinner("Re-evaluating the selected scenarios..."):
                        chosen_ids = [row_labels[label] for label in chosen]
                        chosen_rows = [history[history["id"] == rid].iloc[0].to_dict()
                                       for rid in chosen_ids]
                        scenarios = compare_scenarios(chosen_rows)
                except Exception as exc:
                    st.warning(f"Comparison unavailable. ({exc})")
                if scenarios:
                    st.markdown("**What changed between the scenarios**")
                    st.dataframe(build_diff_table(scenarios), hide_index=True,
                                 width="stretch")
                    st.info(comparison_takeaway(scenarios))
                    st.pyplot(plot_range_comparison(scenarios))
                    plt.close("all")
                    with st.expander("Salary vs experience"):
                        st.pyplot(plot_curves_comparison(scenarios))
                        plt.close("all")

# ================================================================ history page
elif page == NAV_HISTORY:
    st.subheader("My saved predictions")
    if not logged_in:
        st.info("Log in or register to save predictions and see them here.")
    else:
        history = db.list_predictions(username)
        if history.empty:
            st.info("No saved predictions yet — make a prediction and click "
                    "**Save this prediction to my history**.")
        else:
            st.write(f"**{len(history)}** saved prediction(s). "
                     "Select rows to delete them.")
            display = pd.DataFrame({
                "When": pd.to_datetime(history["created_at"]).dt.strftime("%d %b %Y, %H:%M"),
                "Job title": history["job_title"],
                "Category": history["category"],
                "State": history["state"],
                "Type": history["emp_type"],
                "Exp (yrs)": history["experience_years"],
                "Education": [EDU_LEVELS[int(v)]
                              if pd.notna(v) and 0 <= int(v) < len(EDU_LEVELS)
                              else EDU_LEVELS[0]
                              for v in history["edu_level"]],
                "Skills": history["skills"],
                "Range (RM)": [f"{lo:,.0f} – {hi:,.0f}"
                               for lo, hi in zip(history["pred_low"], history["pred_high"])],
                "Estimate (RM)": [f"{p:,.0f}" for p in history["pred_point"]],
                "Offered (RM)": [f"{s:,.0f}" if s > 0 else "—"
                                 for s in history["salary_offered"]],
                "Verdict": [v if v else "—" for v in history["verdict"]],
            })
            selection = st.dataframe(display, hide_index=True, key="history_table",
                                     on_select="rerun", selection_mode="multi-row")
            selected_rows = selection.selection.rows   # positions into `history`
            selected_ids = history.iloc[selected_rows]["id"].tolist()
            col_del, col_clear = st.columns(2)
            col_del.button("Delete selected", disabled=not selected_rows,
                           on_click=db.delete_predictions, args=(username, selected_ids))
            confirm_clear = col_clear.checkbox("Confirm clearing ALL history")
            col_clear.button("Clear all history", disabled=not confirm_clear,
                             on_click=db.clear_history, args=(username,))
            st.caption("To compare saved predictions side by side, open the "
                       "**Compare Predictions** page.")

# ================================================================ profile page
elif page == NAV_PROFILE:
    st.subheader("Your profile")
    if not logged_in:
        st.info("Log in or register to keep a profile — it prefills the "
                "prediction form on every login.")
    else:
        st.write(f"**{display_name}** · username: {username}")
        st.caption("Your profile stores personal details only — education, "
                   "location, experience and skills. Job title, category and "
                   "employment type belong to each individual prediction and "
                   "are not stored here.")
        # Seed the editor from the database the first time this page renders
        # (and again after logout wiped the keys). The DB stays the source of
        # truth; unsaved edits are discarded when the user navigates away.
        if "profile_edu" not in st.session_state:
            saved = db.load_profile(username) or {}
            edu_saved = saved.get("edu_level", 0)
            st.session_state["profile_edu"] = (
                EDU_LEVELS[edu_saved] if 0 <= edu_saved < len(EDU_LEVELS) else EDU_LEVELS[0])
            st.session_state["profile_state"] = (
                saved.get("state") if saved.get("state") in state_options else None)
            st.session_state["profile_exp"] = max(0, min(saved.get("experience_years", 0), 20))
            st.session_state["profile_skills"] = [s for s in saved.get("skills", [])
                                                  if s in all_skills]
        st.selectbox("Highest education", EDU_LEVELS, key="profile_edu")
        st.selectbox("Where are you based?", state_options, index=None,
                     placeholder="Choose a state (optional)", key="profile_state")
        st.slider("Years of work experience", 0, 20, key="profile_exp")
        st.multiselect("Your skills", all_skills, key="profile_skills",
                       help="Skills detected in Malaysian job ads — pick "
                            "everything that applies.")
        st.button("Save profile", type="primary",
                  on_click=save_profile_from_page, args=(username,))
        if st.session_state.pop("profile_saved", False):
            st.success("Profile saved — the prediction form now uses these "
                       "details, and they will be prefilled on every login.")

# ================================================================== about page
# Deliberately short and disclaimer-toned (v6.1): users need to know what the
# numbers mean and how far to trust them — not the training details.
else:
    st.subheader("About the model")
    st.write("The salary estimates on this dashboard are produced by a "
             "machine-learning model trained on real Malaysian job "
             "advertisements. They describe what the advertised job market "
             "pays for a profile like yours not what any individual "
             "employer will offer.")
    st.write("The band around each estimate shows where half of comparable "
             "advertisements pay, and the single figure is the trained "
             "model's best estimate within that range. The explanation under "
             "a prediction shows which of your inputs pushed the estimate up "
             "or down, and skill suggestions only include skills that are "
             "genuinely common in higher-paying advertisements for your role.")
    st.write("**Please treat the numbers as guidance, not a promise.** Actual "
             "salaries depend on things no model can see such as company "
             "budgets, negotiation, the exact responsibilities of a job. So, "
             "use the estimates to inform your expectations, not to decide them.")
