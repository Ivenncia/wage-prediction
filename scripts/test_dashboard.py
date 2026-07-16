"""AppTest verification for the v6 dashboard (auth hub, guest mode, top
navigation + user menu, onboarding for new registrations, Predict page with
concept-level explanations and evidence-based skill recommendations, What-if
scenario comparison, History, Profile page, About page).

Run from the project root:  .venv\\Scripts\\python.exe scripts\\test_dashboard.py

Uses Streamlit's built-in testing harness (streamlit.testing.v1.AppTest) to
drive dashboards/app.py without a browser. The dashboard's SQLite module is
pointed at a TEMPORARY database file first, so these tests never touch the
real data/dashboard.db.

Known harness limits (verified manually in a browser instead):
- login cookies (AppTest has no browser cookies),
- st.dataframe row selection (Delete selected in the history page),
- real SMTP email delivery.
"""

import sys
import tempfile
from pathlib import Path

import bcrypt
import joblib
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "dashboards"))
import db       # the same module objects app.py imports -> overrides reach the app
import emailer

# NEVER send real emails from tests — .streamlit/secrets.toml may hold real SMTP
# credentials. Forcing False also exercises the on-screen fallback path
# deterministically; real delivery is verified manually in a browser.
emailer.send_password_email = lambda *args, **kwargs: False

from streamlit.testing.v1 import AppTest

APP_PATH = str(ROOT / "dashboards" / "app.py")

# All tests share one throw-away database (later tests reuse earlier accounts)
db.DB_PATH = Path(tempfile.mkdtemp()) / "test_dashboard.db"
print(f"Temporary test database: {db.DB_PATH}\n")

PASSED = []


def check(label, condition, detail=""):
    if condition:
        PASSED.append(label)
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        sys.exit(1)


def fresh_app():
    return AppTest.from_file(APP_PATH, default_timeout=300)


def session_get(at, key, default=None):
    """AppTest's session_state proxy has no .get() — emulate it."""
    try:
        return at.session_state[key]
    except KeyError:
        return default


def find_button(at, label_part):
    matches = [b for b in at.button if label_part in (b.label or "")]
    assert matches, f"button containing '{label_part}' not found"
    return matches[0]


def has_button(at, label_part):
    return bool([b for b in at.button if label_part in (b.label or "")])


def text_inputs(at, label):
    """All text inputs with this exact label, in render order.
    Hub order: login(Username, Password) -> register(First name, Last name,
    Email, Username, Password, Repeat password) -> forgot(Username)."""
    return [t for t in at.text_input if t.label == label]


def all_text(at):
    """Every user-visible text snippet: markdown/write plus captions and the
    status boxes (AppTest exposes them as separate element lists)."""
    parts = [(m.value or "") for m in at.markdown]
    for attr in ("caption", "success", "info", "warning", "error"):
        try:
            parts += [(e.value or "") for e in getattr(at, attr)]
        except Exception:
            pass   # element type not present in this Streamlit version
    return " | ".join(parts)


def do_login(at, user, password):
    text_inputs(at, "Username")[0].input(user)
    text_inputs(at, "Password")[0].input(password)
    find_button(at, "Login").click()
    at.run()


EDU_LEVELS = ["Not specified", "SPM / secondary school", "Diploma",
              "Bachelor's degree", "Master's / PhD"]   # must match app.py
NAV_PREDICT = "Predict"                                # must match app.py
NAV_WHATIF = "What-if Analysis"
NAV_HISTORY = "History"
NAV_PROFILE = "Profile"
NAV_ABOUT = "About the model"
NAV_PAGES = [NAV_PREDICT, NAV_WHATIF, NAV_HISTORY, NAV_PROFILE, NAV_ABOUT]


def go_to(at, nav_option):
    """Switch the top navigation (segmented control) and rerun."""
    at.segmented_control(key="nav").set_value(nav_option)
    at.run()


def do_predict(at):
    """Fill the sidebar form and click Predict (assumes app past the hub)."""
    at.selectbox(key="job_title_input").select("Data Analyst")
    at.run()  # on_change fires -> category should auto-fill
    at.selectbox(key="state_input").select("Kuala Lumpur")
    at.slider(key="experience_input").set_value(5)
    at.selectbox(key="edu_input").select("Bachelor's degree")
    at.multiselect(key="skills_input").select("python")
    at.multiselect(key="skills_input").select("sql")
    at.number_input(key="salary_input").set_value(4000)
    find_button(at, "Predict my market salary").click()
    at.run()


# ---------------------------------------------------------------- 1: entry hub
print("1. Entry hub renders for anonymous visitors")
at = fresh_app()
at.run()
check("no uncaught exception", not at.exception, str(at.exception))
check("login/register/forgot + guest button present",
      text_inputs(at, "Username") and text_inputs(at, "Repeat password")
      and has_button(at, "Continue as guest"))
check("prediction UI hidden before auth",
      not has_button(at, "Predict my market salary"))
seeded = db.load_credentials()["usernames"]
check("demo accounts seeded from config.yaml into SQLite",
      set(seeded) == {"demo_user", "supervisor"}, str(set(seeded)))

# ------------------------------------------------------------------- 2: login
print("2. Login: wrong then correct password; user menu in the top bar")
at = fresh_app()
at.run()
do_login(at, "demo_user", "wrong_password")
check("wrong password rejected", at.session_state["authentication_status"] is False)
check("error message shown", any("incorrect" in e.value for e in at.error))

at = fresh_app()
at.run()
do_login(at, "demo_user", "demo123")
check("demo_user/demo123 accepted (seed hashes still valid)",
      at.session_state["authentication_status"] is True)
check("main app rendered after login (no onboarding for seeded accounts)",
      has_button(at, "Predict my market salary"))
check("user menu holds logout + change password",
      has_button(at, "Logout") and bool(text_inputs(at, "Current password")))
check("top navigation shows the five pages",
      list(at.segmented_control(key="nav").options) == NAV_PAGES,
      str(list(at.segmented_control(key="nav").options)))

# -------------------------------------------------- 3: guest predict + autofill
print("3. Guest mode: autocomplete, category auto-fill, prediction outputs")
at = fresh_app()
at.run()
find_button(at, "Continue as guest").click()
at.run()
check("guest reaches the prediction form", not at.exception
      and has_button(at, "Predict my market salary"))
check("welcome empty state shown before the first prediction",
      "Welcome" in all_text(at))
check("transparency caption removed (no error figure, no dataset size)",
      "typical prediction error" not in all_text(at)
      and "31,406" not in all_text(at))
edu_box = at.selectbox(key="edu_input")
check("education selectbox: 5 ordinal levels, default 'Not specified'",
      list(edu_box.options) == EDU_LEVELS and edu_box.value == "Not specified",
      f"{list(edu_box.options)} / {edu_box.value}")

do_predict(at)
check("no uncaught exception during prediction", not at.exception, str(at.exception))
autofilled = session_get(at, "category_input")
check("category auto-filled from job title 'Data Analyst'",
      autofilled == "Information & Communication Technology", str(autofilled))
result = session_get(at, "last_result")
check("result stored in session state", bool(result))
low, point, high = result["low"], result["point"], result["high"]
check("range is ordered: low <= estimate <= high", low <= point <= high,
      f"{low:.0f}/{point:.0f}/{high:.0f}")
expected_verdict = "BELOW" if 4000 < low else ("ABOVE" if 4000 > high else "WITHIN")
check(f"verdict consistent with range (RM4,000 -> {expected_verdict})",
      result["verdict"] == expected_verdict, result["verdict"])
check("SHAP explanation computed", "error" not in result["shap"],
      result["shap"].get("error"))
check("skill recommendations computed", "error" not in result["tips"],
      result["tips"].get("error"))
check("hero estimate metric displayed", len(at.metric) == 1, str(len(at.metric)))
print(f"       Data Analyst / KL / 5 yrs / python+sql -> "
      f"RM {low:,.0f} - {high:,.0f} (estimate RM {point:,.0f}), verdict {result['verdict']}")
check("guest sees log-in-to-save prompt, not a save button",
      has_button(at, "Log in / Register to save")
      and not has_button(at, "Save this prediction"))

# --- long-standing checks that must survive the redesign ----------------------
state_opts = list(at.selectbox(key="state_input").options)
check("state dropdown hides 'Malaysia' and 'Others' (16 real states left)",
      "Malaysia" not in state_opts and "Others" not in state_opts
      and len(state_opts) == 16, str(state_opts))
curve = result.get("exp_curve")
check("experience curve computed (21 positive RM values)",
      curve is not None and len(curve) == 21 and all(v > 0 for v in curve),
      str(curve)[:80] if curve else "None")
page_text = all_text(at)
check("no explanation/tips warnings",
      not any("unavailable" in (w.value or "") for w in at.warning))

# --- v6: concept-level explanation ---------------------------------------------
check("baseline anchor sentence present",
      "A typical Malaysian job ad pays around" in page_text)
check("explanation uses concept sentences (increased/reduced the estimate)",
      "increased the estimate." in page_text or "reduced the estimate." in page_text,
      page_text[:400])
check("factor rows carry RM effects", "≈ +RM" in page_text or "≈ −RM" in page_text)
check("old per-feature phrasing is gone",
      "Working in your favour" not in page_text
      and "in your job title'" not in page_text
      and "' in your job title" not in page_text)
expander_labels = [(getattr(e, "label", "") or "") for e in at.expander]
check("'Advanced model explanation' expander present (waterfall inside)",
      any("Advanced model explanation" in lbl for lbl in expander_labels),
      str(expander_labels))
check("old expander/section titles are gone",
      "Detailed SHAP breakdown" not in page_text
      and "How to raise your market value" not in page_text)

# --- v6: career improvement opportunities --------------------------------------
check("career section present with the three levers",
      "Career improvement opportunities" in page_text
      and "Skills worth learning" in page_text
      and "Experience outlook" in page_text and "**Education**" in page_text)
tips = result["tips"]
check("skill recommendations carry evidence fields",
      "recs" in tips and "group_label" in tips, str(tips.keys()))
if tips["recs"]:
    check("recommendation cards show frequency + supporting-ad evidence",
          "Common in" in page_text and "Based on" in page_text
          and "Model-estimated difference" in page_text, page_text[-600:])
    # Re-verify every displayed skill against the artifact: recommendations
    # must be backed by the data, never by a positive model effect alone.
    stats = joblib.load(ROOT / "models" / "skill_stats.joblib")
    if tips["group_kind"] == "title":
        group = stats["by_title"][tips["group_label"].lower()]
        min_support = 10
    else:
        group = stats["by_category"][tips["group_label"]]
        min_support = 20
    ok = all(rec["gain"] > 50
             and group["skills"][rec["skill"]][0] == rec["n_ads"]
             and rec["n_ads"] >= min_support and rec["share"] >= 0.15
             for rec in tips["recs"])
    check("every recommended skill passes the relevance thresholds", ok, str(tips))
    print(f"       evidence group: {tips['group_kind']} '{tips['group_label']}' -> "
          + ", ".join(f"{r['skill']} ({r['share']:.0%}, +RM{r['gain']:,.0f})"
                      for r in tips["recs"]))
else:
    check("honest fallback shown when no skill passes the evidence filter",
          "No missing skill is both common" in page_text
          or "Not enough advertisements" in page_text, page_text[-600:])
edu_uplift = result.get("edu_uplift")
check("education uplift computed for a Bachelor's profile (next = Master's/PhD)",
      edu_uplift is not None and edu_uplift["next_level"] == "Master's / PhD",
      str(edu_uplift))
n_images = len(at.get("image"))
check("charts rendered (range band, concept bars, advanced view, curve)",
      n_images >= 4, f"images rendered = {n_images}")

# --- v6: form only on the Predict page, values survive the round-trip ----------
go_to(at, NAV_HISTORY)
check("guest history page asks to log in",
      any("Log in or register" in (i.value or "") for i in at.info))
check("prediction form hidden on the history page",
      not has_button(at, "Predict my market salary"))
go_to(at, NAV_ABOUT)
about_text = all_text(at)
check("about page reads as a short disclaimer",
      "machine-learning model" in about_text
      and "trained model's best estimate" in about_text
      and "guidance, not a promise" in about_text, about_text[:400])
check("about page has no report content (comparison/limitations/dataset size)",
      "Model comparison" not in about_text
      and "Honest limitations" not in about_text
      and "31,406" not in about_text
      and len(at.dataframe) == 0)
go_to(at, NAV_PREDICT)
check("form and result survive a nav round-trip",
      at.selectbox(key="job_title_input").value == "Data Analyst"
      and len(at.metric) == 1
      and session_get(at, "last_result") is not None)

# --------------------------- 4: guest result survives login and AUTO-saves
print("4. Guest result survives logging in and is saved automatically")
find_button(at, "Log in / Register to save").click()
at.run()
check("back on the entry hub", bool(text_inputs(at, "Repeat password")))
check("prediction kept in session while logging in",
      session_get(at, "last_result") is not None)
do_login(at, "demo_user", "demo123")
check("result still rendered after login", len(at.metric) == 1)
history = db.list_predictions("demo_user")
check("prediction AUTO-saved right after login (no Save click)",
      len(history) == 1, str(len(history)))
row = history.iloc[0]
check("saved row matches the prediction (incl. education level)",
      row["job_title"] == "Data Analyst" and row["experience_years"] == 5
      and row["verdict"] == expected_verdict
      and abs(row["pred_point"] - point) < 1
      and int(row["edu_level"]) == EDU_LEVELS.index("Bachelor's degree"))
check("auto-save confirmation + double-save prevented",
      any("automatically" in s.value for s in at.success)
      and not has_button(at, "Save this prediction"))
at.run()  # plain rerun: nothing saves twice
check("still exactly one saved row after a plain rerun",
      len(db.list_predictions("demo_user")) == 1)
go_to(at, NAV_HISTORY)
check("history table rendered on the history page", len(at.dataframe) >= 1)
hist_df = at.dataframe[0].value
check("history table shows the education column",
      "Education" in hist_df.columns
      and hist_df["Education"].iloc[0] == "Bachelor's degree",
      str(list(hist_df.columns)))
check("history page points to the What-if page for comparisons",
      "What-if Analysis" in all_text(at))

# ------------------------------------------------------------- 5: registration
print("5. Registration: valid, duplicate, weak password")
at = fresh_app()
at.run()
text_inputs(at, "First name")[0].input("Test")
text_inputs(at, "Last name")[0].input("User")
text_inputs(at, "Email")[0].input("test.user@example.com")
text_inputs(at, "Username")[1].input("testuser")       # [0] is the login tab's
text_inputs(at, "Password")[1].input("Test@1234")      # [0] is the login tab's
text_inputs(at, "Repeat password")[0].input("Test@1234")
find_button(at, "Register").click()
at.run()
check("registration succeeds", any("created" in s.value for s in at.success),
      "; ".join(e.value for e in at.error))
users = db.load_credentials()["usernames"]
check("new account persisted to SQLite", "testuser" in users)
check("password stored as working bcrypt hash",
      bcrypt.checkpw(b"Test@1234", users["testuser"]["password"].encode()))

text_inputs(at, "First name")[0].input("Test")
text_inputs(at, "Last name")[0].input("User")
text_inputs(at, "Email")[0].input("other@example.com")
text_inputs(at, "Username")[1].input("testuser")
text_inputs(at, "Password")[1].input("Test@1234")
text_inputs(at, "Repeat password")[0].input("Test@1234")
find_button(at, "Register").click()
at.run()
check("duplicate username rejected with an error", len(at.error) > 0)

text_inputs(at, "First name")[0].input("Weak")
text_inputs(at, "Last name")[0].input("Pass")
text_inputs(at, "Email")[0].input("weak@example.com")
text_inputs(at, "Username")[1].input("weakuser")
text_inputs(at, "Password")[1].input("abc")
text_inputs(at, "Repeat password")[0].input("abc")
find_button(at, "Register").click()
at.run()
check("weak password rejected with an error", len(at.error) > 0)
check("weak-password account NOT persisted",
      "weakuser" not in db.load_credentials()["usernames"])

# ------------------------------------------- 6: onboarding for new registrations
print("6. Onboarding: shown once for new accounts, skippable, saves a profile")
check("freshly registered account is marked for onboarding",
      db.needs_onboarding("testuser"))
at = fresh_app()
at.run()
do_login(at, "testuser", "Test@1234")
check("registered account can log in", at.session_state["authentication_status"] is True)
onboarding_text = all_text(at)
check("onboarding shown instead of the app on first login",
      "Welcome, Test User" in onboarding_text
      and has_button(at, "Skip for now") and has_button(at, "Save & continue")
      and not has_button(at, "Predict my market salary"), onboarding_text[:300])
find_button(at, "Skip for now").click()
at.run()
check("skipping lands in the app with no profile saved",
      has_button(at, "Predict my market salary")
      and not db.needs_onboarding("testuser")
      and db.load_profile("testuser") is None)
at = fresh_app()
at.run()
do_login(at, "testuser", "Test@1234")
check("onboarding never re-appears after being skipped",
      has_button(at, "Predict my market salary")
      and not has_button(at, "Skip for now"))

# The save path, with a second fresh account created directly in the database
db.add_user("onboarder", "On Boarder", "onboarder@example.com",
            bcrypt.hashpw(b"Onboard@123", bcrypt.gensalt()).decode())
at = fresh_app()
at.run()
do_login(at, "onboarder", "Onboard@123")
check("onboarding shown for the second new account",
      has_button(at, "Save & continue"))
at.selectbox(key="onboard_edu").select("Diploma")
at.selectbox(key="onboard_state").select("Penang")
at.slider(key="onboard_exp").set_value(4)
at.multiselect(key="onboard_skills").select("excel")
find_button(at, "Save & continue").click()
at.run()
check("no uncaught exception after onboarding save", not at.exception, str(at.exception))
prof = db.load_profile("onboarder")
check("onboarding saved the profile (personal fields only)",
      prof == {"state": "Penang", "experience_years": 4,
               "edu_level": EDU_LEVELS.index("Diploma"), "skills": ["excel"]},
      str(prof))
check("form prefilled right after onboarding",
      at.selectbox(key="state_input").value == "Penang"
      and at.slider(key="experience_input").value == 4
      and at.selectbox(key="edu_input").value == "Diploma"
      and list(at.multiselect(key="skills_input").value) == ["excel"]
      and at.selectbox(key="job_title_input").value is None)

# ------------- 6b: v6.1 regressions — new accounts start with a clean form
print("6b. Stale-form fix, pending-save contract, logout wipe")


def do_register(at, first, last, email, username, password):
    text_inputs(at, "First name")[0].input(first)
    text_inputs(at, "Last name")[0].input(last)
    text_inputs(at, "Email")[0].input(email)
    text_inputs(at, "Username")[1].input(username)
    text_inputs(at, "Password")[1].input(password)
    text_inputs(at, "Repeat password")[0].input(password)
    find_button(at, "Register").click()
    at.run()


# (i) guest inputs survive one form-less run (Streamlit drops widget state
# only after a full absent run), so a guest who goes straight to the login
# form would leak their inputs into the next account — the fresh-login reset
# must clear them. demo_user has no saved profile yet at this point.
at = fresh_app()
at.run()
find_button(at, "Continue as guest").click()
at.run()
at.selectbox(key="job_title_input").select("Data Analyst")
at.run()   # autofill sets the category to ICT
find_button(at, "Log in / Register").click()   # the guest popover button
at.run()
check("guest inputs still in session on the hub (the leak path is real)",
      session_get(at, "job_title_input") == "Data Analyst")
do_login(at, "demo_user", "demo123")
check("fresh login resets the form (no guest leftovers, 0 years)",
      at.selectbox(key="job_title_input").value is None
      and at.selectbox(key="category_input").value == "Accounting"
      and at.slider(key="experience_input").value == 0,
      f"{session_get(at, 'job_title_input')}/{session_get(at, 'category_input')}/"
      f"{session_get(at, 'experience_input')}")

# (i-b) the same for a brand-new account, through register + onboarding-skip
at = fresh_app()
at.run()
find_button(at, "Continue as guest").click()
at.run()
at.selectbox(key="job_title_input").select("Data Analyst")
at.run()
at.multiselect(key="skills_input").select("python")
at.run()
find_button(at, "Log in / Register").click()
at.run()
do_register(at, "Stale", "User", "stale@example.com", "staleuser", "Stale@1234")
do_login(at, "staleuser", "Stale@1234")
check("onboarding shown for the new account", has_button(at, "Skip for now"))
find_button(at, "Skip for now").click()
at.run()
check("skipping onboarding lands on a CLEAN form (no leftovers, 0 years)",
      at.selectbox(key="job_title_input").value is None
      and list(at.multiselect(key="skills_input").value) == []
      and at.selectbox(key="category_input").value == "Accounting"
      and at.slider(key="experience_input").value == 0
      and session_get(at, "last_result") is None,
      f"{session_get(at, 'job_title_input')}/{session_get(at, 'skills_input')}/"
      f"{session_get(at, 'category_input')}/{session_get(at, 'experience_input')}")

# (ii) the 'Log in / Register to save' contract must survive the wipe: a guest
# who registered specifically to save a prediction keeps it (and their form)
at = fresh_app()
at.run()
find_button(at, "Continue as guest").click()
at.run()
do_predict(at)
saved_point = session_get(at, "last_result")["point"]
find_button(at, "Log in / Register to save").click()
at.run()
do_register(at, "Save", "User", "save@example.com", "saveuser", "Save@1234")
do_login(at, "saveuser", "Save@1234")
find_button(at, "Skip for now").click()
at.run()
saved_rows = db.list_predictions("saveuser")
check("guest prediction auto-saved to the new account despite onboarding",
      len(saved_rows) == 1 and abs(saved_rows.iloc[0]["pred_point"] - saved_point) < 1,
      str(len(saved_rows)))
# The form itself resets during the multi-run registration (Streamlit drops
# widget state after one form-less run) — the inputs live on in the result.
check("result still rendered for the pending-save flow",
      len(at.metric) == 1
      and session_get(at, "last_result") is not None
      and session_get(at, "last_result")["saved"] is True)

# (iii) logout (now inside the user-menu popover) still wipes the form
at = fresh_app()
at.run()
do_login(at, "demo_user", "demo123")
at.selectbox(key="job_title_input").select("Data Analyst")
at.run()
find_button(at, "Logout").click()
at.run()
at.run()   # streamlit-authenticator applies the logout mid-run; settle once
check("logout returns to the entry hub", has_button(at, "Continue as guest"))
find_button(at, "Continue as guest").click()
at.run()
check("logout wiped the form for the next visitor",
      at.selectbox(key="job_title_input").value is None
      and session_get(at, "last_result") is None)

# ------------------------- 7: forgot password (email send forced to fail)
print("7. Forgot password with email delivery unavailable: on-screen fallback")
old_hash = db.load_credentials()["usernames"]["testuser"]["password"]
at = fresh_app()
at.run()
text_inputs(at, "Username")[2].input("testuser")       # [2] = the forgot tab's
find_button(at, "Submit").click()
at.run()
check("fallback warning shown (email could not be sent)",
      any("could not be sent" in w.value for w in at.warning),
      "; ".join(e.value for e in at.error))
codes = at.get("code")
check("new password displayed on screen", len(codes) > 0)
new_password = codes[0].value
new_hash = db.load_credentials()["usernames"]["testuser"]["password"]
check("new hash persisted to SQLite", new_hash != old_hash)
check("displayed password matches the stored hash",
      bcrypt.checkpw(new_password.encode(), new_hash.encode()))
check("old password no longer works",
      not bcrypt.checkpw(b"Test@1234", new_hash.encode()))

at = fresh_app()
at.run()
text_inputs(at, "Username")[2].input("nobody_here")
find_button(at, "Submit").click()
at.run()
check("unknown username reported", any("not found" in e.value for e in at.error))

# ------------------------------------------------- 8: history delete functions
print("8. History delete/clear (db level; row selection UI needs a browser)")
ids = [db.save_prediction("testuser", {
    "job_title": f"Job {i}", "category": "Accounting", "state": "Selangor",
    "emp_type": "Full time", "experience_years": i, "skills": ["excel"],
    "salary_offered": 0.0, "pred_low": 3000.0, "pred_point": 3500.0,
    "pred_high": 4000.0, "verdict": ""}) for i in range(2)]
db.delete_predictions("testuser", [ids[0]])
check("delete_predictions removes only the selected row",
      db.list_predictions("testuser")["id"].tolist() == [ids[1]])
db.delete_predictions("demo_user", [ids[1]])  # wrong owner -> must be a no-op
check("users cannot delete other users' rows",
      len(db.list_predictions("testuser")) == 1)
db.clear_history("testuser")
check("clear_history empties the user's history",
      db.list_predictions("testuser").empty)

# ------------------------------------- 9: education level changes the prediction
print("9. Education level actually changes the prediction")
at = fresh_app()
at.run()
find_button(at, "Continue as guest").click()
at.run()
at.selectbox(key="job_title_input").select("Data Analyst")
at.run()
find_button(at, "Predict my market salary").click()
at.run()   # education left at the default "Not specified"
point_no_edu = session_get(at, "last_result")["point"]
at.selectbox(key="edu_input").select("Bachelor's degree")
find_button(at, "Predict my market salary").click()
at.run()
point_degree = session_get(at, "last_result")["point"]
check("estimate moves when education changes (input is not cosmetic)",
      abs(point_degree - point_no_edu) > 1,
      f"no edu RM{point_no_edu:,.0f} vs degree RM{point_degree:,.0f}")
print(f"       Data Analyst, no education RM {point_no_edu:,.0f} -> "
      f"Bachelor's degree RM {point_degree:,.0f}")

# ---------------------------------- 10: Profile page: save + prefill on login
print("10. Profile page: personal fields only, prefills the form on login")
at = fresh_app()
at.run()
do_login(at, "demo_user", "demo123")
go_to(at, NAV_PROFILE)
check("profile editor rendered (personal fields, no job fields)",
      at.selectbox(key="profile_edu") is not None
      and not has_button(at, "Predict my market salary"))
at.selectbox(key="profile_edu").select("Diploma")
at.selectbox(key="profile_state").select("Penang")
at.slider(key="profile_exp").set_value(7)
at.multiselect(key="profile_skills").select("python")
find_button(at, "Save profile").click()
at.run()
check("profile-saved confirmation shown",
      any("Profile saved" in (s.value or "") for s in at.success))
prof = db.load_profile("demo_user")
check("profile persisted with personal fields only",
      prof == {"state": "Penang", "experience_years": 7,
               "edu_level": EDU_LEVELS.index("Diploma"), "skills": ["python"]},
      str(prof))

at = fresh_app()          # brand-new session: nothing in session state
at.run()
do_login(at, "demo_user", "demo123")
check("no uncaught exception on prefilled login", not at.exception, str(at.exception))
check("personal fields prefilled; job fields start fresh",
      at.selectbox(key="state_input").value == "Penang"
      and at.slider(key="experience_input").value == 7
      and at.selectbox(key="edu_input").value == "Diploma"
      and list(at.multiselect(key="skills_input").value) == ["python"]
      and at.selectbox(key="job_title_input").value is None,
      f"{session_get(at, 'job_title_input')}/{session_get(at, 'state_input')}/"
      f"{session_get(at, 'experience_input')}/{session_get(at, 'edu_input')}")

# ------------------------------------- 11: What-if scenario comparison
print("11. What-if page: pick 2 saved predictions, diff table + takeaway")
# Give demo_user a second, different history row so 2 rows can be compared.
# edu_level=1 here + the auto-saved Bachelor's row -> both education paths run.
db.save_prediction("demo_user", {
    "job_title": "Software Engineer", "category":
    "Information & Communication Technology", "state": "Selangor",
    "emp_type": "Full time", "experience_years": 3, "edu_level": 1,
    "skills": ["python", "sql"], "salary_offered": 0.0, "pred_low": 4000.0,
    "pred_point": 5000.0, "pred_high": 6000.0, "verdict": ""})

at = fresh_app()
at.run()
do_login(at, "demo_user", "demo123")
go_to(at, NAV_WHATIF)
check("prediction form hidden on the what-if page",
      not has_button(at, "Predict my market salary"))
picker = at.multiselect(key="whatif_pick")
check("scenario picker lists both saved predictions", len(picker.options) == 2,
      str(picker.options))
picker.select(picker.options[0])
picker.select(picker.options[1])
at.run()
find_button(at, "Compare scenarios").click()
at.run()
check("no uncaught exception during comparison", not at.exception, str(at.exception))
check("no comparison warning",
      not any("Comparison unavailable" in (w.value or "") for w in at.warning),
      "; ".join((w.value or "") for w in at.warning))
diff_tables = [d.value for d in at.dataframe
               if "Factor" in getattr(d.value, "columns", [])]
check("what-changed diff table rendered", len(diff_tables) == 1,
      f"{len(at.dataframe)} dataframes")
diff = diff_tables[0]
check("diff table has Scenario A/B columns + a salary estimate row",
      list(diff.columns) == ["Factor", "Scenario A", "Scenario B"]
      and "Salary estimate" in diff["Factor"].values, str(diff))
# Both rows are ICT with 2 skills, so those factors must NOT be listed
check("diff table only lists factors that differ",
      "Category" not in diff["Factor"].values
      and "Skills" not in diff["Factor"].values, str(diff["Factor"].values))
whatif_text = all_text(at)
check("one-line takeaway names the driving concepts",
      ("higher** than" in whatif_text and "mainly because of" in whatif_text)
      or "almost the same" in whatif_text, whatif_text[-500:])
n_images = len(at.get("image"))
check("range chart + experience curves rendered", n_images >= 2,
      f"images rendered = {n_images}")
check("re-evaluation caption shown", "current model" in whatif_text)
print("       takeaway: " + next((i.value for i in at.info if "Scenario" in (i.value or "")
                                  or "almost the same" in (i.value or "")), "(none)"))

# ----------------------------------------- 12: pre-v6 database migration
print("12. Migration: a v3-era database gains edu_level/profiles/onboarded")
import sqlite3

old_db_path = db.DB_PATH
mig_path = Path(tempfile.mkdtemp()) / "v3_schema.db"
conn = sqlite3.connect(mig_path)
conn.execute("""CREATE TABLE users (username TEXT PRIMARY KEY, name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL)""")
conn.execute("""INSERT INTO users VALUES ('olduser', 'Old User', 'old@example.com',
                'hash', '2026-07-14T12:00:00')""")
conn.execute("""CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL, created_at TEXT NOT NULL, job_title TEXT,
                category TEXT, state TEXT, emp_type TEXT, experience_years INTEGER,
                skills TEXT, salary_offered REAL, pred_low REAL, pred_point REAL,
                pred_high REAL, verdict TEXT)""")
conn.execute("""INSERT INTO predictions (username, created_at, job_title, category,
                state, emp_type, experience_years, skills, salary_offered,
                pred_low, pred_point, pred_high, verdict)
                VALUES ('olduser', '2026-07-14T12:00:00', 'Clerk', 'Accounting',
                'Johor', 'Full time', 2, 'excel', 0.0, 2500.0, 2800.0, 3100.0, '')""")
conn.commit()
conn.close()
try:
    db.DB_PATH = mig_path
    db.init_db()
    conn = sqlite3.connect(mig_path)
    pred_cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    user_cols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    check("edu_level column added to the old predictions table",
          "edu_level" in pred_cols, str(pred_cols))
    check("profiles table created", "profiles" in tables, str(tables))
    check("onboarded column added; existing account grandfathered",
          "onboarded" in user_cols and not db.needs_onboarding("olduser"),
          str(user_cols))
    old_hist = db.list_predictions("olduser")
    check("pre-v4 row still listable, edu_level empty (-> 'Not specified')",
          len(old_hist) == 1 and pd.isna(old_hist["edu_level"].iloc[0]))
finally:
    db.DB_PATH = old_db_path

print(f"\nAll {len(PASSED)} checks passed.")
