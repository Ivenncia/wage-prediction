"""AppTest verification for the v6 dashboard (auth hub, guest mode, top
navigation + user menu, onboarding for new registrations, Predict page with
concept-level explanations and evidence-based skill recommendations, Compare
Predictions scenario comparison, History, Profile page, About page).

Run from the project root:  .venv\\Scripts\\python.exe scripts\\test_dashboard.py

Uses Streamlit's built-in testing harness (streamlit.testing.v1.AppTest) to
drive dashboards/app.py without a browser. The dashboard's database module
(dashboards/db.py, Supabase) is pointed at an IN-MEMORY fake client first, so
these tests never touch the network or real data.

Known harness limits (verified manually in a browser instead):
- login cookies (AppTest has no browser cookies),
- st.dataframe row selection (Delete selected in the history page),
- real SMTP email delivery.
"""

import hashlib
import sys
import time
from pathlib import Path

import bcrypt
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "dashboards"))
import db       # the same module objects app.py imports -> overrides reach the app
import emailer

# NEVER send real emails from tests — .streamlit/secrets.toml may hold real SMTP
# credentials. Forcing False also exercises the on-screen fallback path
# deterministically; real delivery is verified manually in a browser.
emailer.send_reset_link_email = lambda *args, **kwargs: False

from streamlit.testing.v1 import AppTest

APP_PATH = str(ROOT / "dashboards" / "app.py")
# Injected via AppTest secrets; >= 32 bytes so PyJWT raises no key-length warning
COOKIE_KEY = "test_cookie_signing_key_for_apptest_only_0000"


# --------------------------------------------------------------- fake database
# The app stores everything in Supabase through dashboards/db.py. Tests must
# not depend on the network or touch real data, so db._client() is replaced
# with an in-memory fake that mimics the small slice of the supabase query API
# db.py actually uses. Every real db.py code path (dict building, DataFrame
# construction, skills parsing) still runs — only the HTTP layer is faked.
# This is the same test seam the SQLite version offered via db.DB_PATH.
class FakeResult:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    """One pending operation on one table, built fluently like the real client:
    table(x).select/insert/upsert/update/delete ... .eq/.in_/.order ... .execute()."""

    PRIMARY_KEYS = {"users": "username", "profiles": "username", "predictions": "id"}

    def __init__(self, client, table):
        self._client = client
        self._table = table
        self._op = None
        self._payload = None
        self._filters = []
        self._order = []

    def select(self, *_columns):          # column list ignored: db.py reads by key
        self._op = "select"
        return self

    def insert(self, row):
        self._op, self._payload = "insert", dict(row)
        return self

    def upsert(self, row):
        self._op, self._payload = "upsert", dict(row)
        return self

    def update(self, values):
        self._op, self._payload = "update", dict(values)
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, column, value):
        self._filters.append(lambda r, c=column, v=value: r.get(c) == v)
        return self

    def in_(self, column, values):
        self._filters.append(lambda r, c=column, v=list(values): r.get(c) in v)
        return self

    def order(self, column, desc=False):
        self._order.append((column, desc))
        return self

    def _matching(self, rows):
        return [r for r in rows if all(f(r) for f in self._filters)]

    def execute(self):
        rows = self._client.tables[self._table]
        pk = self.PRIMARY_KEYS[self._table]
        if self._op == "select":
            result = [dict(r) for r in self._matching(rows)]
            for column, desc in reversed(self._order):   # stable multi-key sort
                result.sort(key=lambda r: r[column], reverse=desc)
            return FakeResult(result)
        if self._op == "insert":
            row = dict(self._payload)
            if self._table == "predictions":
                row["id"] = self._client.next_id
                self._client.next_id += 1
            if any(r[pk] == row.get(pk) for r in rows):
                raise Exception(f'duplicate key violates "{self._table}_pkey"')
            rows.append(row)
            return FakeResult([dict(row)])
        if self._op == "upsert":
            row = dict(self._payload)
            existing = [r for r in rows if r[pk] == row[pk]]
            if existing:
                existing[0].update(row)
            else:
                rows.append(row)
            return FakeResult([dict(row)])
        if self._op == "update":
            matched = self._matching(rows)
            for r in matched:
                r.update(self._payload)
            return FakeResult([dict(r) for r in matched])
        if self._op == "delete":
            removed = self._matching(rows)
            self._client.tables[self._table] = [
                r for r in rows if not any(r is x for x in removed)]
            return FakeResult([dict(r) for r in removed])
        raise AssertionError(f"no operation set for table {self._table}")


class FakeSupabaseClient:
    def __init__(self):
        self.tables = {"users": [], "predictions": [], "profiles": []}
        self.next_id = 1

    def table(self, name):
        return FakeQuery(self, name)


FAKE_DB = FakeSupabaseClient()
db._client = lambda: FAKE_DB
print("Using an in-memory fake Supabase client (no network, no real data)\n")

# Two pre-seeded TEST accounts (bcrypt hashes for demo123 / super123). They
# exist only inside this fake store — the shipped app has no seeded accounts;
# every real account comes from in-app registration.
for _u, _n, _e, _h in [
    ("demo_user", "Demo User", "demo_user@example.com",
     "$2b$12$iO7iFvRtfQA2henT8sQIFuPfup7ZhzbTY5U7GNzvIrP8kkUS6s68u"),
    ("supervisor", "Supervisor", "supervisor@example.com",
     "$2b$12$KnCNvw1Jz9EnH9eNJUkQ7eM.g6g/yx4vtB17ceCCE/7F/ZE4YMmu2"),
]:
    FAKE_DB.tables["users"].append({
        "username": _u, "name": _n, "email": _e, "password_hash": _h,
        "created_at": "2026-07-17T00:00:00", "onboarded": True})

PASSED = []


def check(label, condition, detail=""):
    if condition:
        PASSED.append(label)
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        sys.exit(1)


def fresh_app():
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    # The cookie signing key lives in st.secrets now (not in config.yaml)
    at.secrets["auth"] = {"cookie_key": COOKIE_KEY}
    return at


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


def run_with_password_section_open(at):
    """Run the app with the 'Change password' section expanded.

    The section only renders its fields while it is open, and AppTest does not
    carry an expander's open state from one run to the next (a browser does),
    so the state is re-asserted before every run."""
    at.session_state["pw_expander"] = True
    at.run()
    return at


def password_fields(at):
    """The three change-password inputs, by label -> current value.
    Empty dict when the section is closed and the fields are not rendered."""
    labels = ("Current password", "New password", "Repeat new password")
    return {t.label: t.value for t in at.text_input if t.label in labels}


def submit_password_change(at, current, new, repeat):
    at.text_input(key="pw_current").input(current)
    at.text_input(key="pw_new").input(new)
    at.text_input(key="pw_repeat").input(repeat)
    at.button(key="pw_submit").click()
    return run_with_password_section_open(at)


def stored_password_hash(user):
    return [r for r in FAKE_DB.tables["users"] if r["username"] == user][0]["password_hash"]


def do_login(at, user, password):
    text_inputs(at, "Username")[0].input(user)
    text_inputs(at, "Password")[0].input(password)
    find_button(at, "Login").click()
    at.run()


def do_register(at, first, last, email, username, password):
    """Fill and submit the hub's register tab. Since v6.3 a successful
    registration logs the account in automatically in the same run."""
    text_inputs(at, "First name")[0].input(first)
    text_inputs(at, "Last name")[0].input(last)
    text_inputs(at, "Email")[0].input(email)
    text_inputs(at, "Username")[1].input(username)       # [0] is the login tab's
    text_inputs(at, "Password")[1].input(password)       # [0] is the login tab's
    text_inputs(at, "Repeat password")[0].input(password)
    find_button(at, "Register").click()
    at.run()


EDU_LEVELS = ["Not specified", "SPM / secondary school", "Diploma",
              "Bachelor's degree", "Master's / PhD"]   # must match app.py
NAV_PREDICT = "Predict"                                # must match app.py
NAV_COMPARE = "Compare Predictions"
NAV_HISTORY = "History"
NAV_PROFILE = "Profile"
NAV_ABOUT = "About the model"
NAV_PAGES = [NAV_PREDICT, NAV_COMPARE, NAV_HISTORY, NAV_PROFILE, NAV_ABOUT]


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
check("demo-account hint removed from the landing page",
      "Demo accounts" not in all_text(at) and "demo123" not in all_text(at))
seeded = db.load_credentials()["usernames"]
check("test accounts visible through db.load_credentials()",
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
check("demo_user/demo123 accepted (test-seed hashes valid)",
      at.session_state["authentication_status"] is True)
check("main app rendered after login (no onboarding for pre-seeded accounts)",
      has_button(at, "Predict my market salary"))
check("user menu holds logout + a change-password section",
      has_button(at, "Logout")
      and any(e.label == "Change password" for e in at.expander))
check("change-password fields stay closed until the section is opened",
      not text_inputs(at, "Current password"))
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
welcome_text = all_text(at)
check("welcome card uses the v7.4 wording",
      "whether it is below, within or above the market range" in welcome_text
      and "the factors behind your estimate" in welcome_text
      and "in plain language" not in welcome_text, welcome_text[:400])
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
check("hero estimate metric displayed with the v7.4 label",
      len(at.metric) == 1
      and at.metric[0].label == "Estimated advertised monthly salary",
      str([m.label for m in at.metric]))
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

# --- v6.3: technical model copy removed from the results page ------------------
check("no percentile language anywhere on the results page",
      "percentile" not in page_text and "blue band" not in page_text
      and "P25" not in page_text and "P75" not in page_text)
check("SHAP interaction caveat removed",
      "RM effects are approximate" not in page_text
      and "background sample" not in page_text)
check("factor figures carry the don't-add-up note (v7.4)",
      "will not reproduce the final estimate" in page_text)
check("'Why this estimate?' shows the how-to-read legend (v7.5)",
      "How to read this" in page_text
      and "depends on that single factor" in page_text)
check("experience chart carries the one-sentence disclaimer (v7.6)",
      "holding everything else in your profile fixed" in page_text
      and "under-represented" not in page_text)

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

# --- v6: career improvement section --------------------------------------------
check("career section present with the three levers (v7.6 title)",
      "Career improvement" in page_text
      and "improvement opportunities" not in page_text
      and "Skills worth learning" in page_text
      and "Experience outlook" in page_text and "**Education**" in page_text)
tips = result["tips"]
check("skill recommendations carry evidence fields",
      "recs" in tips and "group_label" in tips, str(tips.keys()))
if tips["recs"]:
    check("recommendation cards show frequency + model difference, no ad-count "
          "caption or 'never enough' tail (v7.6)",
          "Common in" in page_text and "Based on" not in page_text
          and "never enough" not in page_text
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

# --- v7.6: charts stop at 10 years; a >10-year profile must still work ---------
at.slider(key="experience_input").set_value(15)
find_button(at, "Predict my market salary").click()
at.run()
check("15-year profile predicts cleanly (chart shows 0-10, marker skipped)",
      not at.exception
      and session_get(at, "last_result")["inputs"]["experience_years"] == 15
      and len(at.get("image")) >= n_images,
      str(at.exception))
check("curve still computed to 20 years for the experience outlook",
      len(session_get(at, "last_result")["exp_curve"]) == 21)
do_predict(at)   # restore the 5-year prediction — the next section auto-saves it

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
check("history page points to the Compare Predictions page for comparisons",
      "Compare Predictions" in all_text(at))

# ------------------------------------------------------------- 5: registration
print("5. Registration: auto-login, multi-word usernames, duplicate, weak password")
at = fresh_app()
at.run()
do_register(at, "Test", "User", "test.user@example.com", "testuser", "Test@1234")
check("registration auto-logs the new account in",
      session_get(at, "authentication_status") is True
      and session_get(at, "username") == "testuser",
      "; ".join(e.value for e in at.error))
check("welcome message + onboarding shown right after registering",
      any("signed in as" in (s.value or "") for s in at.success)
      and has_button(at, "Skip for now"), all_text(at)[:300])
users = db.load_credentials()["usernames"]
check("new account persisted to the database", "testuser" in users)
check("password stored as working bcrypt hash",
      bcrypt.checkpw(b"Test@1234", users["testuser"]["password"].encode()))

# The registering session is now logged in, so duplicate/weak attempts need a
# fresh hub. Both fail, so they can share one session (the hub stays up).
at = fresh_app()
at.run()
do_register(at, "Test", "User", "other@example.com", "testuser", "Test@1234")
check("duplicate username rejected with an error", len(at.error) > 0)

do_register(at, "Weak", "Pass", "weak@example.com", "weakuser", "abc")
check("weak password rejected with an error", len(at.error) > 0)
check("weak-password account NOT persisted",
      "weakuser" not in db.load_credentials()["usernames"])

# --- v6.3: usernames may be 1-3 words ("Lulu Man") ----------------------------
at = fresh_app()
at.run()
do_register(at, "Lulu", "Man", "lulu.man@example.com", "Lulu Man", "Lulu@1234")
check("two-word username registers and is auto-logged in",
      session_get(at, "authentication_status") is True
      and session_get(at, "username") == "lulu man",
      "; ".join(e.value for e in at.error))
check("two-word account stored lowercased in the database (library lowercases)",
      "lulu man" in db.load_credentials()["usernames"])

at = fresh_app()
at.run()
do_login(at, "Lulu Man", "Lulu@1234")   # login lowercases the typed username too
check("two-word account can log in from a fresh session",
      session_get(at, "authentication_status") is True)

at = fresh_app()
at.run()
do_register(at, "Four", "Words", "four.words@example.com",
            "one two three four", "Four@1234")
check("four-word username rejected with an error", len(at.error) > 0)
check("four-word account NOT persisted",
      "one two three four" not in db.load_credentials()["usernames"])

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
# v7.4: the reset must ASSIGN every form key, never pop it — a popped key is
# resurrected by the browser's widget manager (the v7.2 lesson), which is how
# one account's inputs leaked into the next on the same tab. AppTest cannot
# replay the browser side, so it asserts the server half: every key present.
FORM_KEYS = ["job_title_input", "category_input", "state_input", "type_input",
             "experience_input", "edu_input", "skills_input", "salary_input"]
check("every form key is app-assigned after login (nothing left popped for "
      "the browser to resurrect)",
      all(session_get(at, k, "MISSING") != "MISSING" for k in FORM_KEYS),
      str({k: session_get(at, k, "MISSING") for k in FORM_KEYS}))

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
check("onboarding shown right after registering (auto-login)",
      has_button(at, "Skip for now"))
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

# ------------------------- 7: forgot password = emailed reset link (v7.2)
# The email send is forced to fail (see top of file), so the app must show
# the reset link on screen — which also hands the test the token to open the
# reset landing page with, via AppTest's query_params support.
print("7. Forgot password: reset link, landing page, auto sign-in")
old_hash = db.load_credentials()["usernames"]["testuser"]["password"]
at = fresh_app()
at.run()
text_inputs(at, "Email")[1].input("test.user@example.com")  # [0] = register tab's
find_button(at, "Send reset link").click()
at.run()
check("fallback warning shown (email could not be sent)",
      any("could not be sent" in w.value for w in at.warning),
      "; ".join(e.value for e in at.error))
codes = at.get("code")
check("reset link displayed on screen", len(codes) > 0)
reset_link = codes[0].value
check("link points at the app with a reset_token parameter",
      reset_link.startswith("http://localhost:8501/?reset_token="), reset_link)
reset_token = reset_link.split("reset_token=", 1)[1]
token_row = [r for r in FAKE_DB.tables["users"] if r["username"] == "testuser"][0]
check("only the token's hash is stored, with an expiry in the future",
      token_row.get("reset_token_hash") not in (None, reset_token)
      and token_row.get("reset_token_expires", 0) > time.time())
check("password hash untouched by requesting a link",
      db.load_credentials()["usernames"]["testuser"]["password"] == old_hash)

# Unknown email + invalid format -> explicit errors (student's decision)
text_inputs(at, "Email")[1].input("nobody@example.com")
find_button(at, "Send reset link").click()
at.run()
check("unknown email reported", any("No account uses" in e.value for e in at.error))
text_inputs(at, "Email")[1].input("not-an-email")
find_button(at, "Send reset link").click()
at.run()
check("invalid email format reported",
      any("valid email" in e.value for e in at.error))

# Open the reset link: weak password / mismatch rejected, then success
at = fresh_app()
at.query_params["reset_token"] = reset_token
at.run()
check("reset landing page renders instead of the hub",
      has_button(at, "Set new password and sign in")
      and not has_button(at, "Continue as guest"))

at.text_input(key="rp_new").input("abc")
at.text_input(key="rp_repeat").input("abc")
find_button(at, "Set new password and sign in").click()
at.query_params["reset_token"] = reset_token
at.run()
check("weak new password rejected on the reset page",
      any("Password must" in e.value for e in at.error))

at.text_input(key="rp_new").input("Res3t@Pass1")
at.text_input(key="rp_repeat").input("Different@1")
find_button(at, "Set new password and sign in").click()
at.query_params["reset_token"] = reset_token
at.run()
check("mismatched passwords rejected on the reset page",
      any("do not match" in e.value for e in at.error))

at.text_input(key="rp_new").input("Res3t@Pass1")
at.text_input(key="rp_repeat").input("Res3t@Pass1")
find_button(at, "Set new password and sign in").click()
at.query_params["reset_token"] = reset_token   # still in the URL when submitting
at.run()
check("successful reset signs the user straight in",
      at.session_state["authentication_status"] is True
      and at.session_state["username"] == "testuser")
check("signed-in confirmation shown in the app body",
      any("you are signed in" in s.value for s in at.success))
new_hash = db.load_credentials()["usernames"]["testuser"]["password"]
check("new password persisted to the database",
      bcrypt.checkpw(b"Res3t@Pass1", new_hash.encode()))
check("old password no longer works",
      not bcrypt.checkpw(b"Test@1234", new_hash.encode()))
token_row = [r for r in FAKE_DB.tables["users"] if r["username"] == "testuser"][0]
check("token cleared after use (single use)",
      token_row.get("reset_token_hash") is None
      and token_row.get("reset_token_expires") is None)

# A spent (or garbage) token and an expired token both show the dead-link card
at = fresh_app()
at.query_params["reset_token"] = reset_token
at.run()
check("reused link is rejected",
      any("invalid or has expired" in e.value for e in at.error)
      and not has_button(at, "Set new password and sign in"))

expired_token = "expired-token-for-test"
db.set_reset_token("testuser", hashlib.sha256(expired_token.encode()).hexdigest(),
                   int(time.time()) - 60)   # expiry already in the past
at = fresh_app()
at.query_params["reset_token"] = expired_token
at.run()
check("expired link is rejected (same message as an invalid one)",
      any("invalid or has expired" in e.value for e in at.error))
db.clear_reset_token("testuser")

# Fresh login with the new password still works through the normal hub
at = fresh_app()
at.run()
do_login(at, "testuser", "Res3t@Pass1")
check("new password logs in through the hub",
      at.session_state["authentication_status"] is True)

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
# v7.4: saving the profile applies it to the prediction form immediately —
# the student should not need to log out and back in to see it.
go_to(at, NAV_PREDICT)
check("saved profile applied to the form in the SAME session",
      at.selectbox(key="state_input").value == "Penang"
      and at.slider(key="experience_input").value == 7
      and at.selectbox(key="edu_input").value == "Diploma"
      and list(at.multiselect(key="skills_input").value) == ["python"],
      f"{session_get(at, 'state_input')}/{session_get(at, 'experience_input')}/"
      f"{session_get(at, 'edu_input')}/{session_get(at, 'skills_input')}")

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

# ------------------------------------- 11: Compare Predictions (scenarios)
print("11. Compare Predictions page: pick 2 saved rows, diff table + takeaway")
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
go_to(at, NAV_COMPARE)
check("prediction form hidden on the Compare Predictions page",
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
check("re-evaluation caveat removed (v6.3)", "current model" not in whatif_text)
print("       takeaway: " + next((i.value for i in at.info if "Scenario" in (i.value or "")
                                  or "almost the same" in (i.value or "")), "(none)"))

# --------------------------------- 12: db-layer safety nets (Supabase version)
# The old SQLite migration test is gone: the schema is now managed once in the
# Supabase SQL editor (docs/DEPLOYMENT.md), not by the app. What remains worth
# testing at this layer is the empty-history contract and profile isolation.
print("12. Database layer: empty history shape, per-user profile isolation")
empty = db.list_predictions("nobody_at_all")
check("empty history still carries the full column set",
      empty.empty and list(empty.columns) == db.PREDICTION_COLUMNS,
      str(list(empty.columns)))
check("profiles are per-user (unknown user -> None)",
      db.load_profile("nobody_at_all") is None)
check("save_profile keeps ONE row per user (upsert, not append)",
      len(FAKE_DB.tables["profiles"]) == len(
          {r["username"] for r in FAKE_DB.tables["profiles"]})
      and len(FAKE_DB.tables["profiles"]) >= 2)   # onboarder + demo_user saved above

# --------------------------------------------- 13: change password (v7.1)
# The form in the user menu is rendered by app.py, not by the library's
# reset_password() widget, so that its inputs have keys and can be cleared.
# All validation still goes through the library's authentication controller.
# Runs last: it changes the supervisor account's password for good.
print("13. Change password: reveal toggle, validation, clearing on success/close")
at = fresh_app()
at.run()
do_login(at, "supervisor", "super123")
run_with_password_section_open(at)
check("opening the section renders three empty password fields",
      list(password_fields(at).values()) == ["", "", ""],
      str(password_fields(at)))
check("fields are masked by default",
      [t.proto.type for t in at.text_input if "password" in t.label.lower()] == [1, 1, 1])
check("browser autofill disabled (autocomplete=off on all three fields)",
      all(t.proto.autocomplete == "off" for t in at.text_input
          if "password" in t.label.lower()))
check("built-in reveal eye hidden by the scoped style block (v7.2)",
      any(".st-key-pw_fields" in str(getattr(h, "value", "")
                                     or getattr(h.proto, "body", ""))
          for h in at.get("html")))


def rendered_order(at, labels):
    """(element type, label) pairs in render order, filtered to the given
    labels — used to assert the on-screen ordering of widgets."""
    def walk(block, out):
        for child in getattr(block, "children", {}).values():
            out.append((type(child).__name__, getattr(child, "label", None)))
            walk(child, out)
        return out
    return [pair for pair in walk(at._tree, []) if pair[1] in labels]


check("'Show passwords' sits below the inputs (v7.2 layout)",
      rendered_order(at, ("Repeat new password", "Show passwords"))
      == [("TextInput", "Repeat new password"), ("Checkbox", "Show passwords")],
      str(rendered_order(at, ("Repeat new password", "Show passwords"))))

at.checkbox(key="pw_show").set_value(True)
run_with_password_section_open(at)
check("'Show passwords' unmasks all three fields (type 0 = plain text)",
      [t.proto.type for t in at.text_input if "password" in t.label.lower()] == [0, 0, 0])
at.checkbox(key="pw_show").set_value(False)
run_with_password_section_open(at)

submit_password_change(at, "wrong_password", "Sup3r@New1", "Sup3r@New1")
check("wrong current password rejected",
      any("incorrect" in e.value for e in at.error))
check("a rejected attempt keeps what was typed (only one field needs fixing)",
      password_fields(at)["Current password"] == "wrong_password")
check("a rejected attempt leaves the stored hash alone",
      bcrypt.checkpw(b"super123", stored_password_hash("supervisor").encode()))

submit_password_change(at, "super123", "Sup3r@New1", "Sup3r@New2")
check("mismatched new passwords rejected",
      any("do not match" in e.value for e in at.error))

submit_password_change(at, "super123", "abc", "abc")
check("weak new password rejected (library policy still applies)",
      any("Password must" in e.value for e in at.error))

submit_password_change(at, "super123", "Sup3r@New1", "Sup3r@New1")
check("successful change confirmed on screen",
      any("Password changed" in s.value for s in at.success))
check("all three fields cleared after a successful change",
      list(password_fields(at).values()) == ["", "", ""],
      str(password_fields(at)))
check("new password persisted to the database",
      bcrypt.checkpw(b"Sup3r@New1", stored_password_hash("supervisor").encode()))
check("old password no longer works",
      not bcrypt.checkpw(b"super123", stored_password_hash("supervisor").encode()))
run_with_password_section_open(at)
check("the confirmation does not linger on the next interaction",
      not any("Password changed" in s.value for s in at.success))

# Closing the section: the fields stop rendering, so Streamlit drops their
# state and reopening always starts blank. In a browser the expander's
# on_change callback clears them as well; AppTest cannot click an expander
# header, so the close is driven through its session state instead.
at.text_input(key="pw_current").input("half_typed_secret")
run_with_password_section_open(at)
check("half-typed password is present before the section is closed",
      password_fields(at)["Current password"] == "half_typed_secret")
at.session_state["pw_expander"] = False
at.run()
check("closing the section stops rendering the fields", password_fields(at) == {})
check("closing the section drops the password state",
      not [k for k in at.session_state.filtered_state
           if k.startswith("pw_") and k != "pw_expander"],
      str([k for k in at.session_state.filtered_state if k.startswith("pw_")]))
run_with_password_section_open(at)
check("reopening the section shows empty fields",
      list(password_fields(at).values()) == ["", "", ""],
      str(password_fields(at)))

# v7.3 regression: log out while the Change-password section is OPEN. The
# library applies logout MID-run, after the password widgets were instantiated,
# and Streamlit forbids ASSIGNING session state to instantiated widget keys —
# v7.2's logout wipe did exactly that and crashed the deployed app with
# StreamlitAPIException. The section must stay open during the click run to
# reproduce the browser condition (AppTest drops expander state between runs,
# which is how the original suite missed it).
at.text_input(key="pw_current").input("typed-before-logout")
at.session_state["pw_expander"] = True   # keep the section open, browser-style
find_button(at, "Logout").click()
at.run()
check("logout with the password section open raises no exception (v7.3)",
      not at.exception,
      "; ".join(e.message for e in at.exception))
at.run()   # the library applies its logout mid-run; settle once
check("logout after using the password form lands on the hub",
      has_button(at, "Continue as guest") and not at.exception, str(at.exception))
at = fresh_app()
at.run()
do_login(at, "supervisor", "Sup3r@New1")
check("the new password logs the account in",
      at.session_state["authentication_status"] is True)
run_with_password_section_open(at)
check("password fields are empty for the next login (no logout wipe needed)",
      list(password_fields(at).values()) == ["", "", ""],
      str(password_fields(at)))

print(f"\nAll {len(PASSED)} checks passed.")
