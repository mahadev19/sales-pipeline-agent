"""
app.py — Streamlit Web UI for the Sales Pipeline Agent
========================================================

PURPOSE:
    Provides a rich, interactive browser-based dashboard for the multi-agent
    B2B sales pipeline. Replaces the need to run CLI commands directly.

PAGES:
    1. 🚀 Run Pipeline    — Trigger the agent pipeline and see live results
    2. 📋 CRM Dashboard   — Browse, filter, and update all CRM leads
    3. 🎯 Priority Dashboard — Ranked leads with charts and KPI metrics
    4. 📊 Monitoring      — System health: metrics, runs, and error logs
    5. 🧠 Memory Explorer  — Browse and semantically search ChromaDB lead memory

USAGE:
    streamlit run app.py

    Open http://localhost:8501 in your browser.
"""

import json
import os
import re
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# Plotly is used by Page 6 (Execution Traces) for the Gantt-style bar chart
# and the run comparison chart. It provides far richer interactivity than
# Streamlit's built-in charts for time-series visualization.
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# Make sure all sibling modules (tools/, memory/, agents/) are importable
# when the app is launched from the project root via `streamlit run app.py`.
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CRM_FILE = ROOT_DIR / "data" / "crm.json"
METRICS_FILE = ROOT_DIR / "data" / "metrics.json"
EVAL_LOG_FILE = ROOT_DIR / "data" / "eval_log.jsonl"
AGENT_LOG_FILE = ROOT_DIR / "agent_log.txt"
TRACES_DIR = ROOT_DIR / "data" / "traces"
MCP_URL = os.getenv("MCP_URL", "http://localhost:8001")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — Must be the very first Streamlit call
# Sets the browser tab title, icon, and sidebar layout mode.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Pipeline Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium dark-mode UI with glassmorphism effects
# Injects custom CSS to override Streamlit's default styles for a polished look.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import modern Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global dark background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
    }

    /* Expander styling */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
    }

    /* Info/success/warning boxes */
    .stAlert {
        border-radius: 12px;
    }

    /* Score badge styles (used via st.markdown) */
    .badge-hot {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 13px; display: inline-block;
    }
    .badge-warm {
        background: linear-gradient(135deg, #f9ca24, #f0932b);
        color: #1a1a1a; padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 13px; display: inline-block;
    }
    .badge-cold {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 13px; display: inline-block;
    }
    .badge-none {
        background: rgba(255,255,255,0.15);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 13px; display: inline-block;
    }

    /* Priority bar colors */
    .bar-hot   { background: linear-gradient(90deg, #ff6b6b, #ee5a24); border-radius: 4px; height: 8px; }
    .bar-warm  { background: linear-gradient(90deg, #f9ca24, #f0932b); border-radius: 4px; height: 8px; }
    .bar-cold  { background: linear-gradient(90deg, #74b9ff, #0984e3); border-radius: 4px; height: 8px; }

    /* Company card */
    .company-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Page header */
    .page-header {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #a29bfe, #6c5ce7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .page-subtitle {
        color: rgba(255,255,255,0.5);
        font-size: 14px;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS — Shared utilities used across multiple pages
# ─────────────────────────────────────────────────────────────────────────────

def load_crm() -> dict:
    """Load and parse the CRM JSON file. Returns empty structure on failure."""
    if not CRM_FILE.exists():
        return {"leads": [], "contacts": [], "deals": []}
    try:
        with open(CRM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"⚠️ Failed to read CRM: {e}")
        return {"leads": [], "contacts": [], "deals": []}


def save_crm(data: dict) -> bool:
    """Write updated CRM dict back to crm.json. Returns True on success."""
    try:
        with open(CRM_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"⚠️ Failed to save CRM: {e}")
        return False


def load_metrics() -> dict:
    """Load metrics.json, returning defaults if the file is missing or malformed."""
    if not METRICS_FILE.exists():
        return {}
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def tier_badge_html(tier: str | None) -> str:
    """Return an HTML badge string for a given tier (Hot / Warm / Cold / None)."""
    tier_str = (tier or "").strip().title()
    css_class = {"Hot": "badge-hot", "Warm": "badge-warm", "Cold": "badge-cold"}.get(tier_str, "badge-none")
    label = tier_str if tier_str in ("Hot", "Warm", "Cold") else "—"
    return f'<span class="{css_class}">{label}</span>'


def score_color(score) -> str:
    """Return a color hex string based on the numeric lead score."""
    if score is None:
        return "#aaaaaa"
    s = int(score)
    if s >= 80:
        return "#ff6b6b"   # red-hot
    elif s >= 60:
        return "#f9ca24"   # amber-warm
    else:
        return "#74b9ff"   # blue-cold


def parse_log_runs(log_path: Path, max_lines: int = 2000) -> list[dict]:
    """
    Parse agent_log.txt for COMPANY_START / COMPANY_END pairs.
    Returns a list of run records with company name, status, and timestamp.
    Only reads the last `max_lines` lines of the log file for performance.
    """
    if not log_path.exists():
        return []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-max_lines:]
    except Exception:
        return []

    runs = {}  # keyed by company name
    records = []

    for line in lines:
        line = line.strip()
        # Match COMPANY_START log entries
        if "COMPANY_START" in line:
            # Extract timestamp and company name
            ts_match = re.search(r"\[([^\]]+)\]", line)
            co_match = re.search(r"company=([^\s|]+)", line)
            if ts_match and co_match:
                runs[co_match.group(1)] = {
                    "company": co_match.group(1),
                    "started_at": ts_match.group(1),
                    "status": "running",
                    "ended_at": None,
                }
        # Match COMPANY_END log entries
        elif "COMPANY_END" in line:
            ts_match = re.search(r"\[([^\]]+)\]", line)
            co_match = re.search(r"company=([^\s|]+)", line)
            status_match = re.search(r"status=(\w+)", line)
            if co_match:
                company = co_match.group(1)
                rec = runs.get(company, {"company": company, "started_at": None})
                rec["ended_at"] = ts_match.group(1) if ts_match else None
                rec["status"] = status_match.group(1) if status_match else "unknown"
                records.append(rec)

    return records[-50:]  # Return last 50 runs


def get_error_lines(log_path: Path, max_errors: int = 30) -> list[str]:
    """Extract error and warning lines from agent_log.txt."""
    if not log_path.exists():
        return []
    errors = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                # Look for error/warning keywords in log entries
                if any(kw in stripped.upper() for kw in ("ERROR", "FAILED", "EXCEPTION", "WARNING", "CRITICAL")):
                    errors.append(stripped)
    except Exception:
        pass
    return errors[-max_errors:]


def get_priority_leads(crm_data: dict) -> list[dict]:
    """
    Calculate priority scores for all leads using the priority formula:
        Priority = (score * 0.4) + recency_bonus + status_bonus + (eval_quality * 0.1)
    Returns leads sorted by priority descending.
    """
    # Load eval quality scores from eval_log.jsonl
    eval_scores = {}
    if EVAL_LOG_FILE.exists():
        try:
            with open(EVAL_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            company = rec.get("company", "")
                            score = rec.get("research_eval_score", 0)
                            if company:
                                eval_scores[company.lower()] = score
                        except Exception:
                            pass
        except Exception:
            pass

    STATUS_BONUS = {
        "new": 10, "approved": 10, "contacted": 20,
        "replied": 30, "qualified": 30,
        "meeting_booked": 40, "proposal": 40,
        "won": 50, "closed_won": 50,
    }

    dt_now = datetime.now(timezone.utc)
    prioritized = []

    for lead in crm_data.get("leads", []):
        score = lead.get("score") or 0
        status = (lead.get("status") or "new").lower()
        created_at_str = lead.get("created_at", "")
        company = lead.get("company", "Unknown")

        # Calculate recency bonus based on creation date
        rec_bonus = 0
        try:
            dt_created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            hours_old = (dt_now - dt_created).total_seconds() / 3600
            if hours_old <= 24:
                rec_bonus = 20
            elif hours_old <= 168:
                rec_bonus = 10
        except Exception:
            pass

        stat_bonus = STATUS_BONUS.get(status, 0)
        eval_qual = eval_scores.get(company.lower(), 0)
        priority = int(round((score * 0.4) + rec_bonus + stat_bonus + (eval_qual * 0.1)))

        prioritized.append({
            **lead,
            "priority": priority,
            "recency_bonus": rec_bonus,
            "status_bonus": stat_bonus,
        })

    return sorted(prioritized, key=lambda x: x["priority"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# The sidebar contains the app logo, page navigation radio buttons, and
# a quick-status footer showing CRM lead count and memory stats.
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / branding
    st.markdown("### 🤖 Sales Pipeline Agent")
    st.markdown("<small style='color:rgba(255,255,255,0.4)'>Powered by Google ADK + FastMCP</small>", unsafe_allow_html=True)
    st.divider()

    # Main navigation — each option maps to a page rendering function below
    page = st.radio(
        "Navigation",
        options=[
            "🚀 Run Pipeline",
            "📋 CRM Dashboard",
            "🎯 Priority Dashboard",
            "📊 Monitoring",
            "🧠 Memory Explorer",
            "🔍 Execution Traces",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Quick CRM Stats in Sidebar ──────────────────────────────────────────
    # Shows live counts without requiring the user to navigate to CRM page
    crm_sidebar = load_crm()
    leads_sidebar = crm_sidebar.get("leads", [])
    hot_count = sum(1 for l in leads_sidebar if l.get("tier") == "Hot")
    warm_count = sum(1 for l in leads_sidebar if l.get("tier") == "Warm")
    cold_count = sum(1 for l in leads_sidebar if l.get("tier") == "Cold")

    st.markdown("**📊 Quick Stats**")
    st.markdown(f"Total Leads: **{len(leads_sidebar)}**")
    st.markdown(f"🔥 Hot: **{hot_count}** &nbsp; 🌤 Warm: **{warm_count}** &nbsp; ❄️ Cold: **{cold_count}**",
                unsafe_allow_html=True)

    # Memory stats — load from ChromaDB if available
    try:
        from memory.vector_memory import get_memory_stats
        mem_count = get_memory_stats()
        st.markdown(f"🧠 Memory: **{mem_count}** stored leads")
    except Exception:
        st.markdown("🧠 Memory: *unavailable*")

    st.divider()
    st.caption("Run: `streamlit run app.py`")


# =============================================================================
# PAGE 1: 🚀 RUN PIPELINE
# =============================================================================
def page_run_pipeline():
    """
    Pipeline Runner Page
    ────────────────────
    Lets the user input company names, configure options, and trigger the
    main.py pipeline. Displays live status updates and results per company
    once the pipeline finishes.

    Implementation Note:
        We call `main.py` as a subprocess rather than importing it directly.
        This avoids event-loop conflicts between Streamlit's runtime and
        the asyncio loops used inside the ADK agent pipeline.
    """
    st.markdown('<div class="page-header">🚀 Run Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Research, score, and draft outreach for new companies</div>',
                unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        # Main company input — accepts comma-separated list
        companies_input = st.text_input(
            "Enter company names (comma separated)",
            placeholder="e.g. Stripe, Notion, HubSpot",
            help="The pipeline will research each company individually.",
        )
    with col2:
        # Auto-approve toggle — passes --auto-approve flag to main.py
        auto_approve = st.toggle(
            "Auto-approve outreach",
            value=False,
            help="If ON, skips the human review step and saves all drafts automatically.",
        )

    # Optional: fast mode skips Gemini API for outreach drafting
    fast_mode = st.checkbox(
        "⚡ Fast mode (template-based drafts, no API calls for outreach)",
        value=False,
        help="Uses pre-written templates for email/LinkedIn drafts. Much faster.",
    )

    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    # ── Pipeline Execution ────────────────────────────────────────────────────
    if run_btn:
        companies_raw = companies_input.strip()
        if not companies_raw:
            st.warning("⚠️ Please enter at least one company name.")
            return

        # Parse and clean company names
        companies = [c.strip() for c in companies_raw.split(",") if c.strip()]
        st.info(f"🎯 Running pipeline for: **{', '.join(companies)}**")

        # Build the command to invoke main.py via the venv python
        venv_python = str(ROOT_DIR / "venv" / "Scripts" / "python.exe")
        if not Path(venv_python).exists():
            venv_python = sys.executable  # fallback to current interpreter

        cmd = [
            venv_python,
            str(ROOT_DIR / "main.py"),
            "--companies", ",".join(companies),
        ]
        if auto_approve:
            cmd.append("--auto-approve")
        if fast_mode:
            cmd.append("--fast")

        # ── Progress Display ─────────────────────────────────────────────────
        # We display a progress bar and simulated status messages.
        # The real pipeline runs in the background; we poll stdout for live output.
        progress_bar = st.progress(0, text="Starting pipeline...")
        status_area = st.empty()  # placeholder for dynamic status messages
        log_expander = st.expander("📜 Live Pipeline Logs", expanded=False)
        log_output = log_expander.empty()

        full_log = []
        step = 0
        total_steps = len(companies) * 4  # Research, Score, Draft, Save per company
        current_company_idx = 0

        # Status message patterns to display nicely in the UI
        STATUS_PATTERNS = {
            "Researching":  "🔍 Researching {company}...",
            "Scoring":      "📊 Scoring {company}...",
            "Drafting":     "✍️  Drafting outreach for {company}...",
            "Saving":       "💾 Saving {company} to CRM...",
            "COMPANY_END":  "✅ Done with {company}!",
        }

        try:
            # Launch the pipeline subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(ROOT_DIR),
            )

            current_company = companies[0] if companies else ""

            # Read subprocess output line by line for live display
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                full_log.append(line)

                # Detect which company is being processed
                for c in companies:
                    if c.lower() in line.lower() and "research" in line.lower():
                        current_company = c
                        current_company_idx = companies.index(c)

                # Map log lines to friendly status messages
                display_msg = None
                for keyword, template in STATUS_PATTERNS.items():
                    if keyword.lower() in line.lower():
                        display_msg = template.format(company=current_company)
                        step = min(step + 1, total_steps)
                        break

                if display_msg:
                    pct = int((step / total_steps) * 100) if total_steps > 0 else 0
                    progress_bar.progress(pct, text=display_msg)
                    status_area.markdown(f"**{display_msg}**")

                # Update the live log pane
                log_output.code("\n".join(full_log[-40:]), language="")

            process.wait()

        except FileNotFoundError:
            st.error("❌ Could not find Python interpreter or main.py. Check your setup.")
            return
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            return

        progress_bar.progress(100, text="✅ Pipeline complete!")
        status_area.success("✅ Pipeline completed! Scroll down to see results.")

        # ── Results Display ───────────────────────────────────────────────────
        # After the pipeline, reload CRM and show each processed company
        st.divider()
        st.subheader("📦 Results")

        crm_data = load_crm()
        all_leads = crm_data.get("leads", [])

        found_any = False
        for company in companies:
            # Find the most recent lead matching this company name
            matching = [
                l for l in all_leads
                if l.get("company", "").lower() == company.lower()
            ]
            if not matching:
                st.warning(f"⚠️ No CRM record found for **{company}** — pipeline may have failed.")
                continue

            found_any = True
            lead = matching[-1]  # Use the most recently added record

            # ── Expandable Result Card per Company ───────────────────────────
            # Each company gets an expander with all pipeline outputs inside.
            with st.expander(
                f"📊 {lead.get('company', company)} — Score: {lead.get('score', '?')}/100",
                expanded=True,
            ):
                c1, c2, c3 = st.columns([1, 1, 2])

                with c1:
                    # Color-coded score badge (green=Hot, orange=Warm, blue=Cold)
                    tier = lead.get("tier", "")
                    st.markdown(f"**Tier:** {tier_badge_html(tier)}", unsafe_allow_html=True)
                    score_val = lead.get("score")
                    if score_val is not None:
                        st.metric("Score", f"{score_val}/100")
                    else:
                        st.metric("Score", "N/A")

                with c2:
                    st.markdown(f"**Status:** `{lead.get('status', 'unknown')}`")
                    st.markdown(f"**Contact:** {lead.get('name', 'Unknown')}")
                    st.markdown(f"**Title:** {lead.get('title', '—')}")

                with c3:
                    st.markdown(f"**Score Justification:**")
                    st.info(lead.get("score_justification") or "No justification recorded.")

                st.divider()

                # ── Research Findings Table ───────────────────────────────────
                st.markdown("**🔬 Research Findings**")
                research_fields = {
                    "Industry":        lead.get("industry", "—"),
                    "Company Size":    lead.get("company_size", "—"),
                    "Annual Revenue":  lead.get("annual_revenue", "—"),
                    "Email":           lead.get("email", "—"),
                    "Phone":           lead.get("phone", "—"),
                    "Source":          lead.get("source", "—"),
                }
                # Convert to a 2-column DataFrame for clean display
                df_research = pd.DataFrame(
                    [(k, v) for k, v in research_fields.items()],
                    columns=["Field", "Value"]
                )
                st.dataframe(df_research, use_container_width=True, hide_index=True)

                st.divider()

                # ── Email Draft — Editable Text Box ──────────────────────────
                st.markdown("**✉️ Email Draft** *(editable)*")
                email_key = f"email_{lead.get('id', company)}"
                edited_email = st.text_area(
                    "Email Draft",
                    value=lead.get("email_draft", ""),
                    height=200,
                    key=email_key,
                    label_visibility="collapsed",
                )

                # ── LinkedIn Message — Editable Text Box ─────────────────────
                st.markdown("**💼 LinkedIn Message** *(editable)*")
                li_key = f"li_{lead.get('id', company)}"
                edited_linkedin = st.text_area(
                    "LinkedIn Draft",
                    value=lead.get("linkedin_draft", ""),
                    height=80,
                    key=li_key,
                    label_visibility="collapsed",
                )

                # ── Approve / Skip Action Buttons ─────────────────────────────
                btn_col1, btn_col2, _ = st.columns([1, 1, 4])
                with btn_col1:
                    if st.button(f"✅ Approve", key=f"approve_{lead.get('id', company)}"):
                        # Save edited drafts and mark as approved
                        crm_data = load_crm()
                        for l in crm_data["leads"]:
                            if l.get("id") == lead.get("id"):
                                l["status"] = "approved"
                                l["email_draft"] = edited_email
                                l["linkedin_draft"] = edited_linkedin
                                l["status_updated_at"] = datetime.now(timezone.utc).isoformat()
                        if save_crm(crm_data):
                            st.success(f"✅ {company} approved and saved!")
                        else:
                            st.error("Failed to save CRM.")

                with btn_col2:
                    if st.button(f"⏭️ Skip", key=f"skip_{lead.get('id', company)}"):
                        # Mark lead as skipped without saving drafts
                        crm_data = load_crm()
                        for l in crm_data["leads"]:
                            if l.get("id") == lead.get("id"):
                                l["status"] = "skipped"
                                l["status_updated_at"] = datetime.now(timezone.utc).isoformat()
                        if save_crm(crm_data):
                            st.warning(f"⏭️ {company} marked as skipped.")
                        else:
                            st.error("Failed to save CRM.")

        if not found_any:
            st.info("No results to display. Check the pipeline logs above for errors.")


# =============================================================================
# PAGE 2: 📋 CRM DASHBOARD
# =============================================================================
def page_crm_dashboard():
    """
    CRM Dashboard Page
    ──────────────────
    Loads all leads from data/crm.json, displays them in an interactive
    sortable/filterable table. Clicking a lead reveals a detailed panel
    with status update, note-adding, and score-editing actions.

    Data mutations (status, score, notes) write back to crm.json immediately.
    """
    st.markdown('<div class="page-header">📋 CRM Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Manage and update all your sales leads in one place</div>',
                unsafe_allow_html=True)

    crm_data = load_crm()
    all_leads = crm_data.get("leads", [])

    if not all_leads:
        st.info("No leads found in CRM. Run the pipeline to add leads.")
        return

    # ── Filter Controls ────────────────────────────────────────────────────────
    # Three filter columns: Tier, Status, Date Range
    st.subheader("🔍 Filters")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        # Tier filter — multi-select (default: all)
        all_tiers = sorted(set(l.get("tier") or "Unknown" for l in all_leads))
        selected_tiers = st.multiselect("Filter by Tier", options=all_tiers, default=all_tiers)

    with fc2:
        # Status filter — multi-select (default: all)
        all_statuses = sorted(set(l.get("status") or "unknown" for l in all_leads))
        selected_statuses = st.multiselect("Filter by Status", options=all_statuses, default=all_statuses)

    with fc3:
        # Date range filter — based on created_at field
        st.markdown("**Filter by Date Range**")
        dr_col1, dr_col2 = st.columns(2)
        with dr_col1:
            date_from = st.date_input("From", value=None, key="crm_date_from")
        with dr_col2:
            date_to = st.date_input("To", value=None, key="crm_date_to")

    # ── Apply Filters ─────────────────────────────────────────────────────────
    filtered_leads = []
    for lead in all_leads:
        tier = lead.get("tier") or "Unknown"
        status = lead.get("status") or "unknown"

        # Tier and status filters
        if tier not in selected_tiers:
            continue
        if status not in selected_statuses:
            continue

        # Date range filter on created_at
        created_raw = lead.get("created_at", "")
        if created_raw and (date_from or date_to):
            try:
                created_dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00")).date()
                if date_from and created_dt < date_from:
                    continue
                if date_to and created_dt > date_to:
                    continue
            except Exception:
                pass

        filtered_leads.append(lead)

    st.caption(f"Showing **{len(filtered_leads)}** of **{len(all_leads)}** leads")
    st.divider()

    # ── Leads Table ──────────────────────────────────────────────────────────
    # Build a DataFrame for sortable display. The st.dataframe widget provides
    # built-in column sorting by clicking headers.
    if not filtered_leads:
        st.warning("No leads match the current filters.")
        return

    table_rows = []
    for lead in filtered_leads:
        score_val = lead.get("score")
        table_rows.append({
            "ID": lead.get("id", "—"),
            "Company": lead.get("company", "—"),
            "Contact": lead.get("name", "—"),
            "Title": lead.get("title", "—"),
            "Industry": lead.get("industry", "—"),
            "Score": score_val if score_val is not None else "—",
            "Tier": lead.get("tier") or "—",
            "Status": lead.get("status") or "—",
            "Created": lead.get("created_at", "—")[:10] if lead.get("created_at") else "—",
        })

    df_leads = pd.DataFrame(table_rows)

    # Display the sortable table. selection_mode="single-row" lets users click a row.
    selection = st.dataframe(
        df_leads,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",       # triggers a rerun when the user selects a row
        selection_mode="single-row",
    )

    # ── Lead Detail Panel ─────────────────────────────────────────────────────
    # When a row is selected in the table, show a full detail view below.
    selected_rows = selection.selection.get("rows", []) if hasattr(selection, "selection") else []

    if selected_rows:
        selected_idx = selected_rows[0]
        lead = filtered_leads[selected_idx]
        lead_id = lead.get("id")

        st.divider()
        st.subheader(f"📌 Lead Details: {lead.get('company', 'Unknown')}")

        d1, d2, d3 = st.columns([1, 1, 1])

        with d1:
            st.markdown(f"**Company:** {lead.get('company', '—')}")
            st.markdown(f"**Contact:** {lead.get('name', '—')}")
            st.markdown(f"**Title:** {lead.get('title', '—')}")
            st.markdown(f"**Email:** {lead.get('email') or '—'}")
            st.markdown(f"**Phone:** {lead.get('phone') or '—'}")

        with d2:
            st.markdown(f"**Score:** {lead.get('score', '—')}")
            st.markdown(f"**Tier:** {tier_badge_html(lead.get('tier'))}", unsafe_allow_html=True)
            st.markdown(f"**Status:** `{lead.get('status', '—')}`")
            st.markdown(f"**Source:** {lead.get('source', '—')}")
            st.markdown(f"**Industry:** {lead.get('industry') or '—'}")

        with d3:
            st.markdown(f"**Created:** {lead.get('created_at', '—')[:19] if lead.get('created_at') else '—'}")
            st.markdown(f"**Last Updated:** {lead.get('status_updated_at', '—')[:19] if lead.get('status_updated_at') else '—'}")
            notes_raw = lead.get("notes", "")
            if isinstance(notes_raw, list):
                # Notes can be a list of timestamped dicts or a plain string
                notes_display = "\n".join(
                    n.get("note", str(n)) if isinstance(n, dict) else str(n)
                    for n in notes_raw
                )
            else:
                notes_display = str(notes_raw)
            st.markdown(f"**Notes:** {notes_display or '—'}")

        st.divider()

        # ── Action Buttons ─────────────────────────────────────────────────────
        st.markdown("**⚙️ Lead Actions**")
        act1, act2, act3 = st.columns(3)

        with act1:
            # Update Status dropdown
            st.markdown("**Update Status**")
            valid_statuses = [
                "new", "contacted", "replied", "meeting_booked",
                "won", "lost", "nurture", "approved", "skipped", "archived",
            ]
            current_status = lead.get("status", "new")
            current_idx = valid_statuses.index(current_status) if current_status in valid_statuses else 0
            new_status = st.selectbox(
                "Status",
                options=valid_statuses,
                index=current_idx,
                key=f"status_select_{lead_id}",
                label_visibility="collapsed",
            )
            if st.button("💾 Update Status", key=f"btn_status_{lead_id}"):
                crm_data = load_crm()
                for l in crm_data["leads"]:
                    if l.get("id") == lead_id:
                        old_status = l.get("status", "unknown")
                        l["status"] = new_status
                        l["status_updated_at"] = datetime.now(timezone.utc).isoformat()
                        # Append to timeline if it exists
                        timeline = l.get("timeline", [])
                        timeline.append({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "type": "status_change",
                            "old_status": old_status,
                            "new_status": new_status,
                            "description": f"Status updated via Web UI: {old_status} → {new_status}",
                        })
                        l["timeline"] = timeline
                if save_crm(crm_data):
                    st.success(f"✅ Status updated to `{new_status}`")
                    st.rerun()

        with act2:
            # Add Note text input
            st.markdown("**Add Note**")
            new_note = st.text_input(
                "Note",
                placeholder="Enter a note...",
                key=f"note_input_{lead_id}",
                label_visibility="collapsed",
            )
            if st.button("📝 Add Note", key=f"btn_note_{lead_id}"):
                if new_note.strip():
                    crm_data = load_crm()
                    for l in crm_data["leads"]:
                        if l.get("id") == lead_id:
                            existing_notes = l.get("notes", [])
                            # Normalize notes to a list
                            if isinstance(existing_notes, str):
                                existing_notes = [existing_notes] if existing_notes else []
                            note_entry = {
                                "note": new_note.strip(),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                            existing_notes.append(note_entry)
                            l["notes"] = existing_notes
                            # Also add to timeline
                            timeline = l.get("timeline", [])
                            timeline.append({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "type": "note",
                                "note": new_note.strip(),
                                "description": f"Note added via Web UI: {new_note.strip()}",
                            })
                            l["timeline"] = timeline
                    if save_crm(crm_data):
                        st.success("📝 Note added!")
                        st.rerun()
                else:
                    st.warning("Please enter a note first.")

        with act3:
            # Update Score number input
            st.markdown("**Update Score**")
            current_score = lead.get("score") or 0
            new_score = st.number_input(
                "Score (0–100)",
                min_value=0,
                max_value=100,
                value=int(current_score),
                key=f"score_input_{lead_id}",
                label_visibility="collapsed",
            )
            score_reason = st.text_input(
                "Reason",
                placeholder="Reason for score change...",
                key=f"score_reason_{lead_id}",
            )
            if st.button("🎯 Update Score", key=f"btn_score_{lead_id}"):
                crm_data = load_crm()
                for l in crm_data["leads"]:
                    if l.get("id") == lead_id:
                        old_score = l.get("score", 0)
                        l["score"] = new_score
                        # Recalculate tier based on new score
                        if new_score >= 80:
                            l["tier"] = "Hot"
                        elif new_score >= 60:
                            l["tier"] = "Warm"
                        else:
                            l["tier"] = "Cold"
                        l["score_justification"] = score_reason or f"Score updated via Web UI"
                        l["scored_at"] = datetime.now(timezone.utc).isoformat()
                        # Append to timeline
                        timeline = l.get("timeline", [])
                        timeline.append({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "type": "score_update",
                            "old_score": old_score,
                            "new_score": new_score,
                            "reason": score_reason,
                            "description": f"Score updated via Web UI: {old_score} → {new_score}",
                        })
                        l["timeline"] = timeline
                if save_crm(crm_data):
                    st.success(f"🎯 Score updated to **{new_score}** (Tier: {crm_data['leads'][-1].get('tier', '?')})")
                    st.rerun()

        # ── Email & LinkedIn Draft Display ─────────────────────────────────────
        if lead.get("email_draft"):
            with st.expander("✉️ Email Draft"):
                st.text_area("Email", value=lead["email_draft"], height=200,
                             key=f"detail_email_{lead_id}", label_visibility="collapsed")

        if lead.get("linkedin_draft"):
            with st.expander("💼 LinkedIn Message"):
                st.text_area("LinkedIn", value=lead["linkedin_draft"], height=80,
                             key=f"detail_li_{lead_id}", label_visibility="collapsed")

        # ── Lead Timeline ──────────────────────────────────────────────────────
        timeline = lead.get("timeline", [])
        if timeline:
            with st.expander("📅 Lead Timeline"):
                for event in reversed(timeline):  # Most recent first
                    ts = event.get("timestamp", "")[:19]
                    desc = event.get("description", str(event))
                    st.markdown(f"**{ts}** — {desc}")


# =============================================================================
# PAGE 3: 🎯 PRIORITY DASHBOARD
# =============================================================================
def page_priority_dashboard():
    """
    Priority Dashboard Page
    ────────────────────────
    Computes priority scores for all leads and displays:
      - KPI metric cards at the top (Total Leads, Hot Leads, Avg Score, Approval Rate)
      - Priority-ranked lead cards with colored bars
      - Bar chart: leads by tier
      - Line chart: leads added over time (by date)

    Uses the same priority formula as tools/priority_dashboard.py.
    """
    st.markdown('<div class="page-header">🎯 Priority Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Ranked leads with actionable insights and charts</div>',
                unsafe_allow_html=True)

    crm_data = load_crm()
    all_leads = crm_data.get("leads", [])

    if not all_leads:
        st.info("No leads found in CRM. Run the pipeline first.")
        return

    # ── Top-Level KPI Metric Cards ────────────────────────────────────────────
    # Four cards: Total Leads | Hot Leads | Avg Score | Approval Rate
    metrics_data = load_metrics()

    total_leads = len(all_leads)
    hot_leads = sum(1 for l in all_leads if l.get("tier") == "Hot")
    scored_leads = [l.get("score") for l in all_leads if l.get("score") is not None]
    avg_score = round(sum(scored_leads) / len(scored_leads), 1) if scored_leads else 0
    approval_rate = metrics_data.get("human_approval_rate", 0)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("📊 Total Leads", total_leads)
    with kpi2:
        st.metric("🔥 Hot Leads", hot_leads, delta=f"{round(hot_leads/total_leads*100)}% of total" if total_leads else None)
    with kpi3:
        st.metric("⭐ Avg Score", f"{avg_score}/100")
    with kpi4:
        st.metric("✅ Approval Rate", f"{approval_rate:.1f}%")

    st.divider()

    # ── Charts Row ────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Bar chart: Count of leads by tier (Hot / Warm / Cold)
        st.subheader("🏷️ Leads by Tier")
        tier_counts = {"Hot": 0, "Warm": 0, "Cold": 0, "Unscored": 0}
        for lead in all_leads:
            t = lead.get("tier")
            if t in tier_counts:
                tier_counts[t] += 1
            else:
                tier_counts["Unscored"] += 1

        df_tiers = pd.DataFrame(
            [{"Tier": k, "Count": v} for k, v in tier_counts.items() if v > 0]
        )
        if not df_tiers.empty:
            # Set Tier as index so Streamlit bar_chart uses it as X axis
            st.bar_chart(df_tiers.set_index("Tier"), color="#6c5ce7")

    with chart_col2:
        # Line chart: Leads added over time (grouped by created_at date)
        st.subheader("📈 Leads Added Over Time")
        date_counts = {}
        for lead in all_leads:
            created_raw = lead.get("created_at", "")
            if created_raw:
                try:
                    date_str = created_raw[:10]  # "YYYY-MM-DD"
                    date_counts[date_str] = date_counts.get(date_str, 0) + 1
                except Exception:
                    pass

        if date_counts:
            df_dates = pd.DataFrame(
                [{"Date": k, "Leads Added": v} for k, v in sorted(date_counts.items())]
            )
            st.line_chart(df_dates.set_index("Date"))
        else:
            st.info("No date data available.")

    st.divider()

    # ── Priority-Ranked Lead Cards ────────────────────────────────────────────
    # Leads are ranked by the priority formula and displayed as visual cards
    # with a colored progress bar proportional to priority score.
    st.subheader("🏆 Priority-Ranked Leads")
    prioritized = get_priority_leads(crm_data)

    for rank, lead in enumerate(prioritized, 1):
        company = lead.get("company", "Unknown")
        score = lead.get("score") or 0
        tier = lead.get("tier") or "Cold"
        status = lead.get("status") or "unknown"
        priority = lead.get("priority", 0)
        name = lead.get("name", "Unknown")

        # Determine bar color class based on tier
        bar_class = {"Hot": "bar-hot", "Warm": "bar-warm"}.get(tier, "bar-cold")
        # Bar width is percentage of max priority (capped at 100)
        bar_width = min(100, priority)

        # Render priority card using inline HTML for visual richness
        st.markdown(f"""
        <div class="company-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:18px; font-weight:700; color:white;">
                    #{rank} &nbsp; {company}
                </span>
                <span>{tier_badge_html(tier)}</span>
            </div>
            <div style="display:flex; gap:24px; color:rgba(255,255,255,0.6); font-size:13px; margin-bottom:10px;">
                <span>👤 {name}</span>
                <span>📊 Score: <b style="color:{score_color(score)}">{score}/100</b></span>
                <span>🔖 Status: <b>{status}</b></span>
                <span>🎯 Priority: <b style="color:#a29bfe">{priority}</b></span>
            </div>
            <div class="{bar_class}" style="width:{bar_width}%; height:6px; border-radius:4px; margin-top:4px;"></div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: 📊 MONITORING
# =============================================================================
def page_monitoring():
    """
    Monitoring Page
    ───────────────
    Reads data/metrics.json and agent_log.txt to surface:
      - Key system health metrics as st.metric cards
      - A table of recent pipeline runs with timestamps and status
      - An error log section showing any failures or warnings

    This page is especially useful for post-run auditing and debugging.
    """
    st.markdown('<div class="page-header">📊 Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">System health, pipeline runs, and error tracking</div>',
                unsafe_allow_html=True)

    metrics = load_metrics()

    # ── Refresh Button ─────────────────────────────────────────────────────────
    # Metrics and logs are read fresh on each page load, but users can manually
    # trigger a rerun to refresh after a pipeline completes.
    if st.button("🔄 Refresh Metrics"):
        # Try to recompute metrics from logs before displaying
        try:
            from tools.monitor import compute_metrics, save_metrics
            fresh_metrics = compute_metrics()
            save_metrics(fresh_metrics)
            metrics = fresh_metrics
            st.success("✅ Metrics refreshed from logs.")
        except Exception as e:
            st.warning(f"⚠️ Could not recompute metrics: {e}. Showing cached values.")
        st.rerun()

    # ── Key Metric Cards ───────────────────────────────────────────────────────
    # Display each metric as a Streamlit metric widget with meaningful labels
    st.subheader("📈 Key Metrics")

    if not metrics:
        st.warning("No metrics data found. Run the pipeline first, then refresh.")
    else:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("🏭 Leads Processed", metrics.get("total_leads_processed", 0))
        with m2:
            st.metric("🔬 Avg Research Quality", f"{metrics.get('avg_research_quality', 0):.1f}%")
        with m3:
            st.metric("✉️ Avg Email Quality", f"{metrics.get('avg_email_quality', 0):.1f}%")
        with m4:
            st.metric("✅ Approval Rate", f"{metrics.get('human_approval_rate', 0):.1f}%")
        with m5:
            st.metric("⏱️ Avg Pipeline Time", f"{metrics.get('avg_pipeline_time_s', 0):.1f}s")
        with m6:
            err_rate = metrics.get("error_rate", 0)
            # Color error rate red if high
            st.metric("⚠️ Error Rate", f"{err_rate:.1f}%", delta=None)

        st.divider()

        # ── Tier Distribution ──────────────────────────────────────────────────
        st.subheader("🏷️ Tier Distribution (All Time)")
        tiers = metrics.get("tier_distribution", {})
        td1, td2, td3 = st.columns(3)
        with td1:
            st.metric("🔥 Hot", tiers.get("Hot", 0))
        with td2:
            st.metric("🌤 Warm", tiers.get("Warm", 0))
        with td3:
            st.metric("❄️ Cold", tiers.get("Cold", 0))

        # ── Raw Counts ────────────────────────────────────────────────────────
        raw = metrics.get("raw_counts", {})
        if raw:
            st.divider()
            st.subheader("📋 Raw Counts")
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.metric("Total Runs", raw.get("total_runs", 0))
            with rc2:
                st.metric("Failed Runs", raw.get("failed_runs", 0))
            with rc3:
                st.metric("Approved", raw.get("approved", 0))
            with rc4:
                st.metric("Skipped", raw.get("skipped", 0))

        # ── Most Common Industries ─────────────────────────────────────────────
        industries = metrics.get("most_common_industries", [])
        if industries:
            st.divider()
            st.subheader("🏢 Most Common Industries")
            df_ind = pd.DataFrame(industries, columns=["Industry", "Count"])
            st.dataframe(df_ind, use_container_width=True, hide_index=True)

    # ── Recent Pipeline Runs Table ─────────────────────────────────────────────
    st.divider()
    st.subheader("🕐 Recent Pipeline Runs")

    runs = parse_log_runs(AGENT_LOG_FILE)
    if runs:
        df_runs = pd.DataFrame(runs)
        # Rename columns for readability
        df_runs = df_runs.rename(columns={
            "company": "Company",
            "started_at": "Started At",
            "ended_at": "Ended At",
            "status": "Status",
        })
        st.dataframe(df_runs, use_container_width=True, hide_index=True)
    else:
        st.info("No pipeline run records found in agent_log.txt.")

    # ── Error Log Section ──────────────────────────────────────────────────────
    # Shows the most recent errors/warnings from the agent log for debugging
    st.divider()
    st.subheader("🚨 Error Log")

    error_lines = get_error_lines(AGENT_LOG_FILE)
    if error_lines:
        # Display errors in a scrollable code block
        st.code("\n".join(error_lines), language="")
    else:
        if AGENT_LOG_FILE.exists():
            st.success("✅ No errors found in agent_log.txt — all clear!")
        else:
            st.info("agent_log.txt not found. Run the pipeline to generate logs.")

    # ── Raw Log Viewer ─────────────────────────────────────────────────────────
    # Allow the user to view the last N lines of the full agent log
    with st.expander("📜 View Raw Agent Log (last 100 lines)"):
        if AGENT_LOG_FILE.exists():
            try:
                with open(AGENT_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                    last_lines = f.readlines()[-100:]
                st.code("".join(last_lines), language="")
            except Exception as e:
                st.error(f"Could not read log: {e}")
        else:
            st.info("No log file found.")


# =============================================================================
# PAGE 5: 🧠 MEMORY EXPLORER
# =============================================================================
def page_memory_explorer():
    """
    Memory Explorer Page
    ─────────────────────
    Connects to ChromaDB (stored in data/chroma_db/) and displays:
      - A count of all stored lead profiles
      - A table of all stored leads with their metadata
      - A semantic search box to find similar past leads

    This page showcases the RAG (Retrieval-Augmented Generation) memory
    system that improves research quality for recurring industries.
    """
    st.markdown('<div class="page-header">🧠 Memory Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Explore ChromaDB long-term lead memory and semantic search</div>',
                unsafe_allow_html=True)

    # ── ChromaDB Connection ────────────────────────────────────────────────────
    # Import vector_memory module which initializes the ChromaDB client.
    # We wrap this in try/except to gracefully handle missing chromadb package.
    try:
        from memory.vector_memory import collection, get_memory_stats, recall_similar_leads
        memory_available = True
    except ImportError:
        st.error("❌ ChromaDB not installed. Run: `pip install chromadb`")
        memory_available = False
    except Exception as e:
        st.error(f"❌ Failed to connect to ChromaDB: {e}")
        memory_available = False

    if not memory_available:
        return

    # ── Memory Stats Header ────────────────────────────────────────────────────
    total_stored = get_memory_stats()
    stat1, stat2, _ = st.columns([1, 1, 3])
    with stat1:
        st.metric("🧠 Leads in Memory", total_stored)
    with stat2:
        db_path = ROOT_DIR / "data" / "chroma_db"
        db_size_mb = 0
        if db_path.exists():
            db_size_mb = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file()) / (1024 * 1024)
        st.metric("💾 DB Size", f"{db_size_mb:.2f} MB")

    st.divider()

    # ── All Stored Leads Table ─────────────────────────────────────────────────
    # Retrieve all stored lead profiles from ChromaDB and display in a table.
    st.subheader("📚 All Stored Lead Profiles")

    if total_stored == 0:
        st.info("No leads stored in memory yet. Run the pipeline to populate memory.")
    else:
        try:
            # collection.get() returns all documents, metadatas, and ids
            all_data = collection.get(include=["documents", "metadatas"])
            ids = all_data.get("ids", [])
            documents = all_data.get("documents", [])
            metadatas = all_data.get("metadatas", [])

            table_rows = []
            for doc_id, doc, meta in zip(ids, documents, metadatas):
                meta = meta or {}
                table_rows.append({
                    "Company": doc_id,
                    "Industry": meta.get("industry", "—"),
                    "Score": meta.get("score", "—"),
                    "Profile Preview": (doc or "")[:120] + "..." if doc and len(doc) > 120 else doc or "—",
                })

            df_memory = pd.DataFrame(table_rows)
            st.dataframe(df_memory, use_container_width=True, hide_index=True)

            # ── Full Profile Viewer ────────────────────────────────────────────
            # Let the user pick a specific company to see its full stored profile
            selected_company = st.selectbox(
                "View full profile for:",
                options=["— Select a company —"] + ids,
                key="memory_profile_select",
            )
            if selected_company and selected_company != "— Select a company —":
                company_idx = ids.index(selected_company)
                st.markdown(f"**Full stored profile for {selected_company}:**")
                st.code(documents[company_idx], language="")

        except Exception as e:
            st.error(f"⚠️ Failed to retrieve memory data: {e}")

    st.divider()

    # ── Semantic Search Box ────────────────────────────────────────────────────
    # Users can type a company name or description to find similar past leads.
    # Results are ranked by ChromaDB's vector similarity score.
    st.subheader("🔍 Find Similar Leads")
    st.caption("Uses ChromaDB vector similarity to find past leads with similar industry, pain points, or context.")

    search_query = st.text_input(
        "Find similar leads to...",
        placeholder="e.g. 'fintech startup struggling with compliance automation'",
        help="Enter a company name, industry, or description to find semantically similar past leads.",
    )

    top_n = st.slider("Number of results", min_value=1, max_value=10, value=3, key="memory_top_n")

    if st.button("🔍 Search Memory", type="primary"):
        if not search_query.strip():
            st.warning("Please enter a search query.")
        elif total_stored == 0:
            st.info("Memory is empty. Run the pipeline to populate it.")
        else:
            with st.spinner("Searching ChromaDB..."):
                try:
                    # Use the recall_similar_leads function from vector_memory.py
                    # It queries ChromaDB and returns top-n similar lead profiles.
                    results = recall_similar_leads(search_query.strip(), n=top_n)

                    if not results:
                        st.info("No similar leads found. Try a different search query.")
                    else:
                        st.markdown(f"**Found {len(results)} similar lead(s):**")
                        for i, match in enumerate(results, 1):
                            company_id = match.get("id", "Unknown")
                            doc = match.get("document", "")
                            meta = match.get("metadata") or {}

                            # Similarity score is not directly returned by recall_similar_leads,
                            # but we can show the rank as a proxy for relevance
                            with st.expander(f"#{i} — {company_id}", expanded=(i == 1)):
                                col_a, col_b = st.columns([1, 2])
                                with col_a:
                                    st.markdown(f"**Company:** {company_id}")
                                    st.markdown(f"**Industry:** {meta.get('industry', '—')}")
                                    st.markdown(f"**Score:** {meta.get('score', '—')}/100")
                                    st.markdown(f"**Rank:** #{i} most similar")
                                with col_b:
                                    st.markdown("**Stored Profile:**")
                                    st.code(doc[:500] + ("..." if len(doc) > 500 else ""), language="")

                except Exception as e:
                    st.error(f"❌ Search failed: {e}")


# =============================================================================
# PAGE 6: 🔍 EXECUTION TRACES
# =============================================================================
def page_execution_traces():
    """
    Execution Traces Page
    ─────────────────────
    Visualizes the per-agent execution timeline for every pipeline run stored
    in data/traces/. For each run, shows:
      - A summary table (agent, duration, status, model, tokens)
      - A horizontal Gantt-style bar chart using Plotly showing relative timing
      - Input/output previews per agent in collapsible expanders
      - A "Compare Runs" section for A/B timing analysis between two runs

    WHY EXECUTION TRACING MATTERS:
        In multi-agent pipelines, performance bottlenecks are invisible without
        instrumentation. The trace view answers:
        1. Which agent is the slowest? (Gantt chart immediately reveals this)
        2. Did switching models change latency? (Compare Runs shows before/after)
        3. What input/output was produced at each stage? (Expanders show this)
        4. Did any agent use a fallback? (Status column flags these)

    IMPLEMENTATION NOTE:
        Trace files are saved by the Tracer class in tools/tracer.py after
        each run_company_pipeline() call in main.py. Each file is named:
            data/traces/{CompanyName}_{YYYYMMDD_HHMMSS}.json
        The list_traces() function groups files by company name for the
        company dropdown.
    """
    st.markdown('<div class="page-header">🔍 Execution Traces</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Per-agent timing, model usage, and I/O inspection across all pipeline runs</div>',
        unsafe_allow_html=True,
    )

    # ── Load Available Traces ─────────────────────────────────────────────────
    # list_traces() scans data/traces/ and groups .json files by company name.
    # If the directory doesn't exist yet (no runs yet), show a helpful message.
    try:
        from tools.tracer import list_traces, load_trace
        company_traces = list_traces(TRACES_DIR)
    except Exception as e:
        st.error(f"❌ Failed to load traces: {e}")
        return

    if not company_traces:
        st.info(
            "📂 No trace files found in `data/traces/`. "
            "Run the pipeline first — traces are saved automatically after each company run."
        )
        # Show what will appear once traces exist
        with st.expander("ℹ️ What will appear here after running the pipeline?"):
            st.markdown("""
            After each pipeline run, you'll see:
            - **Execution Timeline Table**: Agent-by-agent duration, status, model used
            - **Gantt Chart**: Horizontal bar chart showing each agent's relative time
            - **I/O Previews**: Input/output per agent in expandable panels
            - **Run Comparison**: Select two runs to compare timing side-by-side
            """)
        return

    # ── Company Selector ──────────────────────────────────────────────────────
    # Dropdown lists all companies that have at least one trace file.
    company_list = sorted(company_traces.keys())
    sel_col1, sel_col2 = st.columns([2, 1])

    with sel_col1:
        selected_company = st.selectbox(
            "🏢 Select company to inspect",
            options=company_list,
            key="trace_company_select",
        )

    # ── Run Selector ──────────────────────────────────────────────────────────
    # Dropdown lists all trace files for the selected company.
    # Files are named with timestamps so we sort newest-first.
    run_files = company_traces.get(selected_company, [])
    run_labels = []
    for fp in run_files:
        # Extract timestamp from filename: {company}_{YYYYMMDD}_{HHMMSS}.json
        parts = fp.stem.rsplit("_", 2)
        if len(parts) >= 3:
            date_str, time_str = parts[-2], parts[-1]
            try:
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                label = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                label = fp.stem
        else:
            label = fp.stem
        run_labels.append(label)

    with sel_col2:
        selected_run_label = st.selectbox(
            "🕐 Select run (by timestamp)",
            options=run_labels,
            key="trace_run_select",
        )

    # Resolve selected run file
    selected_run_idx = run_labels.index(selected_run_label) if selected_run_label in run_labels else 0
    selected_run_file = run_files[selected_run_idx]

    # ── Load Trace Data ───────────────────────────────────────────────────────
    trace = load_trace(selected_run_file)
    spans = trace.get("spans", [])
    total_ms = trace.get("total_duration_ms", 0)
    run_ts = trace.get("run_timestamp", "")

    if not spans:
        st.warning("No span data found in this trace file.")
        return

    st.divider()

    # ── Execution Timeline Header ─────────────────────────────────────────────
    # Shows the company name, run timestamp, and total pipeline duration
    # as a prominent header above the timeline table.
    run_display_ts = run_ts[:19].replace("T", " ") if run_ts else selected_run_label
    st.subheader(f"📋 EXECUTION TRACE — {selected_company} ({run_display_ts})",
                 divider="rainbow")

    # ── Summary Timeline Table ────────────────────────────────────────────────
    # Each row = one agent span. Columns: Agent | Duration | Status | Model | Tokens
    # The duration is shown in milliseconds for sub-second agents and as
    # "X.Xs" for slow agents to match what the user sees in the terminal.
    span_sum_ms = sum(s.get("duration_ms", 0) for s in spans)

    table_rows = []
    for span in spans:
        dur_ms = span.get("duration_ms", 0)
        # Format duration: < 1000ms → show as ms, >= 1000ms → show as seconds
        if dur_ms < 1000:
            dur_str = f"{dur_ms}ms"
        else:
            dur_str = f"{dur_ms/1000:.1f}s"

        status = span.get("status", "unknown")
        # Map status codes to emoji indicators
        status_icon = {"success": "✅", "error": "❌", "fallback": "⚠️"}.get(status, "❓")

        table_rows.append({
            "Agent": span.get("agent_name", "—"),
            "Duration": dur_str,
            "Status": f"{status_icon} {status}",
            "Model": span.get("model_used", "—"),
            "In Tokens ~": span.get("input_tokens_estimated", 0),
            "Out Tokens ~": span.get("output_tokens_estimated", 0),
        })

    # Add TOTAL row at the bottom of the table
    if total_ms < 1000:
        total_dur_str = f"{total_ms}ms"
    else:
        total_dur_str = f"{total_ms/1000:.1f}s"

    table_rows.append({
        "Agent": "⏱️ TOTAL",
        "Duration": total_dur_str,
        "Status": "",
        "Model": "",
        "In Tokens ~": sum(s.get("input_tokens_estimated", 0) for s in spans),
        "Out Tokens ~": sum(s.get("output_tokens_estimated", 0) for s in spans),
    })

    df_trace = pd.DataFrame(table_rows)
    st.dataframe(df_trace, use_container_width=True, hide_index=True)

    st.divider()

    # ── Gantt-Style Horizontal Bar Chart ─────────────────────────────────────
    # Shows each agent's execution duration as a colored horizontal bar,
    # proportional to total pipeline time. Longer bars = bigger bottlenecks.
    #
    # WHY GANTT OVER A PIE CHART:
    #   Gantt charts preserve the sequential execution order of agents, making
    #   it immediately obvious which agent runs first and which is the bottleneck.
    #   Pie charts lose ordering information and are harder to compare across runs.
    st.subheader("📊 Gantt-Style Duration Breakdown")

    if PLOTLY_AVAILABLE and spans:
        # Build a Plotly horizontal bar chart (Gantt-style)
        # Each agent gets a bar proportional to its duration_ms.
        # Colors: success=purple, error=red, fallback=orange, other=blue.
        STATUS_COLORS = {
            "success":  "#6c5ce7",
            "error":    "#ff6b6b",
            "fallback": "#f0932b",
            "stub":     "#74b9ff",
            "rules":    "#55efc4",
            "template": "#a29bfe",
        }

        agent_names = [s.get("agent_name", "Agent") for s in spans]
        durations_ms = [s.get("duration_ms", 0) for s in spans]
        statuses = [s.get("status", "unknown") for s in spans]
        models = [s.get("model_used", "?") for s in spans]
        colors = [STATUS_COLORS.get(st_val, "#636e72") for st_val in statuses]

        # Create a simple horizontal bar chart using Plotly
        # The bars are cumulative (stacked at the correct X position to
        # simulate a real Gantt chart without needing start/end times in
        # the trace format).
        fig = go.Figure()

        # Calculate cumulative start times for proper Gantt positioning
        cumulative_start = 0
        for i, (agent, dur, status_val, model, color) in enumerate(
            zip(agent_names, durations_ms, statuses, models, colors)
        ):
            pct = (dur / max(span_sum_ms, 1)) * 100
            fig.add_trace(go.Bar(
                name=agent,
                x=[dur],
                y=[agent],
                orientation="h",
                marker_color=color,
                text=f"{dur/1000:.2f}s ({pct:.1f}%)",
                textposition="auto",
                hovertemplate=(
                    f"<b>{agent}</b><br>"
                    f"Duration: {dur}ms ({dur/1000:.2f}s)<br>"
                    f"Model: {model}<br>"
                    f"Status: {status_val}<br>"
                    f"Share: {pct:.1f}% of total"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=f"Agent Execution Time Breakdown — {selected_company}",
            xaxis_title="Duration (milliseconds)",
            yaxis_title="Agent",
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=300 + len(spans) * 40,
            margin=dict(l=20, r=20, t=50, b=40),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                categoryorder="array",
                categoryarray=list(reversed(agent_names)),  # Preserve execution order top-to-bottom
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Fallback if plotly is not installed: render ASCII progress bars
        # using Streamlit's native st.progress() widget.
        st.info("Install plotly (`pip install plotly`) for the interactive Gantt chart.")
        max_dur = max((s.get("duration_ms", 1) for s in spans), default=1)
        for span in spans:
            agent = span.get("agent_name", "Agent")
            dur_ms = span.get("duration_ms", 0)
            pct = min(dur_ms / max_dur, 1.0)
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.markdown(f"**{agent}**")
            with col_b:
                st.progress(pct, text=f"{dur_ms}ms")

    st.divider()

    # ── Input/Output Previews ─────────────────────────────────────────────────
    # Each agent's span is shown in a collapsible expander with its
    # input_summary and output_summary. This helps diagnose agent failures:
    # "What did LeadResearcher receive as input?" and
    # "What did it return as output?"
    st.subheader("🔬 Agent I/O Previews")

    for span in spans:
        agent = span.get("agent_name", "Agent")
        status = span.get("status", "unknown")
        model = span.get("model_used", "?")
        dur_ms = span.get("duration_ms", 0)
        status_icon = {"success": "✅", "error": "❌", "fallback": "⚠️"}.get(status, "❓")

        expander_label = (
            f"{status_icon} {agent} — {model} — "
            f"{dur_ms/1000:.2f}s"
        )

        with st.expander(expander_label):
            io_col1, io_col2 = st.columns(2)

            with io_col1:
                st.markdown("**📥 Input Summary**")
                input_text = span.get("input_summary", "(none)")
                st.code(input_text or "(empty)", language="")

                in_tok = span.get("input_tokens_estimated", 0)
                st.caption(f"~{in_tok} tokens estimated")

            with io_col2:
                st.markdown("**📤 Output Summary**")
                output_text = span.get("output_summary", "(none)")
                st.code(output_text or "(empty)", language="")

                out_tok = span.get("output_tokens_estimated", 0)
                st.caption(f"~{out_tok} tokens estimated")

            # Show full span metadata
            st.markdown("**📋 Span Metadata**")
            meta_cols = st.columns(4)
            with meta_cols[0]:
                st.metric("Duration", f"{dur_ms}ms")
            with meta_cols[1]:
                st.metric("Status", status)
            with meta_cols[2]:
                st.metric("Model", model)
            with meta_cols[3]:
                total_tok = in_tok + out_tok
                st.metric("Total Tokens ~", total_tok)

    st.divider()

    # ── Compare Runs Section ──────────────────────────────────────────────────
    # Select two runs for the same company and display a side-by-side timing
    # comparison. This is useful for:
    #   - Benchmarking model changes (did switching to gemini-2.0-flash save time?)
    #   - Measuring the impact of the --fast flag
    #   - Detecting regressions (did a code change make the pipeline slower?)
    st.subheader("⚖️ Compare Two Runs")
    st.caption(
        "Select two runs for the same company to compare agent timing side-by-side. "
        "Useful for benchmarking model changes or the --fast flag."
    )

    if len(run_files) < 2:
        st.info(
            f"Only 1 run found for **{selected_company}**. "
            "Run the pipeline again to enable comparison."
        )
    else:
        # Run A and Run B selectors
        cmp_col1, cmp_col2 = st.columns(2)

        with cmp_col1:
            run_a_label = st.selectbox(
                "Run A (Baseline)",
                options=run_labels,
                index=0,
                key="cmp_run_a",
            )
        with cmp_col2:
            run_b_label = st.selectbox(
                "Run B (Comparison)",
                options=run_labels,
                index=min(1, len(run_labels) - 1),
                key="cmp_run_b",
            )

        if run_a_label == run_b_label:
            st.warning("Please select two different runs to compare.")
        else:
            run_a_idx = run_labels.index(run_a_label)
            run_b_idx = run_labels.index(run_b_label)
            trace_a = load_trace(run_files[run_a_idx])
            trace_b = load_trace(run_files[run_b_idx])

            spans_a = {s["agent_name"]: s for s in trace_a.get("spans", [])}
            spans_b = {s["agent_name"]: s for s in trace_b.get("spans", [])}

            # All agents that appear in either run
            all_agents = sorted(set(list(spans_a.keys()) + list(spans_b.keys())))

            comparison_rows = []
            for agent in all_agents:
                span_a = spans_a.get(agent, {})
                span_b = spans_b.get(agent, {})
                dur_a = span_a.get("duration_ms", 0)
                dur_b = span_b.get("duration_ms", 0)

                # Calculate difference and determine which is faster
                diff_ms = dur_b - dur_a
                if dur_a == 0 and dur_b == 0:
                    winner = "—"
                elif dur_a == 0:
                    winner = "B only"
                elif dur_b == 0:
                    winner = "A only"
                elif diff_ms < -50:     # Run B is >50ms faster than A
                    winner = f"✅ B faster by {abs(diff_ms)}ms"
                elif diff_ms > 50:      # Run A is >50ms faster than B
                    winner = f"✅ A faster by {diff_ms}ms"
                else:
                    winner = "≈ Equal"   # Within 50ms noise floor

                comparison_rows.append({
                    "Agent": agent,
                    f"Run A ({run_a_label})": f"{dur_a}ms" if dur_a else "—",
                    f"Run B ({run_b_label})": f"{dur_b}ms" if dur_b else "—",
                    "Δ (B - A)": f"{'+' if diff_ms > 0 else ''}{diff_ms}ms" if (dur_a and dur_b) else "—",
                    "Result": winner,
                })

            # Add totals row
            total_a = trace_a.get("total_duration_ms", 0)
            total_b = trace_b.get("total_duration_ms", 0)
            diff_total = total_b - total_a
            comparison_rows.append({
                "Agent": "⏱️ TOTAL",
                f"Run A ({run_a_label})": f"{total_a}ms",
                f"Run B ({run_b_label})": f"{total_b}ms",
                "Δ (B - A)": f"{'+' if diff_total > 0 else ''}{diff_total}ms",
                "Result": (
                    f"✅ B is {abs(diff_total/1000):.1f}s faster"
                    if diff_total < -100 else
                    f"✅ A is {abs(diff_total/1000):.1f}s faster"
                    if diff_total > 100 else "≈ Similar"
                ),
            })

            df_comparison = pd.DataFrame(comparison_rows)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)

            # Plotly side-by-side bar chart for visual comparison
            if PLOTLY_AVAILABLE:
                common_agents = [a for a in all_agents if a in spans_a and a in spans_b]
                if common_agents:
                    durs_a = [spans_a[a].get("duration_ms", 0) for a in common_agents]
                    durs_b = [spans_b[a].get("duration_ms", 0) for a in common_agents]

                    fig_cmp = go.Figure(data=[
                        go.Bar(
                            name=f"Run A ({run_a_label})",
                            x=common_agents,
                            y=durs_a,
                            marker_color="#6c5ce7",
                            text=[f"{d}ms" for d in durs_a],
                            textposition="auto",
                        ),
                        go.Bar(
                            name=f"Run B ({run_b_label})",
                            x=common_agents,
                            y=durs_b,
                            marker_color="#00b894",
                            text=[f"{d}ms" for d in durs_b],
                            textposition="auto",
                        ),
                    ])
                    fig_cmp.update_layout(
                        title=f"Run A vs Run B — {selected_company}",
                        xaxis_title="Agent",
                        yaxis_title="Duration (ms)",
                        barmode="group",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white", family="Inter"),
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                        height=400,
                        margin=dict(l=20, r=20, t=50, b=40),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER — Renders the selected page based on sidebar radio selection
# ─────────────────────────────────────────────────────────────────────────────
if page == "🚀 Run Pipeline":
    page_run_pipeline()
elif page == "📋 CRM Dashboard":
    page_crm_dashboard()
elif page == "🎯 Priority Dashboard":
    page_priority_dashboard()
elif page == "📊 Monitoring":
    page_monitoring()
elif page == "🧠 Memory Explorer":
    page_memory_explorer()
elif page == "🔍 Execution Traces":
    page_execution_traces()
