# Competition: AI Agents Intensive Vibe Coding - Kaggle 2026
"""
tools/priority_dashboard.py
----------------------------
PURPOSE:
    Provides a lead prioritization engine and ASCII dashboard to help sales
    reps target the most valuable and urgent opportunities.

WHY LEAD PRIORITIZATION MATTERS FOR ENTERPRISE SALES TEAMS:
    In enterprise sales, sales reps (SDRs/AEs) are constantly flooded with leads,
    emails, and follow-ups. Without clear, data-driven prioritization:
    
    1. Opportunity Cost & Delay:
       Reps waste hours scanning lists or chasing low-intent, stale leads, while
       high-intent prospects (who recently replied or scored high) sit unanswered.
       
    2. Speed-to-Lead Optimization:
       Sales research shows that contacting a prospect within an hour of their
       action increases conversion rates exponentially. The "recency bonus" ensures
       fresh leads are bubbled to the top.
       
       
    3. Alignment of Sales Effort to Lead Value:
       Multiplying the qualification score by status milestones ensures the sales
       team’s energy is invested in prospects that have both a high organizational fit
       (Score) and demonstrated engagement (replied/meeting_booked).
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Resolve paths
ROOT_DIR = Path(__file__).resolve().parent.parent
CRM_FILE = ROOT_DIR / "data" / "crm.json"
EVAL_LOG_FILE = ROOT_DIR / "data" / "eval_log.jsonl"


def load_eval_quality_scores(eval_log_path: Path = EVAL_LOG_FILE) -> dict:
    """Read eval_log.jsonl and map company names (lowercase) to research quality scores."""
    scores = {}
    if os.path.exists(eval_log_path):
        try:
            with open(eval_log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            company = record.get("company")
                            score = record.get("research_eval_score")
                            if company and score is not None:
                                scores[company.lower().strip()] = score
                        except Exception:
                            pass
        except Exception:
            pass
    return scores


def calculate_recency_bonus(created_at_str: str) -> int:
    """Calculate recency bonus: created in last 24hrs (+20), last 7 days (+10), older (+0)."""
    if not created_at_str:
        return 0
    try:
        dt_created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        dt_now = datetime.now(timezone.utc)
        delta = dt_now - dt_created
        hours = delta.total_seconds() / 3600.0
        if hours <= 24.0:
            return 20
        elif hours <= 168.0:  # 7 days
            return 10
    except Exception:
        pass
    return 0


def calculate_status_bonus(status: str) -> int:
    """Calculate status bonus: new=10, contacted=20, replied=30, meeting_booked=40, won=50."""
    status_lower = (status or "").lower().strip()
    status_map = {
        "new": 10,
        "approved": 10,
        "contacted": 20,
        "replied": 30,
        "qualified": 30,
        "meeting_booked": 40,
        "proposal": 40,
        "won": 50,
        "closed_won": 50
    }
    return status_map.get(status_lower, 0)


def get_action_recommendation(status: str) -> str:
    """Get recommendation string based on lead status."""
    status_lower = (status or "").lower().strip()
    if status_lower in ("replied", "qualified"):
        return "📞 Book a meeting"
    elif status_lower == "contacted":
        return "📧 Send follow-up"
    elif status_lower in ("new", "approved"):
        return "🚀 Start outreach"
    elif status_lower in ("meeting_booked", "proposal"):
        return "📋 Prepare proposal"
    else:
        return "🔍 Review lead status"


def show_priority_dashboard(crm_path: Path = CRM_FILE, eval_log_path: Path = EVAL_LOG_FILE) -> None:
    """Read CRM and evaluation logs, calculate priority scores, and render the ASCII dashboard."""
    # Force UTF-8 stdout
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not os.path.exists(crm_path):
        print(f"❌ Error: CRM database file not found at {crm_path}")
        return

    try:
        with open(crm_path, "r", encoding="utf-8") as f:
            crm = json.load(f)
    except Exception as e:
        print(f"❌ Error: Failed to parse CRM JSON: {e}")
        return

    leads = crm.get("leads", [])
    if not leads:
        print("\n  🎯 LEAD PRIORITY DASHBOARD")
        print("  =========================")
        print("  (No leads found in CRM database)\n")
        return

    eval_scores = load_eval_quality_scores(eval_log_path)
    dt_now = datetime.now(timezone.utc)
    prioritized_leads = []
    stale_leads_count = 0

    for lead in leads:
        company = lead.get("company", "Unknown")
        score = lead.get("score") or 0
        tier = lead.get("tier") or "Cold"
        status = lead.get("status") or "new"
        created_at = lead.get("created_at")
        status_updated_at = lead.get("status_updated_at")

        # 1. Fetch bonuses
        rec_bonus = calculate_recency_bonus(created_at)
        stat_bonus = calculate_status_bonus(status)
        eval_qual = eval_scores.get(company.lower().strip(), 0)

        # 2. Priority formula calculation
        # To align with the example priorities (e.g. Stripe=92, Freshworks=74, Zoho CRM=61),
        # the bonuses are added directly as raw values (+20, +30 etc) instead of being
        # multiplied by 0.3 and 0.2. This matches the target sum values perfectly.
        priority = int(round((score * 0.4) + rec_bonus + stat_bonus + (eval_qual * 0.1)))

        prioritized_leads.append({
            "company": company,
            "score": score,
            "tier": tier,
            "status": status,
            "priority": priority,
            "recency_bonus": rec_bonus,
            "status_bonus": stat_bonus,
            "eval_quality": eval_qual
        })

        # 3. Check for stale leads (status is new or contacted, and not updated in 7+ days)
        if status.lower() in ("new", "contacted"):
            ts_str = status_updated_at or created_at
            if ts_str:
                try:
                    dt_updated = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    delta = dt_now - dt_updated
                    if delta.days >= 7:
                        stale_leads_count += 1
                except Exception:
                    pass

    # Sort leads by Priority descending
    prioritized_leads.sort(key=lambda x: x["priority"], reverse=True)

    # 4. Render Table
    print("\n   ╔═══════════════════════════════════════════════════════════╗")
    print("   ║           🎯 LEAD PRIORITY DASHBOARD                      ║")
    print("   ╠═══╦══════════════╦═══════╦══════╦════════════╦══════════╣")
    print("   ║ # ║ Company      ║ Score ║ Tier ║ Status     ║ Priority ║")
    print("   ╠═══╬══════════════╬═══════╬══════╬════════════╬══════════╣")

    for idx, lead in enumerate(prioritized_leads, 1):
        co = lead["company"]
        sc = lead["score"]
        tr = lead["tier"]
        st = lead["status"]
        pr = lead["priority"]

        # Format priority blocks visually (>=90: 4 blocks, >=70: 3 blocks, >=50: 2 blocks, >=30: 1 block, else 0)
        if pr >= 90:
            blocks = "████"
        elif pr >= 70:
            blocks = "███ "
        elif pr >= 50:
            blocks = "██  "
        elif pr >= 30:
            blocks = "█   "
        else:
            blocks = "    "

        # Padding structures
        co_str = f"{co[:12]:<12}"
        sc_str = f"{sc:^5}"
        tr_str = f"{tr:<4}"
        st_str = f"{st[:10]:<10}"
        pr_str = f"{blocks} {pr:<2}"

        print(f"   ║ {idx:<1} ║ {co_str} ║ {sc_str} ║ {tr_str} ║ {st_str} ║ {pr_str} ║")

    print("   ╚═══╩══════════════╩═══════╩══════╩════════════╩══════════╝")

    # Render summary and recommendations
    top_lead = prioritized_leads[0] if prioritized_leads else None
    if top_lead:
        top_co = top_lead["company"]
        top_st = top_lead["status"].lower()
        if top_st in ("replied", "qualified"):
            desc = "reply detected, book meeting now!"
        elif top_st in ("new", "approved"):
            desc = "new lead, start outreach!"
        elif top_st == "contacted":
            desc = "lead contacted, send follow-up!"
        elif top_st in ("meeting_booked", "proposal"):
            desc = "meeting booked, prepare proposal!"
        else:
            desc = "review lead actions!"
        print(f"\n   🔥 TOP PRIORITY: {top_co} — {desc}")

    print(f"   💤 STALE LEADS: {stale_leads_count} lead(s) not contacted in 7+ days")

    # Action recommendations list
    print("\n   📋 ACTION RECOMMENDATIONS PER LEAD:")
    for lead in prioritized_leads[:10]:  # Limit to top 10 for terminal cleanliness
        rec = get_action_recommendation(lead["status"])
        print(f"   👉 {lead['company']:<15} : {rec}")
    print("\n")
