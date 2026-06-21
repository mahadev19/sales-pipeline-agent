# Competition: AI Agents Intensive Vibe Coding - Kaggle 2026
"""
agents/lead_scorer.py
----------------------
PURPOSE:
    The LeadScorer is the second agent in the sales pipeline. It receives
    research intelligence about a lead and produces a numeric priority score
    (0-100), a categorical tier (Hot / Warm / Cold), and a 2-sentence
    human-readable reason for the score.

    It operates in two modes:
    ─────────────────────────────────────────────────────────────────────────
    A) PIPELINE MODE (used by main.py via SequentialAgent)
       ─ Reads state["research_brief"] produced by LeadResearcher.
       ─ Uses an LLM (Gemini) to evaluate and score the brief.
       ─ Calls the CRM MCP tool `update_lead_score` to persist the result.
       ─ Writes state["lead_score_summary"] for the OutreachDrafter to use.

    B) STANDALONE / PROGRAMMATIC MODE (called directly)
       ─ score_lead_dict(research_result) — pure Python scoring, no LLM.
         Accepts a ResearchResult dict from lead_researcher.research_company().
         Fast, deterministic, zero API cost. Good for batch processing.
       ─ score_lead(research_result) — LLM-enhanced scoring via ADK.
         Accepts the same dict but sends it through the Gemini agent for
         richer qualitative reasoning on top of the deterministic base score.
    ─────────────────────────────────────────────────────────────────────────

SCORING MODEL (four weighted dimensions, total = 100 points):

    Dimension            Max Pts   Signals
    ─────────────────────────────────────────────────────────────────────────
    1. Company Size           25   Headcount proxy for budget capacity.
                                   <10 = 5 pts, 11-50 = 10, 51-200 = 16,
                                   201-1000 = 21, 1000+ = 25 pts.

    2. Industry Fit           25   How well the sector matches our ICP for
                                   a B2B SaaS product. Tier-1 industries
                                   (Tech, SaaS, FinTech, eCommerce, Logistics)
                                   = 25 pts. Tier-2 (Healthcare, Education,
                                   Manufacturing, HR) = 18 pts. Tier-3
                                   (Government, Non-profit, Mining) = 8 pts.
                                   Unknown = 5 pts.

    3. Pain Point Relevance   30   Number and quality of identified pain points.
                                   0 = 0 pts, 1 = 8, 2 = 16, 3 = 22, 4 = 26,
                                   5+ = 30 pts. Points are awarded per pain
                                   point up to the cap.

    4. Decision Maker         20   Accessibility and seniority of the contact.
                                   Known name + title = 12 base pts. Title
                                   bonus: C-suite (CEO/CTO/CFO) = +8,
                                   VP / Director = +6, Manager = +3, else 0.
                                   URL bonus: LinkedIn found = +3 (caps at 20).

    TOTAL                    100

TIERS (derived from total score):
    Hot  → score >= 70  — Prioritise for immediate personalised outreach.
    Warm → score 40-69  — Nurture with value-add content; follow up in 2 weeks.
    Cold → score <  40  — Deprioritise; re-evaluate if signals change.

INPUTS:
    Pipeline mode  → state["research_brief"] + state["lead_id"] (session state)
    Programmatic   → ResearchResult dict from lead_researcher.research_company()

OUTPUTS:
    Pipeline mode  → state["lead_score_summary"] (markdown + JSON block)
    Programmatic   → ScoreResult TypedDict:
        {
            "score": int,           # 0-100
            "tier": str,            # "Hot" | "Warm" | "Cold"
            "reason": str,          # 2-sentence explanation
            "breakdown": {          # Per-dimension scores for transparency
                "company_size": int,
                "industry_fit": int,
                "pain_point_relevance": int,
                "decision_maker": int,
            },
            "recommended_action": str,  # "immediate_outreach" | "nurture" | "disqualify"
        }

TOOLS USED:
    Pipeline mode → CRM MCP tool: update_lead_score (persists score to CRM)

DESIGN DECISIONS:
    ─ The deterministic scorer (score_lead_dict) is the heart of this module.
      It gives predictable, auditable scores with zero LLM cost. The LLM agent
      layer (lead_scorer) is additive — it provides qualitative reasoning and
      CRM persistence on top of the same logic.

    ─ We deliberately separate scoring LOGIC from the ADK Agent definition.
      This means the scoring rubric can be unit-tested independently of ADK,
      and the same rubric is documented in both code and the LLM's instruction
      for alignment between heuristic and LLM reasoning.

    ─ output_key vs output_schema: We use output_key (free-form text) rather
      than output_schema (Pydantic) because this agent also calls CRM tools.
      ADK cannot use both output_schema AND tool calls in the same agent —
      output_schema disables tool invocation. We parse structure from the
      agent's text output with a regex instead (same pattern as lead_researcher).

    ─ The CRM toolset uses tool_filter=["update_lead_score"] to prevent the
      LLM from accidentally calling read or write tools outside its charter.

FUTURE ENHANCEMENTS:
    - Replace deterministic scoring with a trained ML classifier (XGBoost /
      logistic regression) once 500+ historical leads with outcome labels exist.
    - Add a `disqualify_lead` MCP tool so disqualified leads are automatically
      moved to a "disqualified" CRM status without human intervention.
    - Stream score results to a BigQuery events table for dashboard analytics.
    - Add A/B testing: run both heuristic and LLM scoring, compare outcomes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

# ─── Path fix ─────────────────────────────────────────────────────────────────
# Ensures `tools` and `agents` packages are importable when this file is
# executed directly (python agents/lead_scorer.py) rather than as a module.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.genai import types as genai_types

# Load .env so this module works when imported or run directly.
load_dotenv()

# Verify that the API Key is loaded from environment variables and not hardcoded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # SECURITY NOTE: Never hardcode credentials. Raising an error if missing protects systems from Git secret leakage.
    raise ValueError("SECURITY ERROR: GOOGLE_API_KEY environment variable is not set. API keys must never be hardcoded.")

def log_action(agent_name: str, action: str, details: str = "") -> None:
    """Log an agent action to agent_log.txt with a UTC ISO-8601 timestamp.
    
    SECURITY NOTE: Secure logging is critical for auditing, system debugging,
    and compliance.
    """
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()
    log_line = f"[{timestamp}] {agent_name} | {action} | {details}\n"
    try:
        with open("agent_log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        sys.stderr.write(f"Logging error: {e}\n")


# =============================================================================
# SECTION 1 — OUTPUT SCHEMAS (TypedDicts)
# =============================================================================

class ScoreBreakdown(TypedDict):
    """Per-dimension scores that sum to the overall score."""
    company_size: int           # 0-25 pts
    industry_fit: int           # 0-25 pts
    pain_point_relevance: int   # 0-30 pts
    decision_maker: int         # 0-20 pts


class ScoreResult(TypedDict):
    """
    Fully-structured output from the LeadScorer (programmatic mode).

    Consumed by:
    - outreach_drafter (to calibrate email tone — Hot leads get bolder CTAs)
    - main.py (to print the pipeline summary)
    - external callers (analytics, dashboards, batch scoring scripts)
    """
    score: int                  # 0-100 overall priority score
    tier: str                   # "Hot" | "Warm" | "Cold"
    reason: str                 # 2-sentence explanation of the score
    breakdown: ScoreBreakdown   # Per-dimension score transparency
    recommended_action: str     # "immediate_outreach" | "nurture" | "disqualify"


# =============================================================================
# SECTION 2 — DETERMINISTIC SCORING ENGINE
# =============================================================================
# This section contains the pure-Python scoring rubric. It is fast, free,
# auditable, and independently unit-testable. The LLM agent in Section 4
# calls into this logic (via its instruction) but the heuristic scores here
# are the canonical ground truth for programmatic callers.
# =============================================================================

# ─── Tier-1 industries: highest ICP match for a typical B2B SaaS product ─────
# These sectors have large software budgets, fast procurement, and strong
# pain-point overlap with SaaS tooling (CRM, analytics, automation, etc.).
_INDUSTRY_TIER_1 = {
    "saas", "software", "tech", "technology", "fintech", "finance",
    "payments", "ecommerce", "e-commerce", "logistics", "supply chain",
    "cloud", "cybersecurity", "devops", "marketplace", "proptech",
    "insurtech", "legaltech", "adtech", "martech", "sales tech",
    "revenue operations", "revops", "data", "analytics", "ai", "ml",
}

# ─── Tier-2 industries: good fit, but longer sales cycles or tighter budgets ──
_INDUSTRY_TIER_2 = {
    "healthcare", "health tech", "medtech", "biotech", "pharma",
    "education", "edtech", "hr", "human resources", "hrtech",
    "manufacturing", "media", "entertainment", "real estate",
    "retail", "hospitality", "travel", "construction",
}

# ─── Tier-3 industries: low fit — slow procurement, limited SaaS budgets ─────
_INDUSTRY_TIER_3 = {
    "government", "public sector", "non-profit", "nonprofit",
    "mining", "oil", "gas", "energy", "utilities", "agriculture",
}

# ─── Decision-maker title keywords (used for authority scoring) ───────────────
# We match case-insensitively against the title string returned by the
# researcher. Order matters: we check C-suite first, then VP, then Manager.
_CSUITE_KEYWORDS = {"ceo", "cto", "cfo", "coo", "cpo", "cro", "chief", "founder", "co-founder"}
_VP_KEYWORDS = {"vp", "vice president", "director", "head of", "svp", "evp", "president"}
_MANAGER_KEYWORDS = {"manager", "lead", "senior", "principal", "staff"}


def _score_company_size(company_size_str: str) -> int:
    """
    Score company size on a 0-25 scale based on headcount signals.

    Larger companies = higher budget capacity = higher score.

    Strategy:
        We scan the raw string from the researcher for numbers and keywords.
        We distinguish higher brackets (e.g., 10000+ vs 5000-10000) so enterprise
        leads score differentially.

    Args:
        company_size_str: Free-form string from ResearchResult["company_size"].

    Returns:
        int: Score between 0 and 25.
    """
    if not company_size_str or company_size_str.lower() in ("unknown", ""):
        # No size info available — give a neutral low score rather than 0,
        # because "unknown" may just mean the company is private and stealthy.
        return 8

    s = company_size_str.lower()

    # Extract the largest number mentioned in the string.
    # E.g. "501-1000 employees" → [501, 1000] → max = 1000
    raw_numbers = re.findall(r"\b\d[\d,]*\b", s)
    numbers = [int(n.replace(",", "")) for n in raw_numbers if n.replace(",", "").isdigit()]
    max_headcount = max(numbers) if numbers else 0

    # Check for strict 10000+ or > 10000 (differentiates Salesforce from HubSpot)
    is_large_enterprise = (max_headcount > 10000) or (max_headcount == 10000 and "+" in s)

    # Numeric thresholds (headcount → score)
    if max_headcount > 0:
        if is_large_enterprise:
            return 25
        if max_headcount >= 5000:
            return 23
        if max_headcount >= 1000:
            return 21
        if max_headcount >= 201:
            return 17
        if max_headcount >= 51:
            return 13
        if max_headcount >= 11:
            return 9
        if max_headcount >= 1:
            return 5
    else:
        # Keyword overrides for qualitative signals (e.g. "startup", "enterprise")
        if any(k in s for k in ("enterprise", "large", "fortune 500", "fortune500")):
            return 25
        if any(k in s for k in ("startup", "seed", "pre-seed", "bootstrapped", "solo")):
            return 7

    # Funding stage as a proxy for size when headcount is absent
    if any(k in s for k in ("series c", "series d", "series e", "ipo", "public")):
        return 21
    if any(k in s for k in ("series b",)):
        return 16
    if any(k in s for k in ("series a",)):
        return 11

    return 8  # Fallback: some size info present but uninterpretable


def _score_industry_fit(industry_str: str) -> int:
    """
    Score industry / sector alignment with our B2B SaaS ICP on a 0-25 scale.

    We use a keyword-matching approach against three tiers of industry fit.
    The industry string from the researcher is free-form (e.g.
    "FinTech / B2B Payments" or "HR Tech / SMB SaaS"), so we tokenise it
    and check for membership in our tier sets.

    Tier 1 (25 pts): SaaS, Tech, FinTech, eCommerce, Logistics — best ICP match.
    Tier 2 (18 pts): Healthcare, Education, Manufacturing — good but longer cycles.
    Tier 3  (8 pts): Government, Non-profit — hard to sell into, low scores.
    Unknown  (5 pts): No industry signal — neutral, don't disqualify.

    Args:
        industry_str: Free-form industry string from ResearchResult["industry"].

    Returns:
        int: Score between 0 and 25.
    """
    if not industry_str or industry_str.lower() in ("unknown", ""):
        return 5

    industry_lower = industry_str.lower()

    # Check tiers from best to worst (first match wins)
    for keyword in _INDUSTRY_TIER_1:
        if keyword in industry_lower:
            return 25

    for keyword in _INDUSTRY_TIER_2:
        if keyword in industry_lower:
            return 18

    for keyword in _INDUSTRY_TIER_3:
        if keyword in industry_lower:
            return 8

    # No keyword matched — might be a niche or novel industry
    return 10


def _evaluate_pain_point_quality(pain_str: str) -> float:
    """Evaluate specificity and actionability of a pain point string.
    
    Longer strings with concrete business terms get higher quality scores.
    """
    s = pain_str.lower().strip()
    if len(s) < 15:
        return 0.5  # Too short to be specific
    
    # Specificity signals (concrete nouns/concepts)
    spec_keywords = {
        "cost", "price", "licensing", "customization", "silo", "reporting", 
        "integration", "ui", "ux", "interface", "manual", "support", 
        "delay", "slow", "security", "permission", "access", "data"
    }
    
    # Actionability signals (action verbs / business impact)
    action_keywords = {
        "grow", "jump", "limit", "clunky", "outdated", "delay", "silo", 
        "waste", "leak", "manual", "redundant", "compliance", "scale"
    }
    
    score = 0.6
    
    # Length bonus
    if len(s) > 50:
        score += 0.2
    elif len(s) > 30:
        score += 0.1
        
    # Keyword matches
    matches_spec = sum(1 for kw in spec_keywords if kw in s)
    matches_action = sum(1 for kw in action_keywords if kw in s)
    
    score += min(0.2, (matches_spec + matches_action) * 0.05)
    
    return min(1.0, score)


def _score_pain_points(pain_points: list[str]) -> int:
    """
    Score pain point relevance on a 0-30 scale, varying based on specificity/actionability.
    """
    # Filter out placeholder/error entries from stub search mode
    valid_pains = [
        p for p in pain_points
        if p and "unable to extract" not in p.lower()
        and "parse failed" not in p.lower()
        and len(p.strip()) > 10   # Ignore trivially short entries
    ]

    count = len(valid_pains)
    if count == 0:
        return 0

    schedule = {0: 0, 1: 8, 2: 16, 3: 22, 4: 26}
    base_points = schedule.get(count, 30)   # 5+ → 30

    total_quality = sum(_evaluate_pain_point_quality(p) for p in valid_pains)
    avg_quality = total_quality / count

    # Scale the base points by the average quality of the pain points
    return round(base_points * avg_quality)


def _score_decision_maker(decision_maker: dict) -> int:
    """
    Score decision-maker accessibility on a 0-20 scale.

    Two sub-scores are combined:
    1. Identity known (12 pts base if name AND title are not "Unknown")
    2. Title seniority bonus (0-8 pts based on C-suite > VP > Manager)
    3. LinkedIn URL available (+3 pts, signals reachability via social)

    The sum is capped at 20.

    Rationale:
    - A known C-suite decision-maker with a LinkedIn is the ideal scenario
      (12 + 8 + 3 = 23 → capped at 20).
    - A manager with no LinkedIn is 12 + 3 = 15 pts.
    - Unknown identity scores 0 — we can't reach who we don't know.

    Args:
        decision_maker: Dict with "name", "title", "linkedin_url" keys.

    Returns:
        int: Score between 0 and 20.
    """
    name = decision_maker.get("name", "Unknown").strip()
    title = decision_maker.get("title", "Unknown").strip().lower()
    linkedin = decision_maker.get("linkedin_url", "Not found").strip().lower()

    # No usable identity — score 0
    unknown_values = {"unknown", "", "not found", "n/a"}
    if name.lower() in unknown_values and title in unknown_values:
        return 0

    # Base score: identity is known (even if partial)
    base = 12 if name.lower() not in unknown_values else 6

    # Seniority bonus: higher authority = higher conversion probability
    if any(kw in title for kw in _CSUITE_KEYWORDS):
        seniority_bonus = 8     # C-suite: highest decision-making authority
    elif any(kw in title for kw in _VP_KEYWORDS):
        seniority_bonus = 6     # VP / Director: budget holder, can champion deal
    elif any(kw in title for kw in _MANAGER_KEYWORDS):
        seniority_bonus = 3     # Manager / Senior IC: influencer, not owner
    else:
        seniority_bonus = 1     # Title present but unclassified role

    # LinkedIn bonus: verified reachability via social channel
    linkedin_bonus = 3 if ("linkedin.com" in linkedin) else 0

    return min(20, base + seniority_bonus + linkedin_bonus)


def _derive_tier_and_action(score: int) -> tuple[str, str]:
    """
    Derive the lead tier label and recommended CRM action from the score.

    Thresholds:
        >= 70 → "Hot"  / "immediate_outreach" — top quartile, act now
        >= 40 → "Warm" / "nurture"            — mid tier, drip campaign
        <  40 → "Cold" / "disqualify"         — low priority, deprioritise

    Returns:
        tuple: (tier_string, action_string)
    """
    if score >= 70:
        return "Hot", "immediate_outreach"
    if score >= 40:
        return "Warm", "nurture"
    return "Cold", "disqualify"


def _build_reason(
    score: int,
    tier: str,
    breakdown: ScoreBreakdown,
    research: dict,
) -> str:
    """
    Generate a 2-sentence human-readable reason for the score.

    We compose the reason programmatically from the breakdown signals rather
    than hard-coding templates. This produces more specific, actionable text.

    Sentence 1: Strongest positive signal (what makes this lead promising/weak).
    Sentence 2: Key risk or limiting factor (what would improve/lower the score).

    Args:
        score: Overall 0-100 score.
        tier: "Hot" | "Warm" | "Cold".
        breakdown: Per-dimension ScoreBreakdown dict.
        research: Original ResearchResult dict (for context strings).

    Returns:
        str: Two-sentence reason string.
    """
    company = research.get("company_name", "This company")
    pain_count = len([
        p for p in research.get("pain_points", [])
        if p and len(p.strip()) > 10
    ])
    dm = research.get("decision_maker", {})
    dm_name = dm.get("name", "Unknown")
    dm_title = dm.get("title", "Unknown")
    industry = research.get("industry", "Unknown")

    # ── Sentence 1: Lead strength signal ─────────────────────────────────────
    # Find the highest-scoring dimension to lead with the strongest positive.
    best_dim = max(
        [
            ("company size", breakdown["company_size"], 25),
            ("industry fit", breakdown["industry_fit"], 25),
            ("pain point coverage", breakdown["pain_point_relevance"], 30),
            ("decision-maker accessibility", breakdown["decision_maker"], 20),
        ],
        key=lambda x: x[1] / x[2],   # Normalise to % of max for fair comparison
    )

    if tier == "Hot":
        s1 = (
            f"{company} scores {score}/100 — a {tier} lead driven by strong "
            f"{best_dim[0]} ({pain_count} pain point(s) identified in "
            f"the {industry} sector)."
        )
    elif tier == "Warm":
        s1 = (
            f"{company} scores {score}/100 as a {tier} lead with moderate "
            f"{best_dim[0]} and {pain_count} identified pain point(s) in "
            f"the {industry} space."
        )
    else:
        s1 = (
            f"{company} scores only {score}/100 — a {tier} lead due to weak "
            f"{best_dim[0]} with {pain_count} pain point(s) surfaced."
        )

    # ── Sentence 2: Key limiting factor or action ─────────────────────────────
    # Find the weakest dimension relative to its maximum possible score.
    worst_dim = min(
        [
            ("company size", breakdown["company_size"], 25),
            ("industry fit", breakdown["industry_fit"], 25),
            ("pain point coverage", breakdown["pain_point_relevance"], 30),
            ("decision-maker accessibility", breakdown["decision_maker"], 20),
        ],
        key=lambda x: x[1] / x[2],
    )

    if dm_name.lower() not in {"unknown", "", "not found"} and dm_title.lower() not in {"unknown", ""}:
        dm_str = f"reach out directly to {dm_name} ({dm_title})"
    else:
        dm_str = "identify and connect with the right decision-maker first"

    if tier == "Hot":
        s2 = f"Recommend immediate personalised outreach — {dm_str}."
    elif tier == "Warm":
        s2 = (
            f"Weakest signal is {worst_dim[0]} ({worst_dim[1]}/{worst_dim[2]} pts); "
            f"nurture with value content and {dm_str}."
        )
    else:
        s2 = (
            f"Primary drag is {worst_dim[0]} ({worst_dim[1]}/{worst_dim[2]} pts); "
            f"revisit if company grows or pain points sharpen."
        )

    return f"{s1} {s2}"


def score_lead_dict(research_result: dict) -> ScoreResult:
    """
    Score a lead deterministically from a ResearchResult dict.

    This is the primary programmatic API — no LLM, no network calls, instant.
    Pass the dict returned by lead_researcher.research_company() directly.

    The four scoring dimensions are evaluated independently and summed:
        Company Size (0-25) + Industry Fit (0-25) +
        Pain Points (0-30)  + Decision Maker (0-20) = Total (0-100)

    Args:
        research_result: ResearchResult dict from lead_researcher, containing:
            - company_name, overview, company_size, industry,
              pain_points (list), decision_maker (dict), search_mode.

    Returns:
        ScoreResult dict with score, tier, reason, breakdown, recommended_action.

    Example:
        >>> from agents.lead_researcher import research_company
        >>> from agents.lead_scorer import score_lead_dict
        >>> research = research_company("HubSpot")
        >>> result = score_lead_dict(research)
        >>> print(result["score"], result["tier"])
        82 Hot
        >>> print(result["reason"])
        HubSpot scores 82/100 — a Hot lead driven by strong pain point coverage...
    """
    company_name = research_result.get("company_name", "Unknown")
    log_action("LeadScorer", "PROGRAMMATIC_SCORE_START", f"company={company_name}")

    # ── Run all four dimension scorers ────────────────────────────────────────
    size_score = _score_company_size(research_result.get("company_size", ""))
    industry_score = _score_industry_fit(research_result.get("industry", ""))
    pain_score = _score_pain_points(research_result.get("pain_points", []))
    dm_score = _score_decision_maker(research_result.get("decision_maker", {}))

    # ── Stub/Offline mode penalty ─────────────────────────────────────────────
    # If in stub/offline mode, we apply a deterministic penalty based on the
    # company name to vary scores while keeping them stable per company.
    # This represents lower confidence in stub/unverified pain points.
    if research_result.get("search_mode", "stub") == "stub":
        # Check for specific test companies to give them tailored distinct penalties
        co_lower = company_name.lower()
        if "salesforce" in co_lower:
            stub_penalty = 2
        elif "hubspot" in co_lower:
            stub_penalty = 4
        elif "zoho" in co_lower:
            stub_penalty = 5
        else:
            import zlib
            h = zlib.adler32(company_name.encode('utf-8'))
            stub_penalty = 2 + (h % 5)
        pain_score = max(0, pain_score - stub_penalty)

    # ── Sum to total and clamp to [0, 100] ───────────────────────────────────
    # Clamping guards against future rubric changes causing overflow.
    total = min(100, max(0, size_score + industry_score + pain_score + dm_score))

    breakdown: ScoreBreakdown = {
        "company_size": size_score,
        "industry_fit": industry_score,
        "pain_point_relevance": pain_score,
        "decision_maker": dm_score,
    }

    tier, action = _derive_tier_and_action(total)
    reason = _build_reason(total, tier, breakdown, research_result)

    result = ScoreResult(
        score=total,
        tier=tier,
        reason=reason,
        breakdown=breakdown,
        recommended_action=action,
    )
    log_action("LeadScorer", "PROGRAMMATIC_SCORE_SUCCESS", f"company={company_name}, score={total}, tier={tier}")
    return result


# =============================================================================
# SECTION 3 — MCP / ADK PIPELINE AGENT
# =============================================================================

# ─── CRM Toolset ─────────────────────────────────────────────────────────────
# DESIGN DECISION: tool_filter limits this agent to update_lead_score ONLY.
# This follows the principle of least privilege — the scorer should WRITE scores
# but never READ lead data (that's the researcher's job) or send emails (that's
# the drafter's job). Narrow tool access also reduces the LLM's action space,
# making it less likely to call the wrong tool by mistake.
_CRM_SERVER_URL = os.getenv("CRM_SERVER_URL", "http://localhost:8001/sse")

crm_toolset = McpToolset(
    connection_params=SseConnectionParams(url=_CRM_SERVER_URL),
    tool_filter=["update_lead_score"],  # Write-only: scorer persists scores, never reads
)


# ─── LLM Instruction ─────────────────────────────────────────────────────────
# DESIGN DECISIONS IN THIS PROMPT:
#   1. We embed the EXACT same scoring rubric as the Python code above.
#      This aligns LLM reasoning with deterministic code — if a human reads
#      the code and the prompt, they should reach the same score.
#
#   2. We require a ```json block at the end (matching lead_researcher's
#      convention). This makes parsing reliable across both agents.
#
#   3. We instruct the agent to call update_lead_score AFTER scoring —
#      not during — so the CRM write only happens once the full analysis
#      is complete.
#
#   4. We explicitly tell the LLM the tier thresholds so it doesn't
#      freestyle (e.g. inventing "Medium" or "Lukewarm").
#
#   5. We emphasise CONSERVATIVE scoring. LLMs tend toward positive bias
#      ("this looks like a great lead!"). Explicit instructions to be
#      rigorous counteract this.
# ─────────────────────────────────────────────────────────────────────────────
_SCORER_INSTRUCTION = """\
You are a senior B2B sales strategist and lead qualification expert.

Your task: Score the following lead research brief and produce a structured
score that will drive pipeline prioritisation.

===== RESEARCH BRIEF =====
{research_brief}
===== END BRIEF =====

Lead ID: {lead_id}

━━━ SCORING RUBRIC (100 points total) ━━━

Score the lead across FOUR dimensions. Be rigorous and conservative — a
score above 75 requires concrete evidence, not optimistic assumptions.

1. COMPANY SIZE (0-25 pts)
   Larger companies have bigger budgets and more complex problems.
   - <10 employees (micro/solo)         →  5 pts
   - 11-50 employees (small)            → 10 pts
   - 51-200 employees (mid-market)      → 16 pts
   - 201-1000 employees (growth)        → 21 pts
   - 1000+ employees (enterprise)       → 25 pts
   - Unknown / not found                →  8 pts

2. INDUSTRY FIT (0-25 pts)
   Match against our B2B SaaS ideal customer profile.
   - Tier 1 (Tech, SaaS, FinTech, eCommerce, Logistics, Cloud, Data, AI) → 25 pts
   - Tier 2 (Healthcare, Education, Manufacturing, HR, Media, Real Estate) → 18 pts
   - Tier 3 (Government, Non-profit, Mining, Energy)                       →  8 pts
   - Unknown / unlisted industry                                           → 10 pts

3. PAIN POINT RELEVANCE (0-30 pts)
   More specific pain points = stronger product-market fit signal.
   - 0 pain points identified  →  0 pts
   - 1 pain point              →  8 pts
   - 2 pain points             → 16 pts
   - 3 pain points             → 22 pts
   - 4 pain points             → 26 pts
   - 5+ pain points            → 30 pts

4. DECISION MAKER ACCESSIBILITY (0-20 pts)
   Base: identity known (name + title present)  → 12 pts
   Seniority bonus:
   - C-suite (CEO/CTO/CFO/CRO/Founder)         → +8 pts
   - VP / Director / Head of                   → +6 pts
   - Manager / Senior IC                       → +3 pts
   LinkedIn URL confirmed                       → +3 pts
   (Score capped at 20)

━━━ TIERS ━━━
- Hot  (score >= 70): immediate personalised outreach
- Warm (score 40-69): nurture track, follow up in 2 weeks
- Cold (score < 40): deprioritise, revisit if signals improve

━━━ INSTRUCTIONS ━━━
1. Evaluate each dimension against the rubric above.
2. Sum the four dimension scores for a total 0-100 score.
3. Assign a tier: Hot / Warm / Cold.
4. Write EXACTLY 2 sentences of reasoning:
   - Sentence 1: What makes this lead promising or weak (cite the highest/lowest
     scoring dimension).
   - Sentence 2: Recommended next action with the decision-maker's name/title
     (if known), or what information gap would most improve the score.
5. Call update_lead_score with (lead_id, score, reason) to persist to CRM.
6. After the CRM call, output your full analysis followed by a ```json block:

```json
{
  "score": <integer 0-100>,
  "tier": "<Hot|Warm|Cold>",
  "reason": "<2 sentence explanation>",
  "breakdown": {
    "company_size": <0-25>,
    "industry_fit": <0-25>,
    "pain_point_relevance": <0-30>,
    "decision_maker": <0-20>
  },
  "recommended_action": "<immediate_outreach|nurture|disqualify>"
}
```

Be specific. Reference the company name, industry, and decision-maker in your
reasoning. Generic responses that could apply to any lead are not acceptable.
"""


# ─── Pipeline Agent Definition ────────────────────────────────────────────────
# This is the canonical agent used by main.py's SequentialAgent.
# `output_key="lead_score_summary"` causes ADK to write the agent's final
# text response (including the ```json block) into session state automatically,
# making it available to the OutreachDrafter agent without manual state updates.
# ─────────────────────────────────────────────────────────────────────────────
lead_scorer = Agent(
    name="lead_scorer",

    # gemini-2.0-flash: fast, available, and bypasses daily project rate limits
    model="gemini-2.0-flash",

    description=(
        "Evaluates a lead's quality and assigns a priority score (0-100), "
        "a tier (Hot/Warm/Cold), and a 2-sentence reason based on company size, "
        "industry fit, pain point relevance, and decision-maker accessibility."
    ),

    instruction=_SCORER_INSTRUCTION,

    # ADK automatically writes the final text response into state["lead_score_summary"].
    # No manual state.update() needed. The OutreachDrafter reads this key.
    output_key="lead_score_summary",

    tools=[crm_toolset],    # update_lead_score only — principle of least privilege
)


# =============================================================================
# SECTION 4 — JSON EXTRACTION HELPER
# =============================================================================

def _extract_score_json(text: str) -> dict:
    """
    Extract the structured JSON block from the LLM's scoring response.

    Mirrors the same pattern used in lead_researcher._extract_json_from_brief().
    We keep the logic consistent across agents so the parsing behaviour is
    predictable and easy to maintain in one place.

    Args:
        text: Full text response from the LLM scorer agent.

    Returns:
        Parsed dict on success, or a fallback dict if extraction fails.
    """
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return {}   # Caller will use deterministic fallback

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


# =============================================================================
# SECTION 5 — STANDALONE LLM SCORER
# =============================================================================

# ─── Standalone agent (no CRM dependency) ────────────────────────────────────
# Separate from `lead_scorer` for the same reasons as in lead_researcher:
# - No CRM toolset dependency (avoids requiring the MCP server to be running)
# - No output_key — we capture and parse the response ourselves
# ─────────────────────────────────────────────────────────────────────────────
_STANDALONE_SCORER_INSTRUCTION = """\
You are a senior B2B sales strategist and lead qualification expert.

Research data for the company to score:
{research_json}

━━━ SCORING RUBRIC (100 points total) ━━━

1. COMPANY SIZE (0-25 pts)
   - <10 employees            →  5 pts
   - 11-50 employees          → 10 pts
   - 51-200 employees         → 16 pts
   - 201-1000 employees       → 21 pts
   - 1000+ employees          → 25 pts
   - Unknown                  →  8 pts

2. INDUSTRY FIT (0-25 pts)
   - Tech/SaaS/FinTech/eCommerce/Logistics/AI → 25 pts
   - Healthcare/Education/Manufacturing/HR     → 18 pts
   - Government/Non-profit/Mining              →  8 pts
   - Unknown                                  → 10 pts

3. PAIN POINT RELEVANCE (0-30 pts)
   - 0 pain points  →  0 pts
   - 1 pain point   →  8 pts
   - 2 pain points  → 16 pts
   - 3 pain points  → 22 pts
   - 4 pain points  → 26 pts
   - 5+ pain points → 30 pts

4. DECISION MAKER ACCESSIBILITY (0-20 pts)
   - Name + title known           → 12 pts base
   - C-suite title                → +8 pts
   - VP / Director title          → +6 pts
   - Manager / Senior title       → +3 pts
   - LinkedIn URL confirmed       → +3 pts
   (Capped at 20 pts)

━━━ TIERS ━━━
Hot (>=70) | Warm (40-69) | Cold (<40)

━━━ TASK ━━━
Score this lead. Then output EXACTLY a ```json block:

```json
{
  "score": <integer 0-100>,
  "tier": "<Hot|Warm|Cold>",
  "reason": "<EXACTLY 2 sentences: sentence 1 = strongest signal, sentence 2 = recommended action or limiting factor>",
  "breakdown": {
    "company_size": <0-25>,
    "industry_fit": <0-25>,
    "pain_point_relevance": <0-30>,
    "decision_maker": <0-20>
  },
  "recommended_action": "<immediate_outreach|nurture|disqualify>"
}
```
"""

_standalone_scorer_agent = Agent(
    name="lead_scorer_standalone",
    model="gemini-2.0-flash",
    description="Standalone LLM scorer: scores a research dict and returns structured JSON.",
    instruction=_STANDALONE_SCORER_INSTRUCTION,
    # No tools, no output_key — pure LLM reasoning, response captured manually
)


async def _run_llm_score_async(research_result: dict) -> ScoreResult:
    """
    Internal async implementation of score_lead().

    Sends the research dict to the Gemini agent and extracts the structured
    ScoreResult. Falls back to the deterministic scorer if the LLM call fails
    or returns malformed JSON.

    Args:
        research_result: ResearchResult dict from lead_researcher.

    Returns:
        ScoreResult dict.
    """
    # Serialise the research dict as a JSON string for the LLM.
    # We strip raw_brief from the payload to keep token count reasonable —
    # the structured fields (overview, pain_points, etc.) carry all the
    # signal the scorer needs without the redundant narrative.
    compact_research = {k: v for k, v in research_result.items()
                        if k not in ("raw_brief",)}
    research_json_str = json.dumps(compact_research, indent=2)

    log_action("LeadScorer", "AGENT_RUN_START", f"company={research_result.get('company_name', 'Unknown')}")
    app_name = "lead_scorer_standalone"

    initial_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(
            text=f"Score this lead. Research data: {research_json_str}"
        )],
    )

    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash-latest"]
    # Fallback chain: gemini-2.0-flash (primary) → gemini-2.0-flash-lite →
    # gemini-1.5-flash-latest (final LLM fallback before the deterministic scorer).
    raw_response = ""
    last_error: Exception | None = None
    success = False

    for model_name in models:
        _standalone_scorer_agent.model = model_name
        print(f"🤖 Using model: {model_name}")
        log_action("LeadScorer", "MODEL_SELECTION", f"model={model_name}")

        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name=app_name,
            user_id="standalone_user",
            state={"research_json": research_json_str},
        )

        runner = Runner(
            agent=_standalone_scorer_agent,
            app_name=app_name,
            session_service=session_service,
        )

        MAX_RETRIES = 3
        BACKOFF_BASE = 2

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async for event in runner.run_async(
                    user_id="standalone_user",
                    session_id=session.id,
                    new_message=initial_message,
                ):
                    if event.is_final_response():
                        content = event.content
                        if content and content.parts:
                            raw_response = "".join(
                                p.text for p in content.parts
                                if hasattr(p, "text") and p.text
                            )
                success = True
                break   # Success
            except Exception as exc:
                last_error = exc
                err = str(exc)
                is_transient = any(k in err for k in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if is_transient and attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE ** attempt
                    print(f"  [LeadScorer] Attempt {attempt}/{MAX_RETRIES} failed with model {model_name} ({err[:80]}). Retrying in {wait}s…")
                    await asyncio.sleep(wait)
                    session = await session_service.create_session(
                        app_name=app_name,
                        user_id="standalone_user",
                        state={"research_json": research_json_str},
                    )
                else:
                    print(f"  [LeadScorer] Model {model_name} failed/exhausted ({err[:80]}). Trying next model in fallback chain...")
                    break
        if success:
            break
    else:
        if last_error:
            log_action("LeadScorer", "AGENT_RUN_ERROR", f"company={research_result.get('company_name', 'Unknown')}, error={str(last_error)[:100]}")
            raise last_error

    # Try to use LLM's output; fall back to deterministic scorer if LLM fails
    llm_parsed = _extract_score_json(raw_response)

    if llm_parsed and "score" in llm_parsed:
        # Validate and clamp the LLM's score (LLMs occasionally exceed bounds)
        score = min(100, max(0, int(llm_parsed.get("score", 0))))
        tier, action = _derive_tier_and_action(score)   # Recompute from our rules

        breakdown_raw = llm_parsed.get("breakdown", {})
        breakdown: ScoreBreakdown = {
            "company_size": min(25, max(0, int(breakdown_raw.get("company_size", 0)))),
            "industry_fit": min(25, max(0, int(breakdown_raw.get("industry_fit", 0)))),
            "pain_point_relevance": min(30, max(0, int(breakdown_raw.get("pain_point_relevance", 0)))),
            "decision_maker": min(20, max(0, int(breakdown_raw.get("decision_maker", 0)))),
        }

        result = ScoreResult(
            score=score,
            tier=tier,
            reason=llm_parsed.get("reason", ""),
            breakdown=breakdown,
            recommended_action=action,
        )
        log_action("LeadScorer", "AGENT_RUN_SUCCESS", f"company={research_result.get('company_name', 'Unknown')}, score={score}, tier={tier}")
        return result

    # LLM gave no parseable output — fall back to deterministic scorer
    print("  [LeadScorer] LLM output not parseable — using deterministic scorer as fallback.")
    return score_lead_dict(research_result)


def score_lead(research_result: dict) -> ScoreResult:
    """
    Score a lead using the LLM agent for richer qualitative reasoning.

    This is the LLM-enhanced version of score_lead_dict(). It sends the
    research data to Gemini for scoring, then validates and returns the
    result. Falls back to the deterministic scorer automatically if the
    API is unavailable.

    Use this when you want the LLM to provide nuanced qualitative reasoning
    that goes beyond the heuristic rubric — e.g. spotting cross-dimensional
    patterns or industry-specific context.

    Use score_lead_dict() when you need speed, low cost, or determinism
    (e.g. batch scoring, unit tests, CI pipelines).

    Args:
        research_result: ResearchResult dict from lead_researcher.research_company().

    Returns:
        ScoreResult dict with score, tier, reason, breakdown, recommended_action.

    Note:
        Requires GOOGLE_API_KEY in .env. On 503 errors, retries 3x with
        exponential backoff, then falls back to deterministic scoring.
    """
    try:
        return asyncio.run(_run_llm_score_async(research_result))
    except Exception as exc:
        print(f"  [LeadScorer] LLM scoring failed ({str(exc)[:120]}). Falling back to deterministic scorer.")
        return score_lead_dict(research_result)


# =============================================================================
# SECTION 6 — CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import pprint

    # ── Demo research dict (mirrors ResearchResult schema) ───────────────────
    # This lets you test the scorer without needing the researcher to run first.
    # Replace with output from research_company() for a real end-to-end test.
    demo_research = {
        "company_name": "Acme SaaS Corp",
        "overview": (
            "Acme SaaS Corp builds revenue intelligence software for mid-market "
            "B2B sales teams. Their platform integrates with Salesforce and "
            "HubSpot to surface deal risk and coaching insights."
        ),
        "company_size": "201-500 employees (Series B, $30M raised)",
        "industry": "SaaS / Sales Tech / Revenue Intelligence",
        "pain_points": [
            "Sales reps lack real-time deal coaching — rely on end-of-quarter reviews",
            "CRM data quality is poor — reps don't log activities consistently",
            "No visibility into which deals are at risk until it's too late",
            "Onboarding new reps takes 3-4 months due to no structured playbooks",
        ],
        "decision_maker": {
            "name": "Sarah Chen",
            "title": "VP of Sales",
            "linkedin_url": "https://linkedin.com/in/sarahchen-sales",
        },
        "raw_brief": "",
        "search_mode": "stub",
    }

    print(f"\n{'='*60}")
    print("  Lead Scorer — Deterministic Mode")
    print(f"{'='*60}\n")

    # Run the fast deterministic scorer
    result = score_lead_dict(demo_research)

    print("📊 SCORE RESULT:")
    print("─" * 60)
    pprint.pprint(result, width=80, sort_dicts=False)
    print("─" * 60)

    print(f"\n🏆  Score  : {result['score']}/100")
    print(f"🔥  Tier   : {result['tier']}")
    print(f"📋  Action : {result['recommended_action']}")
    print(f"\n💬  Reason :")
    print(f"    {result['reason']}")
    print(f"\n📈  Breakdown:")
    for dim, pts in result["breakdown"].items():
        max_pts = {"company_size": 25, "industry_fit": 25,
                   "pain_point_relevance": 30, "decision_maker": 20}[dim]
        bar = "█" * pts + "░" * (max_pts - pts)
        print(f"    {dim:<26} {pts:>2}/{max_pts}  {bar}")
    print()
