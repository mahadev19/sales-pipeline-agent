"""
main.py — Sales Pipeline Orchestrator
======================================

PURPOSE:
    Command-line entry point for the multi-agent B2B sales pipeline.
    Accepts a list of company names, runs each through a three-stage
    AI agent pipeline, persists results to the CRM, and prints a
    formatted summary table.

PIPELINE FLOW (per company):
    1. LeadResearcher   — Web search → structured intelligence dict
    2. LeadScorer       — Deterministic 4-dimension scoring → score + tier
    3. OutreachDrafter  — Tone-calibrated email + LinkedIn drafts
    4. CRM Persistence  — add_lead() saves everything to crm_server.py
    5. Status Update    — update_lead_status() marks lead as "contacted"

USAGE:
    # Single company:
    python main.py --companies "Stripe"

    # Multiple companies (comma-separated):
    python main.py --companies "Notion,HubSpot,Figma"

    # Legacy mode — process a specific CRM lead ID:
    python main.py --lead-id lead_001

    # Show what this pipeline can do (Agent Skills summary):
    python main.py --agent-skill

    # Fast mode — use template drafter (no API calls for drafts):
    python main.py --companies "Stripe" --fast

    # Skip CRM server (useful if mcp_server not running):
    python main.py --companies "Stripe" --no-crm

ENVIRONMENT VARIABLES (.env file):
    GOOGLE_API_KEY             — Gemini API key (Google AI Studio)
    GOOGLE_GENAI_USE_VERTEXAI  — "False" for AI Studio, "True" for Vertex AI
    GOOGLE_CLOUD_PROJECT       — GCP project ID (Vertex AI only)
    GOOGLE_CLOUD_LOCATION      — GCP region, e.g. "us-central1"
    SEARCH_API_KEY             — Google Custom Search API key (optional)
    SEARCH_ENGINE_ID           — Custom Search Engine ID (optional)
    CRM_SERVER_URL             — MCP server URL (default: http://localhost:8001/sse)

ARCHITECTURE NOTES:
    ─ This file contains TWO orchestration modes that coexist:

      A) COMPANY-NAME MODE (new, this file's primary purpose):
         Uses the standalone programmatic APIs of each agent:
           research_company()    — agents/lead_researcher.py
           score_lead_dict()     — agents/lead_scorer.py
           draft_outreach_fast() — agents/outreach_drafter.py
         Calls MCP CRM tools directly via Python (no ADK session needed).
         Deterministic, fast, no Gemini API calls for scorer or drafter.

      B) LEAD-ID MODE (legacy, preserved for backward-compatibility):
         Uses the ADK SequentialAgent + Runner + InMemorySessionService.
         Processes a single CRM lead_id through the full LLM pipeline.
         Requires all three MCP toolsets to be connected.

    ─ We deliberately keep the two modes in the same file so operators have
      a single entry point regardless of their workflow. The --companies flag
      triggers company-name mode; --lead-id triggers legacy mode.

    ─ The summary table is printed using pure stdlib (no tabulate dep)
      with fixed-width column formatting for terminal readability.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

# Force UTF-8 output on Windows terminals (cp1252 default can't encode many chars).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# Google ADK imports — used by legacy lead-id mode
from google.adk.agents import SequentialAgent
try:
    from google.adk.agents import Workflow as _Pipeline
except ImportError:
    _Pipeline = SequentialAgent   # Fallback for older ADK versions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

import re

# Ensure parent directory for logs or local log folder is accessible.
# Load .env variables
load_dotenv()

# Verify that the API Key is loaded and not hardcoded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # SECURITY NOTE: Loading API keys from environment variables prevents secrets exposure in Git.
    sys.stderr.write("SECURITY ERROR: GOOGLE_API_KEY environment variable is not set. API keys must never be hardcoded.\n")
    sys.exit(2)

def log_action(agent_name: str, action: str, details: str = "") -> None:
    """Log an agent action to agent_log.txt with a UTC ISO-8601 timestamp.
    
    SECURITY NOTE: Secure logging is critical for auditing, system debugging,
    intrusion detection, and compliance.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    log_line = f"[{timestamp}] {agent_name} | {action} | {details}\n"
    try:
        # Append to agent_log.txt in the current workspace directory
        with open("agent_log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        sys.stderr.write(f"Logging error: {e}\n")

def sanitize_company_name(name: str) -> str:
    """Sanitize and validate a company name input.
    
    SECURITY NOTE: Input sanitization protects against prompt injection,
    denial of service, and directory traversal if names are used in paths.
    """
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Company name cannot be empty or whitespace-only.")
    if len(cleaned) > 80:
        raise ValueError(f"Company name is suspiciously long ({len(cleaned)} chars). Max length is 80 characters.")
    # Reject tags, control characters, and common injection symbols
    if any(char in cleaned for char in ("\x00", "\n", "\r", "\t", "<", ">", "\\")):
        raise ValueError("Company name contains invalid control characters or HTML tags.")
    return cleaned

def scan_for_secrets(text: str) -> list[str]:
    """Scan text for patterns resembling API keys or secrets (Data Loss Prevention).
    
    SECURITY NOTE: Output scanning helps prevent data loss/exfiltration by
    detecting if the LLM mistakenly generated or leaked internal API credentials.
    """
    if not text:
        return []
    
    matches = []
    # Test for standard Google API keys, OpenAI keys, and .env key formats
    patterns = [
        ("Google/Gemini API Key", r"AIzaSy[A-Za-z0-9_-]{35}"),
        ("Dotenv Google API Key format", r"AQ\.[A-Za-z0-9_-]{30,}"),
        ("Generic OpenAI/SaaS Secret", r"sk-[a-zA-Z0-9_-]{32,}")
    ]
    for label, regex in patterns:
        found = re.findall(regex, text)
        if found:
            for f in found:
                matches.append(f"{label}: {f[:8]}...")
    return matches

# ---------------------------------------------------------------------------
# Agent imports — standalone programmatic APIs
# ---------------------------------------------------------------------------
# We import the standalone functions (not just the ADK agent objects) so that
# company-name mode can call them directly without going through ADK sessions.
# The ADK agent objects (lead_researcher, lead_scorer, outreach_drafter) are
# still imported for legacy lead-id mode's SequentialAgent.
# ---------------------------------------------------------------------------
from agents.lead_researcher import (
    lead_researcher,
    research_company,
    ResearchResult,
)
from agents.lead_scorer import (
    lead_scorer,
    score_lead_dict,
    ScoreResult,
)
from agents.outreach_drafter import (
    outreach_drafter,
    draft_outreach_fast,
    DraftResult,
)
from agents.reviewer_agent import (
    review_email,
)

# Execution tracer — records per-agent timing, model, I/O previews for
# the Streamlit trace visualization page (Page 6 in app.py).
from tools.tracer import Tracer

# CRM server functions — called directly in company-name mode
# We import them as Python functions rather than going through the MCP HTTP
# layer. This works because both this file and crm_server.py run in the same
# Python process (or can be imported). In production, you'd call the MCP
# tools via McpToolset instead.
from mcp_server.crm_server import (
    add_lead,
    update_lead_status,
    get_all_leads,
    get_leads_by_tier,
)


# =============================================================================
# SECTION 1 — AGENT SKILL CARD
# =============================================================================
# The --agent-skill flag prints this card. It describes what the pipeline
# does, what tools each agent uses, and how they chain together.
#
# COMPETITION NOTE: This is the "Agent Skills" demonstration required by the
# competition brief. It shows that the pipeline is a composable, multi-agent
# system where each agent has a clearly defined skill, tool set, and contract.
# =============================================================================

AGENT_SKILL_CARD = """
╔══════════════════════════════════════════════════════════════════════════╗
║          SALES PIPELINE AGENT — SKILL CARD                              ║
║          Built with Google ADK + FastMCP                                ║
╚══════════════════════════════════════════════════════════════════════════╝

OVERVIEW:
  An end-to-end AI sales pipeline that transforms a raw company name into a
  scored lead with personalised outreach drafts — fully automated, zero manual
  research required.

PIPELINE: LeadResearcher → LeadScorer → OutreachDrafter → CRM

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT 1 — LEAD RESEARCHER
  Role    : Web intelligence gathering
  Input   : Company name (string) or CRM lead_id
  Output  : Structured research dict with:
              - Company overview, size, industry
              - 3-5 specific B2B SaaS pain points
              - Decision-maker name, title, LinkedIn URL
  Tools   : search_web (Google Custom Search / stub)
             get_lead   (CRM MCP server)
  Model   : gemini-flash-latest
  Mode    : Dual — standalone (no LLM for parsing) or pipeline (ADK session)

AGENT 2 — LEAD SCORER
  Role    : 4-dimension priority scoring (0-100)
  Input   : ResearchResult dict from LeadResearcher
  Output  : ScoreResult with score, tier, reason, breakdown
  Scoring : Company Size (0-25) + Industry Fit (0-25)
             + Pain Points (0-30) + Decision Maker (0-20)
  Tiers   : Hot (>=70) | Warm (40-69) | Cold (<40)
  Tools   : update_lead_score (CRM MCP server — pipeline mode)
  Model   : gemini-flash-latest (LLM mode) or pure Python (deterministic)
  Mode    : Dual — deterministic (fast, zero cost) or LLM-enhanced

AGENT 3 — OUTREACH DRAFTER
  Role    : Tone-calibrated outreach copywriting
  Input   : ResearchResult + ScoreResult
  Output  : Cold email (subject + <=150 word body)
             LinkedIn connection note (<=300 chars)
  Strategy: 6 prompting techniques:
             1. Persona anchoring
             2. Anti-pattern injection (20 forbidden phrases blacklisted)
             3. Pain-point anchoring (must cite specific research findings)
             4. Tier-aware tone modulation (Hot/Warm/Cold tone changes)
             5. JSON format enforcement
             6. Character-count constraints
  Tools   : get_lead, add_outreach_draft (CRM MCP server)
  Model   : gemini-flash-latest (LLM mode) or Python templates (fast)

CRM SERVER — FastMCP
  Role    : Single source of truth for lead data
  Storage : data/crm.json (JSON file, atomic writes)
  Tools   : add_lead, get_all_leads, get_leads_by_tier,
             update_lead_status, get_lead, list_leads,
             update_lead_score, add_outreach_draft
  Design  : Result-object pattern (no exceptions across MCP boundary)
             Self-healing missing-file creation
             Atomic write via .tmp + rename (crash-safe)
             Upsert semantics (idempotent re-runs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE:
  python main.py --companies "Stripe,Notion,HubSpot"
  python main.py --companies "Figma" --fast
  python main.py --lead-id lead_001          # Legacy CRM lead mode
  python main.py --agent-skill               # This card

TECH STACK:
  Google ADK (SequentialAgent, Runner, InMemorySessionService)
  FastMCP (MCP server — tools over HTTP/SSE)
  Gemini Flash (gemini-flash-latest via Google AI Studio)
  Python 3.12 | TypedDict schemas | Atomic JSON persistence
"""


# =============================================================================
# SECTION 2 — RESULT DATA STRUCTURES
# =============================================================================

class PipelineResult(NamedTuple):
    """
    Holds the complete output for one company processed through the pipeline.

    Using NamedTuple (vs dataclass or dict) for:
    - Immutability — results should not be mutated after production
    - Unpacking support — easy to spread into print/table functions
    - Zero extra dependencies

    The `error` field is non-empty when any stage failed. We return a
    PipelineResult (not raise) even on failure so batch processing can
    continue with the next company and collect all results for the summary.
    """
    company: str            # Input company name (as provided by user)
    score: int              # 0-100 priority score (0 if failed)
    tier: str               # "Hot" | "Warm" | "Cold" | "ERROR"
    status: str             # CRM pipeline status after processing
    lead_id: str            # CRM lead ID (e.g. "lead_004") or empty
    email_subject: str      # Generated email subject line
    word_count: int         # Email body word count
    linkedin_chars: int     # LinkedIn note character count
    duration_s: float       # Wall-clock seconds for this company
    search_mode: str        # "live" | "stub" | "error"
    error: str              # Non-empty string if any stage failed
    research_eval: int = 0  # Research quality score (0-100)
    email_eval: int = 0     # Email quality score (0-100)


# =============================================================================
# SECTION 3 — PIPELINE STAGES
# =============================================================================
# Each stage is a pure function that takes well-typed inputs and returns
# well-typed outputs. Errors are caught and returned as strings so the
# orchestrator can collect all results before printing the summary.
#
# DESIGN DECISION: Stages are synchronous functions (not async coroutines)
# because all three standalone agent APIs (research_company, score_lead_dict,
# draft_outreach_fast) are synchronous. The async ADK sessions are only used
# internally by those functions. Making the outer loop sync keeps main.py
# simple and avoids asyncio.run() nesting issues on Windows.
# =============================================================================

def stage_research(company: str, similar_leads: str = "") -> tuple[ResearchResult | None, str]:
    """
    Stage 1: Run the LeadResearcher for a company name.

    Calls research_company() which internally spins up an ADK agent session,
    runs web searches (or stubs), and returns a structured ResearchResult dict.

    Error handling:
    - research_company() has its own internal retry loop (3x, backoff).
    - It NEVER raises — on failure it returns a dict with search_mode="error".
    - We check for search_mode="error" here and surface it as a stage error.

    Args:
        company: Company name string (e.g. "Stripe").
        similar_leads: Context string containing matching leads.

    Returns:
        (ResearchResult, "") on success, or (None, error_message) on failure.
    """
    print(f"    [1/3] Researching {company}...", end=" ", flush=True)
    t0 = time.time()
    log_action("Orchestrator", "STAGE_RESEARCH_START", f"company={company}")

    try:
        result = research_company(company, similar_leads)

        # research_company() returns an error-keyed dict on failure
        if result.get("search_mode") == "error":
            err = result.get("error", "Research failed with unknown error")
            print(f"FAILED ({time.time()-t0:.1f}s)")
            log_action("Orchestrator", "STAGE_RESEARCH_FAILED", f"company={company}, error={err[:100]}")
            return None, f"Research error: {err[:120]}"

        mode = result.get("search_mode", "stub")
        pain_count = len([p for p in result.get("pain_points", []) if p and len(p) > 10])
        print(f"OK [{mode}, {pain_count} pain pts] ({time.time()-t0:.1f}s)")
        log_action("Orchestrator", "STAGE_RESEARCH_SUCCESS", f"company={company}, mode={mode}, pain_points={pain_count}")
        return result, ""

    except Exception as exc:
        print(f"FAILED ({time.time()-t0:.1f}s)")
        log_action("Orchestrator", "STAGE_RESEARCH_EXCEPTION", f"company={company}, error={str(exc)[:100]}")
        return None, f"Research exception: {str(exc)[:120]}"


def stage_score(research: ResearchResult) -> tuple[ScoreResult | None, str]:
    """
    Stage 2: Score the research result using the deterministic scorer.

    Uses score_lead_dict() (pure Python, no API calls) for speed and
    reliability. The LLM-based score_lead() is available for richer
    qualitative reasoning but requires API access and is slower.

    DESIGN DECISION: We default to the deterministic scorer in main.py
    because:
    1. The same Gemini API that powers the researcher may be rate-limited —
       adding two more LLM calls per company (scorer + drafter) triples the
       503 risk.
    2. The deterministic scorer is perfectly adequate for prioritisation —
       it uses explicit rubric weights, not LLM guesswork.
    3. The LLM scorer adds nuanced qualitative reasoning but the delta in
       actionability for the sales rep is minimal.

    Use `score_lead()` instead if richer justifications are needed.

    Args:
        research: ResearchResult dict from stage_research().

    Returns:
        (ScoreResult, "") on success, or (None, error_message) on failure.
    """
    print(f"    [2/3] Scoring...", end=" ", flush=True)
    t0 = time.time()
    log_action("Orchestrator", "STAGE_SCORE_START", f"company={research.get('company_name', 'Unknown')}")

    try:
        result = score_lead_dict(research)
        print(
            f"OK [{result['score']}/100 — {result['tier']}] ({time.time()-t0:.1f}s)"
        )
        log_action("Orchestrator", "STAGE_SCORE_SUCCESS", f"company={research.get('company_name', 'Unknown')}, score={result['score']}, tier={result['tier']}")
        return result, ""

    except Exception as exc:
        print(f"FAILED ({time.time()-t0:.1f}s)")
        log_action("Orchestrator", "STAGE_SCORE_EXCEPTION", f"company={research.get('company_name', 'Unknown')}, error={str(exc)[:100]}")
        return None, f"Scoring exception: {str(exc)[:120]}"


def stage_draft(
    research: ResearchResult,
    score: ScoreResult,
    fast: bool = True,
) -> tuple[DraftResult | None, str]:
    """
    Stage 3: Generate outreach email and LinkedIn note.

    Two modes controlled by the `fast` parameter:
    - fast=True  (default): draft_outreach_fast() — template-based, instant
    - fast=False: draft_outreach() — LLM-generated via Gemini, ~10-20s

    In fast mode, the template engine still personalises drafts by injecting
    the specific company name, pain points, decision-maker name, and tier
    tone — it just doesn't use an LLM for the prose generation.

    Args:
        research: ResearchResult from stage_research().
        score: ScoreResult from stage_score().
        fast: If True, use template drafter (no API). Default True.

    Returns:
        (DraftResult, "") on success, or (None, error_message) on failure.
    """
    mode_label = "template" if fast else "LLM"
    print(f"    [3/3] Drafting outreach [{mode_label}]...", end=" ", flush=True)
    t0 = time.time()
    company = research.get("company_name", "Unknown")
    log_action("Orchestrator", "STAGE_DRAFT_START", f"company={company}, mode={mode_label}")

    try:
        if fast:
            from agents.outreach_drafter import draft_outreach_fast
            result = draft_outreach_fast(research, score)
        else:
            from agents.outreach_drafter import draft_outreach
            result = draft_outreach(research, score)

        print(
            f"OK [{result['word_count']}w email, {result['char_count']}ch LI] "
            f"({time.time()-t0:.1f}s)"
        )
        log_action("Orchestrator", "STAGE_DRAFT_SUCCESS", f"company={company}, mode={mode_label}, words={result['word_count']}, chars={result['char_count']}")
        return result, ""

    except Exception as exc:
        print(f"FAILED ({time.time()-t0:.1f}s)")
        log_action("Orchestrator", "STAGE_DRAFT_EXCEPTION", f"company={company}, error={str(exc)[:100]}")
        return None, f"Draft exception: {str(exc)[:120]}"


def stage_persist(
    research: ResearchResult,
    score: ScoreResult,
    draft: DraftResult,
    use_crm: bool = True,
    status: str = "contacted",
    original_email_draft: str = "",
) -> tuple[str, str, str]:
    """
    Stage 4: Persist the pipeline results to the CRM.

    Calls two CRM tools:
    1. add_lead()           — Creates or updates the lead record with score,
                              tier, email draft, and LinkedIn draft.
    2. update_lead_status() — Marks the lead as the reviewed/approved stage.

    If use_crm=False (--no-crm flag), this stage is skipped and the lead_id
    is returned as "local-only".

    DESIGN DECISION: We call the CRM functions directly as Python imports
    rather than via HTTP to the MCP server. This works because:
    - In this process, both main.py and crm_server.py share the same Python
      runtime, so imports work directly.
    - It avoids an extra HTTP round-trip per company.
    - The CRM tools still enforce all validation rules (score range, tier
      validation, atomic writes, etc.) — the MCP boundary only matters for
      cross-process calls.

    In a production setup where the CRM server is a separate Cloud Run
    service, you'd replace these direct calls with httpx requests or
    McpToolset calls.

    Args:
        research:             ResearchResult (provides company name).
        score:                ScoreResult (provides score, tier, reason).
        draft:                DraftResult (provides email_subject, email_body, linkedin_message).
        use_crm:              If False, skip CRM persistence. Default True.
        status:               The target pipeline status to set. Default "contacted".
        original_email_draft: Original email draft before reviewer reflection (optional).

    Returns:
        (lead_id, crm_status, error_message) — error is empty on success.
    """
    if not use_crm:
        return "local-only", "new", ""

    print(f"    [4/4] Saving to CRM...", end=" ", flush=True)
    t0 = time.time()

    company = research.get("company_name", "Unknown")
    email_full = f"Subject: {draft['email_subject']}\n\n{draft['email_body']}"
    log_action("Orchestrator", "STAGE_PERSIST_START", f"company={company}, use_crm={use_crm}")

    dm = research.get("decision_maker", {})
    dm_name = dm.get("name", "")
    dm_title = dm.get("title", "")

    try:
        # ── Upsert the lead with score + drafts ───────────────────────────────
        # add_lead() creates a new record if the company doesn't exist, or
        # updates an existing one if it does. This makes pipeline re-runs
        # idempotent — no duplicate records accumulate.
        crm_result = add_lead(
            company=company,
            score=score["score"],
            tier=score["tier"],
            email_draft=email_full,
            linkedin_draft=draft["linkedin_message"],
            name=dm_name,
            title=dm_title,
            original_email_draft=original_email_draft,
        )

        if crm_result.get("status") != "success":
            err = crm_result.get("error", "Unknown CRM error")
            print(f"FAILED ({time.time()-t0:.1f}s)")
            log_action("Orchestrator", "STAGE_PERSIST_FAILED", f"company={company}, error={err[:100]}")
            return "", "new", f"CRM add_lead error: {err}"

        lead_id = crm_result.get("lead_id", "")

        # ── Advance status to target ──────────────────────────────────────────
        status_result = update_lead_status(company=company, status=status)
        final_status = status_result.get("new_status", status)

        print(f"OK [{lead_id}] ({time.time()-t0:.1f}s)")
        log_action("Orchestrator", "STAGE_PERSIST_SUCCESS", f"company={company}, lead_id={lead_id}, status={final_status}")
        return lead_id, final_status, ""

    except Exception as exc:
        print(f"FAILED ({time.time()-t0:.1f}s)")
        log_action("Orchestrator", "STAGE_PERSIST_EXCEPTION", f"company={company}, error={str(exc)[:100]}")
        return "", "new", f"CRM exception: {str(exc)[:120]}"


# =============================================================================
# SECTION 4 — COMPANY-NAME PIPELINE ORCHESTRATOR
# =============================================================================

def run_company_pipeline(
    company: str,
    fast: bool = True,
    use_crm: bool = True,
    auto_approve: bool = False,
) -> PipelineResult:
    """
    Run the full research → score → draft → persist pipeline for one company.

    This is the core orchestration function. It calls each stage in order,
    short-circuiting on failure (printing the error and returning a
    PipelineResult with tier="ERROR" so the summary table is always complete).

    The function is SYNCHRONOUS even though the underlying agent functions use
    async ADK internally. Each agent function uses asyncio.run() internally,
    which manages its own event loop. This is safe because our outer loop is
    also synchronous — we never call run_company_pipeline() from within an
    async context.

    Args:
        company:  Company name to research (e.g. "Stripe").
        fast:     If True, use template drafter (no LLM for drafts).
        use_crm:  If True, persist results to the CRM server.

    Returns:
        PipelineResult NamedTuple with all fields populated.
    """
    wall_start = time.time()

    # ── Execution Tracer ──────────────────────────────────────────────────────
    # Instantiate a fresh Tracer for this company run. Each run produces one
    # trace file in data/traces/ for the Streamlit visualization page.
    # The Tracer records start/end times, model used, and I/O previews for
    # every agent span, enabling post-run performance analysis and debugging.
    tracer = Tracer()

    print(f"\n  Processing: {company}")
    print(f"  {'─' * 50}")

    # ── RAG Pattern: Semantic Retrieval Phase ─────────────────────────────────
    # Before we execute the research stage, we check our long-term memory (ChromaDB)
    # for any similar leads that we have processed in the past.
    #
    # How this implements the Retrieval-Augmented Generation (RAG) pattern:
    # 1. Retrieval: Query the ChromaDB collection using semantic similarity search
    #    based on the company name to fetch similar profiles (if any exist).
    # 2. Augmentation: If similar historical leads are found, we compile their stored
    #    documents and pass them as context under "Similar past leads: {matches}"
    #    into the Prompt / Instruction of the research agent.
    # 3. Generation: The researcher agent uses this context to align its industry classification,
    #    identify common/standard pain points, and target decision-makers with similar titles,
    #    thereby generating a richer and more context-aware brief.
    print("🧠 Checking long-term memory for similar leads...")
    similar_leads_context = ""
    try:
        from memory.vector_memory import recall_similar_leads
        matches = recall_similar_leads(company, n=3)
        if matches:
            print(f"🧠 Found {len(matches)} similar past leads — using as context")
            similar_leads_context = f"Similar past leads: {matches}"
    except Exception as e:
        print(f"    [!] Long-term memory query failed: {e}")

    # ── Stage 1: Research ─────────────────────────────────────────────────────
    # Tracer span: records the company name as the input and the full research
    # dict as the output for the LeadResearcher. model_used reflects whether
    # the web search ran live (gemini-2.0-flash) or in stub mode.
    tracer.start_span("LeadResearcher", input_data=company + (f" [context: {similar_leads_context[:60]}]" if similar_leads_context else ""))
    research, research_err = stage_research(company, similar_leads_context)
    tracer.end_span(
        "LeadResearcher",
        output=research or {"error": research_err},
        status="error" if research_err else research.get("search_mode", "success"),
        model_used=research.get("search_mode", "stub") if research else "error",
    )
    if research_err:
        tracer.save_trace(company)
        return PipelineResult(
            company=company, score=0, tier="ERROR", status="research_failed",
            lead_id="", email_subject="", word_count=0, linkedin_chars=0,
            duration_s=time.time()-wall_start,
            search_mode="error", error=research_err,
        )

    # ── Stage 2: Score ────────────────────────────────────────────────────────
    # Tracer span: inputs are the research dict fields that drive scoring.
    # model_used is "rules" for the deterministic scorer (no LLM API call).
    tracer.start_span("LeadScorer", input_data={
        "company": research.get("company_name", company),
        "industry": research.get("industry", ""),
        "pain_points_count": len(research.get("pain_points", [])),
    })
    score, score_err = stage_score(research)
    tracer.end_span(
        "LeadScorer",
        output=score or {"error": score_err},
        status="error" if score_err else "success",
        model_used="rules",   # Deterministic scorer uses no LLM
    )
    if score_err:
        tracer.save_trace(company)
        return PipelineResult(
            company=company, score=0, tier="ERROR", status="score_failed",
            lead_id="", email_subject="", word_count=0, linkedin_chars=0,
            duration_s=time.time()-wall_start,
            search_mode=research.get("search_mode", "stub"), error=score_err,
        )

    # ── Stage 3: Draft ────────────────────────────────────────────────────────
    # Tracer span: inputs are the score tier (drives tone) and pain points
    # (drives personalization). model_used is "template" in fast mode.
    tracer.start_span("OutreachDrafter", input_data={
        "tier": score.get("tier", ""),
        "score": score.get("score", 0),
        "mode": "template" if fast else "gemini-2.0-flash",
    })
    draft, draft_err = stage_draft(research, score, fast=fast)
    tracer.end_span(
        "OutreachDrafter",
        output=draft or {"error": draft_err},
        status="error" if draft_err else "success",
        model_used="template" if fast else "gemini-2.0-flash",
    )
    if draft_err:
        # Drafting failure is non-fatal — we can still persist score without drafts
        print(f"    [!] Draft failed: {draft_err[:80]}")
        # Build a minimal placeholder draft so CRM persistence can still run
        draft = DraftResult(
            email_subject=f"Re: {company}",
            email_body="[Draft generation failed — please create manually]",
            linkedin_message=f"Connecting with {company} team.",
            word_count=0, char_count=0,
            tier=score["tier"], tone_notes="",
        )
        draft_err = ""   # Treat as non-fatal going forward

    # ── Review / Reflection Stage ─────────────────────────────────────────────
    # Evaluate the generated email draft using the ReviewerAgent.
    # Score on 5 dimensions (1-10 each). If score < 35, rewrite the email.
    original_email_full = f"Subject: {draft['email_subject']}\n\n{draft['email_body']}"
    
    print(f"    [3b] Reviewing email draft (Reflection)...", end=" ", flush=True)
    t_rev = time.time()
    # Tracer span: ReviewerAgent scores and optionally rewrites the email.
    # model_used reflects the Gemini model used internally by review_email().
    tracer.start_span("ReviewerAgent", input_data={
        "subject": draft["email_subject"][:80],
        "body_preview": draft["email_body"][:80],
    })
    reviewer_status = "success"
    reviewer_model = "gemini-2.0-flash"
    try:
        review_res = review_email(
            company_name=company,
            email_subject=draft["email_subject"],
            email_body=draft["email_body"],
            pain_points=research.get("pain_points", []),
        )
        orig_reviewer_score = review_res["original_score"].get("total", 0)
        improved_reviewer_score = review_res["improved_score"].get("total", 0)
        
        print(f"OK [Score: {orig_reviewer_score}/50] ({time.time()-t_rev:.1f}s)")
        print(f"✍️  Reviewer scored email: {orig_reviewer_score}/50")
        
        if review_res.get("is_rewritten", False):
            print(f"✍️  Improved to: {improved_reviewer_score}/50")
            draft["email_subject"] = review_res["final_email"]["subject"]
            draft["email_body"] = review_res["final_email"]["body"]
            draft["word_count"] = len(draft["email_body"].split())

        tracer.end_span(
            "ReviewerAgent",
            output={
                "original_score": orig_reviewer_score,
                "improved_score": improved_reviewer_score,
                "is_rewritten": review_res.get("is_rewritten", False),
            },
            status=reviewer_status,
            model_used=reviewer_model,
        )
        log_action(
            "Orchestrator",
            "STAGE_REVIEW_SUCCESS",
            f"company={company}, original_score={orig_reviewer_score}/50, rewritten={review_res.get('is_rewritten', False)}, final_score={improved_reviewer_score}/50",
        )
    except Exception as exc:
        tracer.end_span("ReviewerAgent", output={"error": str(exc)},
                        status="error", model_used=reviewer_model)
        print(f"FAILED ({time.time()-t_rev:.1f}s)")
        print(f"    [!] Review failed with error: {exc}")
        log_action("Orchestrator", "STAGE_REVIEW_FAILED", f"company={company}, error={exc}")

    # ── Human-in-the-Loop (HITL) Review Phase ─────────────────────────────────
    # WHY HUMAN-IN-THE-LOOP (HITL) MATTERS FOR ENTERPRISE SALES AUTOMATION SAFETY:
    # Enterprise sales automation operates at the intersection of brand reputation,
    # compliance, and high-value revenue generation. Purely autonomous agents can
    # make critical errors (e.g. hallucinations, awkward phrasing, misinterpreting
    # recent news, or ignoring blacklisted terms) that damage prospect relationships
    # or violate privacy/anti-spam regulations (like CAN-SPAM and GDPR).
    #
    # Integrating a HITL safety control ensures:
    # 1. Verification of factual accuracy: Humans catch AI-hallucinated details.
    # 2. Brand alignment: Operators adjust tone to match enterprise standards.
    # 3. Dynamic overrides: Reps can inject unique personal touchpoints.
    # 4. Compliance enforcement: Ensures no sensitive/forbidden content is sent.
    crm_status = "approved"

    if not auto_approve:
        # 1. Print drafted outreach clearly to the terminal
        print(f"\n========================================================")
        print(f"  HUMAN REVIEW FOR: {company}")
        print(f"========================================================")
        print(f"  [EMAIL SUBJECT] : {draft['email_subject']}")
        print(f"  [EMAIL BODY]    :")
        print(f"  ------------------------------------------------------")
        for line in draft["email_body"].split("\n"):
            print(f"  {line}")
        print(f"  ------------------------------------------------------")
        print(f"  [LINKEDIN NOTE] : {draft['linkedin_message']}")
        print(f"========================================================")

        # 2. Ask for approval
        choice = ""
        while True:
            print("\n─────────────────────────────────────")
            print("👤 HUMAN REVIEW REQUIRED")
            print("─────────────────────────────────────")
            print(f"Approve this outreach for {company}?")
            print("[A] Approve  [E] Edit  [S] Skip  [Q] Quit")
            sys.stdout.write("> ")
            sys.stdout.flush()
            choice = sys.stdin.readline().strip()
            # Handle empty/newline and map to upper
            if choice:
                choice = choice[0].upper()
            if choice in ("A", "E", "S", "Q"):
                break
            print("[!] Invalid option. Please enter A, E, S, or Q.")

        # Log decision to agent_log.txt
        timestamp = datetime.now(timezone.utc).isoformat()
        log_line = f"HUMAN_DECISION: {company} → {choice} at {timestamp}\n"
        try:
            with open("agent_log.txt", "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            sys.stderr.write(f"Logging error: {e}\n")

        # 3. Handle each choice
        if choice == "Q":
            print("\n[!] Exiting pipeline cleanly.")
            sys.exit(0)
        elif choice == "S":
            crm_status = "skipped"
            print(f"⏭️ Skipping {company} (saving with status 'skipped').")
        elif choice == "E":
            print("\nEdit email subject (press Enter to keep):")
            sys.stdout.write("> ")
            sys.stdout.flush()
            new_subject = sys.stdin.readline().strip()
            if new_subject:
                draft["email_subject"] = new_subject

            print("Edit email body (press Enter to keep):")
            sys.stdout.write("> ")
            sys.stdout.flush()
            new_body = sys.stdin.readline().strip()
            if new_body:
                draft["email_body"] = new_body

            # Recalculate metrics
            draft["word_count"] = len(draft["email_body"].split())
            crm_status = "approved"
            print("✏️ Draft updated successfully.")
        else: # Choice A
            crm_status = "approved"
            print("✅ Outreach approved.")
    else:
        # Auto-approved (no review)
        crm_status = "approved"

    # ── Stage 4: Persist to CRM ───────────────────────────────────────────────
    # We call the CRM functions directly as Python imports
    # rather than via HTTP to the MCP server.
    lead_id, crm_status, crm_err = stage_persist(
        research, score, draft, use_crm, status=crm_status, original_email_draft=original_email_full
    )
    if crm_err:
        print(f"    [!] CRM save failed: {crm_err[:80]}")
        # CRM failure is non-fatal — the pipeline still produced useful output
        crm_status = "crm_error"

    # ── RAG Pattern: Memory Storage Phase ─────────────────────────────────────
    # Once the entire pipeline runs successfully and generates our final structured data,
    # we store the lead's company profile, score, and outreach draft in our ChromaDB vector database.
    # This acts as long-term memory, ensuring that future runs on similar companies can retrieve this lead
    # as context (completing the RAG store-and-recall cycle).
    if research and score and not research_err and not score_err:
        try:
            from memory.vector_memory import store_lead_memory
            store_lead_memory(
                company=research.get("company_name", company),
                research_dict=research,
                score=score["score"],
                email_draft=draft["email_body"] if draft else "",
            )
            print("🧠 Stored lead to long-term memory")
        except Exception as e:
            print(f"    [!] Storing to long-term memory failed: {e}")

    # ── Output Secrets Scanning ───────────────────────────────────────────────
    # SECURITY NOTE: Accidental credential leakage is a major vulnerability.
    # We scan the generated output from each agent stage to warn the user if
    # any key-like strings were outputted.
    if research:
        detected = scan_for_secrets(research.get("raw_brief", ""))
        if detected:
            print(f"\n    [WARNING] SECURITY: Potential secret(s) detected in Research brief: {', '.join(detected)}")
            log_action("Orchestrator", "SECURITY_ALERT_SECRETS_DETECTED", f"company={company}, stage=research, findings={detected}")
    if score:
        detected = scan_for_secrets(score.get("reason", ""))
        if detected:
            print(f"\n    [WARNING] SECURITY: Potential secret(s) detected in Score justification: {', '.join(detected)}")
            log_action("Orchestrator", "SECURITY_ALERT_SECRETS_DETECTED", f"company={company}, stage=scorer, findings={detected}")
    if draft:
        detected = scan_for_secrets(draft.get("email_body", "")) + scan_for_secrets(draft.get("linkedin_message", ""))
        if detected:
            print(f"\n    [WARNING] SECURITY: Potential secret(s) detected in Outreach draft: {', '.join(detected)}")
            log_action("Orchestrator", "SECURITY_ALERT_SECRETS_DETECTED", f"company={company}, stage=drafter, findings={detected}")

    duration = time.time() - wall_start

    # ── Evaluation Framework Phase ────────────────────────────────────────────
    research_eval_score = 0
    email_eval_score = 0
    try:
        from tools.evaluator import evaluate_pipeline_run
        eval_report = evaluate_pipeline_run(
            company=company,
            research=research,
            score_dict=score,
            email_draft=draft,
            time_taken=duration
        )
        research_eval_score = eval_report.get("research_eval_score", 0)
        email_eval_score = eval_report.get("email_eval_score", 0)
        log_action(
            "Orchestrator",
            "STAGE_EVALUATION_SUCCESS",
            f"company={company}, research_score={research_eval_score}, email_score={email_eval_score}"
        )
    except Exception as e:
        print(f"    [!] Evaluation framework execution failed: {e}")
        log_action("Orchestrator", "STAGE_EVALUATION_FAILED", f"company={company}, error={e}")

    # ── Save Execution Trace ──────────────────────────────────────────────────
    # Serialize all recorded spans to data/traces/{company}_{timestamp}.json.
    # The trace file is used by the Streamlit Page 6 (Execution Traces) to
    # visualize the Gantt chart, run comparisons, and I/O inspections.
    try:
        trace_path = tracer.save_trace(company)
        print(f"  📍 Trace saved: {trace_path.name}")
        log_action("Orchestrator", "TRACE_SAVED", f"company={company}, file={trace_path.name}")
    except Exception as e:
        print(f"    [!] Trace save failed: {e}")

    print(f"  {'─' * 50}")
    print(f"  Done in {duration:.1f}s  |  {score['tier']} ({score['score']}/100)  |  Research: {research_eval_score}%  |  Email: {email_eval_score}%")

    return PipelineResult(
        company=research.get("company_name", company),
        score=score["score"],
        tier=score["tier"],
        status=crm_status,
        lead_id=lead_id,
        email_subject=draft["email_subject"],
        word_count=draft["word_count"],
        linkedin_chars=draft["char_count"],
        duration_s=duration,
        search_mode=research.get("search_mode", "stub"),
        error="",
        research_eval=research_eval_score,
        email_eval=email_eval_score,
    )


# =============================================================================
# SECTION 5 — SUMMARY TABLE PRINTER
# =============================================================================

def print_summary_table(results: list[PipelineResult]) -> None:
    """
    Print a formatted summary table of all pipeline results to stdout.

    Layout:
        Company | Score | Tier | Status | Lead ID | Email (words) | LI (chars) | Time

    Design decisions:
    - Pure stdlib — no tabulate, rich, or other dependencies.
    - Fixed-width columns with truncation for long company names.
    - Tier column is right-padded so Hot/Warm/Cold align visually.
    - Error rows are marked with [ERR] in the Tier column.
    - Footer row shows totals: lead count by tier and total processing time.

    Args:
        results: List of PipelineResult NamedTuples from run_company_pipeline().
    """
    if not results:
        print("\n  [!] No results to display.\n")
        return

    # ── Column widths ─────────────────────────────────────────────────────────
    W_CO = 22    # Company
    W_SC = 6     # Score
    W_TR = 6     # Tier
    W_RE = 9     # Research%
    W_EM = 8     # Email%
    W_TM = 7     # Time

    def trunc(s: str, w: int) -> str:
        """Truncate string to width, adding ellipsis if needed."""
        s = str(s)
        return s if len(s) <= w else s[:w-2] + ".."

    # ── Header ────────────────────────────────────────────────────────────────
    header = (
        f"{'Company':<{W_CO}}  "
        f"{'Score':>{W_SC}}  "
        f"{'Tier':<{W_TR}}  "
        f"{'Research%':>{W_RE}}  "
        f"{'Email%':>{W_EM}}  "
        f"{'Time':>{W_TM}}"
    )
    sep = "─" * len(header)

    print(f"\n{'='*70}")
    print("  SALES PIPELINE RESULTS")
    print(f"{'='*70}")
    print(f"  {header}")
    print(f"  {sep}")

    # ── Data rows ─────────────────────────────────────────────────────────────
    hot_count = warm_count = cold_count = err_count = 0
    total_time = 0.0

    for r in results:
        total_time += r.duration_s

        # Tier display: append indicator for clarity
        if r.error:
            tier_display = "[ERR]"
            err_count += 1
        elif r.tier == "Hot":
            tier_display = "Hot"
            hot_count += 1
        elif r.tier == "Warm":
            tier_display = "Warm"
            warm_count += 1
        elif r.tier == "Cold":
            tier_display = "Cold"
            cold_count += 1
        else:
            tier_display = r.tier
            err_count += 1

        score_display = str(r.score) if not r.error else "—"
        research_eval_display = f"{r.research_eval}%" if not r.error else "—"
        email_eval_display = f"{r.email_eval}%" if not r.error else "—"
        mode_suffix = "*" if r.search_mode == "stub" else ""

        row = (
            f"  {trunc(r.company, W_CO):<{W_CO}}  "
            f"{score_display+mode_suffix:>{W_SC}}  "
            f"{tier_display:<{W_TR}}  "
            f"{research_eval_display:>{W_RE}}  "
            f"{email_eval_display:>{W_EM}}  "
            f"{r.duration_s:>{W_TM}.1f}s"
        )
        print(row)

        # Print error detail on the next line for failed rows
        if r.error:
            print(f"    {'└── ERROR: ' + r.error[:70]}")

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"  {sep}")

    tier_summary = f"Hot:{hot_count}  Warm:{warm_count}  Cold:{cold_count}"
    if err_count:
        tier_summary += f"  Errors:{err_count}"

    footer_left = f"  {len(results)} companies  |  {tier_summary}"
    footer_right = f"Total: {total_time:.1f}s"
    padding = len(header) - len(footer_left.strip()) - len(footer_right) + 2
    print(f"{footer_left}{' ' * max(1, padding)}{footer_right}")
    print(f"{'='*70}")

    # ── Legend ────────────────────────────────────────────────────────────────
    print("  * = stub search mode (set SEARCH_API_KEY for live data)")
    print("  Score: Company Size (0-25) + Industry (0-25) + Pain Pts (0-30) + DM (0-20)")
    print()


# =============================================================================
# SECTION 6 — LEGACY LEAD-ID PIPELINE (ADK SequentialAgent mode)
# =============================================================================
# This section preserves the original pipeline from the previous main.py.
# It processes a single CRM lead_id using the ADK SequentialAgent with full
# LLM calls in all three agents and CRM tool calls via McpToolset.
#
# WHEN TO USE: When you have an existing CRM lead_id and want the full LLM
# pipeline with richer research (CRM data + web search combined).
# =============================================================================

# ── ADK Pipeline (preserved from original main.py) ───────────────────────────
# The SequentialAgent wires the three ADK Agent objects together.
# State flows: lead_researcher → [research_brief] → lead_scorer →
#              [lead_score_summary] → outreach_drafter → [outreach_draft]
# ─────────────────────────────────────────────────────────────────────────────
# Use Workflow (preferred) with SequentialAgent fallback for older ADK installs.
_adk_pipeline = _Pipeline(
    name="sales_pipeline",
    description=(
        "End-to-end sales pipeline: researches a CRM lead, scores them, "
        "and drafts personalised outreach."
    ),
    sub_agents=[
        lead_researcher,    # Stage 1: CRM fetch + web research → research_brief
        lead_scorer,        # Stage 2: Score brief → lead_score_summary
        outreach_drafter,   # Stage 3: Draft email + LinkedIn → outreach_draft
    ],
)

_APP_NAME = "sales_pipeline_agent"
_session_service = InMemorySessionService()
_runner = Runner(
    agent=_adk_pipeline,
    app_name=_APP_NAME,
    session_service=_session_service,
)


async def _run_lead_id_pipeline(lead_id: str, user_id: str = "sales_user") -> dict:
    """
    Execute the ADK SequentialAgent pipeline for a given CRM lead_id.

    DESIGN: This async function is the same as the original main.py's
    run_pipeline(). We keep it here to preserve backward-compatibility with
    any scripts or integrations that import and call it.

    Args:
        lead_id: CRM lead ID (e.g. "lead_001").
        user_id: Session owner ID (default "sales_user").

    Returns:
        dict with research_brief, lead_score_summary, outreach_draft.
    """
    print(f"\n{'='*60}")
    print(f"  ADK Pipeline — CRM Lead: {lead_id}")
    print(f"{'='*60}\n")

    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash-latest"]
    success = False
    last_error = None

    for model_name in models:
        # Set the model on the sub-agents before running
        lead_researcher.model = model_name
        lead_scorer.model = model_name
        outreach_drafter.model = model_name

        print(f"🤖 Using model: {model_name}")
        log_action("Orchestrator", "MODEL_SELECTION", f"model={model_name}")

        # Seed the session with lead_id so all agents' {lead_id} placeholders resolve
        session = await _session_service.create_session(
            app_name=_APP_NAME,
            user_id=user_id,
            state={"lead_id": lead_id},
        )

        initial_message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(
                text=f"Process lead {lead_id} through the full sales pipeline."
            )],
        )

        print(f"[>>] Stage 1 — Lead Research starting (using {model_name})...\n")

        try:
            # Stream ADK events — we print each agent's final response as it arrives.
            # ADK's SequentialAgent yields events for each sub-agent in order.
            # is_final_response() is True only for the terminal event of each agent.
            async for event in _runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=initial_message,
            ):
                if event.is_final_response():
                    agent_name = getattr(event, "author", "pipeline")
                    content = event.content
                    if content and content.parts:
                        for part in content.parts:
                            if hasattr(part, "text") and part.text:
                                text = part.text
                                preview = text[:500] + "..." if len(text) > 500 else text
                                print(f"\n[{agent_name.upper()}] → {preview}\n")
                                # SECURITY NOTE: Accidental credential leakage is a major vulnerability.
                                # We scan the generated output from each agent stage to warn the user if
                                # any key-like strings were outputted.
                                detected = scan_for_secrets(text)
                                if detected:
                                    print(f"\n    [WARNING] SECURITY: Potential secret(s) detected in {agent_name} output: {', '.join(detected)}")
                                    log_action(agent_name.capitalize(), "SECURITY_ALERT_SECRETS_DETECTED", f"findings={detected}")
            success = True
            break
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            is_transient = any(k in err_str for k in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if is_transient:
                print(f"  [Orchestrator] Pipeline run failed with model {model_name} ({err_str[:80]}). Trying next model in fallback chain...")
            else:
                print(f"  [Orchestrator] Non-transient error in pipeline run with model {model_name}: {err_str[:80]}. Trying next model...")
    else:
        if last_error:
            raise last_error

    # Retrieve the final session state to extract all output keys
    completed = await _session_service.get_session(
        app_name=_APP_NAME,
        user_id=user_id,
        session_id=session.id,
    )
    state = completed.state if completed else {}

    print(f"\n{'='*60}")
    print("  ADK Pipeline Complete!")
    print(f"{'='*60}")
    print(f"  Lead ID       : {lead_id}")
    print(f"  Research Brief: {'[OK]' if state.get('research_brief') else '[!!] Missing'}")
    print(f"  Score Summary : {'[OK]' if state.get('lead_score_summary') else '[!!] Missing'}")
    print(f"  Outreach Draft: {'[OK] Saved to CRM' if state.get('outreach_draft') else '[!!] Missing'}")
    print(f"{'='*60}\n")

    return {
        "lead_id": lead_id,
        "research_brief": state.get("research_brief"),
        "lead_score_summary": state.get("lead_score_summary"),
        "outreach_draft": state.get("outreach_draft"),
    }


# =============================================================================
# SECTION 7 — CLI ARGUMENT PARSER
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.

    We separate parser construction from parsing (rather than building inline)
    so it can be imported and tested by unit tests without triggering sys.argv
    side effects.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Sales Pipeline Agent — Research, score, and draft outreach for B2B leads.\n"
            "Powered by Google ADK + FastMCP."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --companies "Stripe"
  python main.py --companies "Notion,HubSpot,Figma"
  python main.py --companies "Stripe" --no-fast        # LLM drafts (slower, richer)
  python main.py --companies "Stripe" --no-crm         # Skip CRM persistence
  python main.py --lead-id lead_001                    # Legacy CRM lead mode
  python main.py --agent-skill                         # Show agent skill card
        """,
    )

    # ── Company-name mode flags ───────────────────────────────────────────────
    parser.add_argument(
        "--companies",
        type=str,
        default="",
        help=(
            "Comma-separated list of company names to process. "
            "E.g. --companies \"Stripe,Notion,HubSpot\""
        ),
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        default=False,
        help=(
            "Use LLM-based outreach drafter instead of template mode. "
            "Slower (~15s/company) but produces richer, more personalised drafts."
        ),
    )
    parser.add_argument(
        "--no-crm",
        action="store_true",
        default=False,
        help=(
            "Skip CRM persistence. Run pipeline locally without saving results. "
            "Useful when mcp_server/crm_server.py is not running."
        ),
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=False,
        help="Skip Human-in-the-Loop review and automatically approve generated outreach drafts.",
    )

    # ── Legacy lead-id mode flags ─────────────────────────────────────────────
    parser.add_argument(
        "--lead-id",
        type=str,
        default="",
        help=(
            "LEGACY MODE: CRM lead ID to process via full ADK SequentialAgent. "
            "E.g. --lead-id lead_001. Requires CRM MCP server running."
        ),
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="sales_user",
        help="ADK session user ID (legacy mode only). Default: sales_user.",
    )

    # ── Meta flags ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--agent-skill",
        action="store_true",
        default=False,
        help=(
            "Print the Agent Skill Card describing what this pipeline does, "
            "which agents it uses, and what tools each agent has access to."
        ),
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help="Print the production monitoring dashboard and exit.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Print the lead priority dashboard and exit.",
    )

    return parser


# =============================================================================
# SECTION 8 — MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """
    Main entry point — parse CLI args and dispatch to the appropriate mode.

    Returns:
        Exit code: 0 = success, 1 = partial failure, 2 = complete failure.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # ── --monitor: print the production monitoring dashboard and exit ─────────
    if args.monitor:
        try:
            from tools.monitor import compute_metrics, save_metrics, print_dashboard
            metrics = compute_metrics()
            save_metrics(metrics)
            print_dashboard(metrics)
        except Exception as e:
            sys.stderr.write(f"Error running monitor dashboard: {e}\n")
            return 1
        return 0

    # ── --dashboard: print the lead priority dashboard and exit ───────────────
    if args.dashboard:
        try:
            from tools.priority_dashboard import show_priority_dashboard
            show_priority_dashboard()
        except Exception as e:
            sys.stderr.write(f"Error running priority dashboard: {e}\n")
            return 1
        return 0

    # ── --agent-skill: print the skill card and exit ──────────────────────────
    # This is the Agent Skills demonstration feature for the competition.
    # It shows what the pipeline can do without running any agents.
    if args.agent_skill:
        print(AGENT_SKILL_CARD)
        return 0

    # ── --lead-id: legacy ADK SequentialAgent mode ───────────────────────────
    if args.lead_id:
        log_action("Orchestrator", "LEGACY_PIPELINE_START", f"lead_id={args.lead_id}")
        result = asyncio.run(
            _run_lead_id_pipeline(lead_id=args.lead_id, user_id=args.user_id)
        )
        if not all([
            result["research_brief"],
            result["lead_score_summary"],
            result["outreach_draft"],
        ]):
            print("[!!] Pipeline completed with missing outputs.")
            log_action("Orchestrator", "LEGACY_PIPELINE_FAILED", f"lead_id={args.lead_id}")
            return 1
        print("[OK] Pipeline completed successfully.")
        log_action("Orchestrator", "LEGACY_PIPELINE_SUCCESS", f"lead_id={args.lead_id}")
        return 0

    # ── --companies: company-name pipeline mode ───────────────────────────────
    if not args.companies.strip():
        parser.print_help()
        print("\n[!] Please provide --companies, --lead-id, or --agent-skill.\n")
        return 2

    # Parse company list — trim whitespace, skip empty entries
    companies = [c.strip() for c in args.companies.split(",") if c.strip()]
    if not companies:
        print("[!] No valid company names found in --companies argument.")
        return 2

    # ── RATE LIMITING ──────────────────────────────────────────────────────────
    # SECURITY NOTE: Enforcing a strict limit on the number of items processed per
    # run prevents resource abuse, protects external API quotas, and guards
    # against "denial of wallet" (excessive token charges).
    MAX_COMPANIES_LIMIT = 10
    if len(companies) > MAX_COMPANIES_LIMIT:
        print(f"\n[!] SECURITY ERROR: Rate limit exceeded. Maximum of {MAX_COMPANIES_LIMIT} companies are allowed per run.", file=sys.stderr)
        print(f"    You requested to process {len(companies)} companies. Please reduce the batch size.", file=sys.stderr)
        log_action("Orchestrator", "RATE_LIMIT_EXCEEDED", f"requested={len(companies)}")
        return 2

    # ── INPUT SANITIZATION & VALIDATION ─────────────────────────────────────────
    # SECURITY NOTE: Validating inputs helps mitigate injection vulnerabilities
    # (prompt injection, script injection) and avoids buffer overflow/DoS in downstream tools.
    sanitized_companies: list[str] = []
    for co in companies:
        try:
            sanitized = sanitize_company_name(co)
            sanitized_companies.append(sanitized)
        except ValueError as val_err:
            print(f"\n[!] SECURITY ERROR: Input validation failed for '{co}': {val_err}", file=sys.stderr)
            log_action("Orchestrator", "INPUT_VALIDATION_FAILED", f"input={co[:20]}, error={val_err}")
            return 2
    companies = sanitized_companies

    fast_mode = not args.no_fast      # Default: fast (template drafter)
    use_crm = not args.no_crm         # Default: use CRM

    # ── Pipeline banner ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SALES PIPELINE AGENT")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Companies : {len(companies)}")
    print(f"  Draft mode: {'Template (fast)' if fast_mode else 'LLM (richer)'}")
    print(f"  CRM       : {'Enabled' if use_crm else 'Disabled (--no-crm)'}")
    print(f"  Search    : {'Live' if os.getenv('SEARCH_API_KEY') else 'Stub (set SEARCH_API_KEY for live)'}")
    print(f"{'='*70}")

    log_action("Orchestrator", "PIPELINE_START", f"companies={','.join(companies)}")

    # ── Run pipeline for each company ─────────────────────────────────────────
    results: list[PipelineResult] = []
    total_start = time.time()

    for i, company in enumerate(companies, 1):
        print(f"\n[{i}/{len(companies)}] Starting pipeline for: {company}")
        log_action("Orchestrator", "COMPANY_START", f"company={company}")
        result = run_company_pipeline(
            company=company,
            fast=fast_mode,
            use_crm=use_crm,
            auto_approve=args.auto_approve or os.getenv("AUTO_APPROVE", "false").lower() == "true",
        )
        results.append(result)
        status_label = "FAILED" if result.error else "SUCCESS"
        log_action("Orchestrator", "COMPANY_END", f"company={company}, status={status_label}, score={result.score}")

    total_elapsed = time.time() - total_start

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary_table(results)
    print(f"  Total wall-clock time: {total_elapsed:.1f}s for {len(companies)} company/companies\n")

    # ── Optionally print full outreach drafts for review ─────────────────────
    # Only print drafts if there's exactly one company (avoids flooding the
    # terminal when processing many leads in batch mode).
    if len(results) == 1 and not results[0].error:
        _print_draft_preview(results[0], companies[0])

    # ── Exit code: 0 if all succeeded, 1 if any partial failures ─────────────
    errors = [r for r in results if r.error]
    exit_code = 0
    if len(errors) == len(results):
        exit_code = 2    # All failed
    elif errors:
        exit_code = 1    # Some failed

    # ── Write metrics to data/metrics.json after every run ─────────────────────
    try:
        from tools.monitor import compute_metrics, save_metrics
        metrics = compute_metrics()
        save_metrics(metrics)
        print("📊 Production metrics updated successfully in data/metrics.json")
    except Exception as e:
        print(f"    [!] Failed to update production metrics: {e}")

    log_action("Orchestrator", "PIPELINE_END", f"processed={len(results)}, exit_code={exit_code}")
    return exit_code


def _print_draft_preview(result: PipelineResult, company: str) -> None:
    """
    Print the outreach drafts for a single-company run for easy copy-paste.

    Only shown in single-company mode to avoid overwhelming batch output.
    The drafts are stored in the CRM — this is just a terminal preview.

    Args:
        result: PipelineResult for the company.
        company: Original company name input.
    """
    # Fetch the latest draft from CRM to show the actual saved version
    try:
        from mcp_server.crm_server import get_all_leads
        all_leads = get_all_leads()
        lead = next(
            (l for l in all_leads.get("leads", [])
             if l.get("company", "").lower() == result.company.lower()),
            None,
        )
        if not lead:
            return

        # Get full lead record for the drafts
        from mcp_server.crm_server import get_lead
        full = get_lead(result.lead_id)
        if full.get("status") != "success":
            return

        lead_data = full["lead"]
        email_draft = lead_data.get("email_draft", "")
        linkedin_draft = lead_data.get("linkedin_draft", "")

        if email_draft or linkedin_draft:
            print(f"\n{'='*70}")
            print(f"  OUTREACH DRAFTS — {result.company}")
            print(f"{'='*70}\n")

            if email_draft:
                print("  [EMAIL DRAFT]")
                print("  " + "-" * 60)
                for line in email_draft.split("\n"):
                    print(f"  {line}")
                print("  " + "-" * 60)

            if linkedin_draft:
                print(f"\n  [LINKEDIN CONNECTION NOTE]")
                print("  " + "-" * 60)
                print(f"  {linkedin_draft}")
                print(f"  ({len(linkedin_draft)}/300 characters)")
                print("  " + "-" * 60)
            print()

    except Exception:
        pass   # Draft preview is optional — silently skip on any error


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    sys.exit(main())
