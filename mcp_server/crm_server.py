"""
mcp_server/crm_server.py
========================

PURPOSE:
    A FastMCP-based Model Context Protocol (MCP) server that acts as the
    single source of truth for all CRM data in the sales pipeline.

    ADK agents connect to this server over HTTP/SSE and call its tools
    exactly as they would call any other Python function — but the calls
    cross process boundaries, enabling the server to be deployed separately
    from the agents in production (e.g. as a Cloud Run service).

MCP SERVER DESIGN:
    ─────────────────────────────────────────────────────────────────────────
    WHY MCP INSTEAD OF DIRECT FILE ACCESS?
    ─────────────────────────────────────────────────────────────────────────
    The simplest alternative would be to have each agent read/write crm.json
    directly. We chose MCP instead for three reasons:

    1. SEPARATION OF CONCERNS
       Agents describe WHAT to do ("score this lead"). The MCP server handles
       HOW data is stored. Swapping the JSON backend for PostgreSQL later
       requires zero changes to any agent — only this file changes.

    2. SINGLE WRITER
       Multiple agents run concurrently in a SequentialAgent. Direct file
       access would require each agent to implement its own file locking.
       The MCP server serialises writes through a single process, eliminating
       race conditions without any locking code in the agents.

    3. TOOL DISCOVERABILITY
       ADK's McpToolset auto-discovers the tool list from the server's MCP
       manifest. Adding a new tool here makes it immediately available to any
       agent that specifies it in tool_filter — no agent code changes needed.

    ─────────────────────────────────────────────────────────────────────────
    TOOL DESIGN PHILOSOPHY
    ─────────────────────────────────────────────────────────────────────────
    Each tool follows the same contract:
    - Returns a dict with a top-level "status" key: "success" | "error".
    - On error, includes an "error" key with a human-readable message.
    - On success, includes the relevant data key(s).
    - NEVER raises an exception to the caller — errors are returned as
      structured dicts so ADK agents can read and react to them gracefully.

    This "result object" pattern is preferable to raw exceptions for MCP
    tools because:
    - The LLM can read the error message and decide what to do next.
    - No exception propagation through the MCP protocol layer.
    - Consistent shape makes agent error-handling code simpler.

    ─────────────────────────────────────────────────────────────────────────
    DATA SCHEMA (crm.json)
    ─────────────────────────────────────────────────────────────────────────
    {
      "leads": [
        {
          "id":              str  — Unique ID, auto-generated as "lead_NNN"
          "company":         str  — Company name (primary identifier for new tools)
          "name":            str  — Contact person's full name
          "title":           str  — Contact's job title
          "email":           str  — Contact email address
          "phone":           str  — Contact phone number
          "industry":        str  — Industry/sector string
          "company_size":    str  — Headcount range or description
          "annual_revenue":  str  — Estimated revenue bracket
          "source":          str  — Lead acquisition source
          "status":          str  — Pipeline stage (see VALID_STATUSES)
          "score":           int|null  — 0-100 priority score (null = unscored)
          "tier":            str|null  — "Hot"|"Warm"|"Cold" (null = unscored)
          "score_justification": str  — Reason for score (set by lead_scorer)
          "scored_at":       str  — ISO timestamp of last scoring
          "email_draft":     str  — Cold email draft (subject + body)
          "linkedin_draft":  str  — LinkedIn connection note
          "outreach_drafts": list — Historical drafts [{draft_id, subject, body, ...}]
          "notes":           str  — Free-text notes
          "created_at":      str  — ISO timestamp of lead creation
          "status_updated_at": str — ISO timestamp of last status change
        }
      ],
      "contacts": [],   — Reserved for future contact-level records
      "deals":    []    — Reserved for future deal tracking
    }

    ─────────────────────────────────────────────────────────────────────────

TOOLS EXPOSED:
    NEW (added in this version):
    ─ add_lead(company, score, tier, email_draft, linkedin_draft)
                        → Creates a new lead record. Auto-generates ID.
    ─ get_all_leads()   → Returns all leads with key pipeline fields.
    ─ get_leads_by_tier(tier) → Filters leads by Hot/Warm/Cold tier.
    ─ update_lead_status(company, status)
                        → Moves a lead by company name through pipeline stages.

    EXISTING (preserved for backward-compatibility with ADK pipeline agents):
    ─ list_leads()      → Returns all leads (legacy, use get_all_leads).
    ─ get_lead(lead_id) → Returns one lead by ID.
    ─ update_lead_score(lead_id, score, justification) → Persists a score.
    ─ add_outreach_draft(lead_id, subject, body) → Appends an email draft.

HOW TO RUN:
    # In a separate terminal (keep running while agents execute):
    python mcp_server/crm_server.py
    # Listens on http://0.0.0.0:8001 (SSE transport).

HOW AGENTS CONNECT:
    from google.adk.tools.mcp_tool import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

    McpToolset(
        connection_params=SseConnectionParams(url="http://localhost:8001/sse"),
        tool_filter=["add_lead", "get_all_leads", "get_leads_by_tier"],
    )

FUTURE ENHANCEMENTS:
    - Replace JSON file backend with SQLite (zero-config) then PostgreSQL.
    - Add API key authentication (X-API-Key header validation middleware).
    - Emit webhook events on high-value lead creation (score > 80) to trigger
      Slack notifications or Zapier automations.
    - Add pagination to get_all_leads() for large datasets.
    - Containerise with Docker; deploy to Cloud Run with a mounted volume
      or Cloud SQL backend for production durability.
    - Add a leads_summary() tool for dashboard/analytics consumption.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP


# =============================================================================
# SECTION 1 — BOOTSTRAP
# =============================================================================

# Resolve the data file path relative to THIS file so the server works
# regardless of which directory it is launched from.
# e.g. `python mcp_server/crm_server.py`  or  `python -m mcp_server.crm_server`
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "crm.json"

VALID_STATUSES = {
    "new",          # Just entered the pipeline — not yet contacted
    "contacted",    # First outreach sent
    "replied",      # Contact replied
    "meeting_booked",# Meeting booked with contact
    "won",          # Won the deal
    "lost",         # Lost the deal
    "nurture",      # Nurturing lead for future potential
    "archived",     # Archived lead
    # Legacy statuses for backward compatibility
    "qualified",    
    "proposal",     
    "closed_won",   
    "closed_lost",  
    "disqualified", 
    "approved",     
    "skipped",      
}

# Tier values that the lead_scorer agent assigns.
VALID_TIERS = {"Hot", "Warm", "Cold"}

# FastMCP server instance.
# The `instructions` string is sent to the LLM as part of the MCP context
# window when an agent first connects. It helps the LLM understand WHEN to
# call which tool, reducing unnecessary or incorrect tool calls.
mcp = FastMCP(
    name="crm-server",
    instructions=(
        "You are the CRM data service for a B2B sales pipeline. "
        "Use add_lead to create leads after research and scoring. "
        "Use get_all_leads or get_leads_by_tier to browse the pipeline. "
        "Use update_lead_status to advance a lead through pipeline stages. "
        "Use get_lead(lead_id) to fetch a specific lead's full record. "
        "Always check the returned 'status' field — 'error' means the "
        "operation failed and the 'error' field explains why."
    ),
)


# Health check endpoint for Cloud Run/monitoring
@mcp.custom_route("/health", methods=["GET"])
def health_check(request):
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok"})


# =============================================================================
# SECTION 2 — DATA LAYER HELPERS
# =============================================================================
# All disk I/O is isolated in these two functions. The rest of the code uses
# only _load_crm() and _save_crm(), making it easy to swap the storage backend
# (e.g. replace with SQLite calls) without touching any tool logic.
# =============================================================================

def _ensure_data_file() -> None:
    """
    Ensure the CRM data file and its parent directory exist.

    Called at the start of every _load_crm() so the server is self-healing:
    if someone deletes crm.json or the data/ directory, the next tool call
    regenerates a valid empty structure rather than crashing.

    This is the PRIMARY error-handling strategy for the "missing file" case.
    We choose auto-recreation over a hard error because:
    - In development, the file is frequently deleted during resets.
    - An empty-but-valid CRM is more useful than a crashed server.
    - Each tool still returns its own error if a specific record is missing.
    """
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        # Write a valid empty CRM structure so subsequent reads never fail
        # with a JSON decode error on an empty file.
        empty_crm = {
            "leads": [],
            "contacts": [],   # Reserved for future contact-level records
            "deals": [],      # Reserved for future deal tracking
            "_meta": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "description": "Sales pipeline CRM — auto-created by crm_server.py",
            },
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(empty_crm, f, indent=2)


def _load_crm() -> dict:
    """
    Load and return the entire CRM data structure from disk.

    Error handling strategy:
    ─ FileNotFoundError: Auto-creates the file via _ensure_data_file().
    ─ json.JSONDecodeError: The file is corrupt (e.g. interrupted write).
      We back it up with a timestamp suffix and start fresh. This prevents
      a permanently broken server due to a single bad write.
    ─ PermissionError / OSError: Re-raised as these require manual intervention
      (disk full, wrong file permissions) that we cannot self-heal.

    Returns:
        dict: Full CRM data with at least {"leads": [], "contacts": [], "deals": []}
    """
    _ensure_data_file()

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure required top-level keys exist (schema forward-compatibility)
            data.setdefault("leads", [])
            data.setdefault("contacts", [])
            data.setdefault("deals", [])
            return data

    except json.JSONDecodeError as exc:
        # Corrupt file — back it up and start fresh.
        # The timestamp suffix makes it easy to recover data manually.
        backup_path = DATA_FILE.with_suffix(
            f".corrupt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        DATA_FILE.rename(backup_path)
        print(
            f"[crm_server] WARNING: crm.json was corrupt ({exc}). "
            f"Backed up to {backup_path.name} and reset to empty."
        )
        # Recreate a clean file and return empty structure
        _ensure_data_file()
        return {"leads": [], "contacts": [], "deals": []}


def _save_crm(data: dict) -> None:
    """
    Persist the CRM data structure to disk atomically.

    ATOMIC WRITE STRATEGY:
    We write to a temporary file (.tmp) first, then rename it over the real
    file. This guarantees that readers never see a partial write (which
    would produce corrupt JSON if the process crashes mid-write).

    On POSIX systems, rename() is atomic. On Windows, it is atomic for
    files on the same drive, which is always the case here.

    Args:
        data: Full CRM dict to persist.

    Raises:
        OSError: If the disk is full or the directory is not writable.
    """
    _ensure_data_file()

    tmp_path = DATA_FILE.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Atomic rename: replace the real file with the tmp file
        tmp_path.replace(DATA_FILE)
    except Exception:
        # Clean up the tmp file if anything went wrong
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _generate_lead_id(existing_leads: list) -> str:
    """
    Generate the next sequential lead ID (e.g. "lead_004").

    We use a sequential numeric suffix rather than a UUID because:
    - Sequential IDs are human-readable in logs and the CRM UI.
    - They make it obvious when a lead was added relative to others.
    - UUIDs are more appropriate when multiple writers could conflict;
      since we have a single-writer MCP server, that risk is eliminated.

    The ID is derived from the current count of ALL leads (including
    historical / deleted ones if we ever implement soft-delete). This
    avoids ID reuse if leads are ever removed from the list.

    Args:
        existing_leads: Current list of lead dicts in the CRM.

    Returns:
        str: New unique lead ID in the format "lead_NNN" (zero-padded to 3).
    """
    # Find the highest existing numeric suffix to avoid collisions
    max_num = 0
    for lead in existing_leads:
        lead_id = lead.get("id", "")
        if lead_id.startswith("lead_"):
            try:
                num = int(lead_id.split("_", 1)[1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass
    return f"lead_{max_num + 1:03d}"


def _find_lead_by_company(leads: list, company: str) -> dict | None:
    """
    Find the first lead whose company name matches (case-insensitive).

    Case-insensitive matching prevents failures when the researcher returns
    "HubSpot" but the CRM stores "hubspot" or "HUBSPOT". We use the first
    match to handle duplicate company names gracefully (last-writer-wins
    for updates; first-found for reads).

    Args:
        leads: List of lead dicts.
        company: Company name to search for.

    Returns:
        The matching lead dict, or None if not found.
    """
    company_lower = company.strip().lower()
    for lead in leads:
        if lead.get("company", "").strip().lower() == company_lower:
            return lead
    return None


# =============================================================================
# SECTION 3 — NEW TOOLS (added per user request)
# =============================================================================

@mcp.tool()
def add_lead(
    company: str,
    score: int,
    tier: str,
    email_draft: str,
    linkedin_draft: str,
    name: str = "",
    title: str = "",
    original_email_draft: str = "",
) -> dict:
    """Add a new lead to the CRM with research score and outreach drafts.

    Called by the OutreachDrafter agent (or external scripts) after the full
    research → scoring → drafting pipeline completes for a company.

    If a lead for this company already exists, the record is UPDATED with the
    new score, tier, and drafts rather than creating a duplicate. This makes
    the tool idempotent — re-running the pipeline for the same company
    refreshes the CRM record without creating noise.

    Args:
        company:              Company name (e.g. "Notion", "HubSpot").
        score:                Lead priority score, integer 0-100.
        tier:                 Qualification tier: "Hot", "Warm", or "Cold".
        email_draft:          Full cold email draft (subject + body, plain text).
        linkedin_draft:       LinkedIn connection note (max 300 characters).
        name:                 Contact person's full name (optional).
        title:                Contact's job title (optional).
        original_email_draft: Original email draft before reflection/review (optional).

    Returns:
        dict: {"status": "success", "lead_id": str, "action": "created"|"updated"}
              or {"status": "error", "error": str} on failure.
    """
    # ── Input validation ───────────────────────────────────────────────────────
    # We validate BEFORE loading the file to avoid unnecessary I/O on bad input.
    if not company or not company.strip():
        return {"status": "error", "error": "company cannot be empty."}

    if not isinstance(score, int) or not 0 <= score <= 100:
        return {
            "status": "error",
            "error": f"score must be an integer between 0 and 100, got: {score!r}",
        }

    if tier not in VALID_TIERS:
        return {
            "status": "error",
            "error": f"tier must be one of {sorted(VALID_TIERS)}, got: {tier!r}",
        }

    # LinkedIn connection notes have a hard 300-character limit enforced by
    # the LinkedIn platform. We truncate rather than reject so the pipeline
    # always succeeds — the sales rep will see the truncation and can edit.
    if len(linkedin_draft) > 300:
        linkedin_draft = linkedin_draft[:297] + "..."

    now = datetime.now(timezone.utc).isoformat()
    crm = _load_crm()
    leads = crm["leads"]

    # ── Upsert logic: update existing record or create new one ─────────────────
    # DESIGN DECISION: Upsert (update-or-insert) prevents duplicate company
    # records when the pipeline is re-run for the same company. A pure insert
    # would clutter the CRM with near-identical records after each re-run.
    existing = _find_lead_by_company(leads, company)

    if existing:
        # UPDATE path — refresh score, tier, and drafts; preserve other fields
        existing["score"] = score
        existing["tier"] = tier
        existing["score_justification"] = f"Scored by pipeline: {tier} ({score}/100)"
        existing["scored_at"] = now
        existing["email_draft"] = email_draft
        existing["original_email_draft"] = original_email_draft
        existing["linkedin_draft"] = linkedin_draft
        existing["status_updated_at"] = now
        if name:
            existing["name"] = name.strip()
        if title:
            existing["title"] = title.strip()
        # Preserve existing status — we don't reset pipeline progress on re-score
        _save_crm(crm)
        return {
            "status": "success",
            "lead_id": existing["id"],
            "action": "updated",
            "company": company,
            "score": score,
            "tier": tier,
        }

    # CREATE path — build a complete new lead record
    new_id = _generate_lead_id(leads)
    new_lead = {
        "id": new_id,
        "company": company.strip(),
        # Contact fields default to empty — filled later by get_lead / manual entry
        "name": name.strip() if name else "",
        "title": title.strip() if title else "",
        "email": "",
        "phone": "",
        # Research fields populated from researcher output (passed via pipeline)
        "industry": "",
        "company_size": "",
        "annual_revenue": "",
        "source": "pipeline",       # Indicates this lead was created by the AI pipeline
        # Scoring fields
        "status": "new",            # All new pipeline leads start at "new"
        "score": score,
        "tier": tier,
        "score_justification": f"Scored by pipeline: {tier} ({score}/100)",
        "scored_at": now,
        # Outreach drafts
        "email_draft": email_draft,
        "original_email_draft": original_email_draft,
        "linkedin_draft": linkedin_draft,
        "outreach_drafts": [],      # Historical drafts appended by add_outreach_draft
        # Metadata
        "notes": "",
        "created_at": now,
        "status_updated_at": now,
    }

    leads.append(new_lead)
    _save_crm(crm)

    return {
        "status": "success",
        "lead_id": new_id,
        "action": "created",
        "company": company,
        "score": score,
        "tier": tier,
    }


@mcp.tool()
def get_all_leads() -> dict:
    """Return all leads in the CRM with their key pipeline fields.

    Returns a summary view (not full records) to keep the response compact.
    Each item includes the fields most relevant to pipeline management:
    company, tier, score, status, and the existence of outreach drafts.

    For full lead details (all fields including drafts), use get_lead(lead_id).

    Returns:
        dict: {
            "status": "success",
            "leads": [summary_dict, ...],
            "total": int,
            "by_tier": {"Hot": int, "Warm": int, "Cold": int, "unscored": int}
        }
    """
    crm = _load_crm()
    leads = crm.get("leads", [])

    # Build a lightweight summary for each lead.
    # We deliberately exclude email_draft and linkedin_draft from this
    # summary view — they can be multi-hundred-word strings. Including them
    # would make get_all_leads() very token-expensive for LLM callers.
    summaries = []
    tier_counts = {"Hot": 0, "Warm": 0, "Cold": 0, "unscored": 0}

    for lead in leads:
        tier = lead.get("tier")
        if tier in tier_counts:
            tier_counts[tier] += 1
        else:
            tier_counts["unscored"] += 1

        summaries.append({
            "id": lead.get("id"),
            "company": lead.get("company"),
            "name": lead.get("name") or "(contact unknown)",
            "title": lead.get("title") or "",
            "industry": lead.get("industry") or "",
            "status": lead.get("status"),
            "score": lead.get("score"),          # null = not yet scored
            "tier": tier,                         # null = not yet scored
            "has_email_draft": bool(lead.get("email_draft")),
            "has_linkedin_draft": bool(lead.get("linkedin_draft")),
            "created_at": lead.get("created_at"),
        })

    return {
        "status": "success",
        "leads": summaries,
        "total": len(summaries),
        "by_tier": tier_counts,
    }


@mcp.tool()
def get_leads_by_tier(tier: str) -> dict:
    """Return all leads matching a specific qualification tier.

    Use this to pull Hot leads for immediate outreach, Warm leads for a
    nurture campaign, or Cold leads for deprioritisation review.

    Args:
        tier: One of "Hot", "Warm", or "Cold". Case-sensitive.

    Returns:
        dict: {
            "status": "success",
            "tier": str,
            "leads": [summary_dict, ...],
            "total": int
        }
        or {"status": "error", "error": str} for an invalid tier.
    """
    # ── Validate tier input ─────────────────────────────────────────────────
    # We validate the tier BEFORE file I/O. This means invalid tier requests
    # fail fast without touching disk — important for performance at scale.
    if tier not in VALID_TIERS:
        return {
            "status": "error",
            "error": (
                f"Invalid tier '{tier}'. Must be one of: "
                f"{sorted(VALID_TIERS)}. Tier is case-sensitive."
            ),
        }

    crm = _load_crm()
    matching = [
        {
            "id": lead.get("id"),
            "company": lead.get("company"),
            "name": lead.get("name") or "(contact unknown)",
            "title": lead.get("title") or "",
            "status": lead.get("status"),
            "score": lead.get("score"),
            "tier": lead.get("tier"),
            # Include draft presence flags so callers know if outreach is ready
            "has_email_draft": bool(lead.get("email_draft")),
            "has_linkedin_draft": bool(lead.get("linkedin_draft")),
            # Include the email draft itself — useful for Hot leads going straight
            # to outreach without a separate get_lead() call
            "email_draft": lead.get("email_draft", "") if tier == "Hot" else None,
        }
        for lead in crm.get("leads", [])
        if lead.get("tier") == tier
    ]

    # Sort Hot leads by score descending so the highest-priority leads come first
    if tier == "Hot":
        matching.sort(key=lambda x: x.get("score") or 0, reverse=True)

    return {
        "status": "success",
        "tier": tier,
        "leads": matching,
        "total": len(matching),
    }


@mcp.tool()
def update_lead_status(company: str, status: str) -> dict:
    """Advance or update a lead's pipeline status, identified by company name.

    This is the company-name-based version of the existing update_lead_status
    tool (which uses lead_id). It is more convenient for pipeline agents that
    work with company names from research output and may not know the lead_id.

    Valid pipeline stages (in rough progression order):
        new → contacted → qualified → proposal → closed_won
        any stage → closed_lost | disqualified

    Args:
        company: Company name to look up (case-insensitive).
        status:  Target pipeline stage (must be one of the valid statuses).

    Returns:
        dict: {
            "status": "success",
            "lead_id": str,
            "company": str,
            "old_status": str,
            "new_status": str,
            "updated_at": str   (ISO timestamp)
        }
        or {"status": "error", "error": str} on failure.
    """
    # ── Validate status ────────────────────────────────────────────────────────
    if status not in VALID_STATUSES:
        return {
            "status": "error",
            "error": (
                f"Invalid status '{status}'. "
                f"Must be one of: {sorted(VALID_STATUSES)}."
            ),
        }

    if not company or not company.strip():
        return {"status": "error", "error": "company cannot be empty."}

    # ── Load, find, update, save ───────────────────────────────────────────────
    crm = _load_crm()
    lead = _find_lead_by_company(crm["leads"], company)

    if lead is None:
        # Return a helpful error that tells the caller how to proceed.
        # If the company doesn't exist yet, suggest using add_lead() first.
        return {
            "status": "error",
            "error": (
                f"No lead found for company '{company}'. "
                f"If this is a new company, use add_lead() to create it first. "
                f"Company matching is case-insensitive."
            ),
        }

    old_status = lead.get("status", "unknown")
    now = datetime.now(timezone.utc).isoformat()

    lead["status"] = status
    lead["status_updated_at"] = now
    
    # Append status change to timeline
    if "timeline" not in lead:
        lead["timeline"] = []
    lead["timeline"].append({
        "timestamp": now,
        "type": "status_change",
        "old_status": old_status,
        "new_status": status,
        "description": f"Status updated: {old_status} → {status}"
    })
    
    _save_crm(crm)

    return {
        "status": "success",
        "lead_id": lead["id"],
        "company": lead["company"],
        "old_status": old_status,
        "new_status": status,
        "updated_at": now,
    }


@mcp.tool()
def add_lead_note(company: str, note: str) -> dict:
    """Append a timestamped note to the lead's notes array."""
    if not company or not company.strip():
        return {"status": "error", "error": "company cannot be empty."}
    if not note or not note.strip():
        return {"status": "error", "error": "note cannot be empty."}

    crm = _load_crm()
    lead = _find_lead_by_company(crm["leads"], company)
    if lead is None:
        return {
            "status": "error",
            "error": f"No lead found for company '{company}'."
        }

    now = datetime.now(timezone.utc).isoformat()

    # Ensure notes is a list
    if "notes" not in lead or not isinstance(lead["notes"], list):
        existing_notes = lead.get("notes")
        lead["notes"] = []
        if isinstance(existing_notes, str) and existing_notes.strip():
            lead["notes"].append({
                "note": existing_notes,
                "timestamp": lead.get("created_at") or now
            })

    lead["notes"].append({
        "note": note.strip(),
        "timestamp": now
    })

    # Also log to timeline
    if "timeline" not in lead:
        lead["timeline"] = []
    lead["timeline"].append({
        "timestamp": now,
        "type": "note",
        "note": note.strip(),
        "description": f"Note added: {note.strip()}"
    })

    _save_crm(crm)
    return {
        "status": "success",
        "company": lead["company"],
        "note": note.strip(),
        "timestamp": now,
    }


@mcp.tool()
def archive_lead(company: str, reason: str) -> dict:
    """Move lead to archived status, keeping all data intact."""
    if not company or not company.strip():
        return {"status": "error", "error": "company cannot be empty."}

    crm = _load_crm()
    lead = _find_lead_by_company(crm["leads"], company)
    if lead is None:
        return {
            "status": "error",
            "error": f"No lead found for company '{company}'."
        }

    now = datetime.now(timezone.utc).isoformat()
    old_status = lead.get("status", "unknown")

    lead["status"] = "archived"
    lead["status_updated_at"] = now

    # Also log to timeline
    if "timeline" not in lead:
        lead["timeline"] = []
    lead["timeline"].append({
        "timestamp": now,
        "type": "archived",
        "reason": reason,
        "description": f"Lead archived (reason: {reason})"
    })

    _save_crm(crm)
    return {
        "status": "success",
        "company": lead["company"],
        "old_status": old_status,
        "new_status": "archived",
        "reason": reason,
        "timestamp": now,
    }


@mcp.tool()
def get_lead_timeline(company: str) -> dict:
    """Return the chronological history of all status changes and notes for a company."""
    if not company or not company.strip():
        return {"status": "error", "error": "company cannot be empty."}

    crm = _load_crm()
    lead = _find_lead_by_company(crm["leads"], company)
    if lead is None:
        return {
            "status": "error",
            "error": f"No lead found for company '{company}'."
        }

    timeline = lead.get("timeline", [])

    # If timeline is empty but we have creation time, notes, status, reconstruct virtually
    if not timeline:
        timeline = []
        if lead.get("created_at"):
            timeline.append({
                "timestamp": lead.get("created_at"),
                "type": "creation",
                "description": f"Lead created for {lead['company']}"
            })
        if lead.get("score") is not None:
            timeline.append({
                "timestamp": lead.get("scored_at") or lead.get("created_at"),
                "type": "score_init",
                "description": f"Score initialized: {lead.get('score')} ({lead.get('tier')})"
            })
        if lead.get("status") and lead.get("status") != "new":
            timeline.append({
                "timestamp": lead.get("status_updated_at") or lead.get("created_at"),
                "type": "status_change",
                "old_status": "new",
                "new_status": lead.get("status"),
                "description": f"Status updated: new → {lead.get('status')}"
            })
        notes = lead.get("notes")
        if notes:
            if isinstance(notes, list):
                for n in notes:
                    timeline.append({
                        "timestamp": n.get("timestamp"),
                        "type": "note",
                        "note": n.get("note"),
                        "description": f"Note added: {n.get('note')}"
                    })
            elif isinstance(notes, str) and notes.strip():
                timeline.append({
                    "timestamp": lead.get("created_at"),
                    "type": "note",
                    "note": notes,
                    "description": f"Note added: {notes}"
                })

    # Sort in chronological order (ascending timestamp)
    timeline = sorted(timeline, key=lambda x: x.get("timestamp", ""))

    return {
        "status": "success",
        "company": lead["company"],
        "timeline": timeline,
        "total_events": len(timeline),
    }


# =============================================================================
# SECTION 4 — EXISTING TOOLS (preserved for backward-compatibility)
# =============================================================================
# These tools are used by the ADK pipeline agents (lead_researcher,
# lead_scorer, outreach_drafter) via McpToolset with specific tool_filter
# values. They must not be renamed or removed — doing so would break the
# existing pipeline without any code change in the agents.
#
# COMPATIBILITY NOTE: The tool signatures below are identical to the original
# implementation. Only documentation and internal helpers have been updated.
# =============================================================================

@mcp.tool()
def list_leads() -> dict:
    """Return all leads with their id, name, company, status, and score.

    LEGACY TOOL — use get_all_leads() for new code. This tool is preserved
    for backward-compatibility with existing pipeline agents that call it
    via McpToolset with tool_filter=["list_leads"].

    Returns:
        dict: {"status": "success", "leads": [summary, ...], "total": int}
    """
    crm = _load_crm()
    summary = [
        {
            "id": lead.get("id"),
            "name": lead.get("name") or "(unknown)",
            "company": lead.get("company"),
            "title": lead.get("title") or "",
            "industry": lead.get("industry") or "",
            "status": lead.get("status"),
            "score": lead.get("score"),
        }
        for lead in crm.get("leads", [])
    ]
    return {"status": "success", "leads": summary, "total": len(summary)}


@mcp.tool()
def get_lead(lead_id: str) -> dict:
    """Retrieve the complete record for a specific lead by their unique ID.

    Returns all fields including outreach drafts, notes, and timestamps.
    For a lightweight summary of all leads, use get_all_leads() instead.

    Args:
        lead_id: The unique lead identifier (e.g. "lead_001").

    Returns:
        dict: {"status": "success", "lead": full_lead_dict}
              or {"status": "error", "error": str} if not found.
    """
    if not lead_id or not lead_id.strip():
        return {"status": "error", "error": "lead_id cannot be empty."}

    crm = _load_crm()
    for lead in crm.get("leads", []):
        if lead.get("id") == lead_id.strip():
            return {"status": "success", "lead": lead}

    return {
        "status": "error",
        "error": (
            f"Lead '{lead_id}' not found. "
            f"Use get_all_leads() to see available lead IDs."
        ),
    }


@mcp.tool()
def update_lead_score(
    lead_id: str = None,
    score: int = None,
    justification: str = None,
    company: str = None,
    new_score: int = None,
    reason: str = None,
) -> dict:
    """Update lead score and qualification tier.

    Supports both legacy lookup by lead_id (used by ADK agents) and new
    lookup by company name (used for CRM actions and automation).

    Args:
        lead_id:       Unique lead identifier (legacy).
        score:         Numeric priority score (legacy).
        justification: Text explanation of the score (legacy).
        company:       Company name to update (new).
        new_score:     New numeric score 0-100 (new).
        reason:        Explanation for the score update (new).
    """
    now = datetime.now(timezone.utc).isoformat()
    crm = _load_crm()
    leads = crm.get("leads", [])

    # 1. Company-based lookup and update (New API)
    if company is not None or new_score is not None:
        if not company or not company.strip():
            return {"status": "error", "error": "company cannot be empty."}
        if new_score is None:
            return {"status": "error", "error": "new_score cannot be empty."}
        if not isinstance(new_score, int) or not 0 <= new_score <= 100:
            return {
                "status": "error",
                "error": f"new_score must be an integer between 0 and 100, got: {new_score!r}",
            }

        lead = _find_lead_by_company(leads, company)
        if lead is None:
            return {
                "status": "error",
                "error": f"No lead found for company '{company}'."
            }

        old_score = lead.get("score")
        old_score_str = str(old_score) if old_score is not None else "None"

        if new_score >= 70:
            tier = "Hot"
        elif new_score >= 40:
            tier = "Warm"
        else:
            tier = "Cold"

        lead["score"] = new_score
        lead["tier"] = tier
        lead["score_justification"] = reason or ""
        lead["scored_at"] = now

        # Add to timeline
        if "timeline" not in lead:
            lead["timeline"] = []
        
        log_msg = f"Score updated: {old_score_str} → {new_score} (reason: {reason})"
        print(f"[crm_server] {log_msg}")

        # Log change to agent_log.txt
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            log_line = f"[{timestamp}] CRM_Server | SCORE_UPDATE | company={company}, {log_msg}\n"
            with open("agent_log.txt", "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            pass

        lead["timeline"].append({
            "timestamp": now,
            "type": "score_update",
            "old_score": old_score,
            "new_score": new_score,
            "reason": reason,
            "description": log_msg
        })

        _save_crm(crm)
        return {
            "status": "success",
            "lead_id": lead["id"],
            "company": lead["company"],
            "old_score": old_score,
            "new_score": new_score,
            "tier": tier,
            "message": log_msg
        }

    # 2. Lead ID-based lookup and update (Legacy API)
    if not lead_id or not lead_id.strip():
        return {"status": "error", "error": "Either company or lead_id must be provided."}
    if score is None:
        return {"status": "error", "error": "score is required for legacy score update."}
    if not isinstance(score, int) or not 0 <= score <= 100:
        return {
            "status": "error",
            "error": f"score must be an integer between 0 and 100, got: {score!r}",
        }

    for lead in leads:
        if lead.get("id") == lead_id.strip():
            old_score = lead.get("score")
            old_score_str = str(old_score) if old_score is not None else "None"

            if score >= 70:
                tier = "Hot"
            elif score >= 40:
                tier = "Warm"
            else:
                tier = "Cold"

            lead["score"] = score
            lead["tier"] = tier
            lead["score_justification"] = justification or ""
            lead["scored_at"] = now

            # Add to timeline
            if "timeline" not in lead:
                lead["timeline"] = []
            
            log_msg = f"Score updated: {old_score_str} → {score} (justification: {justification})"
            lead["timeline"].append({
                "timestamp": now,
                "type": "score_update",
                "old_score": old_score,
                "new_score": score,
                "reason": justification,
                "description": log_msg
            })

            _save_crm(crm)
            return {
                "status": "success",
                "lead_id": lead_id,
                "score": score,
                "tier": tier,
            }

    return {
        "status": "error",
        "error": f"Lead '{lead_id}' not found. Use get_all_leads() to see valid IDs.",
    }


@mcp.tool()
def add_outreach_draft(lead_id: str, subject: str, body: str) -> dict:
    """Save a personalised outreach email draft linked to a lead record.

    Called by the outreach_drafter agent after generating a personalised email.
    Drafts accumulate in the lead's outreach_drafts list — previous drafts are
    never overwritten, enabling A/B testing of different approaches over time.

    In production, this would queue the draft for human review before sending
    via Gmail API or Outlook API. Currently it stores the draft in the CRM.

    Args:
        lead_id: Unique lead identifier (e.g. "lead_001").
        subject: Email subject line.
        body:    Full email body text (plain text). May include a LinkedIn
                 note appended after a "--- LINKEDIN NOTE:" separator.

    Returns:
        dict: {"status": "success", "lead_id": str, "draft_id": str}
              or {"status": "error", "error": str} if lead not found.
    """
    if not lead_id or not lead_id.strip():
        return {"status": "error", "error": "lead_id cannot be empty."}

    if not subject or not body:
        return {"status": "error", "error": "Both subject and body are required."}

    crm = _load_crm()
    for lead in crm.get("leads", []):
        if lead.get("id") == lead_id:
            # Initialise the drafts list if this lead was created before this
            # field existed (schema migration: backward-compatible default).
            if "outreach_drafts" not in lead:
                lead["outreach_drafts"] = []

            draft_id = f"draft_{len(lead['outreach_drafts']) + 1:03d}"
            draft = {
                "draft_id": draft_id,
                "subject": subject,
                "body": body,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sent": False,          # Set to True after the rep sends it
                "sent_at": None,        # Timestamp when sent (future feature)
            }
            lead["outreach_drafts"].append(draft)
            _save_crm(crm)

            return {
                "status": "success",
                "lead_id": lead_id,
                "draft_id": draft_id,
            }

    return {
        "status": "error",
        "error": (
            f"Lead '{lead_id}' not found. "
            f"Use get_all_leads() to see valid lead IDs."
        ),
    }


# =============================================================================
# SECTION 5 — ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # ── Configuration from environment variables ───────────────────────────────
    # We read host and port from env so the server can be configured for
    # different environments (local dev vs Docker vs Cloud Run) without code
    # changes.
    host = os.getenv("CRM_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("CRM_SERVER_PORT", "8001"))
    transport = os.getenv("CRM_SERVER_TRANSPORT", "sse")

    # ── Pre-flight check ───────────────────────────────────────────────────────
    # Ensure the data file exists before the server starts accepting connections.
    # This gives the operator a clear startup message rather than a runtime
    # error on the first tool call.
    _ensure_data_file()

    lead_count = len(_load_crm().get("leads", []))
    print(f"[crm_server] Data file : {DATA_FILE}")
    print(f"[crm_server] Leads     : {lead_count} existing records")
    print(f"[crm_server] Tools     : add_lead, get_all_leads, get_leads_by_tier,")
    print(f"[crm_server]            update_lead_status (by company), get_lead,")
    print(f"[crm_server]            list_leads, update_lead_score, add_outreach_draft")
    print(f"[crm_server] Starting  : {transport.upper()} on {host}:{port} ...")
    print()

    # TRANSPORT OPTIONS:
    # "sse"   — HTTP/Server-Sent Events. Best for ADK McpToolset (default).
    #           Agents connect via SseConnectionParams(url="http://HOST:PORT/sse")
    # "stdio" — stdin/stdout. Best for MCP clients that launch the server as
    #           a subprocess. Connect via StdioConnectionParams().
    mcp.run(transport=transport, host=host, port=port)
