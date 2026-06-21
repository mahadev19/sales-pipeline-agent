# Competition: AI Agents Intensive Vibe Coding - Kaggle 2026
"""
agents/outreach_drafter.py
---------------------------
PURPOSE:
    The OutreachDrafter is the final agent in the sales pipeline.
    It receives the research intelligence dict (from LeadResearcher) and the
    scoring output dict (from LeadScorer) and produces two personalised,
    professional outreach assets:

        1. A cold email — subject line + body (≤150 words)
        2. A LinkedIn connection request — single message (≤300 characters)

    Both assets are tone-calibrated to the lead's score tier:
        Hot  → Direct, confident, specific urgency signal
        Warm → Value-led, consultative, softer CTA
        Cold → Educational, curiosity-led, no hard ask

    The final drafts are saved to the CRM via the `add_outreach_draft` MCP tool
    so the sales rep can review, edit, and send from their own email/LinkedIn.

    It operates in two modes:
    ─────────────────────────────────────────────────────────────────────────
    A) PIPELINE MODE (used by main.py via SequentialAgent)
       ─ Reads state["research_brief"] and state["lead_score_summary"]
         produced by the previous two agents.
       ─ Calls get_lead (CRM) to confirm the contact's name and email.
       ─ Calls add_outreach_draft (CRM) to persist both email and LinkedIn.
       ─ Writes state["outreach_draft"] for the pipeline summary.

    B) STANDALONE / PROGRAMMATIC MODE (called directly)
       ─ draft_outreach(research_result, score_result) — takes both dicts,
         runs the ADK agent internally, returns a DraftResult TypedDict.
       ─ draft_outreach_fast(research_result, score_result) — pure-Python
         template-based drafting with zero API cost. Useful for testing and
         low-latency batch processing where LLM quality isn't required.
    ─────────────────────────────────────────────────────────────────────────

PROMPTING STRATEGY (detailed, see also inline comments below):
    ─────────────────────────────────────────────────────────────────────────
    The core challenge in cold outreach prompting is avoiding GENERIC output.
    LLMs default to polished-but-forgettable emails ("Hope this finds you
    well..."). Our strategy combats this with four techniques:

    1. PERSONA ANCHORING
       We don't ask the LLM to "write a cold email". We assign it a specific
       expert persona: "a B2B sales strategist who has closed 50+ enterprise
       deals and studied the psychology of cold email response rates." This
       activates more domain-specific knowledge and shifts tone from
       copywriter to advisor.

    2. ANTI-PATTERN INJECTION (negative examples)
       We explicitly list FORBIDDEN phrases (see _FORBIDDEN_PHRASES) and
       inject them into the prompt as a blacklist. LLMs respond well to
       negative constraints — telling it what NOT to do is as effective as
       telling it what to do, and often more so.

    3. PAIN-POINT ANCHORING (specificity forcing)
       We inject the exact pain_points list from the researcher as a
       REQUIRED reference list. The instruction mandates: "You MUST reference
       at least ONE pain point verbatim or paraphrased from this list."
       This forces specificity over generic value props.

    4. TIER-AWARE TONE MODULATION
       The instruction changes based on the score tier (Hot/Warm/Cold).
       A Hot lead's email is more direct and creates urgency. A Cold lead's
       email is educational with no hard ask. This prevents the LLM from
       writing the same email for every lead regardless of qualification level.

    5. FORMAT ENFORCEMENT via JSON BLOCK
       Like our other agents, we ask the LLM to emit a ```json block at the
       end of its response. This decouples the narrative output (for human
       review) from the structured data (for downstream code), and makes
       parsing reliable without brittle string splitting.

    6. CHARACTER COUNT CONSTRAINTS for LinkedIn
       LinkedIn connection requests have a hard 300-character limit. We give
       the LLM this constraint explicitly, and also instruct it to AIM for
       250 characters — giving itself a 50-character safety buffer. The
       parser also validates and trims the final output.
    ─────────────────────────────────────────────────────────────────────────

INPUTS:
    Pipeline mode  → state["research_brief"] + state["lead_score_summary"]
                     + state["lead_id"] (all set by previous agents)
    Programmatic   → ResearchResult dict + ScoreResult dict

OUTPUTS:
    Pipeline mode  → state["outreach_draft"] + CRM record via add_outreach_draft
    Programmatic   → DraftResult TypedDict:
        {
            "email_subject": str,       # 6-10 word subject line
            "email_body": str,          # ≤150 word email body
            "linkedin_message": str,    # ≤300 character connection note
            "word_count": int,          # Actual word count of email body
            "char_count": int,          # Actual char count of LinkedIn message
            "tier": str,                # "Hot" | "Warm" | "Cold" (inherited)
            "tone_notes": str,          # Brief explanation of tone choices made
        }

TOOLS USED:
    Pipeline mode → CRM MCP tools: get_lead, add_outreach_draft

FUTURE ENHANCEMENTS:
    - A/B test 3 subject line variants per lead; track open rates via pixel.
    - Auto-generate a follow-up sequence (email #2 at D+3, #3 at D+7).
    - Integrate Gmail API / Outlook API to queue drafts for 1-click sending.
    - Support multi-language drafting (French, Spanish, German) for EMEA leads.
    - Add a grader agent that scores drafts on personalisation / readability
      before they reach the rep (inner loop quality gate).
    - Maintain brand voice consistency via a fine-tuned style embedding.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import TypedDict

# ─── Path fix ─────────────────────────────────────────────────────────────────
# Makes the module importable whether run as `python agents/outreach_drafter.py`
# or imported from the project root via `from agents.outreach_drafter import ...`
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

# =============================================================================
# KNOWN PAIN POINTS — Company-specific overrides for generic fallback pain points
# =============================================================================
# When the researcher returns the generic stub pain point
# "inefficient manual workflows and processes", this dict provides a
# company-specific replacement so the email hook is still relevant.
# Keys are lowercase company names; "default" is never used (see logic below).
KNOWN_PAIN_POINTS: dict[str, str] = {
    "monday.com": "managing cross-team project visibility and automating repetitive status updates",
    "google": "enterprise developer tooling adoption and internal platform standardization",
    "stripe": "reducing payment fraud and improving developer onboarding experience",
    "default": "inefficient manual workflows and processes"
}

_GENERIC_PAIN_POINT = "inefficient manual workflows and processes"


def _resolve_pain_point(company_name: str, pain_points: list[str]) -> str:
    """
    Return the best primary pain point for outreach.

    If the first pain point is the generic stub value, check KNOWN_PAIN_POINTS
    for a company-specific override. Falls back to the actual pain point from
    the researcher when no override is configured.

    Args:
        company_name: The company being targeted.
        pain_points: List of pain points from the researcher.

    Returns:
        A specific, actionable pain point string.
    """
    primary = pain_points[0] if pain_points else _GENERIC_PAIN_POINT
    KNOWN_PAIN_POINTS["default"] = primary

    co_lower = company_name.lower()
    if _GENERIC_PAIN_POINT.lower() in primary.lower():
        for key in KNOWN_PAIN_POINTS.keys():
            if key != "default" and key in co_lower:
                return KNOWN_PAIN_POINTS[key]
    return KNOWN_PAIN_POINTS["default"]


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
# SECTION 1 — OUTPUT SCHEMA
# =============================================================================

class DraftResult(TypedDict):
    """
    Structured output from the OutreachDrafter (programmatic mode).

    Both assets are ready for the sales rep to review and send.
    The word_count and char_count fields let callers validate compliance
    before storing or sending.
    """
    email_subject: str      # 6-10 word subject line — no clickbait, no ALL CAPS
    email_body: str         # Full email body, ≤150 words, plain text
    linkedin_message: str   # LinkedIn connection note, ≤300 characters
    word_count: int         # Actual email body word count (post-generation)
    char_count: int         # Actual LinkedIn message char count (post-generation)
    tier: str               # "Hot" | "Warm" | "Cold" — inherited from scorer
    tone_notes: str         # Brief explanation of tone strategy used


# =============================================================================
# SECTION 2 — PROMPTING CONSTANTS
# =============================================================================
# These constants are injected into the LLM instruction to enforce quality.
# They are defined here (not inline) so they can be updated without touching
# the instruction template itself — making A/B testing of rules easy.
# =============================================================================

# ── PROMPTING TECHNIQUE: Anti-Pattern Injection ───────────────────────────────
# We explicitly list phrases the LLM must never use.
# Research on cold email response rates consistently shows these phrases
# trigger immediate deletion or spam filtering. By naming them explicitly,
# we override the LLM's training bias toward polished-but-generic output.
# This is more effective than just saying "be specific" — negative constraints
# are highly actionable for instruction-following models.
_FORBIDDEN_PHRASES = [
    "Hope this finds you well",
    "I wanted to reach out",
    "touching base",
    "synergies",
    "low-hanging fruit",
    "circle back",
    "move the needle",
    "game-changer",
    "thought leader",
    "paradigm shift",
    "leverage",
    "disruptive",
    "revolutionary",
    "best-in-class",
    "bleeding edge",
    "world-class",
    "I came across your profile",
    "I noticed you",
    "Quick question",
    "Just following up",
    "Per my last email",
]

# ── PROMPTING TECHNIQUE: Tone Modulation by Tier ─────────────────────────────
# These tone guides are injected into the prompt based on the lead's score tier.
# Hot leads warrant a more direct, confident tone; Cold leads need a lighter,
# educational touch. Matching tone to qualification level increases response rates
# because over-eager emails to cold leads feel pushy, while timid emails to
# hot leads waste urgency signals.
_TONE_GUIDE = {
    "Hot": (
        "TONE: Direct and confident. Reference the most compelling pain point "
        "immediately. Create mild urgency (tie to their growth stage or a recent "
        "event). The CTA can be slightly more specific — propose a concrete "
        "day/time window ('Would Tuesday or Wednesday work for a 15-min call?'). "
        "Avoid being pushy — confident ≠ aggressive."
    ),
    "Warm": (
        "TONE: Value-led and consultative. Lead with insight or a question that "
        "demonstrates understanding of their space, not a pitch. Position yourself "
        "as a peer sharing a relevant observation. Keep the CTA low-friction "
        "('Happy to share what worked for similar teams — would a quick call be "
        "useful?'). No urgency."
    ),
    "Cold": (
        "TONE: Educational and curiosity-led. Do NOT pitch a product. Share one "
        "relevant insight or observation about their industry. End with a simple "
        "yes/no question ('Does this resonate with challenges you're facing?'). "
        "The goal is to start a dialogue, not book a meeting. No CTA for a call."
    ),
}

# ── PROMPTING TECHNIQUE: LinkedIn-Specific Constraints ───────────────────────
# LinkedIn connection requests are fundamentally different from email:
#   - Hard 300-character limit (LinkedIn enforces this)
#   - No subject line — the note IS the message
#   - Must work without any prior relationship context
#   - Profile picture + name are visible, so skip formal introductions
# We document these constraints here and inject them into the prompt.
_LINKEDIN_CONSTRAINTS = """
LinkedIn Connection Note Rules:
- HARD LIMIT: 300 characters. Aim for 240-260 to leave a buffer.
- No greetings like "Hi [Name]," — jump straight to the value.
- ONE sentence max for who you are (company + role implied by profile).
- ONE sentence on why you're connecting (pain point or shared interest).
- Optional: ONE soft question to invite a response.
- NEVER ask for a meeting or a call in the connection request. That comes AFTER they accept.
- NO links, NO attachments, NO emojis (appear unprofessional in B2B).
- Must feel human-written, not templated. Avoid "I'd love to connect."
"""


# =============================================================================
# SECTION 3 — LLM INSTRUCTION BUILDER
# =============================================================================
# We BUILD the instruction dynamically (not hardcode it as a static string)
# because the tone section changes based on the lead tier. This is a key
# PROMPTING STRATEGY DECISION: dynamic few-shot context vs. static instruction.
#
# Alternative considered: One static prompt that mentions all three tiers.
# Rejected: The LLM would get confused about which tone rules apply, and
# might blend "Hot" urgency with "Cold" restraint — producing a middle-ground
# email that works for nobody. Single-tone injection is cleaner.
# =============================================================================

def _build_drafter_instruction(tier: str = "Warm") -> str:
    """
    Build the LLM system instruction, injecting the tier-appropriate tone guide.

    We use an f-string template here (rather than a jinja2 template) to keep
    dependencies minimal. The instruction is assembled at agent-definition time,
    not at prompt time — ADK's {placeholder} substitution handles the runtime
    session state values.

    Args:
        tier: "Hot" | "Warm" | "Cold" — controls tone injection.

    Returns:
        str: Full system instruction for the OutreachDrafter agent.
    """
    tone_section = _TONE_GUIDE.get(tier, _TONE_GUIDE["Warm"])
    forbidden_list = "\n".join(f"   ✗ '{p}'" for p in _FORBIDDEN_PHRASES)

    # ── Why we structure the instruction this way ─────────────────────────────
    # Order in a system prompt matters for instruction-following models:
    #   1. Persona first — establishes the "who" before the "what"
    #   2. Context (research + score) — the LLM needs facts before tasks
    #   3. Constraints BEFORE the task — negative rules read before writing
    #   4. Task steps — numbered for clarity and reliable execution order
    #   5. Output format LAST — format instructions apply after content is ready
    #
    # We deliberately put the forbidden phrases list BEFORE the writing task
    # so the model encodes those constraints before generating any text.
    # Studies on LLM instruction following show constraints are more reliably
    # honoured when listed before the generative task, not after.
    # ─────────────────────────────────────────────────────────────────────────

    return f"""\
You are a B2B sales strategist and copywriter who has closed 50+ enterprise
deals and studied the psychology of cold email response rates extensively.
You write outreach that gets replies — not because it's clever, but because
it's specific, relevant, and respectful of the recipient's time.

━━━ CONTEXT ━━━

Research Brief (from the LeadResearcher agent):
{{research_brief}}

Lead Score Summary (from the LeadScorer agent):
{{lead_score_summary}}

Lead ID: {{lead_id}}

━━━ TONE GUIDANCE ━━━

{tone_section}

━━━ ABSOLUTE CONSTRAINTS (non-negotiable) ━━━

FORBIDDEN PHRASES — never use any of these:
{forbidden_list}

EMAIL CONSTRAINTS:
- Subject line: 6-10 words. Specific hook (reference their company, role, or
  a pain point). No questions marks. No ALL CAPS. No clickbait.
- Body: UNDER 150 WORDS. Count carefully.
- First sentence: Must reference ONE specific fact from the research brief
  (a pain point, company milestone, industry signal, or the decision-maker's
  role). Generic openers are rejected.
- Value proposition: 1-2 sentences MAX. Address ONE pain point specifically —
  do NOT make a generic product pitch.
- CTA: Low-friction. One specific ask. See tone guidance for format.
- Sign-off: "Best," or "Best regards," followed by [Your Name]. No titles.
- NO pricing, NO attachments, NO links (not even your website in email #1).

{_LINKEDIN_CONSTRAINTS}

━━━ PAIN POINTS TO REFERENCE ━━━

You MUST reference at least ONE of these pain points (verbatim or paraphrased)
in the email body. These were identified by the research agent as the most
relevant problems for this company:
{{pain_points_list}}

━━━ STEPS ━━━

1. Call get_lead with {{lead_id}} to confirm the contact's exact name and email.
2. Read the research brief and score summary carefully.
3. Draft the cold email (subject + body ≤150 words).
4. Draft the LinkedIn connection note (≤300 characters; aim for 250).
5. Call add_outreach_draft with:
   - lead_id = {{lead_id}}
   - subject = your email subject line
   - body = email body + "\\n\\n---\\nLINKEDIN NOTE:\\n" + linkedin message
   This saves both assets together in the CRM for the sales rep.
6. After the CRM save, output your full drafts followed by a ```json block:

```json
{{{{
  "email_subject": "<your subject line>",
  "email_body": "<full email body, no subject>",
  "linkedin_message": "<connection note ≤300 chars>",
  "word_count": <integer>,
  "char_count": <integer>,
  "tier": "{tier}",
  "tone_notes": "<1 sentence explaining your key tone decision for this lead>"
}}}}
```

━━━ QUALITY CHECK ━━━

Before emitting the JSON, verify:
☑ Email body is ≤150 words (count them)
☑ Subject line does not contain any forbidden phrase
☑ First line of email body references a specific research fact
☑ LinkedIn note is ≤300 characters
☑ No product pricing or links are included
☑ Tone matches the {tier} tier guidance above
"""


# =============================================================================
# SECTION 4 — MCP TOOLSET AND PIPELINE AGENT
# =============================================================================

# ── CRM Toolset ───────────────────────────────────────────────────────────────
# DESIGN DECISION: We expose TWO tools here (get_lead + add_outreach_draft),
# unlike the researcher (get_lead only) and scorer (update_lead_score only).
#
# Why get_lead? The drafter needs the contact's EXACT name and email for the
# salutation and reply-to. We don't pass these through session state from the
# researcher because state is shared and the researcher may not have confirmed
# them — the CRM is the authoritative source. Fetching fresh also guards
# against stale data if the CRM was updated between agent runs.
#
# Why not add more tools? We exclude list_leads, update_lead_score, etc.
# Least-privilege principle: every tool in the agent's scope is a tool it
# could accidentally call. Narrow scope = fewer failure modes.
# ─────────────────────────────────────────────────────────────────────────────
_CRM_SERVER_URL = os.getenv("CRM_SERVER_URL", "http://localhost:8001/sse")

crm_toolset = McpToolset(
    connection_params=SseConnectionParams(url=_CRM_SERVER_URL),
    tool_filter=["get_lead", "add_outreach_draft"],
)


# ── Pipeline Agent ────────────────────────────────────────────────────────────
# DESIGN DECISION: We use `output_key="outreach_draft"` (free-form text output)
# rather than output_schema (Pydantic). As with the scorer, ADK cannot use
# output_schema AND tool calls in the same agent. Since we need add_outreach_draft
# to fire, we use output_key and parse the JSON block ourselves downstream.
#
# The instruction is built for "Warm" tier by default for the pipeline agent.
# In pipeline mode, the LLM reads the score tier from {lead_score_summary}
# and self-adjusts its tone — the tier injection in the instruction is a hint,
# not a hard constraint. For standalone mode, we build a fresh instruction
# per call using _build_drafter_instruction(tier) with the actual tier value.
# ─────────────────────────────────────────────────────────────────────────────
outreach_drafter = Agent(
    name="outreach_drafter",

    # gemini-2.0-flash: fast, available, and bypasses daily project rate limits
    model="gemini-2.0-flash",

    description=(
        "Writes a personalised cold email (subject + ≤150-word body) and a "
        "LinkedIn connection note (≤300 chars) for a qualified lead, calibrated "
        "to the lead's score tier and specific pain points found by research."
    ),

    # The pipeline instruction uses a default "Warm" tone, but the LLM reads
    # the actual tier from {lead_score_summary} and adjusts accordingly.
    # See _build_drafter_instruction() for why tone is injected this way.
    instruction=_build_drafter_instruction(tier="Warm"),

    # ADK automatically writes the agent's final response into
    # state["outreach_draft"] — consumed by main.py's pipeline summary.
    output_key="outreach_draft",

    tools=[crm_toolset],    # get_lead (confirm contact) + add_outreach_draft (persist)
)


# =============================================================================
# SECTION 5 — JSON EXTRACTION HELPER
# =============================================================================

def _extract_draft_json(text: str) -> dict:
    """
    Extract and validate the structured JSON block from the LLM's draft output.

    DESIGN: We use the same regex-based extraction pattern as the other agents
    (lead_researcher and lead_scorer) to keep parsing behaviour consistent and
    maintainable. The pattern matches a fenced ```json ... ``` block anywhere
    in the response text.

    Post-extraction validation:
    - Trims LinkedIn message to 300 chars if the LLM overran the limit.
      (LLMs occasionally miscount characters — we enforce the constraint in code.)
    - Recalculates word_count and char_count from the actual text so they
      always reflect reality, not the LLM's self-reported numbers.

    Args:
        text: Full text response from the LLM drafter agent.

    Returns:
        Parsed and validated dict, or empty dict if extraction fails.
    """
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return {}

    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}

    # ── Post-parse enforcement ────────────────────────────────────────────────
    # LinkedIn hard-trim: LinkedIn's API rejects notes over 300 characters.
    # We trim rather than error so the pipeline always produces usable output.
    linkedin = parsed.get("linkedin_message", "")
    if len(linkedin) > 300:
        parsed["linkedin_message"] = linkedin[:297] + "..."

    # Email subject clean truncation: cap at 60 characters at a word boundary cleanly
    subject = parsed.get("email_subject", "")
    if len(subject) > 60:
        truncated = subject[:57]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        parsed["email_subject"] = truncated + "..."

    # Recompute counts from actual strings (LLM self-counts are unreliable)
    body = parsed.get("email_body", "")
    parsed["word_count"] = len(body.split())
    parsed["char_count"] = len(parsed.get("linkedin_message", ""))

    return parsed


# =============================================================================
# SECTION 6 — FAST TEMPLATE-BASED DRAFTER (zero-LLM fallback)
# =============================================================================

def draft_outreach_fast(research_result: dict, score_result: dict) -> DraftResult:
    """
    Generate outreach drafts using pure-Python templates — no LLM, no API calls.

    WHEN TO USE:
    - Batch processing hundreds of leads where LLM cost is prohibitive.
    - Unit testing the pipeline without mocking API calls.
    - As a fallback when the Gemini API is unavailable (503 errors).
    - When you need instant output for a live demo.

    QUALITY TRADE-OFF:
    The template produces grammatically correct, non-generic outreach by
    injecting real company data (name, pain point, decision-maker title),
    but lacks the nuanced phrasing and contextual creativity of the LLM
    version. Use LLM mode (draft_outreach) for final sales rep review.

    PROMPTING INSIGHT (applies to templates too):
    Even in templates, we force specificity by always pulling the FIRST
    pain point from the list (index 0) for the email hook. We do NOT use
    a generic placeholder like "your challenges" — a named pain point is
    always more compelling than a vague reference.

    Args:
        research_result: ResearchResult dict from lead_researcher.
        score_result: ScoreResult dict from lead_scorer.

    Returns:
        DraftResult dict ready for the sales rep.
    """
    company = research_result.get("company_name", "your company")
    log_action("OutreachDrafter", "TEMPLATE_DRAFT_START", f"company={company}")
    industry_raw = research_result.get("industry", "your industry")

    # ── Industry tag cleaning ─────────────────────────────────────────────────
    # The LeadResearcher sometimes returns a slash-separated chain of tags, e.g.
    # "Software / E-Commerce / Cloud / Artificial Intelligence / AI".
    # Using the full string in subject lines and email bodies looks awkward and
    # unprofessional. We extract only the first tag for clean, concise copy.
    # Example: "Software / E-Commerce / AI" → "Software"
    industry = industry_raw.split("/")[0].strip() if "/" in industry_raw else industry_raw.strip()

    pain_points = research_result.get("pain_points", [])
    dm = research_result.get("decision_maker", {})
    dm_name = dm.get("name", "")
    dm_title = dm.get("title", "")
    tier = score_result.get("tier", "Warm")
    score = score_result.get("score", 50)

    # Pick the most relevant pain point for the hook, with company-specific override
    # for generic stub values (see KNOWN_PAIN_POINTS and _resolve_pain_point above).
    primary_pain = _resolve_pain_point(company, pain_points)
    short_pain = primary_pain

    # ── Subject line: tier-calibrated hook ───────────────────────────────────
    # Hot: specific and action-oriented
    # Warm: question that creates curiosity without a hard sell
    # Cold: observation-led, no product reference
    if tier == "Hot":
        raw_subject = f"Quick thought on {company}'s {primary_pain.lower()}"
    elif tier == "Warm":
        raw_subject = f"How {industry} teams are solving {primary_pain.lower()}"
    else:
        raw_subject = f"One trend reshaping {industry} in 2025"

    # Cleanly truncate the subject line to a max of 60 characters at a word boundary
    if len(raw_subject) > 60:
        truncated = raw_subject[:57]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        subject = truncated + "..."
    else:
        subject = raw_subject

    # ── Salutation: personalise if DM name is known ───────────────────────────
    # When the name is unknown/empty/generic we fall back to "Hi there," rather
    # than "Hi ," (which looks broken) or the generic "Hi," (which feels cold).
    if dm_name and dm_name.strip().lower() not in ("unknown", "not found", ""):
        first_name = dm_name.split()[0]
        salutation = f"Hi {first_name},"
    else:
        salutation = "Hi there,"

    # ── Body: modular blocks assembled by tier ────────────────────────────────
    overview = research_result.get("overview", f"{company} operates in the {industry} space.")
    overview_short = overview[:120] + "..." if len(overview) > 120 else overview

    if tier == "Hot":
        body = (
            f"{salutation}\n\n"
            f"I've been following {company}'s growth in the {industry} space — "
            f"specifically the challenge of {primary_pain.lower().rstrip('.')}.\n\n"
            f"We've helped similar {industry} teams cut that problem in half within "
            f"90 days, without adding headcount.\n\n"
            f"Worth a 15-minute call this week or next? I can share exactly what worked.\n\n"
            f"Best,\n[Your Name]"
        )
    elif tier == "Warm":
        body = (
            f"{salutation}\n\n"
            f"I came across {company} while researching {industry} companies tackling "
            f"{primary_pain.lower().rstrip('.')}.\n\n"
            f"A few teams I work with faced the same issue — happy to share what "
            f"approaches made the biggest difference, no strings attached.\n\n"
            f"Would a quick conversation be useful?\n\n"
            f"Best,\n[Your Name]"
        )
    else:   # Cold
        body = (
            f"{salutation}\n\n"
            f"I've been studying how {industry} companies are navigating "
            f"{primary_pain.lower().rstrip('.')} — it's one of the most "
            f"common friction points I hear about.\n\n"
            f"We recently published a short breakdown of what's working. "
            f"Happy to share if it would be useful.\n\n"
            f"Does this resonate with anything you're seeing at {company}?\n\n"
            f"Best,\n[Your Name]"
        )

    # ── LinkedIn note: always ≤300 chars, tier-aware ──────────────────────────
    # LinkedIn notes are tight: no room for pleasantries. Jump to the value.
    dm_context = f"({dm_title} roles)" if dm_title and dm_title.lower() != "unknown" else ""
    if tier == "Hot":
        linkedin = (
            f"Researching {company}'s approach to {short_pain[:60].rstrip()} — "
            f"working with several {industry} teams on the same challenge. "
            f"Thought it worth connecting."
        )
    elif tier == "Warm":
        linkedin = (
            f"Following {company}'s work in {industry} {dm_context}. "
            f"We're exploring how teams solve {short_pain[:50].rstrip()} — "
            f"seemed like an interesting area to connect on."
        )
    else:
        linkedin = (
            f"Studying {industry} trends — {primary_pain[:60].rstrip()} "
            f"keeps coming up. Connecting with leaders in the space to compare notes."
        )

    # Enforce 300-char limit
    linkedin = linkedin[:297] + "..." if len(linkedin) > 300 else linkedin

    # Post-process to ensure generic pain point is replaced if company matches
    co_lower = company.lower()
    matched_key = None
    for key in KNOWN_PAIN_POINTS:
        if key != "default" and key in co_lower:
            matched_key = key
            break
    if matched_key:
        override_pain = KNOWN_PAIN_POINTS[matched_key]
        generic_pattern = re.compile(re.escape(_GENERIC_PAIN_POINT), re.IGNORECASE)
        subject = generic_pattern.sub(override_pain, subject)
        body = generic_pattern.sub(override_pain, body)
        linkedin = generic_pattern.sub(override_pain, linkedin)

    # ── Tone notes ────────────────────────────────────────────────────────────
    tone_notes = (
        f"Used {tier.upper()} tier tone: "
        + {"Hot": "direct with urgency, specific pain point hook, concrete CTA.",
           "Warm": "consultative, peer-to-peer, soft no-strings offer.",
           "Cold": "educational, observation-led, low-friction yes/no close."}[tier]
    )

    result = DraftResult(
        email_subject=subject,
        email_body=body,
        linkedin_message=linkedin[:300],
        word_count=len(body.split()),
        char_count=len(linkedin[:300]),
        tier=tier,
        tone_notes=tone_notes,
    )
    log_action("OutreachDrafter", "TEMPLATE_DRAFT_SUCCESS", f"company={company}, words={result['word_count']}, chars={result['char_count']}")
    return result


# =============================================================================
# SECTION 7 — STANDALONE LLM DRAFTER
# =============================================================================

# ── Standalone agent (no CRM dependency) ─────────────────────────────────────
# Separate from `outreach_drafter` so:
# - The MCP server does not need to be running for standalone calls.
# - We can inject the actual tier into the instruction (not "Warm" default).
# - No output_key — we capture and parse the response ourselves.
# ─────────────────────────────────────────────────────────────────────────────
def _build_standalone_agent(tier: str, model: str = "gemini-2.0-flash") -> Agent:
    """
    Build a fresh standalone ADK agent with a tier-specific instruction.

    We create a new Agent instance per call (rather than caching one) because
    the instruction depends on the tier, which varies per lead. ADK Agent
    instances are lightweight (no persistent state), so this is safe.

    DESIGN NOTE: An alternative would be a single agent with a tier parameter
    in the prompt template, resolved from session state. We chose per-call
    instantiation because it's simpler and avoids the risk of the LLM
    misreading a tier variable from a state key with complex formatting.

    Args:
        tier: "Hot" | "Warm" | "Cold"
        model: Model name string.

    Returns:
        ADK Agent configured for standalone outreach drafting.
    """
    return Agent(
        name=f"outreach_drafter_standalone_{tier.lower()}",
        model=model,
        description=f"Standalone {tier}-tier outreach drafter. No CRM tools.",
        instruction=_build_drafter_instruction(tier=tier),
        # No tools, no output_key — we capture raw output and parse JSON ourselves
    )


async def _run_draft_async(
    research_result: dict,
    score_result: dict,
) -> DraftResult:
    """
    Internal async implementation of draft_outreach().

    Runs the LLM agent, extracts the JSON block, validates all constraints,
    and returns a DraftResult. Falls back to draft_outreach_fast() if the
    LLM call fails after retries.

    Args:
        research_result: ResearchResult dict from lead_researcher.
        score_result: ScoreResult dict from lead_scorer.

    Returns:
        DraftResult dict.
    """
    tier = score_result.get("tier", "Warm")
    company = research_result.get("company_name", "the company")
    log_action("OutreachDrafter", "AGENT_RUN_START", f"company={company}, tier={tier}")
    pain_points = research_result.get("pain_points", [])
    # Resolve the best primary pain point, applying company-specific overrides
    # for generic stub values (see KNOWN_PAIN_POINTS and _resolve_pain_point).
    primary_pain_resolved = _resolve_pain_point(company, pain_points)

    # Format pain points as a numbered list for the prompt.
    # Why numbered, not bulleted? Numbered lists signal "evaluate each one"
    # to the LLM, which encourages it to consider all pain points before
    # choosing the best hook — rather than defaulting to the first item.
    # We ensure the resolved primary pain point is always listed first.
    pain_points_for_prompt = pain_points[:]
    if pain_points_for_prompt and primary_pain_resolved != pain_points_for_prompt[0]:
        pain_points_for_prompt = [primary_pain_resolved] + pain_points_for_prompt
    pain_points_formatted = "\n".join(
        f"  {i+1}. {p}" for i, p in enumerate(pain_points_for_prompt)
        if p and len(p.strip()) > 5
    ) or "  (No specific pain points identified — use industry-level insight)"

    # Build a compact research summary for the session state.
    # We omit raw_brief (too long) and keep only the fields the drafter needs.
    research_compact = {
        "company_name": company,
        "overview": research_result.get("overview", ""),
        "company_size": research_result.get("company_size", ""),
        "industry": research_result.get("industry", ""),
        "pain_points": pain_points,
        "decision_maker": research_result.get("decision_maker", {}),
    }

    # Format score context as a clean string for the instruction placeholder.
    score_context = (
        f"Score: {score_result.get('score', 'N/A')}/100 | "
        f"Tier: {tier} | "
        f"Reason: {score_result.get('reason', 'Not provided')} | "
        f"Action: {score_result.get('recommended_action', 'nurture')}"
    )

    app_name = f"outreach_drafter_{tier.lower()}"

    initial_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(
            text=(
                f"Draft outreach for {company} — a {tier} lead "
                f"(score {score_result.get('score', 'N/A')}/100). "
                f"Primary pain point: {primary_pain_resolved}. "
                f"Produce both the cold email and LinkedIn note, then the JSON block."
            )
        )],
    )

    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash-latest"]
    # Fallback chain: gemini-2.0-flash (primary) → gemini-2.0-flash-lite →
    # gemini-1.5-flash-latest (final LLM fallback before template drafter).
    raw_response = ""
    last_error: Exception | None = None
    success = False

    for model_name in models:
        print(f"🤖 Using model: {model_name}")
        log_action("OutreachDrafter", "MODEL_SELECTION", f"model={model_name}")

        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name=app_name,
            user_id="standalone_user",
            state={
                "research_brief": json.dumps(research_compact, indent=2),
                "lead_score_summary": score_context,
                "lead_id": research_result.get("company_name", "standalone"),
                "pain_points_list": pain_points_formatted,
            },
        )

        runner = Runner(
            agent=_build_standalone_agent(tier, model=model_name),
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
                break   # Success — exit retry loop
            except Exception as exc:
                last_error = exc
                err = str(exc)
                is_transient = any(k in err for k in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if is_transient and attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE ** attempt
                    print(f"  [OutreachDrafter] Attempt {attempt}/{MAX_RETRIES} failed with model {model_name} ({err[:80]}). Retrying in {wait}s…")
                    await asyncio.sleep(wait)
                    session = await session_service.create_session(
                        app_name=app_name,
                        user_id="standalone_user",
                        state={
                            "research_brief": json.dumps(research_compact, indent=2),
                            "lead_score_summary": score_context,
                            "lead_id": research_result.get("company_name", "standalone"),
                            "pain_points_list": pain_points_formatted,
                        },
                    )
                else:
                    print(f"  [OutreachDrafter] Model {model_name} failed/exhausted ({err[:80]}). Trying next model in fallback chain...")
                    break
        if success:
            break
    else:
        if last_error:
            log_action("OutreachDrafter", "AGENT_RUN_ERROR", f"company={company}, error={str(last_error)[:100]}")
            raise last_error

    # ── Parse and validate LLM output ────────────────────────────────────────
    parsed = _extract_draft_json(raw_response)

    if parsed and "email_subject" in parsed and "email_body" in parsed:
        # Use LLM output but recompute counts from actual strings
        body = parsed.get("email_body", "")
        linkedin = parsed.get("linkedin_message", "")
        subject = parsed.get("email_subject", "")

        # Post-process to ensure generic pain point is replaced if company matches
        co_lower = company.lower()
        matched_key = None
        for key in KNOWN_PAIN_POINTS:
            if key != "default" and key in co_lower:
                matched_key = key
                break
        if matched_key:
            override_pain = KNOWN_PAIN_POINTS[matched_key]
            generic_pattern = re.compile(re.escape(_GENERIC_PAIN_POINT), re.IGNORECASE)
            subject = generic_pattern.sub(override_pain, subject)
            body = generic_pattern.sub(override_pain, body)
            linkedin = generic_pattern.sub(override_pain, linkedin)

        result = DraftResult(
            email_subject=subject,
            email_body=body,
            linkedin_message=linkedin[:300],    # Hard enforce just in case
            word_count=len(body.split()),
            char_count=len(linkedin[:300]),
            tier=tier,
            tone_notes=parsed.get("tone_notes", ""),
        )
        log_action("OutreachDrafter", "AGENT_RUN_SUCCESS", f"company={company}, words={result['word_count']}, chars={result['char_count']}")
        return result

    # ── Fallback: deterministic template drafter ──────────────────────────────
    # If the LLM produced no parseable output (rare but possible on complex
    # responses), fall back to the template-based drafter. This ensures the
    # pipeline always produces something the sales rep can review.
    print("  [OutreachDrafter] LLM output not parseable — using template fallback.")
    return draft_outreach_fast(research_result, score_result)


def draft_outreach(research_result: dict, score_result: dict) -> DraftResult:
    """
    Generate personalised outreach drafts using the LLM agent.

    This is the primary public API for standalone use. It runs the Gemini
    agent, extracts the structured JSON, validates all constraints (word
    count, char count), and returns a ready-to-review DraftResult.

    Falls back to draft_outreach_fast() automatically on API errors.

    Args:
        research_result: ResearchResult dict from lead_researcher.research_company().
        score_result: ScoreResult dict from lead_scorer.score_lead_dict() or .score_lead().

    Returns:
        DraftResult dict with email_subject, email_body, linkedin_message,
        word_count, char_count, tier, and tone_notes.

    Example:
        >>> from agents.lead_researcher import research_company
        >>> from agents.lead_scorer import score_lead_dict
        >>> from agents.outreach_drafter import draft_outreach
        >>> research = research_company("HubSpot")
        >>> score = score_lead_dict(research)
        >>> draft = draft_outreach(research, score)
        >>> print(draft["email_subject"])
        How HubSpot's RevOps teams are solving pipeline visibility
        >>> print(f"Words: {draft['word_count']}, LinkedIn chars: {draft['char_count']}")
        Words: 132, LinkedIn chars: 248

    Note:
        Requires GOOGLE_API_KEY in .env. Retries 3x on 503/429 errors,
        then falls back to draft_outreach_fast().
    """
    try:
        return asyncio.run(_run_draft_async(research_result, score_result))
    except Exception as exc:
        print(f"  [OutreachDrafter] LLM draft failed ({str(exc)[:120]}). Falling back to template.")
        return draft_outreach_fast(research_result, score_result)


# =============================================================================
# SECTION 8 — CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Force UTF-8 output so the demo prints cleanly on Windows terminals
    # that default to cp1252 (which can't encode many Unicode characters).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # ── Demo data — mirrors real researcher + scorer output ───────────────────
    # You can replace these dicts with actual output from the other agents:
    #   research = research_company("Notion")
    #   score = score_lead_dict(research)
    demo_research = {
        "company_name": "Acme SaaS Corp",
        "overview": (
            "Acme SaaS Corp builds revenue intelligence software for mid-market "
            "B2B sales teams. Their platform integrates with Salesforce and HubSpot."
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

    demo_score = {
        "score": 79,
        "tier": "Hot",
        "reason": (
            "Acme SaaS Corp scores 79/100 — a Hot lead with 4 specific pain points "
            "in the high-ICP Sales Tech space. Recommend immediate outreach to VP "
            "of Sales Sarah Chen who has a confirmed LinkedIn profile."
        ),
        "breakdown": {
            "company_size": 21,
            "industry_fit": 25,
            "pain_point_relevance": 26,
            "decision_maker": 20,
        },
        "recommended_action": "immediate_outreach",
    }

    print(f"\n{'='*60}")
    print("  Outreach Drafter — Template Mode (fast, no API)")
    print(f"{'='*60}\n")

    result = draft_outreach_fast(demo_research, demo_score)

    print("[EMAIL DRAFT]")
    print("-" * 60)
    print(f"SUBJECT: {result['email_subject']}\n")
    print(result["email_body"])
    print("-" * 60)
    word_ok = "[OK]" if result["word_count"] <= 150 else "[OVER LIMIT]"
    print(f"Word count : {result['word_count']}/150 {word_ok}")

    print(f"\n[LINKEDIN NOTE]")
    print("-" * 60)
    print(result["linkedin_message"])
    print("-" * 60)
    char_ok = "[OK]" if result["char_count"] <= 300 else "[OVER LIMIT]"
    print(f"Char count : {result['char_count']}/300 {char_ok}")

    print(f"\nTone  : {result['tier']} tier")
    print(f"Notes : {result['tone_notes']}\n")
