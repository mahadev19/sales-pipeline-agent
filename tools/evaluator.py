# Competition: AI Agents Intensive Vibe Coding - Kaggle 2026
"""
tools/evaluator.py
---------------------
PURPOSE:
    Provides an automated evaluation framework to assess the quality of lead research
    and generated email drafts in the B2B sales pipeline.

WHY EVALUATION FRAMEWORKS MATTER FOR PRODUCTION AI SYSTEMS:
    Deploying LLM-based agents into production presents unique challenges compared
    to traditional deterministic software:
    
    1. Performance Drift & Regression Monitoring:
       LLM APIs are updated frequently, prompts are tweaked, and models can undergo subtle
       behavior changes. An evaluation framework runs automated quality checks to ensure
       changes do not regress output quality (e.g. producing emails that are too long
       or failing to reference researched pain points).
       
    2. Guardrails & Compliance:
       In enterprise sales, outreach drafts must comply with brand rules (e.g. professional
       tone, specific length limits, clear calls to action, and strict character limits
       to avoid spam filters). Automated evaluators act as automated compliance checks.
       
    3. Quantitative Fine-Tuning:
       Instead of "vibe checking" output by reading a few samples, developers can measure
       average quality scores across thousands of runs, enabling data-driven optimization
       of system prompts and model selection.
       
    4. Quality Gates:
       Evaluators can act as programmatic gates in a CI/CD pipeline, automatically rejecting
       PRs if average quality scores drop below acceptable thresholds.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Resolve data file path relative to this workspace
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_LOG_FILE = DATA_DIR / "eval_log.jsonl"


def evaluate_research_quality(research_dict: dict) -> int:
    """Evaluate the quality of the researched lead intelligence.
    
    Returns a score from 0 to 100 based on completeness and specificity.
    """
    if not research_dict:
        return 0
        
    score = 0
    
    # 1. Is company overview more than 20 words? (+20)
    overview = research_dict.get("overview", "") or ""
    if len(overview.split()) > 20:
        score += 20
        
    # 2. Are 3+ pain points identified? (+20)
    pain_points = research_dict.get("pain_points", []) or []
    non_empty_pains = [p for p in pain_points if p and len(p.strip()) > 5]
    if len(non_empty_pains) >= 3:
        score += 20
        
    # 3. Is decision maker name a real person (not placeholder "Alex Smith")? (+20)
    dm = research_dict.get("decision_maker", {}) or {}
    dm_name = (dm.get("name", "") or "").strip()
    dm_name_lower = dm_name.lower()
    # Exclude default/stub placeholders
    invalid_names = {"", "unknown", "alex smith", "[stub] alex smith", "placeholder", "n/a"}
    if dm_name and dm_name_lower not in invalid_names:
        score += 20
        
    # 4. Is LinkedIn URL present? (+20)
    linkedin = (dm.get("linkedin_url", "") or "").strip()
    linkedin_lower = linkedin.lower()
    invalid_urls = {"", "unknown", "not found", "placeholder", "n/a"}
    if linkedin and linkedin_lower not in invalid_urls and "linkedin.com" in linkedin_lower:
        score += 20
        
    # 5. Is industry specific (not generic "Technology")? (+20)
    industry = (research_dict.get("industry", "") or "").strip()
    industry_lower = industry.lower()
    generic_industries = {"", "unknown", "technology", "technology / b2b services", "b2b services"}
    if industry and industry_lower not in generic_industries:
        score += 20
        
    return score


def _parse_email_draft(email_draft: str | dict) -> tuple[str, str]:
    """Parse subject and body out of the email draft payload."""
    if isinstance(email_draft, dict):
        subject = email_draft.get("email_subject") or email_draft.get("subject") or ""
        body = email_draft.get("email_body") or email_draft.get("body") or ""
        return subject.strip(), body.strip()
    elif isinstance(email_draft, str):
        # Handle "Subject: <subject>\n\n<body>"
        lines = email_draft.split("\n")
        subject = ""
        body_lines = []
        for line in lines:
            if line.lower().startswith("subject:"):
                subject = line[8:].strip()
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        # If no Subject: line found, assume first line is subject
        if not subject and lines:
            subject = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
        return subject, body
    return "", ""


def evaluate_email_quality(
    email_draft: str | dict,
    company_name: str = "",
    pain_points: list[str] = None
) -> int:
    """Evaluate the generated cold email draft quality.
    
    Returns a score from 0 to 100 based on brand constraints and personalization.
    """
    subject, body = _parse_email_draft(email_draft)
    if not subject and not body:
        return 0
        
    score = 0
    body_lower = body.lower()
    
    # 1. Subject line under 60 chars? (+20)
    if len(subject) < 60:
        score += 20
        
    # 2. Email body between 80-180 words? (+20)
    word_count = len(body.split())
    if 80 <= word_count <= 180:
        score += 20
        
    # 3. Contains company name? (+20)
    if company_name:
        if company_name.strip().lower() in body_lower:
            score += 20
    else:
        # Standalone check: does it look like it references a company (case-insensitive)
        if any(keyword in body_lower for keyword in ["your team", "at your company"]):
            score += 20
            
    # 4. Contains specific pain point reference? (+20)
    if pain_points:
        # Check if at least one pain point key phrase is referenced
        for p in pain_points:
            words = p.strip().lower().split()
            if len(words) >= 2:
                # check if first 2 words are in the body
                phrase = " ".join(words[:2])
                if phrase in body_lower:
                    score += 20
                    break
        else:
            # Check verbatim
            for p in pain_points:
                if p.strip().lower() in body_lower:
                    score += 20
                    break
    else:
        # Standalone keyword check
        common_b2b_pains = [
            "workflow", "manual", "data visibility", "headcount", "scaling", 
            "customization", "cost", "efficiency", "licensing", "process"
        ]
        if any(pain in body_lower for pain in common_b2b_pains):
            score += 20
            
    # 5. Has clear call to action? (+20)
    cta_patterns = [
        "call", "chat", "conversation", "demo", "calendar", "schedule",
        "worth a 15-minute", "discussion", "time this week"
    ]
    # Look for question marks or CTA action phrases
    if "?" in body or any(cta in body_lower for cta in cta_patterns):
        score += 20
        
    return score


def evaluate_pipeline_run(
    company: str,
    research: dict,
    score_dict: dict,
    email_draft: str | dict,
    time_taken: float
) -> dict:
    """Orchestrate evaluation of a full company pipeline run.
    
    Returns a report dictionary containing scores and metadata.
    """
    research_score = evaluate_research_quality(research)
    
    # Retrieve pain points and company name from research dict if possible
    pain_points = research.get("pain_points", []) if research else []
    canonical_company = research.get("company_name", company) if research else company
    
    email_score = evaluate_email_quality(
        email_draft=email_draft,
        company_name=canonical_company,
        pain_points=pain_points
    )
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "company": company,
        "duration_s": round(time_taken, 2),
        "research_eval_score": research_score,
        "email_eval_score": email_score,
        "metrics": {
            "research_completeness": {
                "overview_len": len((research.get("overview", "") or "").split()) if research else 0,
                "pain_points_count": len(pain_points),
                "decision_maker_found": bool(research.get("decision_maker", {}).get("name")) if research else False,
                "linkedin_found": "linkedin.com" in (research.get("decision_maker", {}).get("linkedin_url", "") or "").lower() if research else False,
                "specific_industry": research.get("industry", "") if research else ""
            },
            "email_outreach": {
                "subject_len": len(_parse_email_draft(email_draft)[0]),
                "body_word_count": len(_parse_email_draft(email_draft)[1].split())
            }
        }
    }
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Append report to JSONL file
    try:
        with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")
    except Exception as e:
        print(f"    [!] Failed to write evaluation log: {e}")
        
    return report
