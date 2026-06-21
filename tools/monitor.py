# Competition: AI Agents Intensive Vibe Coding - Kaggle 2026
"""
tools/monitor.py
---------------------
PURPOSE:
    Provides production monitoring, logging parser, metrics calculation, and
    ASCII dashboard rendering for the multi-agent sales pipeline.

ENTERPRISE PRODUCTION MONITORING IMPORTANCE FOR AI DEPLOYMENTS:
    In enterprise AI architectures, deploying LLM agents is not a "fire-and-forget"
    operation. Production monitoring is essential for several critical reasons:
    
    1. Model Drift & Behavioral Regression:
       LLM providers update underlying models frequently. Subtle updates can cause
       changes in prompt parsing, output formats, or reasoning quality. Continuous
       evaluation and quality monitoring catch regressions before they hit CRM records.
       
    2. Guardrails, Safety & Regulatory Compliance:
       B2B automated outreach must comply with regulations (e.g. GDPR, CAN-SPAM).
       Monitoring drafts for length, spam-like phrasing, and content ensures
       regulatory adherence and protects company brand reputation.
       
    3. Human-in-the-Loop (HITL) Alignment:
       Tracking the ratio of human approvals to edits/skips provides a direct metric
       of agent alignment. A high edit or skip rate indicates that the system prompts
       need calibration or that the model's domain knowledge is lacking.
       
    4. Cost, SLA, and Resource Management:
       LLM APIs introduce latency and token-based costs. Tracking pipeline execution
       times, error rates, and volumes is critical for resource allocation and meeting
       operational SLAs.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

# Resolve workspace directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
AGENT_LOG_FILE = ROOT_DIR / "agent_log.txt"
EVAL_LOG_FILE = DATA_DIR / "eval_log.jsonl"
METRICS_FILE = DATA_DIR / "metrics.json"


def compute_metrics(
    agent_log_path: str | Path = AGENT_LOG_FILE,
    eval_log_path: str | Path = EVAL_LOG_FILE
) -> dict:
    """Read agent logs and evaluation logs to calculate system-wide metrics.
    
    Handles missing files and malformed logs gracefully to ensure production stability.
    """
    # 1. Parse evaluation logs
    eval_records = []
    if os.path.exists(eval_log_path):
        try:
            with open(eval_log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        try:
                            eval_records.append(json.loads(line))
                        except Exception:
                            pass
        except Exception as e:
            print(f"[!] Warning: Failed to read evaluation logs: {e}")

    total_leads_eval = len(eval_records)
    
    # Initialize calculated metrics
    avg_research_score = 0.0
    avg_email_score = 0.0
    avg_pipeline_time = 0.0
    industries = []
    
    if total_leads_eval > 0:
        avg_research_score = sum(r.get("research_eval_score", 0) for r in eval_records) / total_leads_eval
        avg_email_score = sum(r.get("email_eval_score", 0) for r in eval_records) / total_leads_eval
        avg_pipeline_time = sum(r.get("duration_s", 0.0) for r in eval_records) / total_leads_eval
        
        for r in eval_records:
            ind = r.get("metrics", {}).get("research_completeness", {}).get("specific_industry", "")
            if isinstance(ind, str):
                ind = ind.strip()
                if ind and ind.lower() not in ("unknown", "n/a", "placeholder", "generic"):
                    industries.append(ind)
                    
    industry_counts = Counter(industries)
    most_common_industries = industry_counts.most_common(5)

    # 2. Parse agent_log.txt for HITL decisions, tier distributions, and run statuses
    human_approved = 0
    human_skipped = 0
    
    persist_approved = 0
    persist_skipped = 0
    
    tier_counts = {"Hot": 0, "Warm": 0, "Cold": 0}
    total_runs = 0
    failed_runs = 0

    if os.path.exists(agent_log_path):
        try:
            with open(agent_log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Parse Human review decisions
                    # Format: HUMAN_DECISION: {company} → {choice} at {timestamp}
                    if "HUMAN_DECISION:" in line:
                        parts = line.split("HUMAN_DECISION:")
                        if len(parts) >= 2:
                            decision_part = parts[1]
                            if "→" in decision_part:
                                choice_part = decision_part.split("→")[1].strip()
                                if choice_part:
                                    choice = choice_part[0].upper()
                                    if choice in ("A", "E"):
                                        human_approved += 1
                                    elif choice == "S":
                                        human_skipped += 1
                                        
                    # Parse STAGE_PERSIST_SUCCESS statuses (fallback for human decisions)
                    elif "STAGE_PERSIST_SUCCESS" in line:
                        if "status=approved" in line or "status=contacted" in line:
                            persist_approved += 1
                        elif "status=skipped" in line:
                            persist_skipped += 1

                    # Parse Tier distributions
                    if "SCORE_SUCCESS" in line:
                        # Find tier=Hot, tier=Warm, tier=Cold
                        for tier in ("Hot", "Warm", "Cold"):
                            if f"tier={tier}" in line:
                                tier_counts[tier] += 1
                                break

                    # Parse Run statuses (success vs failure)
                    if "COMPANY_END" in line:
                        total_runs += 1
                        if "status=FAILED" in line or "status=error" in line:
                            failed_runs += 1
        except Exception as e:
            print(f"[!] Warning: Failed to read agent logs: {e}")

    # Determine final approved vs skipped numbers
    # If explicit human decision logs exist, prioritize them. Otherwise fallback to persist statuses.
    if (human_approved + human_skipped) > 0:
        approved_count = human_approved
        skipped_count = human_skipped
    else:
        approved_count = persist_approved
        skipped_count = persist_skipped

    total_decisions = approved_count + skipped_count
    human_approval_rate = (approved_count / total_decisions) * 100.0 if total_decisions > 0 else 100.0

    # Fallback for total leads processed: use evaluation records length,
    # or successful runs from logs if eval log is somehow missing/smaller
    successful_runs_from_logs = total_runs - failed_runs
    total_leads_processed = max(total_leads_eval, max(0, successful_runs_from_logs))

    # Error rate
    error_rate = (failed_runs / total_runs) * 100.0 if total_runs > 0 else 0.0

    metrics = {
        "total_leads_processed": total_leads_processed,
        "avg_research_quality": round(avg_research_score, 1),
        "avg_email_quality": round(avg_email_score, 1),
        "human_approval_rate": round(human_approval_rate, 1),
        "avg_pipeline_time_s": round(avg_pipeline_time, 2),
        "error_rate": round(error_rate, 1),
        "tier_distribution": tier_counts,
        "most_common_industries": most_common_industries,
        "raw_counts": {
            "total_runs": total_runs,
            "failed_runs": failed_runs,
            "approved": approved_count,
            "skipped": skipped_count
        }
    }
    return metrics


def save_metrics(metrics: dict, output_path: str | Path = METRICS_FILE) -> None:
    """Save computed metrics to metrics.json file."""
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"[!] Failed to write metrics to {output_path}: {e}")


def print_dashboard(metrics: dict) -> None:
    """Print a clean, premium ASCII box dashboard based on computed metrics."""
    total_leads = metrics.get("total_leads_processed", 0)
    avg_res = metrics.get("avg_research_quality", 0.0)
    avg_em = metrics.get("avg_email_quality", 0.0)
    app_rate = metrics.get("human_approval_rate", 100.0)
    avg_time = metrics.get("avg_pipeline_time_s", 0.0)
    err_rate = metrics.get("error_rate", 0.0)

    # Format values with spacing and symbols
    total_str = f"{total_leads}"
    res_str = f"{round(avg_res)}%"
    em_str = f"{round(avg_em)}%"
    app_str = f"{round(app_rate)}%"
    time_str = f"{avg_time:.1f}s"
    err_str = f"{round(err_rate)}%"

    print("\n")
    print("   ╔══════════════════════════════════════╗")
    print("   ║   SALES PIPELINE — MONITOR DASHBOARD ║")
    print("   ╠══════════════════════════════════════╣")
    print(f"   ║  Total Leads Processed : {total_str:<12} ║")
    print(f"   ║  Avg Research Quality  : {res_str:<12} ║")
    print(f"   ║  Avg Email Quality     : {em_str:<12} ║")
    print(f"   ║  Human Approval Rate   : {app_str:<12} ║")
    print(f"   ║  Avg Pipeline Time     : {time_str:<12} ║")
    print(f"   ║  Error Rate            : {err_str:<12} ║")
    print("   ╚══════════════════════════════════════╝")

    # Tier distribution section
    tiers = metrics.get("tier_distribution", {"Hot": 0, "Warm": 0, "Cold": 0})
    print("\n   [ TIER DISTRIBUTION ]")
    print(f"   🔥 Hot  : {tiers.get('Hot', 0)} lead(s)")
    print(f"   🌤️ Warm : {tiers.get('Warm', 0)} lead(s)")
    print(f"   ❄️ Cold : {tiers.get('Cold', 0)} lead(s)")

    # Industries section
    industries = metrics.get("most_common_industries", [])
    print("\n   [ MOST COMMON INDUSTRIES ]")
    if not industries:
        print("   (No industry data available yet)")
    else:
        for idx, (ind, count) in enumerate(industries, 1):
            print(f"   🏢 {idx}. {ind} ({count})")
    print("\n")
