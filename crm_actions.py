# Competition: AI Agents Intensive Vibe Coding - Kaggle 2026
"""
crm_actions.py
--------------
CLI interface to run CRM updates and timeline history directly from the terminal.

HOW MCP TOOLS ENABLE CRM AUTOMATION:
    Model Context Protocol (MCP) defines a unified, standard interface that allows
    agent systems and LLMs to dynamically query capabilities and execute actions
    against external enterprise CRM databases, separating model cognition from
    persistence layers. By exposing actions such as updating status, score, adding
    notes, and retrieving history as MCP tools, we achieve:
    
    1. Standardized Integration: Any compliant agent platform can immediately discover
       and consume these tools without writing custom wrapper endpoints.
    2. Dynamic Cognition: LLM agents can inspect lead histories, reason about past
       touches, and decide when to record new notes or advance lead pipeline status.
    3. Separation of Concerns & Security: The host process orchestrates writes and
       validations, ensuring data integrity (e.g. status ranges) is checked at the
       database boundary regardless of agent behavior.
"""

import sys
import argparse
from datetime import datetime

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mcp_server.crm_server import (
    update_lead_status,
    add_lead_note,
    update_lead_score,
    get_lead_timeline,
    archive_lead,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="crm_actions.py",
        description="CLI actions for the Sales Pipeline CRM database."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Update Status
    parser_status = subparsers.add_parser("update-status", help="Update lead status")
    parser_status.add_argument("company", type=str, help="Company name (case-insensitive)")
    parser_status.add_argument("status", type=str, help="New pipeline stage")

    # 2. Add Note
    parser_note = subparsers.add_parser("add-note", help="Add a note to a lead")
    parser_note.add_argument("company", type=str, help="Company name (case-insensitive)")
    parser_note.add_argument("note", type=str, help="The note content to append")

    # 3. Update Score
    parser_score = subparsers.add_parser("update-score", help="Update lead score")
    parser_score.add_argument("company", type=str, help="Company name (case-insensitive)")
    parser_score.add_argument("score", type=int, help="Numeric score (0-100)")
    parser_score.add_argument("reason", type=str, help="Reason/Justification for the change")

    # 4. Timeline
    parser_timeline = subparsers.add_parser("timeline", help="Get lead history/timeline")
    parser_timeline.add_argument("company", type=str, help="Company name (case-insensitive)")

    # 5. Archive Lead
    parser_archive = subparsers.add_parser("archive", help="Archive a lead")
    parser_archive.add_argument("company", type=str, help="Company name (case-insensitive)")
    parser_archive.add_argument("reason", type=str, help="Reason for archiving the lead")

    args = parser.parse_args()

    try:
        if args.command == "update-status":
            res = update_lead_status(company=args.company, status=args.status)
            if res.get("status") == "success":
                print(f"✅ Status updated successfully for {args.company}!")
                print(f"   Old Status: {res.get('old_status')}")
                print(f"   New Status: {res.get('new_status')}")
            else:
                sys.stderr.write(f"❌ Error: {res.get('error')}\n")
                sys.exit(1)

        elif args.command == "add-note":
            res = add_lead_note(company=args.company, note=args.note)
            if res.get("status") == "success":
                print(f"✅ Note added successfully to {args.company}!")
                print(f"   Note: '{res.get('note')}'")
            else:
                sys.stderr.write(f"❌ Error: {res.get('error')}\n")
                sys.exit(1)

        elif args.command == "update-score":
            res = update_lead_score(company=args.company, new_score=args.score, reason=args.reason)
            if res.get("status") == "success":
                print(f"✅ {res.get('message')}")
            else:
                sys.stderr.write(f"❌ Error: {res.get('error')}\n")
                sys.exit(1)

        elif args.command == "timeline":
            res = get_lead_timeline(company=args.company)
            if res.get("status") == "success":
                print(f"\n📋 Timeline for {args.company} (Total events: {res.get('total_events')})")
                print("=" * 75)
                for event in res.get("timeline", []):
                    ts = event.get("timestamp", "")
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        ts_formatted = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                    except Exception:
                        ts_formatted = ts
                    
                    desc = event.get("description") or event.get("note") or ""
                    event_type = event.get("type", "").upper()
                    print(f"[{ts_formatted}] {event_type:<15} | {desc}")
                print("=" * 75)
            else:
                sys.stderr.write(f"❌ Error: {res.get('error')}\n")
                sys.exit(1)

        elif args.command == "archive":
            res = archive_lead(company=args.company, reason=args.reason)
            if res.get("status") == "success":
                print(f"✅ Lead {args.company} archived successfully!")
                print(f"   Reason: {res.get('reason')}")
            else:
                sys.stderr.write(f"❌ Error: {res.get('error')}\n")
                sys.exit(1)

    except Exception as e:
        sys.stderr.write(f"❌ CLI execution error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
