# Competition: AI Agents Intensive Vibe Coding - Kaggle 2026
"""
monitor.py
----------
CLI entry point to display the sales pipeline production monitoring dashboard.
"""

import sys

# Force UTF-8 output on Windows terminals (cp1252 default can't encode many chars).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tools.monitor import compute_metrics, save_metrics, print_dashboard

def main() -> None:
    try:
        metrics = compute_metrics()
        save_metrics(metrics)
        print_dashboard(metrics)
    except Exception as e:
        sys.stderr.write(f"Error running monitor dashboard: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
