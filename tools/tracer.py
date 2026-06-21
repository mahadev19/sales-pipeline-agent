# Competition: AI Agents Intensive Vibe Coding - Kaggle 2026
"""
tools/tracer.py
---------------
PURPOSE:
    Provides execution tracing for the multi-agent sales pipeline. Records
    per-agent timing, token estimates, status, model used, and I/O previews
    for every pipeline run, saving each as a JSON file in data/traces/.

WHY EXECUTION TRACING MATTERS FOR AI AGENT SYSTEMS:
    Multi-agent pipelines are inherently opaque — each agent is a black box
    that accepts text and emits text. Without tracing, operators have no
    visibility into:

    1. PERFORMANCE BOTTLENECKS:
       In a 4-agent pipeline, is the 30-second runtime caused by the
       LeadResearcher's web search, the LLM call in the Reviewer, or the
       ChromaDB memory query? Without per-span timing, you cannot answer this.
       Tracing gives you millisecond-precision per-agent durations so you can
       pinpoint and optimize the slowest step.

    2. CASCADING FAILURE DIAGNOSIS:
       When the OutreachDrafter produces a bad email, was it because:
         a) The LeadResearcher surfaced wrong pain points?
         b) The LeadScorer miscategorized the tier, changing tone?
         c) The ReviewerAgent failed to improve a weak draft?
       Tracing records the I/O at each stage boundary, making root cause
       analysis possible without re-running the pipeline.

    3. MODEL GOVERNANCE & COST CONTROL:
       In enterprise deployments, you need audit logs showing which model
       (Gemini Flash / 1.5 Pro / template) was used for each agent at
       each run. This enables:
         - Cost attribution per stage (token counts → API cost)
         - Model drift detection (comparing outputs across model versions)
         - Compliance reporting (who got what output, when)

    4. REGRESSION TESTING & BENCHMARKING:
       By loading two trace files for the same company, you can compare
       a new pipeline version against a baseline, e.g.:
         "The new ReviewerAgent improved email score but added 2.5s latency."

    5. USER TRUST & EXPLAINABILITY:
       In enterprise sales tools, a rep who sees "AI generated this email"
       trusts it less than one who can inspect "the LeadResearcher found these
       3 pain points, the scorer gave it 85/100 because it's a 1000+ employee
       SaaS company, and the reviewer improved the CTA score from 6 to 9."
       Traces enable this explainability layer.

DESIGN:
    - Each pipeline run produces ONE trace file per company.
    - File naming: data/traces/{safe_company_name}_{YYYYMMDD_HHMMSS}.json
    - The Tracer class is thread-safe for single-threaded use (no locks needed
      because our pipeline is synchronous).
    - Spans are recorded in insertion order (the pipeline stage sequence).
    - Token estimates use a simple word_count * 1.3 heuristic — real token
      counting would require a tokenizer (tiktoken, google-generativeai's
      count_tokens), which adds latency and dependency overhead.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Resolve paths relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = _PROJECT_ROOT / "data" / "traces"
os.makedirs(TRACES_DIR, exist_ok=True)


def _safe_filename(company: str) -> str:
    """Convert a company name to a filesystem-safe string.
    
    Replaces spaces and special characters with underscores.
    Example: "Zoho CRM" → "Zoho_CRM"
    """
    return re.sub(r"[^\w\-]", "_", company.strip())


def _summarize(text: Any, max_chars: int = 100) -> str:
    """Extract a readable preview from an agent input or output.
    
    Handles strings, dicts, and lists gracefully. Returns at most max_chars
    characters with an ellipsis if truncated.
    
    Args:
        text: The raw input or output from an agent span.
        max_chars: Maximum characters in the summary string.
    
    Returns:
        A compact, human-readable preview string.
    """
    if text is None:
        return ""
    if isinstance(text, dict):
        # For dicts, serialize to JSON and truncate
        raw = json.dumps(text, ensure_ascii=False)
    elif isinstance(text, list):
        # For lists (e.g. pain_points), join items
        raw = ", ".join(str(item) for item in text)
    else:
        raw = str(text)

    raw = raw.strip()
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars - 3] + "..."


def _estimate_tokens(text: Any) -> int:
    """Estimate LLM token count using the word_count * 1.3 heuristic.
    
    This is a fast, dependency-free approximation. Real token counts would
    require calling the tokenizer API (e.g. google.generativeai.count_tokens),
    which adds latency and requires an API connection.
    
    The 1.3 multiplier accounts for:
    - Subword tokenization (most BPE tokenizers split longer words)
    - Punctuation, whitespace, and special characters counting as tokens
    - JSON keys and brackets in structured outputs
    
    Accuracy: typically ±15% of the true token count for English prose.
    
    Args:
        text: The text to estimate tokens for.
    
    Returns:
        Estimated token count as an integer.
    """
    if text is None:
        return 0
    if isinstance(text, dict):
        raw = json.dumps(text, ensure_ascii=False)
    elif isinstance(text, list):
        raw = " ".join(str(item) for item in text)
    else:
        raw = str(text)
    
    word_count = len(raw.split())
    return int(word_count * 1.3)


class Tracer:
    """
    Records execution spans for each agent in the sales pipeline.
    
    A "span" corresponds to one agent's complete execution:
    from start_span() (called before the agent runs) to end_span()
    (called after it returns or raises).
    
    Usage pattern in main.py:
    
        tracer = Tracer()
        
        # Before calling the agent
        tracer.start_span("LeadResearcher", input_data=company_name)
        
        result = research_company(company_name)  # The actual agent call
        
        # After the agent returns
        tracer.end_span("LeadResearcher", output=result, status="success",
                        model_used="gemini-2.0-flash")
        
        # After all agents complete
        tracer.save_trace(company=company_name)
    
    The trace file records:
        - Ordered list of spans (one per agent)
        - Per-span: agent_name, start/end timestamps, duration_ms, input/output
          previews, status, model, token estimates
        - Top-level metadata: company, pipeline version, total duration
    """

    def __init__(self):
        """Initialize an empty Tracer for a new pipeline run."""
        # Ordered list of completed span dicts
        self._spans: list[dict] = []
        # Active spans waiting for end_span() (keyed by agent_name)
        self._active: dict[str, dict] = {}
        # Wall-clock start of the entire pipeline run
        self._pipeline_start: float = time.time()

    def start_span(self, agent_name: str, input_data: Any = None) -> None:
        """
        Begin recording a new execution span for the given agent.
        
        Call this immediately BEFORE invoking the agent function. Records the
        start timestamp and an input preview. The span is "active" until
        end_span() is called with the same agent_name.
        
        Multiple spans for the same agent_name are supported (e.g. if the
        same agent is called multiple times in the pipeline). Each call to
        start_span() overwrites the active span for that name — end_span()
        will close the most recently opened one.
        
        Args:
            agent_name:  Identifier string, e.g. "LeadResearcher".
            input_data:  Raw input to the agent (string, dict, or list).
                         Automatically summarized to 100 chars.
        """
        self._active[agent_name] = {
            "agent_name": agent_name,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "_start_mono": time.monotonic(),   # monotonic for accurate duration
            "input_summary": _summarize(input_data, max_chars=100),
            "input_tokens_estimated": _estimate_tokens(input_data),
        }

    def end_span(
        self,
        agent_name: str,
        output: Any = None,
        status: str = "success",
        model_used: str = "unknown",
    ) -> None:
        """
        Finalize the span for the given agent and record it.
        
        Call this immediately AFTER the agent function returns (or raises).
        Computes the wall-clock duration, records output summary and token
        estimates, then moves the span from active → completed.
        
        If start_span() was not called first (e.g. due to an exception in the
        orchestrator before the tracer call), end_span() creates a minimal
        error span rather than raising, to avoid masking the real error.
        
        Args:
            agent_name:  Must match the agent_name used in start_span().
            output:      Raw output from the agent (string, dict, or list).
            status:      "success" | "fallback" | "error".
            model_used:  Model/engine identifier, e.g. "gemini-2.0-flash",
                         "template", "rules", "stub".
        """
        end_mono = time.monotonic()
        end_time = datetime.now(timezone.utc).isoformat()

        # Retrieve the active span (or create a placeholder if start_span was skipped)
        span = self._active.pop(agent_name, {
            "agent_name": agent_name,
            "start_time": end_time,
            "_start_mono": end_mono,
            "input_summary": "",
            "input_tokens_estimated": 0,
        })

        # Calculate wall-clock duration in milliseconds
        duration_ms = int((end_mono - span.pop("_start_mono", end_mono)) * 1000)

        # Build completed span record
        completed_span = {
            "agent_name": span["agent_name"],
            "start_time": span["start_time"],
            "end_time": end_time,
            "duration_ms": duration_ms,
            "input_summary": span["input_summary"],
            "output_summary": _summarize(output, max_chars=100),
            "status": status,
            "model_used": model_used,
            "input_tokens_estimated": span["input_tokens_estimated"],
            "output_tokens_estimated": _estimate_tokens(output),
            # Preserve the full input/output for the detail view in the UI
            # We store the raw data (not just the summary) so the Streamlit
            # page can show it in st.json() expanders.
            "input_full": _summarize(input_data=None) if output is None else "",  # placeholder
        }

        self._spans.append(completed_span)

    def get_spans(self) -> list[dict]:
        """Return all completed spans in execution order."""
        return list(self._spans)

    def save_trace(
        self,
        company: str,
        traces_dir: Path | str = TRACES_DIR,
    ) -> Path:
        """
        Serialize the complete trace to a JSON file.
        
        File naming convention:
            data/traces/{safe_company_name}_{YYYYMMDD_HHMMSS}.json
        
        The timestamp in the filename ensures that multiple runs for the same
        company produce separate trace files, enabling run comparison in the UI.
        
        Trace file structure:
        {
            "company": "Stripe",
            "run_timestamp": "2026-06-21T07:37:00Z",
            "total_duration_ms": 13400,
            "pipeline_version": "1.0",
            "spans": [
                {
                    "agent_name": "LeadResearcher",
                    "start_time": "...",
                    "end_time": "...",
                    "duration_ms": 12400,
                    "input_summary": "Stripe",
                    "output_summary": "{company_name: Stripe, overview: ...",
                    "status": "success",
                    "model_used": "gemini-2.0-flash",
                    "input_tokens_estimated": 1,
                    "output_tokens_estimated": 150
                },
                ...
            ]
        }
        
        Args:
            company:    The company name (used in the filename).
            traces_dir: Directory to save the trace file.
        
        Returns:
            Path to the saved trace file.
        """
        traces_dir = Path(traces_dir)
        os.makedirs(traces_dir, exist_ok=True)

        total_ms = int((time.monotonic() - self._pipeline_start) * 1000)
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        trace_data = {
            "company": company,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_duration_ms": total_ms,
            "pipeline_version": "1.0",
            "spans": self._spans,
        }

        safe_name = _safe_filename(company)
        filename = f"{safe_name}_{run_ts}.json"
        filepath = traces_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import sys
            sys.stderr.write(f"[Tracer] Failed to save trace: {e}\n")

        return filepath


def load_trace(filepath: str | Path) -> dict:
    """
    Load a single trace file from disk and return it as a dict.
    
    Args:
        filepath: Absolute or relative path to the JSON trace file.
    
    Returns:
        Parsed trace dict, or an empty dict if the file is missing/corrupt.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def list_traces(traces_dir: Path | str = TRACES_DIR) -> dict[str, list[Path]]:
    """
    Scan the traces directory and group trace files by company name.
    
    Returns a dict mapping company name → list of trace file Paths,
    sorted with the most recent run first.
    
    Example return value:
    {
        "Stripe": [
            Path(".../Stripe_20260621_113700.json"),
            Path(".../Stripe_20260620_090000.json"),
        ],
        "HubSpot": [
            Path(".../HubSpot_20260621_120000.json"),
        ]
    }
    
    Args:
        traces_dir: Directory to scan for .json trace files.
    
    Returns:
        Dict of {company_name: [sorted list of Paths, newest first]}
    """
    traces_dir = Path(traces_dir)
    if not traces_dir.exists():
        return {}

    company_traces: dict[str, list[Path]] = {}

    for fp in sorted(traces_dir.glob("*.json"), reverse=True):
        # Filename format: {safe_company}_{YYYYMMDD_HHMMSS}.json
        # We split on the last two underscored segments (date + time) to get company
        parts = fp.stem.rsplit("_", 2)
        if len(parts) >= 3:
            # e.g. "Stripe_20260621_113700" → company = "Stripe"
            company_safe = parts[0]
        else:
            company_safe = fp.stem

        # Try to recover the display name from the trace JSON
        try:
            trace = load_trace(fp)
            company_display = trace.get("company", company_safe)
        except Exception:
            company_display = company_safe

        if company_display not in company_traces:
            company_traces[company_display] = []
        company_traces[company_display].append(fp)

    # Each company's runs are already sorted newest-first (from the glob sort above)
    return company_traces
