"""
tools/web_search_tool.py
------------------------
PURPOSE:
    Provides a lightweight web search capability for ADK agents.
    The lead_researcher agent uses this tool to gather company intelligence,
    recent news, LinkedIn-style profile data, and industry context for a
    given lead.

HOW IT WORKS (to be implemented):
    1.  Accepts a query string and an optional result-count parameter.
    2.  Calls a search API (e.g. SerpAPI, Google Custom Search, or the ADK
        built-in `google_search` grounding tool when used standalone).
    3.  Returns a list of result dicts, each with 'title', 'url', and 'snippet'.

TOOL CONTRACT (ADK rules):
    - Function signature must use type hints — no defaults allowed on
      parameters that the LLM will supply.
    - Return value must be JSON-serializable (dict or list).
    - Docstring is sent verbatim to the LLM as the tool description — keep it
      clear and concise.

FUTURE ENHANCEMENTS:
    - Add caching (Redis or in-memory LRU) to avoid duplicate API calls for
      the same company within a pipeline run.
    - Support filtering by date range (e.g. "news from last 30 days").
    - Integrate with LinkedIn scraping API for richer profile data.
"""

import os
import httpx
from datetime import datetime, timezone
from google.adk.tools import ToolContext

def log_action(agent_name: str, action: str, details: str = "") -> None:
    """Log tool action to agent_log.txt with a UTC ISO-8601 timestamp.
    
    SECURITY NOTE: Secure logging is critical for auditing, system debugging,
    and compliance.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    log_line = f"[{timestamp}] {agent_name} | {action} | {details}\n"
    try:
        with open("agent_log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration — load from environment or .env via python-dotenv
# ---------------------------------------------------------------------------
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")       # e.g. SerpAPI key
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")   # Google CSE ID
SEARCH_BASE_URL = "https://www.googleapis.com/customsearch/v1"


# ---------------------------------------------------------------------------
# Tool function — registered directly on the LeadResearcher agent
# ---------------------------------------------------------------------------
def search_web(query: str, num_results: int, tool_context: ToolContext) -> dict:
    """Search the web for information about a company, person, or topic.

    Use this to research a lead's company, find recent news, understand
    their product offerings, or gather any publicly available intelligence
    before crafting an outreach message.

    Args:
        query: The search query string (e.g. "TechCorp Inc latest funding round").
        num_results: Number of results to return (1-10).

    Returns:
        dict with 'status' ('success' or 'error') and 'results' list.
        Each result has 'title', 'url', and 'snippet' keys.
    """
    log_action("WebSearchTool", "TOOL_CALL", f"query={query[:50]}, results={num_results}")
    # ------------------------------------------------------------------
    # TODO: Replace this stub with a real API call.
    # Stub returns fake results so the pipeline can be tested end-to-end
    # before API credentials are configured.
    # ------------------------------------------------------------------
    if not SEARCH_API_KEY:
        from agents.lead_researcher import run_free_search
        # Extract the company name from the query (usually the first word)
        company_name = query.split()[0].replace('"', '').replace("'", "").rstrip(',:;.')
        res = run_free_search(company_name, query)
        tool_context.state["last_search_query"] = query
        return {
            "status": "success",
            "results": res["results"],
            "mode": res["mode"]
        }

    # Real implementation (uncomment and fill in when API keys are ready):
    # try:
    #     params = {
    #         "key": SEARCH_API_KEY,
    #         "cx": SEARCH_ENGINE_ID,
    #         "q": query,
    #         "num": min(num_results, 10),
    #     }
    #     response = httpx.get(SEARCH_BASE_URL, params=params, timeout=10.0)
    #     response.raise_for_status()
    #     items = response.json().get("items", [])
    #     results = [
    #         {"title": item["title"], "url": item["link"], "snippet": item["snippet"]}
    #         for item in items
    #     ]
    #     tool_context.state["last_search_query"] = query
    #     return {"status": "success", "results": results}
    # except Exception as e:
    #     return {"status": "error", "error": str(e)}

    return {"status": "error", "error": "Search not implemented yet."}
