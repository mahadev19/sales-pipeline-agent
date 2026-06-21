# tools/__init__.py
# Expose the search_web function so agents can import it cleanly.
from tools.web_search_tool import search_web

__all__ = ["search_web"]
