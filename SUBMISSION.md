# Submission Details — Sales Pipeline Agent

- **Track:** Agents for Business
- **YouTube Video Link:** [TO BE ADDED]

## 🌟 Key Concepts Demonstrated

Our submission showcases five key architectural and developer experience patterns defined in the AI Agents framework:

1. **Multi-Agent Orchestration & Communication:** Coordinates three specialized agents (`LeadResearcher`, `LeadScorer`, and `OutreachDrafter`) passing structured state (intelligence brief, rubrics, drafts) down the pipeline using the Google Agent Development Kit (ADK) event-based runtime.
2. **Model Context Protocol (MCP) Integration:** Implements a fully decoupled CRM database server using `FastMCP` over HTTP/Server-Sent Events (SSE). This separates business data storage from agent cognition, supporting transactional safety (atomic tmp writes) and idempotency.
3. **Agent Skills & Self-Discovery:** Provides the `--agent-skill` CLI capability that dynamically prints a complete summary of the system’s abilities, model selections, tools, andCopywriting heuristic parameters without initiating execution.
4. **Production-Ready Enterprise Security (DLP & Sanitization):** Embeds custom validation to block prompt injections, rate limits batch sizes, and runs dynamic Data Loss Prevention (DLP) regex scanners over generated output to block accidental API key or secret exfiltration.
5. **Robust API Fallbacks & Rate Limit Resilience:** Built-in automatic retry with exponential backoff for Gemini API rate limits (`429` / `503`), combined with offline pre-defined stubs for quick-testing, and a free web scraping search fallback (using DuckDuckGo Lite and BeautifulSoup) for unknown companies.

## 🚀 What Makes This Project Unique

Most AI agent hackathon entries are fragile script wrappers that break easily under rate limits, leak secrets, or make single-shot monolithic prompt queries. This project is unique because it is built like a production-grade, highly resilient B2B enterprise service. By separating data persistence behind a decoupled FastMCP boundary, implementing multi-dimensional heuristic lead-scoring alongside copywriting constraints, and building multi-tiered API resilience (combining automatic retries, free DuckDuckGo web scraping, and robust offline templates), it guarantees 100% execution success and database consistency under real-world network and quota constraints.
