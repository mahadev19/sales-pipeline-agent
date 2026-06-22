# Sales Pipeline Agent 🚀
An End-to-End Autonomous AI Sales Development Representative (SDR) Pipeline Powered by Google ADK & FastMCP.
 
---

## 📌 Problem Statement
Sales development representatives (SDRs) spend over **65% of their day** on administrative and non-selling tasks. The most time-consuming bottleneck is manual prospecting:
1. **Cold Intelligence Gathering:** SDRs manually search Google, LinkedIn, and company websites to understand what a company does, its size, and its industry.
2. **Qualitative Pain Point Identification:** Sifting through public articles and job postings to identify what problems the target company actually faces.
3. **Manual Lead Scoring:** Manually prioritizing leads based on subjective criteria, leading to missed opportunities or wasted effort on low-fit prospects.
4. **Copywriting Friction:** Crafting personalized outreach messages (emails, LinkedIn notes) from scratch, which is slow and often results in generic, low-conversion templates.

This manual prospecting cycle limits pipeline velocity, raises customer acquisition costs (CAC), and lowers reply rates due to generic, low-relevance messaging.

---

## ✨ Solution Overview
The **Sales Pipeline Agent** is an autonomous B2B pipeline that automates the entire SDR prospecting workflow. Given a company name (or raw CRM lead ID), the system:
1. **Gathers Web & CRM Intelligence:** Conducts targeted web searches and fetches CRM profiles to build a comprehensive intelligence brief.
2. **Scores & Qualifies Leads Heuristically:** Evaluates the lead across company size, industry fit, pain point density, and decision-maker availability using a rigorous, multi-dimensional scoring rubric.
3. **Drafts Calibrated Outreach Copy:** Synthesizes the research and scoring signals to draft personalized, professional cold emails and LinkedIn connection notes, modulating tone based on the lead's qualification tier (Hot, Warm, Cold).
4. **Persists Automatically to CRM:** Synchronizes updates back to the CRM data store, preparing the drafts for immediate sales rep review.

---

## 🏗️ Agent Architecture

The orchestrator manages a sequential pipeline of three co-ordinated agents communicating through a shared session state, persisting all results to a decoupled Model Context Protocol (MCP) CRM server.

```
                  ┌─────────────────────────────────────────┐
                  │          Command Line / main.py         │
                  │        (Pipeline Orchestrator)          │
                  └────────────────────┬────────────────────┘
                                       │ Sanitized Inputs
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │         Shared Session State            │
                  │  {company_name, research, score, draft} │
                  └────────────────────┬────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌─────────────────┐           ┌─────────────────┐           ┌──────────────────┐
│ LeadResearcher  │           │   LeadScorer    │           │ OutreachDrafter  │
│ (Gemini 2.5)    │           │  (Gemini 2.5 /  │           │  (Gemini 2.5 /   │
│                 │           │  Deterministic) │           │    Template)     │
└────────┬────────┘           └────────┬────────┘           └────────┬─────────┘
         │                             │                             │
         │                             │                             │
         └─────────────────────────────┼─────────────────────────────┘
                                       │ MCP Boundary
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │           FastMCP CRM Server            │
                  │             (data/crm.json)             │
                  └─────────────────────────────────────────┘
```

### The 3-Agent Pipeline:

1. **Lead Researcher Agent ([agents/lead_researcher.py](file:///c:/Users/MAHADEV/sales-pipeline-agent/agents/lead_researcher.py)):**
   * **Role:** Web intelligence gathering.
   * **Operation:** Connects to the web search tool and local CRM. It conducts multiple targeted searches (overview, size, news, pain points, decision-maker) to build a unified Markdown research brief and structured JSON dataset.
2. **Lead Scorer Agent ([agents/lead_scorer.py](file:///c:/Users/MAHADEV/sales-pipeline-agent/agents/lead_scorer.py)):**
   * **Role:** Multi-dimension priority scoring (0-100).
   * **Operation:** Evaluates Company Size (max 25), Industry Fit (max 25), Pain Point Relevance (max 30), and Decision-Maker Authority/Accessibility (max 20). Automatically classifies leads into tiers: **Hot** (>= 70), **Warm** (40-69), or **Cold** (< 40).
3. **Outreach Drafter Agent ([agents/outreach_drafter.py](file:///c:/Users/MAHADEV/sales-pipeline-agent/agents/outreach_drafter.py)):**
   * **Role:** Calibrated copywriting.
   * **Operation:** Synthesizes research findings and scoring signals. Uses advanced copywriting strategies (anti-pattern blacklists, tier-aware CTAs, character limits) to produce a personalized cold email (under 150 words) and a LinkedIn connection message (under 300 characters).

---

## 🛠️ Key Concepts Demonstrated

### 1. Google ADK (Agent Development Kit)
* **Unified Agent Definitions:** Leverages `google.adk.agents.Agent` with specialized instructions and tools.
* **Orchestration:** Demonstrates both programmatic standalone executions (ideal for high-performance scripting) and sequential pipeline orchestrations via the ADK event runner loop.
* **Shared State Handoff:** Uses `output_key` configurations to automatically stream outputs between agents.

### 2. Model Context Protocol (MCP) Server
* Implemented a decoupled CRM server ([mcp_server/crm_server.py](file:///c:/Users/MAHADEV/sales-pipeline-agent/mcp_server/crm_server.py)) using `FastMCP`.
* Exposes clean, structured tools (`add_lead`, `get_lead`, `update_lead_status`, `update_lead_score`, `add_outreach_draft`) to agents.
* Utilizes **atomic writes** (using `.tmp` replacement) to prevent file corruption and supports **idempotency** (upsert instead of duplicate inserts).

### 3. Agent Skills Summary Card
* Built-in `--agent-skill` CLI flag that displays a structured, human-readable overview of what the pipeline can do, what models it runs, what tools it accesses, and the prompting copy heuristics utilized.

### 4. Enterprise-Grade Security
* **Env-Only Secrets Management:** Never hardcodes API keys. Employs `python-dotenv` and validates keys in `main.py` and agents before execution.
* **Strict Input Sanitization:** Blocks empty strings, dangerously long names (>80 chars), HTML tags, and control characters to prevent prompt injection and buffer overflow exploits.
* **Rate Limiting:** Restricts CLI invocations to a maximum of 10 companies per run to protect API budgets and external service quotas.
* **DLP (Data Loss Prevention) Secrets Scanning:** Actively scans all generated text outputs from agents for patterns matching sensitive keys (e.g. Google API keys, OpenAI tokens) before displaying or persisting results.
* **Audited Execution Logging:** Appends all agent actions, tool calls, inputs, and security alerts to [agent_log.txt](file:///c:/Users/MAHADEV/sales-pipeline-agent/agent_log.txt) with precise ISO-8601 UTC timestamps.

---

## 💻 Tech Stack
* **Framework:** Google Agent Development Kit (ADK)
* **LLM Models:** Gemini 2.5 Flash (`gemini-2.5-flash`)
* **MCP Framework:** FastMCP (SSE transport)
* **Language:** Python 3.12+
* **Dependencies:** `google-genai`, `httpx`, `python-dotenv`, `pydantic`

---

## ⚙️ Setup Instructions

### 1. Clone & Initialize Venv
Ensure Python 3.12+ is installed:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy the example environment template and add your Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GOOGLE_API_KEY=your-gemini-api-key-here
GOOGLE_GENAI_USE_VERTEXAI=False
```

### 3. Start the CRM MCP Server
Open a **separate terminal**, activate the venv, and run:
```bash
.\venv\Scripts\python.exe mcp_server/crm_server.py
```
The server will start listening on `http://localhost:8001/sse`.

### 4. Execute the Pipeline
Run the orchestrator from your main terminal:
```bash
# Display the Agent Skills Card
.\venv\Scripts\python.exe main.py --agent-skill

# Run the pipeline for a single company
.\venv\Scripts\python.exe main.py --companies "Notion"

# Run the pipeline for multiple companies (comma-separated, max 10)
.\venv\Scripts\python.exe main.py --companies "Notion,Stripe,HubSpot"

# Skip CRM persistence (local-only dry run)
.\venv\Scripts\python.exe main.py --companies "Notion" --no-crm

# Use full LLM-based outreach drafting (slower but richer)
.\venv\Scripts\python.exe main.py --companies "Notion" --no-fast
```

---

## 🔌 Running Without API Keys

The Sales Pipeline Agent is designed to be fully testable and resilient under API rate limits or in the complete absence of credentials:

1. **Gemini API Quota/Rate Limit Fallbacks (Offline Stubs):**
   * If the Gemini API key is missing or hits a `429 RESOURCE_EXHAUSTED` / `503 UNAVAILABLE` quota limit, the pipeline automatically intercepts the failure and activates robust local offline fallback profiles.
   * Pre-defined high-fidelity fallback profiles are included for standard test companies (**Salesforce**, **HubSpot**, **Zoho CRM**, and **Notion**) so that judges can see the pipeline run successfully without any API availability constraints.
2. **DuckDuckGo Free Search Company Lookup:**
   * For other companies, when `SEARCH_API_KEY` is not set and Gemini fails, the `LeadResearcher` runs a free live scraping lookup against DuckDuckGo Lite (`https://lite.duckduckgo.com/lite/`).
   * It parses DuckDuckGo result snippet and title tags to extract the company description, decision maker details (from LinkedIn or Crunchbase result titles), and industry keywords.
   * If DuckDuckGo blocks the scraper (e.g. via a CAPTCHA challenge), it falls back gracefully to a generic professional stub profile, guaranteeing 100% pipeline execution safety.
3. **What Judges Will See in Each Mode:**
   * **Full Live Mode:** Uses live Google Custom Search API and Gemini 2.5 Flash to fetch dynamic web insights, score the leads, and draft highly custom copywriting.
   * **Search Stub Mode (`SEARCH_API_KEY` missing):** Runs live Gemini model logic but uses structured search stubs. The final terminal summary table marks the company score with an asterisk `*` (e.g. `89*`) to indicate stub search data was utilized.
   * **Offline Fallback Mode:** The console logs `🌐 Using DuckDuckGo free search for {company}` and falls back to mock profiles, preserving pipeline execution, database writes, and copy outputs.

---

## 📊 Example Output
Running `python main.py --companies "Notion"` outputs the following terminal table:

```
======================================================================
  SALES PIPELINE RESULTS
======================================================================
  Company                  Score  Tier    Status        Lead ID        Email(wds)     LI(chars)     Time
  ──────────────────────────────────────────────────────────────────────────────────────────────────────
  Notion                     100  Hot     contacted     lead_004               71           229     11.4s
  ──────────────────────────────────────────────────────────────────────────────────────────────────────
  1 companies  |  Hot:1  Warm:0  Cold:0                                                       Total: 11.4s
======================================================================
```

It also logs all events to [agent_log.txt](file:///c:/Users/MAHADEV/sales-pipeline-agent/agent_log.txt):
```
[2026-06-20T18:42:13.611172+00:00] Orchestrator | PIPELINE_START | companies=Notion
[2026-06-20T18:42:13.614367+00:00] Orchestrator | STAGE_RESEARCH_START | company=Notion
[2026-06-20T18:42:13.614367+00:00] LeadResearcher | AGENT_RUN_START | company=Notion
[2026-06-20T18:42:18.072077+00:00] WebSearchTool | TOOL_CALL | query=Notion company profile products services target cu, results=3
[2026-06-20T18:42:25.005513+00:00] LeadResearcher | AGENT_RUN_SUCCESS | company=Notion, mode=live
[2026-06-20T18:42:25.012046+00:00] LeadScorer | PROGRAMMATIC_SCORE_SUCCESS | company=Notion, score=100, tier=Hot
[2026-06-20T18:42:25.012046+00:00] OutreachDrafter | TEMPLATE_DRAFT_SUCCESS | company=Notion, words=71, chars=229
[2026-06-20T18:42:25.061630+00:00] Orchestrator | STAGE_PERSIST_SUCCESS | company=Notion, lead_id=lead_004, status=contacted
[2026-06-20T18:42:25.086189+00:00] Orchestrator | PIPELINE_END | processed=1, exit_code=0
```

And saves the generated outreach drafts inside [data/crm.json](file:///c:/Users/MAHADEV/sales-pipeline-agent/data/crm.json):
```json
{
  "id": "lead_004",
  "company": "Notion",
  "name": "Akshay",
  "status": "contacted",
  "score": 100,
  "tier": "Hot",
  "email_draft": "Subject: Quick thought on Notion's Managing complex enterprise-level permissions...\n\nHi Akshay,\n\nI've been following Notion's growth...",
  "linkedin_draft": "Researching Notion's approach to Managing complex enterprise-level permissions... Thought it worth connecting."
}
```

---

## 🌐 Web UI (Streamlit Dashboard)

The project includes a full-featured browser-based dashboard built with **Streamlit**. It provides a no-code interface to run the pipeline, manage CRM leads, and explore memory — all without touching the terminal.

### Starting the Dashboard

```bash
# Activate venv first
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# Launch the Streamlit app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Dashboard Pages

| Page | Description |
|------|-------------|
| 🚀 **Run Pipeline** | Enter company names, toggle auto-approve, click Run. See live logs and results with editable email/LinkedIn drafts and Approve/Skip buttons. |
| 📋 **CRM Dashboard** | Browse all CRM leads in a sortable/filterable table. Click any lead to update its status, add notes, or change its score — all changes write back to `data/crm.json` instantly. |
| 🎯 **Priority Dashboard** | View leads ranked by the priority formula (score × 0.4 + recency + status + eval quality). Includes KPI cards, a tier bar chart, and a leads-over-time line chart. |
| 📊 **Monitoring** | Live system health metrics from `data/metrics.json` and `agent_log.txt`. Shows recent pipeline runs, tier distribution, approval rates, and an error log panel. |
| 🧠 **Memory Explorer** | Browse all leads stored in ChromaDB and run semantic similarity search to find past leads with similar pain points or industry context. |

### Screenshot
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/9eee7998-b931-4a71-bf45-0ca261f9a63d" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/8493d873-bd5b-4eda-91c7-39b70c6eb043" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/3cb70729-da3e-48ce-b730-d64eef342440" />

---


