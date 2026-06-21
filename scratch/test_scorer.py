import sys
from pathlib import Path

# Fix path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.lead_scorer import score_lead_dict

salesforce_research = {
    "company_name": "Salesforce",
    "overview": "Salesforce is a global leader in customer relationship management (CRM) software, helping companies connect with their customers in a whole new way.",
    "company_size": "10000+ employees (Enterprise)",
    "industry": "SaaS / CRM / Enterprise Software",
    "pain_points": [
        "High licensing costs and complex customization processes",
        "Siloed customer data across multiple cloud platforms",
        "Sales reps find the interface clunky and time-consuming to update"
    ],
    "decision_maker": {
        "name": "Marc Benioff",
        "title": "CEO",
        "linkedin_url": "https://linkedin.com/in/marcbenioff"
    },
    "search_mode": "stub"
}

hubspot_research = {
    "company_name": "HubSpot",
    "overview": "HubSpot is a leading customer relationship management (CRM) platform for scaling businesses, providing tools for marketing, sales, and customer service.",
    "company_size": "5000-10000 employees (Enterprise)",
    "industry": "SaaS / Marketing Automation / CRM",
    "pain_points": [
        "Steep price jumps as lead database and contact sizes grow",
        "Reporting limitations for complex multi-product customer journeys",
        "Integration challenges with legacy databases and custom ERPs"
    ],
    "decision_maker": {
        "name": "Yamini Rangan",
        "title": "CEO",
        "linkedin_url": "https://linkedin.com/in/yaminirangan"
    },
    "search_mode": "stub"
}

zoho_research = {
    "company_name": "Zoho CRM",
    "overview": "Zoho CRM is a cloud-based customer relationship management platform designed to help businesses manage sales, marketing, and support in a unified system.",
    "company_size": "5000-10000 employees (Enterprise)",
    "industry": "SaaS / CRM / Business Applications",
    "pain_points": [
        "Clunky and outdated UI compared to newer SaaS products",
        "Delays in customer support response times for complex issues",
        "Limited advanced reporting and custom analytics capabilities"
    ],
    "decision_maker": {
        "name": "Sridhar Vembu",
        "title": "CEO",
        "linkedin_url": "https://linkedin.com/in/sridharvembu"
    },
    "search_mode": "stub"
}

for name, res in [("Salesforce", salesforce_research), ("HubSpot", hubspot_research), ("Zoho CRM", zoho_research)]:
    score_res = score_lead_dict(res)
    print(f"Company: {name}")
    print(f"  Total Score: {score_res['score']}")
    print(f"  Breakdown: {score_res['breakdown']}")
    print(f"  Reason: {score_res['reason']}")
    print()
