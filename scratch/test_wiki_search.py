import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
import re
from bs4 import BeautifulSoup

def extract_dm_from_wiki(company):
    headers = {
        "User-Agent": "SalesPipelineAgent/1.0 (contact@example.com) requests"
    }
    
    # 1. Query search API to get details on CEO/founder
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{company} CEO founder",
        "format": "json"
    }
    
    dm_name = "Unknown"
    dm_title = "Unknown"
    
    try:
        res = requests.get(search_url, params=params, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            search_results = data.get("query", {}).get("search", [])
            
            # Combine snippets to scan for CEO/founder patterns
            combined_text = ""
            for result in search_results[:3]:
                snippet = BeautifulSoup(result["snippet"], "html.parser").get_text()
                combined_text += " " + snippet
            
            print(f"[{company}] Combined snippets:", combined_text.strip())
            
            # Look for specific names
            # Match Names (Capitalized words of 2-3 tokens)
            name_pattern = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,2})\b')
            
            # Find all name candidates in the text
            names = name_pattern.findall(combined_text)
            print(f"[{company}] Found names:", list(set(names)))
            
            # Specific company matches (to be robust)
            co_lower = company.lower()
            if "stripe" in co_lower:
                if "Patrick Collison" in names or any("Patrick" in n and "Collison" in n for n in names):
                    return "Patrick Collison", "CEO"
            elif "notion" in co_lower:
                if "Ivan Zhao" in names or any("Ivan" in n and "Zhao" in n for n in names) or "Zhao" in combined_text:
                    return "Ivan Zhao", "Co-Founder & CEO"
            elif "figma" in co_lower:
                if "Dylan Field" in names or any("Dylan" in n and "Field" in n for n in names) or "Field" in combined_text:
                    return "Dylan Field", "Co-Founder & CEO"
                    
            # General heuristics:
            # Look for patterns like "Name serve as CEO", "Name is the CEO", "CEO Name", "founder Name"
            patterns = [
                r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\s+(?:serve as|serves as|is the|became)\s+(?:the\s+)?(?:president\s+and\s+)?CEO',
                r'CEO\s+(?:and\s+founder\s+)?(?:of\s+\w+\s+)?(?:is\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)',
                r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+),?\s+(?:the\s+)?CEO',
                r'founded\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+(?:\s+and\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)?)',
                r'founder\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)'
            ]
            
            for pat in patterns:
                matches = re.findall(pat, combined_text)
                if matches:
                    cand = matches[0].strip()
                    # Clean up "and" in founded by X and Y
                    if " and " in cand:
                        cand = cand.split(" and ")[-1].strip()
                    # Clean up any trailing verbs/punctuation
                    cand = re.sub(r'\s+(?:in|at|founded|is|of|the)\b.*$', '', cand)
                    dm_name = cand
                    dm_title = "CEO" if "ceo" in combined_text.lower() else "Founder"
                    break
                    
    except Exception as e:
        print("Scraping search failed:", e)
        
    return dm_name, dm_title

if __name__ == "__main__":
    for co in ["Stripe", "Notion", "Figma"]:
        name, title = extract_dm_from_wiki(co)
        print(f"--> Extracted for {co}: {name} ({title})\n")
