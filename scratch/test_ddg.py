import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
import time
from bs4 import BeautifulSoup

def test_fallback_search(company_name):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    }
    
    print("🌐 Trying DuckDuckGo...")
    time.sleep(2)
    
    query = f"{company_name} CEO founder decision maker site:linkedin.com OR site:crunchbase.com"
    url = "https://html.duckduckgo.com/html/"
    
    try:
        response = requests.get(url, params={"q": query}, headers=headers, timeout=10)
        print("DuckDuckGo Status Code:", response.status_code)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            titles = soup.select(".result__title, .result__a")
            snippets = soup.select(".result__snippet")
            print(f"Found {len(titles)} titles and {len(snippets)} snippets")
            
            for i in range(min(len(titles), 3)):
                print(f"Title {i+1}: {titles[i].get_text(strip=True)}")
                snippet_text = snippets[i].get_text(strip=True) if i < len(snippets) else "No snippet"
                print(f"Snippet {i+1}: {snippet_text}")
                
            if titles:
                return True
        else:
            print(f"DuckDuckGo search returned status {response.status_code}")
    except Exception as e:
        print("DuckDuckGo search failed:", e)
        
    print("📖 Trying Wikipedia...")
    # Try company-specific title first
    wiki_titles = [f"{company_name} (company)", company_name]
    for w_title in wiki_titles:
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{w_title.replace(' ', '_')}"
        try:
            wiki_res = requests.get(wiki_url, headers={"User-Agent": headers["User-Agent"]}, timeout=10)
            print(f"Wikipedia ({w_title}) Status Code:", wiki_res.status_code)
            if wiki_res.status_code == 200:
                data = wiki_res.json()
                # Check if it's a disambiguation page
                if data.get("type") == "disambiguation":
                    print(f"Wikipedia page '{w_title}' is a disambiguation page. Trying next...")
                    continue
                description = data.get("extract", "")
                title = data.get("title", "")
                print("Wikipedia Title:", title)
                print("Wikipedia Extract:", description)
                return True
            else:
                print(f"Wikipedia page '{w_title}' returned status:", wiki_res.status_code)
        except Exception as e:
            print(f"Wikipedia lookup for '{w_title}' failed:", e)
        
    print("⚠️ Using stub fallback")
    return False

if __name__ == "__main__":
    test_fallback_search("Stripe")
