import aiohttp
import json

class WikipediaTool:
    def __init__(self):
        self.search_url = "https://en.wikipedia.org/w/api.php"
        self.summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    
    async def search(self, search_term: str) -> dict:
        try:
            headers = {
                'User-Agent': 'AI-Agent-Backend/1.0 (https://github.com/yourusername/ai-agent-backend; your-email@example.com)',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                # First, search for the term to get the correct page title
                search_params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': search_term,
                    'format': 'json',
                    'srlimit': 1
                }
                
                async with session.get(self.search_url, params=search_params, headers=headers) as search_response:
                    if search_response.status == 200:
                        search_data = await search_response.json()
                        search_results = search_data.get('query', {}).get('search', [])
                        
                        if search_results:
                            # Get the title of the first search result
                            page_title = search_results[0]['title']
                            print(f"Found Wikipedia page: {page_title}")
                            
                            # Now get the summary using the correct page title
                            summary_url = self.summary_url + page_title.replace(" ", "_")
                            async with session.get(summary_url, headers=headers) as summary_response:
                                if summary_response.status == 200:
                                    summary_data = await summary_response.json()
                                    return {
                                        "title": summary_data.get("title", page_title),
                                        "summary": summary_data.get("extract", "No summary available."),
                                        "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")
                                    }
                                else:
                                    return {
                                        "error": f"Wikipedia summary API error: {summary_response.status}",
                                        "search_term": search_term,
                                        "page_title": page_title
                                    }
                        else:
                            return {
                                "error": f"No Wikipedia page found for: {search_term}",
                                "search_term": search_term
                            }
                    else:
                        return {
                            "error": f"Wikipedia search API error: {search_response.status}",
                            "search_term": search_term
                        }
                        
        except Exception as e:
            return {"error": f"Wikipedia API call failed: {str(e)}"}