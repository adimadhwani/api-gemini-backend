import aiohttp
import json

class WikipediaTool:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    
    async def search(self, search_term: str) -> dict:
        try:
            headers = {
                'User-Agent': 'AI-Agent-Backend/1.0 (https://github.com/yourusername/ai-agent-backend; your-email@example.com)'
            }
            
            async with aiohttp.ClientSession() as session:
                # Clean the search term for URL
                clean_term = search_term.replace(" ", "_")
                url = self.base_url + clean_term
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "title": data.get("title", search_term),
                            "summary": data.get("extract", "No summary available."),
                            "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                        }
                    else:
                        return {
                            "error": f"Wikipedia API error: {response.status}",
                            "search_term": search_term
                        }
                        
        except Exception as e:
            return {"error": f"Wikipedia API call failed: {str(e)}"}