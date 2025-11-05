import aiohttp
import os

class WeatherTool:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    async def get_weather(self, location: str) -> dict:
        if not self.api_key:
            return {"error": "OpenWeather API key not configured"}
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'q': location,
                    'appid': self.api_key,
                    'units': 'metric'
                }
                
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "location": data.get("name"),
                            "temperature": data["main"]["temp"],
                            "description": data["weather"][0]["description"],
                            "humidity": data["main"]["humidity"],
                            "wind_speed": data["wind"]["speed"]
                        }
                    else:
                        return {"error": f"Weather API error: {response.status}"}
                        
        except Exception as e:
            return {"error": f"Weather API call failed: {str(e)}"}