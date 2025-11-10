import asyncio
import time
import google.generativeai as genai
import os
import json
import re
from app.tools.weather import WeatherTool
from app.tools.wikipedia import WikipediaTool

class ReasoningAgent:
    def __init__(self):
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.0 Flash specifically
        try:
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            print("Successfully loaded model: Gemini 2.0 Flash")
        except Exception as e:
            print(f"Failed to load Gemini 2.0 Flash: {e}")
            # Fallback to other models
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    print(f"Successfully loaded fallback model: {model_name}")
                    break
                except Exception as e2:
                    print(f"Failed to load model {model_name}: {e2}")
                    continue
        
        if not self.model:
            raise Exception("Could not load any Gemini model")
            
        self.weather_tool = WeatherTool()
        self.wikipedia_tool = WikipediaTool()
        
        # Rate limiting variables
        self.last_request_time = 0
        self.min_request_interval = 5.0  # 5 seconds between requests
        self.recent_errors = 0
        
    async def _rate_limit(self):
        """Add delay between requests with error cooldown"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Longer delay if we recently had errors
        base_interval = self.min_request_interval
        if self.recent_errors > 0:
            base_interval = 10.0  # 10 seconds after errors
        
        if time_since_last_request < base_interval:
            wait_time = base_interval - time_since_last_request
            print(f"Rate limiting: Waiting {wait_time:.2f} seconds before next request")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        
    async def process_query(self, query: str) -> dict:
        """Main method to process user queries with reasoning and tool usage"""
        
        # Step 1: Try to analyze query with Gemini, but fallback to keyword analysis if it fails
        try:
            await self._rate_limit()  # Add delay before first Gemini call
            reasoning_plan = await self._analyze_query(query)
            self.recent_errors = max(0, self.recent_errors - 1)  # Reduce error count on success
        except Exception as e:
            print(f"Analysis failed, using keyword fallback: {e}")
            self.recent_errors += 1
            reasoning_plan = await self._keyword_analysis(query)
        
        # Step 2: Execute the plan (call external APIs if needed)
        external_data = await self._execute_plan(reasoning_plan, query)
        
        # Step 3: Generate final response with reasoning
        try:
            await self._rate_limit()  # Add delay before second Gemini call
            final_response = await self._generate_final_response(query, reasoning_plan, external_data)
            self.recent_errors = max(0, self.recent_errors - 1)  # Reduce error count on success
        except Exception as e:
            print(f"Final response failed, using fallback: {e}")
            self.recent_errors += 1
            final_response = await self._fallback_response(query, reasoning_plan, external_data)
        
        return final_response
    
    async def _keyword_analysis(self, query: str) -> dict:
        """Fallback analysis using keyword matching when Gemini fails"""
        query_lower = query.lower()
        
        needs_weather = any(word in query_lower for word in [
            'weather', 'temperature', 'forecast', 'rain', 'snow', 
            'cloud', 'humid', '°c', '°f', 'degrees'
        ])
        
        needs_wikipedia = any(word in query_lower for word in [
            'who', 'what is', 'when was', 'history of', 'invented', 
            'discovered', 'tell me about', 'explain', 'biography'
        ])
        
        return {
            "needs_weather": needs_weather,
            "needs_wikipedia": needs_wikipedia,
            "reasoning": "Using keyword-based analysis due to API limitations"
        }
    
    async def _fallback_response(self, query: str, plan: dict, external_data: dict) -> dict:
        """Generate response without Gemini when API fails"""
        
        # Build answer from available data
        answer_parts = []
        
        if external_data.get("weather") and "error" not in external_data["weather"]:
            weather = external_data["weather"]
            answer_parts.append(
                f"Weather in {weather.get('location', 'the location')}: "
                f"{weather.get('description', 'unknown')}, {weather.get('temperature', 'unknown')}°C"
            )
        
        if external_data.get("wikipedia") and "error" not in external_data["wikipedia"]:
            wiki = external_data["wikipedia"]
            answer_parts.append(
                f"Wikipedia: {wiki.get('summary', 'Information available')}"
            )
        
        if not answer_parts:
            answer_parts.append(f"I understand you're asking about: {query}")
        
        return {
            "reasoning": "Processed with fallback mode due to API rate limits",
            "answer": " | ".join(answer_parts)
        }
    
    async def _analyze_query(self, query: str) -> dict:
        """Use Gemini to analyze the query and decide on tool usage"""
        
        system_prompt ="""
    You are a tool-using agent. You MUST use external tools for factual information.
    
    **MANDATORY RULES:**
    1. ALWAYS use Wikipedia for ANY factual query including:
       - People (living or historical)
       - Places, countries, cities
       - Historical events
       - Scientific concepts
       - Inventions and discoveries
       - Books, movies, art
       - ANY factual information
    
    2. ALWAYS use Weather API for:
       - Current weather conditions
       - Temperature queries
       - Weather forecasts
       - Climate information for specific locations
    
    3. ONLY answer directly for:
       - Conversational questions ("How are you?")
       - Math calculations
       - Programming code
       - Personal opinions
       - Creative writing
    
    **DO NOT** answer factual questions from your own knowledge. ALWAYS use Wikipedia for accuracy and verification.
    
    Respond in JSON format: {"needs_weather": boolean, "needs_wikipedia": boolean, "reasoning": string}
    """
        
        try:
            print("Making analysis request to Gemini...")
            response = self.model.generate_content(system_prompt + "\n\nUser query: " + query)
            response_text = response.text
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                print(f"Analysis result: {analysis}")
                return analysis
            else:
                # Try to parse the entire response as JSON
                analysis = json.loads(response_text)
                return analysis
                
        except Exception as e:
            # Fallback analysis if Gemini fails
            print(f"Gemini analysis failed: {e}")
            query_lower = query.lower()
            return {
                "needs_weather": any(word in query_lower for word in ['weather', 'temperature', 'forecast', 'rain', 'snow', 'cloud']),
                "needs_wikipedia": any(word in query_lower for word in ['who', 'what is', 'when was', 'history of', 'invented', 'tell me about']),
                "reasoning": f"Fallback analysis based on keywords due to error: {str(e)}"
            }
    
    async def _execute_plan(self, plan: dict, query: str) -> dict:
        """Execute the planned actions (call external APIs)"""
        external_data = {}
        
        try:
            if plan.get("needs_weather"):
                location = self._extract_location(query)
                if location:
                    print(f"Fetching weather for: {location}")
                    weather_result = await self.weather_tool.get_weather(location)
                    external_data["weather"] = weather_result
                    print(f"Weather result: {weather_result}")
                else:
                    external_data["weather_error"] = "Could not extract location from query"
            
            if plan.get("needs_wikipedia"):
                search_term = self._extract_search_term(query)
                if search_term:
                    print(f"Fetching Wikipedia info for: {search_term}")
                    wiki_result = await self.wikipedia_tool.search(search_term)
                    external_data["wikipedia"] = wiki_result
                    print(f"Wikipedia result: {wiki_result}")
                else:
                    external_data["wikipedia_error"] = "Could not extract search term from query"
                    
        except Exception as e:
            external_data["errors"] = str(e)
            print(f"Error executing plan: {e}")
            
        return external_data
    
    async def _generate_final_response(self, query: str, plan: dict, external_data: dict) -> dict:
        """Generate the final response combining reasoning and external data"""
        
        system_prompt = """
        You are a helpful AI assistant. Generate a response that includes:
        1. Your reasoning process (why you used certain tools or answered directly)
        2. A clear, helpful answer to the user's query
        3. Integration of any external data you gathered
        
        Format your response exactly as follows:

        REASONING: [Your reasoning here]
        ANSWER: [Your final answer here]

        Be concise but informative in your reasoning. If you used external APIs, mention what data you fetched.
        """
        
        # Build context from external data
        context_parts = []
        if external_data.get("weather"):
            weather = external_data["weather"]
            if "error" not in weather:
                context_parts.append(f"Weather data: {weather}")
        if external_data.get("wikipedia"):
            wiki = external_data["wikipedia"]
            if "error" not in wiki:
                context_parts.append(f"Wikipedia summary: {wiki.get('summary', 'No summary available')}")
        
        context = "\n".join(context_parts) if context_parts else "No external data available"
        
        user_prompt = f"""
        User Query: {query}
        
        My Analysis: {plan.get('reasoning', 'No specific analysis')}
        
        External Data: {context}
        
        Please provide your reasoning and final answer in the specified format.
        """
        
        try:
            print("Making final response request to Gemini...")
            response = self.model.generate_content(system_prompt + "\n\n" + user_prompt)
            response_text = response.text
            print(f"Raw Gemini response: {response_text}")
            
            # Parse the response to separate reasoning and answer
            reasoning, answer = self._parse_response(response_text)
            
            return {
                "reasoning": reasoning,
                "answer": answer
            }
            
        except Exception as e:
            print(f"Gemini response generation failed: {e}")
            # Fallback response
            fallback_reasoning = f"I analyzed your query and decided to {'use external APIs' if external_data else 'answer directly'}. Error during response generation: {str(e)}"
            fallback_answer = self._generate_fallback_answer(query, external_data)
            
            return {
                "reasoning": fallback_reasoning,
                "answer": fallback_answer
            }
    
    def _extract_location(self, query: str) -> str:
        """Extract location from query for weather API"""
        patterns = [
            r"weather in (.+?)(?:\?|$| today| now)",
            r"temperature in (.+?)(?:\?|$| today)",
            r"forecast for (.+?)(?:\?|$| today)",
            r"how.*weather.*in (.+?)(?:\?|$)",
            r"what.*weather.*in (.+?)(?:\?|$)",
            r"weather.*like.*in (.+?)(?:\?|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                location = match.group(1).strip()
                location = re.sub(r'\b(?:today|now|right now|like)\b', '', location).strip()
                return location.title()
        
        # Fallback: look for location-like words (capitalized)
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ['in', 'at', 'for'] and i + 1 < len(words):
                potential_location = words[i + 1].strip('?.!,"')
                if potential_location and potential_location[0].isupper():
                    return potential_location
        
        return None
    
    def _extract_search_term(self, query: str) -> str:
        """Extract search term for Wikipedia"""
        # Remove question marks and common phrases
        clean_query = query.strip('?.!')
        
        # Common patterns
        patterns = [
            r"who (is|was|invented|created|discovered) (.+)",
            r"what is (.+)",
            r"tell me about (.+)",
            r"explain (.+)",
            r"when was (.+)",
            r"history of (.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_query.lower())
            if match:
                return match.group(1).strip().title()
        
        # If no pattern matches, use the main noun phrase
        words = clean_query.split()
        if len(words) > 2:
            # Skip question words and return the main subject
            question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'tell', 'me', 'about', 'explain']
            main_words = [word for word in words if word.lower() not in question_words]
            if main_words:
                return ' '.join(main_words[:3]).title()
        
        return clean_query.title()
    
    def _parse_response(self, response: str) -> tuple:
        """Parse Gemini response into reasoning and answer parts"""
        # Try to find REASONING: and ANSWER: markers
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?=ANSWER:|$)', response, re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r'ANSWER:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
        
        if reasoning_match and answer_match:
            reasoning = reasoning_match.group(1).strip()
            answer = answer_match.group(1).strip()
        else:
            # Fallback: try to split by common patterns
            if "ANSWER:" in response:
                parts = response.split("ANSWER:", 1)
                reasoning = parts[0].replace("REASONING:", "").strip()
                answer = parts[1].strip()
            elif "Answer:" in response:
                parts = response.split("Answer:", 1)
                reasoning = parts[0].replace("Reasoning:", "").strip()
                answer = parts[1].strip()
            else:
                # If no clear format, use first 2 sentences as reasoning, rest as answer
                sentences = [s.strip() for s in response.split('.') if s.strip()]
                if len(sentences) > 1:
                    reasoning = '. '.join(sentences[:2]) + '.'
                    answer = '. '.join(sentences[2:])
                else:
                    reasoning = "I processed your query using available tools and knowledge."
                    answer = response
        
        return reasoning, answer
    
    def _generate_fallback_answer(self, query: str, external_data: dict) -> str:
        """Generate a fallback answer when Gemini fails"""
        if external_data.get("weather") and "error" not in external_data["weather"]:
            weather = external_data["weather"]
            return f"The weather in {weather.get('location', 'that location')} is {weather.get('description', 'unknown')} with a temperature of {weather.get('temperature', 'unknown')}°C."
        
        if external_data.get("wikipedia") and "error" not in external_data["wikipedia"]:
            wiki = external_data["wikipedia"]
            return f"According to Wikipedia: {wiki.get('summary', 'Information not available.')}"
        
        return "I encountered an error while processing your query, but I tried my best to gather relevant information for you."