from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from app.agents.reasoning_agent import ReasoningAgent  # This now uses Gemini
from app.memory.short_term_memory import ShortTermMemory

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Agent Backend with Gemini", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
reasoning_agent = ReasoningAgent()
memory = ShortTermMemory()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    reasoning: str
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Main endpoint that receives user queries and returns AI-generated responses
    with reasoning and external data integration using Google Gemini.
    """
    try:
        # Store query in memory
        memory.add_query(request.query)
        
        # Process query through reasoning agent
        result = await reasoning_agent.process_query(request.query)
        
        return QueryResponse(
            reasoning=result["reasoning"],
            answer=result["answer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/memory")
async def get_memory():
    """Get recent queries from short-term memory"""
    return {"recent_queries": memory.get_recent_queries()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "llm": "gemini-pro"}

@app.get("/")
async def root():
    return {"message": "AI Agent Backend Service with Gemini is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)