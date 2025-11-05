# AI Agent Backend

A FastAPI backend that integrates Google Gemini reasoning with external APIs to create an intelligent AI agent.

## Features

- 🤖 **Google Gemini 2.5 Flash** for intelligent reasoning
- 🌤️ **OpenWeather API** for real-time weather data
- 📚 **Wikipedia API** for factual information
- 🧠 **Intelligent Decision Making** - autonomously decides when to use external APIs
- 💾 **Short-term Memory** - remembers recent queries
- 🚀 **FastAPI** backend with proper error handling

## API Endpoints

- `POST /ask` - Main endpoint for queries
- `GET /memory` - Get recent queries
- `GET /health` - Health check

## Setup

1. Clone repository and install dependencies:
```bash
pip install -r requirements.txt