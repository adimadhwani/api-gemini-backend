import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key present: {bool(api_key)}")

if api_key:
    genai.configure(api_key=api_key)
    
    # Test Gemini 2.0 Flash specifically
    try:
        print("Testing Gemini 2.0 Flash...")
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello! Please respond with 'Gemini 2.0 Flash is working!'")
        print(f"✓ Gemini 2.0 Flash response: {response.text}")
    except Exception as e:
        print(f"✗ Gemini 2.0 Flash failed: {e}")
        
    # List available Gemini models
    print("\nAvailable Gemini models:")
    for model in genai.list_models():
        if 'gemini' in model.name.lower():
            print(f" - {model.name}")