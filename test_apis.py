# test_apis.py
import os
from dotenv import load_dotenv

load_dotenv()

# Test Google API
google_key = os.getenv("GOOGLE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

print(f"Google API Key: {'✅ Set' if google_key else '❌ Missing'}")
print(f"Tavily API Key: {'✅ Set' if tavily_key else '❌ Missing'}")

# Test Google Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
    response = llm.invoke("Hola, esto es un test")
    print("✅ Google Gemini: Funcionando")
except Exception as e:
    print(f"❌ Google Gemini Error: {e}")

# Test Tavily
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    search = TavilySearchResults(max_results=1)
    result = search.invoke("test search")
    print("✅ Tavily: Funcionando")
except Exception as e:
    print(f"❌ Tavily Error: {e}")