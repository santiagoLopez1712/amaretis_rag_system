from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import tool

# .env-Datei laden
load_dotenv()

@tool
def web_search_tool(query: str):
    """Führt eine Websuche durch und gibt Inhalt und Quelle zurück."""
    try:
        search = TavilySearchResults(max_results=3)
        results = search.invoke(query)
        
        if results and len(results) > 0:
            result = results[0]
            content = result.get("content", "")
            source = result.get("url", "Quelle unbekannt")
            return content, source
        else:
            return "Keine Ergebnisse gefunden.", "Quelle unbekannt"
    except Exception as e:
        return f"Fehler bei der Websuche: {e}", "Fehler"

# Modell initialisieren
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    temperature=0.7
)

def store_answer_and_source(question, answer, source):
    """Speichert Antwort und Quelle in Datei"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open("answers_and_sources.txt", "a", encoding="utf-8") as file:
            file.write(f"Timestamp: {timestamp}\n")
            file.write(f"Question: {question}\n")
            file.write(f"Answer: {answer}\n")
            file.write(f"Source: {source}\n")
            file.write("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")

# Agent erstellen
research_agent = create_react_agent(
    model=llm,
    tools=[web_search_tool],
    prompt=(
        "Du bist ein Recherche-Agent für AMARETIS Marketing.\n\n"
        "ANWEISUNGEN:\n"
        "- Helfe NUR bei recherche-bezogenen Aufgaben\n"
        "- Fokus auf Marketing, Kommunikation und Markttrends\n"
        "- Antworte DIREKT mit den Ergebnissen deiner Arbeit\n"
        "- Speichere Antworten mit Quellen nach der Recherche\n"
    ),
)
research_agent.name = "research_agent"

def ask_question_and_save_answer(question):
    """Stellt Frage, speichert Antwort und gibt sie zurück"""
    try:
        # Web-Suche ausführen
        answer, source = web_search_tool.invoke(question)
        
        # Die Antwort speichern
        store_answer_and_source(question, answer, source)
        
        return answer, source
    except Exception as e:
        error_msg = f"Fehler bei der Abfrage: {e}"
        store_answer_and_source(question, error_msg, "Fehler")
        return error_msg, "Fehler"

if __name__ == "__main__":
    # Test
    print("🔍 Web Search Agent Test")
    question = "Was sind aktuelle Marketing-Trends 2024?"
    answer, source = ask_question_and_save_answer(question)
    print(f"Antwort: {answer}")
    print(f"Quelle: {source}")