# web_such_agent.py

import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from typing import Tuple, Any

# --- Korrigierte Importe ---
# Wir verwenden die Standard-LangChain-Agentenstruktur
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents import tool # Für die Deklaration der benutzerdefinierten Tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate # Optional, aber gut für ReAct

# .env-Datei laden
load_dotenv()
logging.basicConfig(level=logging.INFO)

# 1. Modell initialisieren
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    temperature=0.7
)

# 2. Web Search Tool Definition
@tool
def web_search_tool(query: str) -> str:
    """
    Führt eine Websuche durch und gibt den relevanten Inhalt zusammen mit der Quelle zurück.
    MUSS verwendet werden, um aktuelle Informationen zu finden.
    """
    try:
        search = TavilySearchResults(max_results=3)
        # Die Tavily-Suche gibt eine Liste von Dicts zurück, die wir für das ReAct-Agenten-Schema formatieren.
        results = search.invoke(query)
        
        # Formatieren der Ergebnisse für den Agenten
        if results:
            formatted_results = "\n\n".join([
                f"Titel: {r.get('title')}\nInhalt: {r.get('content')}\nQuelle: {r.get('url')}"
                for r in results
            ])
            return formatted_results
        else:
            return "Keine aktuellen Suchergebnisse gefunden."
            
    except Exception as e:
        # Im Fehlerfall eine klare Fehlermeldung als String zurückgeben
        return f"Fehler bei der Websuche: {e}"

# 3. Prompt definieren
agent_prompt = PromptTemplate.from_template(
    """
    Du bist ein Recherche-Agent für AMARETIS Marketing. Deine Aufgabe ist es, externe Informationen und Trends zu finden.

    ANWEISUNGEN:
    - Nutze IMMER das Tool 'web_search_tool', um externe und aktuelle Informationen zu finden.
    - Analysiere die Suchergebnisse und fasse sie präzise zusammen.
    - Helfe NUR bei recherche-bezogenen Aufgaben (Markttrends, neue Technologien, allgemeine Informationen).
    - Antworte DIREKT mit den Ergebnissen deiner Arbeit.

    Frage des Benutzers: {input}
    """
)

# 4. Agenten-Logik (Runnable) erstellen
agent_runnable = create_react_agent(
    llm=llm,
    tools=[web_search_tool],
    prompt=agent_prompt,
)

# 5. AgentExecutor erstellen (DIES IST DIE ENTSCHEIDENDE KORREKTUR)
# Der AgentExecutor ist das Objekt, das von LangGraph als Knoten benötigt wird.
research_agent = AgentExecutor(
    agent=agent_runnable,
    tools=[web_search_tool],
    verbose=False,
    handle_parsing_errors=True # Fängt ReAct-Parsing-Fehler ab
)


# --- Hilfsfunktionen für den Test (entfernt aus der Hauptlogik) ---

def store_answer_and_source(question: str, answer: str, source: str):
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
        logging.error(f"Fehler beim Speichern: {e}")

# Die Testfunktion wird für den Supervisor nicht benötigt, aber zum Testen behalten wir sie.
def run_research_query(question: str) -> Tuple[str, str]:
    """Stellt Frage über AgentExecutor, speichert Antwort und gibt sie zurück"""
    try:
        # Der AgentExecutor wird mit einem dict {"input": ...} aufgerufen
        response = research_agent.invoke({"input": question})
        
        answer = response.get('output', 'Konnte keine Antwort generieren.')
        
        # Für die Quelle müssten wir die Logs des AgentExecutors parsen
        # oder das Tool-Ergebnis speichern. Hier simulieren wir die Quelle einfach.
        source = "Web Search Tool (Tavily)" 
        
        store_answer_and_source(question, answer, source)
        return answer, source
        
    except Exception as e:
        error_msg = f"Fehler bei der Abfrage: {e}"
        store_answer_and_source(question, error_msg, "Fehler")
        return error_msg, "Fehler"

if __name__ == "__main__":
    # Test
    print("🔍 Web Search Agent Test")
    question = "Was sind aktuelle Marketing-Trends 2025?"
    answer, source = run_research_query(question)
    print(f"\nAntwort:\n{answer}")
    print(f"Quelle: {source}")