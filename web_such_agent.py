# web_such_agent.py (Versión con el prompt corregido)

import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchAgent:
    """
    Agente especializado en realizar búsquedas web en tiempo real para obtener
    información actual, noticias y tendencias de mercado.
    """
    name = "research_agent"

    def __init__(self, temperature: float = 0.7):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temperature)
        self.tools = self._setup_tools()
        self.agent: AgentExecutor = self._create_agent()

    def _tool_web_search(self, query: str) -> str:
        """Herramienta que realiza la búsqueda web y formatea los resultados."""
        try:
            search = TavilySearchResults(max_results=3)
            results = search.invoke(query)
            
            if not results:
                return "Keine aktuellen Suchergebnisse gefunden."

            formatted_results = "\n\n".join([
                f"Titel: {r.get('title')}\nInhalt: {r.get('content')}\nQuelle: {r.get('url')}"
                for r in results
            ])
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en la herramienta de búsqueda web: {e}")
            return f"Fehler bei der Websuche: {e}"

    def _setup_tools(self) -> List[Tool]:
        """Configura las herramientas para el agente."""
        return [
            Tool(
                name="web_search_tool",
                func=self._tool_web_search,
                description="Führt eine Websuche durch, um aktuelle Informationen, Nachrichten und Markttrends zu finden. MUSS für externe Fragen verwendet werden."
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        """Crea el AgentExecutor interno."""
        # --- CORRECCIÓN AQUÍ ---
        # El prompt ahora incluye los placeholders {tools} y {tool_names}
        prompt = PromptTemplate.from_template(
            """
            Du bist ein Recherche-Agent für AMARETIS Marketing. Deine Aufgabe ist es, externe Informationen und Trends zu finden.
            
            ANWEISUNGEN:
            - Nutze IMMER das Tool 'web_search_tool', um externe und aktuelle Informationen zu finden.
            - Analysiere die Suchergebnisse und fasse sie präzise zusammen.
            - Helfe NUR bei recherche-bezogenen Aufgaben (Markttrends, neue Technologien, allgemeine Informationen).
            - Antworte DIREKT mit den Ergebnissen deiner Arbeit.

            VERFÜGBARE TOOLS:
            {tools}

            TOOL-NAMEN:
            {tool_names}

            NUTZE DAS FOLGENDE FORMAT:
            Frage: die ursprüngliche Frage des Benutzers
            Gedanke: Deine Analyse der Frage und Entscheidung, welches Tool du nutzen musst.
            Aktion: Der Name des zu nutzenden Tools (aus [{tool_names}])
            Aktionseingabe: Die Eingabe für das Tool.
            Beobachtung: Das Ergebnis des Tools.
            ... (dieser Gedanke/Aktion/Aktionseingabe/Beobachtung kann sich wiederholen)
            Gedanke: Ich habe jetzt die endgültige Antwort.
            Endgültige Antwort: die finale Antwort auf die ursprüngliche Frage.

            BEGINN!

            Frage des Benutzers: {input}
            Gedankengang:
            {agent_scratchpad}
            """
        )

        agent_runnable = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent_runnable,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=4,
        )

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Punto de entrada estándar para LangGraph."""
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: Eingabe für Websuche leer."}
            
        try:
            result = self.agent.invoke({"input": user_input})
            final_output = result.get("output", str(result))
            return {"output": final_output}
        except Exception as e:
            logger.error(f"Error en la invocación del Web Search Agent: {e}")
            return {"output": f"Fehler bei der Web-Recherche: {e}"}

# --- Exportación para el Supervisor ---
research_agent = WebSearchAgent()

# --- Bloque de prueba para ejecución directa del archivo ---
if __name__ == "__main__":
    print("🔍 Web Search Agent Test")
    question = "Was sind aktuelle Marketing-Trends 2025?"
    
    response_dict = research_agent.invoke({"input": question})
    
    print(f"\nAntwort:\n{response_dict.get('output')}")