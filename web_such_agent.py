# web_such_agent.py (Versión final "Analista de Investigación de Élite")

import os
import logging
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

from langchain_google_vertexai import ChatVertexAI
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada.")

class WebSearchAgent:
    name = "research_agent"

    def __init__(self, temperature: float = 0.5):
        self.llm = ChatVertexAI(
            project=PROJECT_ID,
            model="gemini-1.5-flash-001", 
            temperature=temperature
        )
        self.tools = self._setup_tools()
        self.agent: Optional[AgentExecutor] = self._create_agent()

    def _tool_search(self, query: str) -> str:
        """Realiza una búsqueda web y devuelve una lista de URLs y títulos."""
        try:
            search = TavilySearch(max_results=7)
            results = search.invoke(query)
            if not results:
                return "No se encontraron resultados de búsqueda."
            return "\n".join([
                f"[Fuente {i+1}: {r.get('url')}] Título: {r.get('title')}"
                for i, r in enumerate(results)
            ])
        except Exception as e:
            logger.error(f"Error en la herramienta de búsqueda: {e}")
            return f"Error durante la búsqueda web: {e}"

    def _tool_scrape(self, url: str) -> str:
        """Lee y extrae el contenido de texto de una URL específica de forma segura."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if soup.body:
                text_content = soup.body.get_text(' ', strip=True)
            else:
                text_content = soup.get_text(' ', strip=True)

            if not text_content:
                return f"La URL {url} no contiene texto legible."

            return text_content[:4000]
        except Exception as e:
            logger.error(f"Error al leer la URL {url}: {e}")
            return f"Error: No se pudo leer el contenido de la URL {url}."

    def _setup_tools(self) -> List[Tool]:
        """Configura las dos herramientas especializadas."""
        return [
            Tool(
                name="web_search",
                func=self._tool_search,
                description="Úsala PRIMERO para encontrar una lista de URLs relevantes para una pregunta. Devuelve una lista de fuentes y títulos."
            ),
            Tool(
                name="scrape_website_content",
                func=self._tool_scrape,
                description="Úsala DESPUÉS de `web_search` para leer el contenido completo de una URL específica de la lista de resultados."
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        """Crea el AgentExecutor con el prompt de metodología de investigación avanzada."""
        prompt = PromptTemplate.from_template(
            """
            Eres un "Analista de Investigación de Élite". Tu trabajo es responder preguntas con la máxima precisión, profundidad y calidad, siguiendo una metodología estricta.

            **METODOLOGÍA DE INVESTIGACIÓN OBLIGATORIA (SECUENCIAL):**

            1.  **PLANIFICAR**: Usa `web_search` para obtener una lista de fuentes. Usa términos de búsqueda precisos y en el idioma local (ej. alemán para Alemania).
            2.  **SELECCIONAR Y FILTRAR**: Analiza la lista de URLs. **IGNORA directorios genéricos** (como Yelp, Semrush, agentur.de) y **PRIORIZA siempre los sitios web oficiales** de las empresas o fuentes de noticias primarias.
            3.  **LEER EN PROFUNDIDAD**: Usa la herramienta `scrape_website_content` para leer el contenido de las 2-3 fuentes de mayor calidad que seleccionaste. Hazlo **UNA URL A LA VEZ**.
            4.  **SINTETIZAR Y RESPONDER**: Tu trabajo NO está completo hasta que hayas leído varias fuentes. Una vez que tengas suficiente información, sintetiza una respuesta original y coherente. CITA cada dato con `[número]` y añade la lista de URLs en la sección "Fuentes:" al final.

            **HERRAMIENTAS:**
            {tools}

            **FORMATO DE PENSAMIENTO:**
            Thought: [Tu razonamiento siguiendo la metodología paso a paso.]
            Action: [El nombre de la herramienta. Debe ser uno de [{tool_names}]]
            Action Input: [La consulta de búsqueda o la URL a leer.]
            Observation: [El resultado de la herramienta.]
            ... (puedes repetir este ciclo varias veces para leer múltiples fuentes) ...
            Thought: Ya he leído suficiente información de fuentes de alta calidad. Estoy listo para sintetizar la respuesta final.
            Final Answer: [Tu respuesta final, bien redactada, con citas como `[1]`, `[3]` y la lista de fuentes al final.]

            **INICIA AHORA**

            Pregunta del Usuario: {input}
            Historial de Chat: {history}
            Tu Gedankengang:
            {agent_scratchpad}
            """
        )

        agent_runnable = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor(
            agent=agent_runnable,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=120,
        )

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Punto de entrada para LangGraph que pasa correctamente el historial."""
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: La consulta para el agente de investigación está vacía."}
        try:
            history = input_dict.get("history", [])
            result = self.agent.invoke({"input": user_input, "history": history})
            final_output = result.get("output", str(result))
            return {"output": final_output}
        except Exception as e:
            logger.error(f"Error en la invocación del Web Search Agent: {e}")
            return {"output": f"Fehler bei der Web-Recherche: {e}"}

research_agent = WebSearchAgent()

if __name__ == "__main__":
    print("🔍 Web Search Agent Test (Analista de Investigación de Élite)")
    question = "cuales son las 3 empresas de IA más importantes y por qué"
    
    response_dict = research_agent.invoke({"input": question})
    
    print(f"\nRespuesta del Agente:\n{response_dict.get('output')}")