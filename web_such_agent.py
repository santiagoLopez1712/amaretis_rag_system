# web_such_agent.py (Versión optimizada como "Analista de Inteligencia de Mercado")

import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSearchAgent:
    """
    Agente especializado en realizar búsquedas web, sintetizar la información
    y citar las fuentes para actuar como un Analista de Inteligencia de Mercado.
    """
    name = "research_agent"

    def __init__(self, temperature: float = 0.7):
        # Usamos el modelo que has especificado. Si da problemas, 'gemini-pro' es una alternativa estable.
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temperature)
        self.tools = self._setup_tools()
        self.agent: AgentExecutor = self._create_agent()

    def _tool_web_search(self, query: str) -> str:
        """
        --- OPTIMIZACIÓN 1: Herramienta mejorada ---
        Realiza la búsqueda y formatea los resultados de manera estructurada para el LLM.
        """
        try:
            search = TavilySearchResults(max_results=3) # Obtenemos 3 fuentes para sintetizar
            results = search.invoke(query)
            
            if not results:
                return "No se encontraron resultados de búsqueda relevantes para esa consulta."

            # Formateamos los resultados con un índice claro para que el LLM pueda citarlos
            formatted_results = "\n\n".join([
                f"[Fuente {i+1}: {r.get('url')}]\n"
                f"Título: {r.get('title')}\n"
                f"Contenido: {r.get('content')}"
                for i, r in enumerate(results)
            ])
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en la herramienta de búsqueda web: {e}")
            return f"Error durante la búsqueda web: {e}"

    def _setup_tools(self) -> List[Tool]:
        """Configura las herramientas para el agente."""
        return [
            Tool(
                name="web_search_tool",
                func=self._tool_web_search,
                description="Realiza una búsqueda web para encontrar información actual, noticias y tendencias de mercado. Es la única herramienta para preguntas externas."
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        """Crea el AgentExecutor con un prompt de alta calidad para síntesis y citación."""
        
        # --- OPTIMIZACIÓN 2: Prompt completamente reescrito ---
        prompt = PromptTemplate.from_template(
            """
            Eres un "Analista de Inteligencia de Mercado" para AMARETIS. Tu misión es investigar preguntas complejas usando la web, sintetizar los hallazgos en una respuesta coherente y citar siempre tus fuentes.

            **REGLAS CRÍTICAS:**
            1.  **SINTETIZA, NO COPIES**: Lee la información de todas las fuentes (`Observation`) y escribe una respuesta fluida y original con tus propias palabras. No copies y pegues fragmentos.
            2.  **CITA MIENTRAS ESCRIBES**: Después de cada afirmación o dato que provenga de una fuente, AÑADE una cita en formato `[número]`. Ejemplo: "El mercado de IA crecerá un 20% en 2025 [1]".
            3.  **AGREGA UNA SECCIÓN DE FUENTES**: Al final de TODA tu respuesta, añade una sección llamada `Fuentes:` y lista las URLs correspondientes a cada número.
            4.  **USA MÚLTIPLES FUENTES**: Intenta basar tu respuesta en la información de varias de las fuentes proporcionadas para que sea más completa.

            **EJEMPLO DE RESPUESTA FINAL:**
            El marketing digital en 2025 se centrará en la hiper-personalización a través de la IA y el contenido de video de formato corto [1]. Además, la privacidad de los datos se volverá un pilar fundamental en la estrategia de las marcas [2]. Se espera que la inversión en marketing de influencers siga creciendo, pero con un enfoque en micro-influencers más auténticos [1, 3].

            Fuentes:
            [1] https://marketing-trends.com/2025-report
            [2] https://privacy-laws-weekly.com/analysis
            [3] https://influencer-today.com/future-of-marketing

            **WERKZEUGE**:
            {tools}

            **FORMATO DE PENSAMIENTO (Thought/Action/Observation):**
            Thought: [Tu razonamiento detallado sobre cómo abordar la pregunta del usuario.]
            Action: {tool_names}
            Action Input: [La consulta de búsqueda que enviarás a la herramienta.]
            Observation: [El resultado de la herramienta, que será proporcionado por el sistema.]
            ... (puedes repetir este ciclo si la primera búsqueda no es suficiente) ...
            Thought: [Pensamiento final donde resumes los hallazgos y preparas la respuesta sintetizada y citada.]
            Final Answer: [Tu respuesta final, bien redactada, con citas en el texto y la lista de fuentes al final, como en el ejemplo.]

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
            max_iterations=5, # Aumentamos una iteración por si necesita refinar la búsqueda
        )

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Punto de entrada estándar para LangGraph."""
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: La consulta para el agente de investigación está vacía."}
            
        try:
            # Pasamos el historial vacío por ahora, se puede integrar más adelante
            result = self.agent.invoke({"input": user_input, "history": []})
            final_output = result.get("output", str(result))
            return {"output": final_output}
        except Exception as e:
            logger.error(f"Error en la invocación del Web Search Agent: {e}")
            return {"output": f"Fehler bei der Web-Recherche: {e}"}

# --- Exportación para el Supervisor ---
research_agent = WebSearchAgent()

# --- Bloque de prueba para ejecución directa del archivo ---
if __name__ == "__main__":
    print("🔍 Web Search Agent Test (Analista de Inteligencia de Mercado)")
    question = "Was sind aktuelle Marketing-Trends 2025?"
    
    response_dict = research_agent.invoke({"input": question})
    
    print(f"\nRespuesta del Agente:\n{response_dict.get('output')}")