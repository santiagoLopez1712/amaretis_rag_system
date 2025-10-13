# brief_generator_agent.py (Versión final con Vertex AI y memoria corregida)

import os
import logging
import json
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada.")

REGION = os.getenv("GOOGLE_CLOUD_REGION")
if not REGION:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_REGION no está configurada.")

class BriefGeneratorAgent:
    name = "brief_generator_agent"
    
    def __init__(self, vectorstore, temperature: float = 0.7):
        self.llm = ChatVertexAI(project=PROJECT_ID, location=REGION, model="gemini-2.5-pro", temperature=temperature)
        self.vectorstore = vectorstore
        self.tools = self._setup_tools()
        self.agent: Optional[AgentExecutor] = self._create_agent()

    def _tool_search_similar_campaigns(self, query: str) -> str:
        """Busca campañas similares en la base de conocimiento."""
        if not self.vectorstore:
            return "Error: Base de datos de conocimiento (vectorstore) no disponible."
        try:
            docs = self.vectorstore.similarity_search(query, k=3, filter={"type": "campaign"})
            if not docs:
                return "No se encontraron campañas similares."
            results = [{
                "content_summary": doc.page_content[:200] + "...",
                "client": doc.metadata.get("client", "N/A"),
                "campaign_type": doc.metadata.get("campaign_type", "N/A"),
            } for doc in docs]
            return json.dumps(results, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error en la herramienta search_similar_campaigns: {e}")
            return f"Error buscando casos similares: {e}"

    def _tool_analyze_target_segment(self, client_info: str) -> str:
        """Analiza información del cliente y sugiere segmentación."""
        prompt = ChatPromptTemplate.from_template("""
        Analiza la siguiente información del cliente y sugiere:
        1. Segmentación de target principal y secundario
        2. Insights demográficos y psicográficos relevantes
        3. Canales de comunicación recomendados
        4. Tono de comunicación sugerido
        
        Cliente: {client_info}
        
        Responde en formato estructurado para usar en un brief.
        """)
        try:
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"client_info": client_info})
            return response
        except Exception as e:
            logger.error(f"Error en la herramienta analyze_target_segment: {e}")
            return f"Error en análisis de target: {e}"

    def _tool_generate_smart_objectives(self, campaign_info: str) -> str:
        """Genera objetivos SMART basado en info de campaña."""
        prompt = ChatPromptTemplate.from_template("""
        Basándote en esta información de campaña, genera 3-5 objetivos SMART (Specific, Measurable, Achievable, Relevant, Time-bound):
        
        Información: {campaign_info}
        
        Formato:
        1. [Objetivo específico con métrica y plazo]
        2. [Objetivo específico con métrica y plazo]
        etc.
        
        Enfócate en objetivos de marketing realistas y medibles.
        """)
        try:
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"campaign_info": campaign_info})
            return response
        except Exception as e:
            logger.error(f"Error en la herramienta generate_smart_objectives: {e}")
            return f"Error generando objetivos: {e}"
        
    def _setup_tools(self) -> List[Tool]:
        """Configura las herramientas usando los métodos de la clase."""
        tools = [
            Tool(
                name="search_similar_campaigns",
                func=self._tool_search_similar_campaigns,
                description="Busca campañas similares exitosas en la base de conocimiento para encontrar referencias y best practices."
            ),
            Tool(
                name="analyze_target_segment", 
                func=self._tool_analyze_target_segment,
                description="Analiza información de un cliente y genera recomendaciones de segmentación y targeting para el brief."
            ),
            Tool(
                name="generate_smart_objectives",
                func=self._tool_generate_smart_objectives,
                description="Genera objetivos SMART específicos y medibles basados en la información de la campaña."
            )
        ]
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Crea el AgentExecutor para el brief generator."""
        prompt = ChatPromptTemplate.from_template("""
        Eres un "Planificador Táctico" experto en AMARETIS. Tu única función es recibir un objetivo y generar un documento de briefing de marketing detallado y estructurado.

        **PROCESO OBLIGATORIO:**
        1. Analiza la petición del usuario para extraer el cliente, el objetivo y el presupuesto.
        2. Usa `search_similar_campaigns` para encontrar inspiración y casos de éxito.
        3. Usa `analyze_target_segment` para definir la audiencia.
        4. Usa `generate_smart_objectives` para crear los KPIs.
        5. Sintetiza toda la información en un único documento de briefing con la siguiente estructura: Executive Summary, Client Background, Target Audience, Campaign Objectives (SMART), Key Messages, Recommended Channels, KPIs, y Timeline.
        
        **HERRAMIENTAS:**
        {tools}

        **FORMATO DE PENSAMIENTO:**
        Thought: [Tu razonamiento siguiendo el proceso paso a paso.]
        Action: [Herramienta a usar de [{tool_names}]]
        Action Input: [Input para la herramienta.]
        Observation: [Resultado de la herramienta.]
        ... (puedes repetir este ciclo) ...
        Thought: Ya tengo toda la información necesaria para construir el brief final.
        Final Answer: [El documento de briefing completo y bien estructurado.]

        **INICIA AHORA**
        
        Historial de Chat: {history}
        Petición Actual: {input}
        Tu Gedankengang:
        {agent_scratchpad}
        """)
        
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=90
        )
        return executor
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Punto de entrada para LangGraph que pasa correctamente el historial."""
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: La solicitud para generar el brief está vacía."}
        try:
            # Pasa correctamente el historial que recibe del supervisor.
            history = input_dict.get("history", [])
            result = self.agent.invoke({"input": user_input, "history": history})
            final_output = result.get("output", str(result))
            return {"output": final_output}
        except Exception as e:
            logger.error(f"Error en la invocación del Brief Generator Agent: {e}")
            return {"output": f"Fehler bei Brief-Generierung: {e}"}

def create_brief_generator_agent(vectorstore) -> BriefGeneratorAgent:
    """Función de fábrica para crear una instancia del agente."""
    return BriefGeneratorAgent(vectorstore)