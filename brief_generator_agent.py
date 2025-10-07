# brief_generator_agent.py Versión corregida y refactorizada

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # <-- 1. Importación necesaria
import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class BriefGeneratorAgent:
    name = "brief_generator_agent"
    
    def __init__(self, vectorstore, temperature: float = 0.7):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)
        self.vectorstore = vectorstore
        self.tools = self._setup_tools()
        self.agent = self._create_agent()

    # --- 2. Las funciones de las herramientas ahora son métodos de la clase ---

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
            # --- 3. Aplicamos el patrón LCEL con StrOutputParser ---
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
            # --- 4. Aplicamos el patrón LCEL con StrOutputParser ---
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
        # ... (El resto del código para _create_agent, invoke, etc. no necesita cambios)
        prompt = ChatPromptTemplate.from_template("""
        Du bist ein erfahrener Strategic Planner bei AMARETIS, spezialisiert auf die Erstellung professioneller Marketing-Briefings.
        Deine Aufgabe ist es, basierend auf Client-Informationen und Kampagnen-Requirements strukturierte, actionable Briefings zu erstellen.
        
        PROZESS:
        1. Analysiere die Client-Anfrage und identifiziere Schlüssel-Requirements.
        2. Suche nach ähnlichen erfolgreichen Kampagnen als Referenz mit `search_similar_campaigns`.
        3. Analysiere die Zielgruppe mit `analyze_target_segment`.
        4. Generiere SMART-Objectives mit `generate_smart_objectives`.
        5. Erstelle einen strukturierten Brief con todos los elementos.

        Verfügbare Tools: {tools}
        Tool-Namen: {tool_names}
        Aktuelle Anfrage: {input}
        Bisherige Schritte: {agent_scratchpad}
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
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: Entrada de usuario vacía para la generación del brief."}
        try:
            result = self.agent.invoke({"input": user_input, "history": []}) # History se puede añadir más tarde
            final_output = result.get("output", str(result))
            return {"output": final_output}
        except Exception as e:
            logger.error(f"Error en la invocación del Brief Generator Agent: {e}")
            return {"output": f"Fehler bei Brief-Generierung: {e}"}

def create_brief_generator_agent(vectorstore) -> BriefGeneratorAgent:
    return BriefGeneratorAgent(vectorstore)