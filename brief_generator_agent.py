
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import json
from typing import Dict, List, Optional

class BriefGeneratorAgent:
    """
    Agente especializado en generar briefings de marketing
    basado en casos similares y mejores prácticas
    """
    
    def __init__(self, vectorstore, temperature: float = 0.7):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            temperature=temperature
        )
        self.vectorstore = vectorstore
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Configura herramientas específicas para brief generation"""
        
        tools = []
        
        # Tool 1: Búsqueda de casos similares
        def search_similar_campaigns(query: str) -> str:
            """Busca campañas similares en la base de conocimiento"""
            try:
                docs = self.vectorstore.similarity_search(
                    query, 
                    k=5,
                    filter={"type": "campaign"}  # Solo campañas
                )
                results = []
                for doc in docs:
                    metadata = doc.metadata
                    results.append({
                        "content": doc.page_content[:200],
                        "client": metadata.get("client", "Unknown"),
                        "campaign_type": metadata.get("campaign_type", "Unknown"),
                        "results": metadata.get("results", "Unknown")
                    })
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Error buscando casos similares: {e}"
        
        tools.append(Tool(
            name="search_similar_campaigns",
            func=search_similar_campaigns,
            description=(
                "Busca campañas similares exitosas en la base de conocimiento. "
                "Útil para encontrar referencias y best practices para el brief."
            )
        ))
        
        # Tool 2: Análisis de target y segmentación
        def analyze_target_segment(client_info: str) -> str:
            """Analiza información del cliente y sugiere segmentación"""
            prompt = f"""
            Analiza la siguiente información del cliente y sugiere:
            1. Segmentación de target principal y secundario
            2. Insights demográficos y psicográficos relevantes
            3. Canales de comunicación recomendados
            4. Tono de comunicación sugerido
            
            Cliente: {client_info}
            
            Responde en formato estructurado para usar en brief.
            """
            try:
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                return f"Error en análisis de target: {e}"
        
        tools.append(Tool(
            name="analyze_target_segment", 
            func=analyze_target_segment,
            description=(
                "Analiza información del cliente y genera recomendaciones "
                "de segmentación y targeting para incluir en el brief."
            )
        ))
        
        # Tool 3: Generador de objetivos SMART
        def generate_smart_objectives(campaign_info: str) -> str:
            """Genera objetivos SMART basado en info de campaña"""
            prompt = f"""
            Basándote en esta información de campaña, genera 3-5 objetivos SMART 
            (Specific, Measurable, Achievable, Relevant, Time-bound):
            
            Información: {campaign_info}
            
            Formato:
            1. [Objetivo específico con métrica y plazo]
            2. [Objetivo específico con métrica y plazo]
            etc.
            
            Enfócate en objetivos de marketing realistas y medibles.
            """
            try:
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                return f"Error generando objetivos: {e}"
        
        tools.append(Tool(
            name="generate_smart_objectives",
            func=generate_smart_objectives,
            description=(
                "Genera objetivos SMART específicos y medibles "
                "basados en la información de la campaña."
            )
        ))
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Crea el agente ReAct para brief generation"""
        
        prompt = ChatPromptTemplate.from_template("""
Du bist ein erfahrener Strategic Planner bei AMARETIS, spezialisiert auf die Erstellung 
professioneller Marketing-Briefings.

Deine Aufgabe ist es, basierend auf Client-Informationen und Kampagnen-Requirements 
strukturierte, actionable Briefings zu erstellen.

PROZESS:
1. Analysiere die Client-Anfrage und identifiziere Schlüssel-Requirements
2. Suche nach ähnlichen erfolgreichen Kampagnen als Referenz  
3. Analysiere Target-Segmentierung für den spezifischen Client
4. Generiere SMART-Objectives basierend auf den Anforderungen
5. Erstelle einen strukturierten Brief mit allen Elementen

BRIEF-STRUKTUR:
- Executive Summary
- Client Background & Situation
- Target Audience & Segmentation  
- Campaign Objectives (SMART)
- Key Messages & Positioning
- Recommended Channels & Tactics
- Success Metrics & KPIs
- Timeline & Milestones
- References (ähnliche Kampagnen)

Verfügbare Tools: {tools}
Tool-Namen: {tool_names}

Aktuelle Anfrage: {input}
Bisherige Schritte: {agent_scratchpad}

Nutze dieses Format:
Thought: [Deine Analyse der Brief-Anfrage]
Action: [Tool-Name]  
Action Input: [Tool-Input]
Observation: [Tool-Ergebnis]
... (wiederhole bei Bedarf)
Thought: Ich habe alle Informationen für einen vollständigen Brief.
Final Answer: [Strukturierter Marketing-Brief]
""")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools, 
            prompt=prompt
        )
        
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=60
        )
        
        executor.name = "brief_generator_agent"
        return executor
    
    def generate_brief(
        self, 
        client_info: str, 
        campaign_requirements: str,
        additional_context: str = ""
    ) -> str:
        """Genera un brief completo basado en la información proporcionada"""
        
        input_text = f"""
        CLIENT INFORMATION:
        {client_info}
        
        CAMPAIGN REQUIREMENTS:
        {campaign_requirements}
        
        ADDITIONAL CONTEXT:
        {additional_context}
        
        Bitte erstelle einen vollständigen Marketing-Brief basierend auf diesen Informationen.
        """
        
        try:
            result = self.agent.invoke({"input": input_text})
            return result.get("output", result)
        except Exception as e:
            return f"Fehler bei Brief-Generierung: {e}"

# Función de conveniencia para crear el agente
def create_brief_generator_agent(vectorstore) -> BriefGeneratorAgent:
    """Crea instancia del Brief Generator Agent"""
    return BriefGeneratorAgent(vectorstore)