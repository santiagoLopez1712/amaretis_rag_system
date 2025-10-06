import os
import re
import logging
import time
import operator
from datetime import datetime
from typing import Dict, List, Tuple, Any, Annotated, TypedDict
from dotenv import load_dotenv

# --- Importaciones de LangChain/LangGraph ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable 
from langchain.agents import AgentExecutor
from langgraph.graph import StateGraph, END 

# --- Importaciones de Agentes (CORREGIDO) ---
from rag_agent import create_amaretis_rag_agent 
from web_such_agent import research_agent 
from compliance_agent import ComplianceAgent 
from data_analysis_agent import agent as data_analysis_agent 
from brief_generator_agent import BriefGeneratorAgent # Clase confirmada
# CORRECCIÓN: Importamos la clase real y le damos un alias si es necesario (MarketingPipeline)
from integrated_marketing_agent import MarketingPipeline as IntegratedMarketingAgent 

# === Configuración de logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Definición del Estado del Grafo (LangGraph State) ---
class AgentState(TypedDict):
    """Representa el estado del grafo en cada paso."""
    messages: Annotated[List[Any], operator.add] 


class SupervisorManager:
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        self.history: List[Dict[str, str]] = []
        
        self.setup_agents()
        self.setup_supervisor()
        
    def setup_agents(self):
        """Inicializa todos los agentes y establece sus nombres."""
        try:
            # --- 1. RAG Agent Setup (CORREGIDO: Captura el vectorstore) ---
            # self.rag_agent ahora es el AgentExecutor; self.rag_vectorstore es el vectorstore
            self.rag_agent, self.rag_vectorstore = create_amaretis_rag_agent(debug=False) 
            
            if not self.rag_agent:
                raise ValueError("Fallo al inicializar rag_agent.")
            self.rag_agent.name = "rag_agent" 

            # --- 2. Compliance Agent Setup ---
            self.compliance_checker = ComplianceAgent() 
            self.compliance_checker.name = "compliance_agent" 

            # --- 3. Brief Generator Agent (CORREGIDO: Usa el vectorstore) ---
            self.brief_generator_agent = BriefGeneratorAgent(
                vectorstore=self.rag_vectorstore # << USAMOS EL OBJETO OBTENIDO
            )
            self.brief_generator_agent.name = "brief_generator_agent"
            
            # --- 4. Integrated Marketing Agent (CORREGIDO: Usa el vectorstore) ---
            self.integrated_marketing_agent = IntegratedMarketingAgent(
                vectorstore=self.rag_vectorstore # << USAMOS EL OBJETO OBTENIDO
            ) 
            self.integrated_marketing_agent.name = "integrated_marketing_agent"
            
            # --- 5 & 6. Otros Agentes (Asumimos que están pre-inicializados en sus módulos) ---
            if not hasattr(research_agent, 'name'): research_agent.name = "research_agent"
            if not hasattr(data_analysis_agent, 'name'): data_analysis_agent.name = "data_analysis_agent"
            
            self.agent_names = [
                "rag_agent", "research_agent", "data_analysis_agent", 
                "compliance_agent", "brief_generator_agent", "integrated_marketing_agent"
            ]
            
            logger.info(f"Agentes inicializados: {self.agent_names}")
            
        except Exception as e:
            logger.error(f"Error al configurar agentes: {e}")
            raise
    
    def _create_agent_node(self, agent_executor: Any) -> Any:
        """
        [ADAPTADOR DEFENSIVO] Adapta un AgentExecutor/Runnable para que sea 
        compatible con el LangGraph AgentState.
        """
        def agent_node_adapter(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
            user_input = state["messages"][-1]["content"]
            agent_name = getattr(agent_executor, 'name', 'unknown_agent')

            try:
                result = agent_executor.invoke({"input": user_input})
                
                if isinstance(result, dict) and 'output' in result:
                    agent_response = result['output']
                elif isinstance(result, str):
                    agent_response = result
                else:
                    logger.error(f"Agente {agent_name} devolvió un tipo inesperado: {type(result)}.")
                    agent_response = f"Fehler in Agent {agent_name}: Unerwartetes Ergebnisformat ({type(result)})."
                
            except Exception as e:
                logger.error(f"Error durante la ejecución del agente {agent_name}: {e}")
                agent_response = f"Fehler in Agent {agent_name}. Die Anfrage konnte nicht verarbeitet werden: {e}"
            
            return {"messages": [
                {"role": "assistant", "content": agent_response, "name": agent_name}
            ]}
            
        return agent_node_adapter

    def route_question(self, state: AgentState) -> str:
        """
        Nodo de LangGraph que usa el LLM para decidir a qué agente delegar.
        """
        user_input = state["messages"][-1]["content"]
        available_agents = ", ".join(self.agent_names)

        supervisor_prompt_text = (
            "Eres el supervisor central de AMARETIS. Tu tarea es enrutar la pregunta del usuario "
            f"al agente más apropiado. Las opciones de enrutamiento deben ser exactamente: {available_agents}.\n"
            "Roles de los Agentes:\n"
            "- 'rag_agent': Documentos internos, smalltalk y preguntas generales.\n"
            "- 'research_agent': Información en tiempo real, tendencias de mercado, datos recientes.\n"
            "- 'data_analysis_agent': Análisis complejo, estadísticas, comparaciones, gráficos y datos estructurados.\n"
            "- 'brief_generator_agent': Creación de briefings estructurados basados en casos y mejores prácticas.\n"
            "- 'compliance_agent': Análisis legal de contenido, revisión de políticas (DSGVO/GDPR, UWG).\n"
            "- 'integrated_marketing_agent': Estrategia de alto nivel, planificación de campañas holísticas.\n"
            "Regla: Responde SÓLO con el nombre del agente. Prioriza 'rag_agent' si la pregunta es interna o general."
        )
        
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", supervisor_prompt_text),
            ("human", f"Enruta la siguiente pregunta: {user_input}"),
        ])
        
        try:
            messages = route_prompt.format_messages()
            response = self.llm.invoke(messages).content.strip().lower() 
            selected_agent = next((name for name in self.agent_names if name.lower() in response), 'rag_agent')
            
            logger.info(f"Router seleccionado: {selected_agent}")
            return selected_agent
        except Exception as e:
            logger.error(f"Error en el routing LLM: {e}. Defaulting to rag_agent.")
            return 'rag_agent'

    def setup_supervisor(self):
        """Configura el supervisor (LangGraph StateGraph) con los seis nodos."""
        
        workflow = StateGraph(AgentState)
        
        # 1. Definir Nodos (Agents) - Los mismos seis nodos
        workflow.add_node("rag_agent", self._create_agent_node(self.rag_agent))
        workflow.add_node("research_agent", self._create_agent_node(research_agent)) 
        workflow.add_node("data_analysis_agent", self._create_agent_node(data_analysis_agent)) 
        workflow.add_node("compliance_agent", self._create_agent_node(self.compliance_checker)) 
        workflow.add_node("brief_generator_agent", self._create_agent_node(self.brief_generator_agent)) 
        workflow.add_node("integrated_marketing_agent", self._create_agent_node(self.integrated_marketing_agent))
        
        # 2. Definir el nodo de enrutamiento (Router)
        workflow.add_node("router", self.route_question)
        
        # 3. Definir Punto de Entrada
        workflow.set_entry_point("router")
        
        # 4. Definir Transiciones Condicionales (Edges)
        route_map = {name: name for name in self.agent_names}
        
        workflow.add_conditional_edges("router", self.route_question, route_map)
        
        # Transición desde los agentes al final
        for name in self.agent_names:
            workflow.add_edge(name, END)
        
        # 5. Compilar el Grafo
        self.supervisor = workflow.compile()
    
    # --- Métodos de utilidad (Mantener process_question, log, etc. del código completo) ---

    def process_question(self, user_input: str) -> Tuple[str, str]:
        """Procesa una pregunta usando el LangGraph Supervisor"""
        try:
            initial_state = {"messages": [{"role": "user", "content": user_input, "name": "user"}]}
            result = self.supervisor.invoke(initial_state)
            
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                answer_text = last_message.get("content", "No se pudo obtener respuesta del agente.")
                source = f"Supervisor → {last_message.get('name', 'rag_agent')}" 
            else:
                answer_text = "El supervisor no pudo rutear la pregunta. Formato de respuesta inesperado."
                source = "Error de Supervisor"
            
            return answer_text, source
            
        except Exception as e:
            logger.error(f"Error crítico en el flujo del supervisor: {e}")
            return "Lo siento, hubo un error crítico en el sistema de agentes. Por favor, inténtalo de nuevo o reformula la pregunta.", "Error Crítico"
    
    def is_insufficient(self, answer: str, user_input: str = "") -> bool:
        """Verifica si la respuesta es insuficiente (falta de datos o números)"""
        if not answer or not isinstance(answer, str) or len(answer.strip()) < 10:
            return True
        
        insufficient_phrases = [
            "keine daten", "nicht verfügbar", "unbekannt", 
            "weiß ich nicht", "kann ich nicht", "no data",
            "no tengo esa información"
        ]
        if any(phrase in answer.lower() for phrase in insufficient_phrases):
            return True
        
        numeric_keywords = ["wie viel", "umsatz", "gewinn", "zahlen", "betrag", "revenue", "budget", "kosten"]
        if any(kw in user_input.lower() for kw in numeric_keywords):
            number_pattern = r"\d+(?:[\.,]\d+)*\s*(?:[kmb]|million|billion|mio|mrd)?"
            if not re.search(number_pattern, answer.lower()):
                return True
        
        return False
        
    def log_interaction(self, user_input: str, answer: str, source: str):
        """Log de interacciones con manejo de errores"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insuff_flag = "❗" if self.is_insufficient(answer, user_input) else "✅"
            
            log_entry = (
                f"\n⏰ {timestamp}\n"
                f"{insuff_flag} Frage: {user_input}\n"
                f"Antwort: {answer}\n"
                f"Quelle: {source}\n"
                f"{'-' * 60}\n"
            )
            
            with open("chat_log.txt", "a", encoding="utf-8") as f:
                f.write(log_entry)
                
        except Exception as e:
            logger.error(f"Error al escribir log: {e}")
    
    def update_history(self, user_input: str, answer: str):
        """Actualiza el historial de conversación, manteniendo solo los últimos 10 intercambios"""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})
        
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def run_interactive(self):
        """Ejecuta el loop interactivo principal para pruebas de consola"""
        print("\n--- 🧠 AMARETIS Supervisor está listo. ---")
        print("Escribe una pregunta (o 'exit' para salir). Ej: 'Welche Gruppen kooperieren bei Herne?' o 'Tendencias de marketing 2025?'")
        
        while True:
            try:
                user_input = input("\nPregunta: ").strip()
                if user_input.lower() in ["exit", "quit", "salir"]:
                    print("¡Hasta luego!")
                    break
                
                if not user_input:
                    continue
                
                print("\nProcesando...")
                start_time = time.time()
                answer_text, source = self.process_question(user_input)
                end_time = time.time()
                
                print(f"\nRespuesta:\n{answer_text}")
                print(f"\nFuente: {source} (Tiempo: {end_time - start_time:.2f}s)")
                
                # Verificación de suficiencia
                if self.is_insufficient(answer_text, user_input):
                    print("\n⚠️ La respuesta parece incompleta/insuficiente. El sistema registrará esto.")
                
                # Compliance Check 
                try:
                    compliance_result = self.compliance_checker.audit_content(
                        content=answer_text,
                        content_type=source 
                    ) 
                    
                    final_warning = str(compliance_result)
                    print(f"\n⚖️ Compliance Check:\n{final_warning}")
                except Exception as e:
                    logger.warning(f"Error en Compliance Agent: {e}")
                
                # Actualizar historial y log
                self.update_history(user_input, answer_text)
                self.log_interaction(user_input, answer_text, source)
                
            except KeyboardInterrupt:
                print("\n\nProceso interrumpido por el usuario. ¡Hasta luego!")
                break
            except Exception as e:
                logger.error(f"Error inesperado en loop interactivo: {e}")
                print(f"Error inesperado: {e}")

# === Función principal ===
def main():
    try:
        supervisor_manager = SupervisorManager()
        supervisor_manager.run_interactive()
    except Exception as e:
        logger.critical(f"Error FATAL al inicializar supervisor: {e}")
        print(f"\n❌ Error crítico de inicialización. Verifique logs. Error: {e}")

if __name__ == "__main__":
    main()