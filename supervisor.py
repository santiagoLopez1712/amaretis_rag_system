# supervisor.py (Versión final con Vertex AI)

import os
import re
import logging
import time
import operator
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Annotated, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable

from rag_agent import create_amaretis_rag_agent
from web_such_agent import research_agent
from compliance_agent import ComplianceAgent
from data_analysis_agent import agent as data_analysis_agent
from brief_generator_agent import BriefGeneratorAgent
from integrated_marketing_agent import create_integrated_marketing_agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada en el archivo .env")

class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]

class SupervisorManager:

    def __init__(self):
        self.llm = ChatVertexAI(project=PROJECT_ID, model="gemini-1.5-flash-001", temperature=0.7)
        self.history: List[Dict[str, str]] = []
        self.agents = {}
        self.agent_names = []
        self.rag_vectorstore = None
        self.supervisor: Optional[Runnable] = None
        
        self.setup_agents()
        self.setup_supervisor()

    def setup_agents(self):
        try:
            rag_agent_instance, self.rag_vectorstore = create_amaretis_rag_agent(debug=False)
            if not rag_agent_instance: raise ValueError("Fallo al inicializar rag_agent.")
            self.agents[rag_agent_instance.name] = rag_agent_instance
            compliance_agent_instance = ComplianceAgent()
            self.agents[compliance_agent_instance.name] = compliance_agent_instance
            brief_agent_instance = BriefGeneratorAgent(vectorstore=self.rag_vectorstore)
            self.agents[brief_agent_instance.name] = brief_agent_instance
            integrated_agent_instance = create_integrated_marketing_agent(vectorstore=self.rag_vectorstore)
            self.agents[integrated_agent_instance.name] = integrated_agent_instance
            if not hasattr(research_agent, 'name'): research_agent.name = "research_agent"
            self.agents[research_agent.name] = research_agent
            if not hasattr(data_analysis_agent, 'name'): data_analysis_agent.name = "data_analysis_agent"
            self.agents[data_analysis_agent.name] = data_analysis_agent
            self.agent_names = list(self.agents.keys())
            logger.info(f"Agentes inicializados: {self.agent_names}")
        except Exception as e:
            logger.error(f"Error al configurar agentes: {e}", exc_info=True)
            raise

    def _create_agent_node(self, agent_executor: Any) -> Any:
        def agent_node_adapter(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
            user_input = state["messages"][-1]["content"]
            agent_name = getattr(agent_executor, 'name', 'unknown_agent')
            try:
                result = agent_executor.invoke({"input": user_input, "history": self.history})
                agent_response = result.get('output', str(result))
            except Exception as e:
                logger.error(f"Error durante la ejecución del agente {agent_name}: {e}")
                agent_response = f"Fehler in Agent {agent_name}: {e}"
            return {"messages": [{"role": "assistant", "content": agent_response, "name": agent_name}]}
        return agent_node_adapter

    def _handle_simple_query(self, user_input: str) -> Optional[str]:
        input_lower = user_input.lower().strip()
        if len(input_lower.split()) <= 3:
            greetings = ["hola", "hi", "buenos días", "buenas tardes", "hey"]
            if any(greet == input_lower for greet in greetings):
                return "¡Hola! Soy el asistente de AMARETIS. ¿En qué puedo ayudarte hoy?"
        capabilities_q = ["qué puedes hacer", "ayuda", "help", "cuáles son tus funciones"]
        if any(cap in input_lower for cap in capabilities_q):
            return (
                "Soy un sistema de IA multi-agente. Mis capacidades incluyen:\n"
                "- **rag_agent**: Responder preguntas sobre documentos internos y archivos PDF subidos.\n"
                "- **research_agent**: Buscar información actual en la web.\n"
                "- **data_analysis_agent**: Analizar datos de archivos CSV/XLSX y generar gráficos.\n"
                "- **compliance_agent**: Revisar la legalidad de los contenidos.\n"
                "- **brief_generator_agent**: Crear briefs de marketing detallados.\n"
                "- **integrated_marketing_agent**: Diseñar estrategias de campaña holísticas."
            )
        return None

    def _direct_response_node(self, state: AgentState) -> dict:
        user_input = state["messages"][-1]["content"]
        response = self._handle_simple_query(user_input)
        return {"messages": [{"role": "assistant", "content": response, "name": "supervisor"}]}

    def route_question(self, state: AgentState) -> str:
        user_input = state["messages"][-1]["content"]
        if self._handle_simple_query(user_input) is not None:
            logger.info("Pregunta simple detectada. Enrutando a 'direct_response'.")
            return "direct_response"
        logger.info("Pregunta compleja detectada. Usando el LLM-Router...")
        available_agents = ", ".join(self.agent_names)
        supervisor_prompt_text = (
            "Eres el supervisor central de AMARETIS. Tu tarea es enrutar la pregunta del usuario "
            f"al agente más apropiado. Las opciones de enrutamiento deben ser exactamente una de: {available_agents}.\n"
            "Roles de los Agentes:\n"
            "- 'rag_agent': Preguntas sobre documentos internos de la empresa o sobre un archivo PDF que el usuario haya subido.\n"
            "- 'research_agent': Información en tiempo real, tendencias de mercado, datos recientes.\n"
            "- 'data_analysis_agent': Análisis de datos de archivos CSV/XLSX, estadísticas, comparaciones y gráficos.\n"
            "- 'brief_generator_agent': Creación de briefs de marketing detallados.\n"
            "- 'compliance_agent': Análisis legal de contenido publicitario y políticas de datos.\n"
            "- 'integrated_marketing_agent': Estrategias de campaña de alto nivel.\n"
            "Regla: Responde SÓLO con el nombre del agente."
        )
        route_prompt = ChatPromptTemplate.from_messages([("system", supervisor_prompt_text), ("human", f"Enruta la siguiente pregunta: {user_input}")])
        try:
            route_chain = route_prompt | self.llm
            llm_response = route_chain.invoke({})
            response_content = getattr(llm_response, 'content', str(llm_response)).strip().lower()
            selected_agent = next((name for name in self.agent_names if name.lower() in response_content), 'rag_agent')
            logger.info(f"Router LLM seleccionó: {selected_agent}")
            return selected_agent
        except Exception as e:
            logger.error(f"Error crítico en el routing LLM: {e}. Fallback a rag_agent.")
            return 'rag_agent'

    def setup_supervisor(self):
        workflow = StateGraph(AgentState)
        for name, agent in self.agents.items():
            workflow.add_node(name, self._create_agent_node(agent))
        workflow.add_node("direct_response", self._direct_response_node)
        def router_entry_node(state):
            return {}
        workflow.add_node("router", router_entry_node)
        workflow.set_entry_point("router")
        route_map = {name: name for name in self.agent_names}
        route_map["direct_response"] = "direct_response"
        workflow.add_conditional_edges("router", self.route_question, route_map)
        workflow.add_edge("direct_response", END)
        for name in self.agent_names:
            workflow.add_edge(name, END)
        self.supervisor = workflow.compile()

    def process_question(self, user_input: str) -> Tuple[str, str, Optional[str]]:
        if not self.supervisor:
            return "Error: El supervisor no está inicializado.", "Error Crítico", None
        image_path: Optional[str] = None
        try:
            self._clear_figures_directory()
            initial_state = {"messages": [{"role": "user", "content": user_input, "name": "user"}]}
            result = self.supervisor.invoke(initial_state) 
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                answer_text = last_message.get("content", "No se pudo obtener respuesta.")
                source_agent = last_message.get('name', 'N/A')
                source = f"Supervisor → {source_agent}"
                if source_agent == "data_analysis_agent":
                    image_path = self._get_latest_figure()
            else:
                answer_text = "Formato de respuesta inesperado."
                source = "Error de Supervisor"
            return answer_text, source, image_path
        except Exception as e:
            logger.error(f"Error crítico en el flujo del supervisor: {e}", exc_info=True)
            return "Lo siento, hubo un error crítico en el sistema de agentes.", "Error Crítico", None

    def _get_latest_figure(self) -> Optional[str]:
        try:
            figures_path = Path("figures")
            if not figures_path.exists(): return None
            figures = list(figures_path.glob("*.png")) + list(figures_path.glob("*.jpg"))
            if not figures: return None
            latest_figure = max(figures, key=lambda f: f.stat().st_mtime)
            logger.info(f"Figura encontrada: {latest_figure}")
            return str(latest_figure)
        except Exception as e:
            logger.error(f"Error buscando la figura más reciente: {e}")
            return None

    def _clear_figures_directory(self):
        try:
            figures_path = Path("figures")
            if figures_path.exists() and figures_path.is_dir():
                for item in figures_path.iterdir():
                    if item.is_file():
                        item.unlink()
                logger.info("Directorio 'figures' limpiado.")
        except Exception as e:
            logger.error(f"Error limpiando el directorio 'figures': {e}")

    def is_insufficient(self, answer: str, user_input: str = "") -> bool:
        if not answer or not isinstance(answer, str) or len(answer.strip()) < 10: return True
        insufficient_phrases = ["keine daten", "nicht verfügbar", "unbekannt", "weiß ich nicht", "kann ich nicht", "no data"]
        if any(phrase in answer.lower() for phrase in insufficient_phrases): return True
        numeric_keywords = ["wie viel", "umsatz", "gewinn", "zahlen", "betrag", "revenue", "budget", "kosten"]
        if any(kw in user_input.lower() for kw in numeric_keywords):
            if not re.search(r"\d", answer): return True
        return False
        
    def log_interaction(self, user_input: str, answer: str, source: str):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insuff_flag = "❗" if self.is_insufficient(answer, user_input) else "✅"
            log_entry = (f"\n⏰ {timestamp}\n{insuff_flag} Pregunta: {user_input}\nAntwort: {answer}\nQuelle: {source}\n{'-' * 60}\n")
            with open("chat_log.txt", "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Error al escribir log: {e}")
    
    def update_history(self, user_input: str, answer: str):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def run_interactive(self):
        print("\n--- 🧠 AMARETIS Supervisor está listo. ---")
        print("Escribe una pregunta (o 'exit' para salir).")
        while True:
            try:
                user_input = input("\nPregunta: ").strip()
                if user_input.lower() in ["exit", "quit", "salir"]:
                    print("¡Hasta luego!")
                    break
                if not user_input: continue
                
                print("\nProcesando...")
                start_time = time.time()
                answer_text, source, image_path = self.process_question(user_input)
                end_time = time.time()
                
                print(f"\nRespuesta:\n{answer_text}")
                print(f"\nFuente: {source} (Tiempo: {end_time - start_time:.2f}s)")
                if image_path:
                    print(f"🖼️  Imagen generada en: {image_path}")
                
                if self.is_insufficient(answer_text, user_input):
                    print("\n⚠️ La respuesta parece incompleta/insuficiente.")
                
                self.update_history(user_input, answer_text)
                self.log_interaction(user_input, answer_text, source)
            except KeyboardInterrupt:
                print("\n\nProceso interrumpido por el usuario. ¡Hasta luego!")
                break
            except Exception as e:
                logger.error(f"Error inesperado en loop interactivo: {e}", exc_info=True)
                print(f"Error inesperado: {e}")

def main():
    try:
        supervisor_manager = SupervisorManager()
        supervisor_manager.run_interactive()
    except Exception as e:
        logger.critical(f"Error FATAL al inicializar supervisor: {e}", exc_info=True)
        print(f"\n❌ Error crítico de inicialización. Verifique logs y el Traceback. Error: {e}")

if __name__ == "__main__":
    main()