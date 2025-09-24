from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_supervisor import create_supervisor
from rag_agnet_ganzneu import create_agent, setup_tools, load_existing_vectorstore
from web_such_agent import research_agent, ask_question_and_save_answer
from qa_ethics_agent import qa_ethics_agent
from data_analysis_agent import agent as data_analysis_agent
import os
import re
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

# === Configuración de logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Carga de variables de entorno ===
load_dotenv()

class SupervisorManager:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        self.history: List[Dict[str, str]] = []
        self.setup_agents()
        self.setup_supervisor()
        
    def setup_agents(self):
        """Inicializa todos los agentes"""
        try:
            # RAG Agent
            self.vectorstore = load_existing_vectorstore()
            self.tools = setup_tools(self.vectorstore)
            self.rag_agent = create_agent(self.tools)
            self.rag_agent.name = "rag_agent"
            
            # Otros agentes
            research_agent.name = "research_agent"
            data_analysis_agent.name = "data_analysis_agent"
            
        except Exception as e:
            logger.error(f"Error al configurar agentes: {e}")
            raise
    
    def setup_supervisor(self):
        """Configura el supervisor"""
        self.supervisor = create_supervisor(
            model=self.llm,
            agents=[self.rag_agent, research_agent, data_analysis_agent],
            prompt=(
                "You are a supervisor managing three agents:\n"
                "- 'rag_agent': Handles document-based and structured data questions.\n"
                "- 'research_agent': Handles real-time web search questions.\n"
                "- 'data_analysis_agent': Handles data analysis, statistics, CSV/Excel, plotting.\n"
                "Route questions based on:\n"
                "1. If asking for recent data (2024+) → research_agent\n"
                "2. If asking for analysis/statistics/plotting → data_analysis_agent\n"
                "3. Otherwise → rag_agent\n"
                "Always provide complete answers with sources."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile()
    
    def is_smalltalk(self, question: str) -> bool:
        """Detecta smalltalk/saludos"""
        smalltalk_keywords = [
            "hallo", "hi", "wie geht", "guten morgen", "guten abend", "servus",
            "grüß dich", "moin", "hey", "was geht", "wie läufts", "alles klar",
            "was machst du", "wer bist du", "was kannst du"
        ]
        return any(kw in question.lower() for kw in smalltalk_keywords)
    
    def contains_recent_year(self, user_input: str, min_year: int = 2024) -> bool:
        """Verifica si contiene años recientes"""
        years = re.findall(r"\b(20\d{2})\b", user_input)
        return any(int(y) >= min_year for y in years)
    
    def is_analysis_request(self, user_input: str) -> bool:
        """Detecta solicitudes de análisis de datos"""
        analysis_keywords = [
            "analyse", "statistik", "vergleich", "diagramm", "plot", "chart",
            "trend", "durchschnitt", "korrelation", "regression"
        ]
        return any(kw in user_input.lower() for kw in analysis_keywords)
    
    def route_question(self, user_input: str) -> str:
        """Determina qué agente debe manejar la pregunta"""
        if self.is_smalltalk(user_input):
            return "rag_agent"  # Para general chat
        elif self.is_analysis_request(user_input):
            return "data_analysis_agent"
        elif self.contains_recent_year(user_input, 2024):
            return "research_agent"
        else:
            return "rag_agent"
    
    def process_question(self, user_input: str) -> Tuple[str, str]:
        """Procesa una pregunta usando el supervisor"""
        try:
            # Usar el supervisor para routing automático
            result = self.supervisor.invoke({
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            })
            
            # Extraer respuesta del resultado
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                answer_text = last_message.get("content", "No se pudo obtener respuesta")
                source = f"Supervisor → {last_message.get('name', 'unknown_agent')}"
            else:
                answer_text = str(result)
                source = "Supervisor"
            
            return answer_text, source
            
        except Exception as e:
            logger.error(f"Error en supervisor: {e}")
            # Fallback manual
            return self._manual_fallback(user_input)
    
    def _manual_fallback(self, user_input: str) -> Tuple[str, str]:
        """Fallback manual si el supervisor falla"""
        try:
            if self.is_smalltalk(user_input):
                general_chat_tool = self.tools[0] if self.tools else None
                if general_chat_tool:
                    answer_text = general_chat_tool.run(user_input)
                    return answer_text, "RAG-Agent (general_chat)"
            
            # Intentar RAG primero
            rag_result = self.rag_agent.invoke({
                "input": user_input, 
                "history": self.history
            })
            
            if isinstance(rag_result, dict):
                answer_text = rag_result.get("output", "")
            else:
                answer_text = str(rag_result) if rag_result else ""
            
            if not answer_text or len(answer_text.strip()) < 10:
                # Fallback a web search
                answer_text, _ = ask_question_and_save_answer(user_input)
                return answer_text, "Web-Agent (fallback)"
            
            return answer_text, "RAG-Agent"
            
        except Exception as e:
            logger.error(f"Error en fallback manual: {e}")
            return f"Error al procesar la pregunta: {e}", "Error"
    
    def is_insufficient(self, answer: str, user_input: str = "") -> bool:
        """Verifica si la respuesta es insuficiente"""
        if not answer or not isinstance(answer, str) or len(answer.strip()) < 10:
            return True
        
        insufficient_phrases = [
            "keine daten", "nicht verfügbar", "unbekannt", 
            "weiß ich nicht", "kann ich nicht", "no data"
        ]
        if any(phrase in answer.lower() for phrase in insufficient_phrases):
            return True
        
        # Verificar números si se solicitan
        numeric_keywords = ["wie viel", "umsatz", "gewinn", "zahlen", "betrag", "revenue"]
        if any(kw in user_input.lower() for kw in numeric_keywords):
            # Regex mejorado para números
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
        """Actualiza el historial de conversación"""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})
        
        # Mantener solo los últimos 10 intercambios
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def run_interactive(self):
        """Ejecuta el loop interactivo principal"""
        print("\nSupervisor está listo. Escribe una pregunta (o 'exit' para salir):")
        
        while True:
            try:
                user_input = input("\nPregunta: ").strip()
                if user_input.lower() in ["exit", "quit", "salir"]:
                    print("¡Hasta luego!")
                    break
                
                if not user_input:
                    print("Por favor ingresa una pregunta.")
                    continue
                
                print("\nProcesando...")
                answer_text, source = self.process_question(user_input)
                
                print(f"\nRespuesta:\n{answer_text}")
                print(f"\nFuente: {source}")
                
                # Verificación de suficiencia
                if self.is_insufficient(answer_text, user_input):
                    print("\n⚠️ La respuesta parece incompleta. Considera reformular la pregunta.")
                
                # QA/Ethics check
                try:
                    warnings = qa_ethics_agent.run(answer_text, [source])
                    print(f"\n⚖️ QA/Ethik-Prüfung:\n{warnings}")
                except Exception as e:
                    logger.warning(f"Error en QA agent: {e}")
                
                # Actualizar historial y log
                self.update_history(user_input, answer_text)
                self.log_interaction(user_input, answer_text, source)
                
            except KeyboardInterrupt:
                print("\n\nProceso interrumpido por el usuario. ¡Hasta luego!")
                break
            except Exception as e:
                logger.error(f"Error inesperado: {e}")
                print(f"Error inesperado: {e}")

# === Función principal ===
def main():
    try:
        supervisor_manager = SupervisorManager()
        supervisor_manager.run_interactive()
    except Exception as e:
        logger.error(f"Error al inicializar supervisor: {e}")
        print(f"Error crítico: {e}")

if __name__ == "__main__":
    main()

# === Exports para uso externo ===
__all__ = [
    "SupervisorManager", "main"
]