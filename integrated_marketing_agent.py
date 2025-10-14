# integrated_marketing_agent.py (Refactorizado para historial de conversación)

import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada.")

REGION = os.getenv("GOOGLE_CLOUD_REGION")
if not REGION:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_REGION no está configurada.")

def create_marketing_pipeline(vectorstore: Optional[Any], model_name: str, temperature: float) -> Runnable:
    """Crea la cadena de LangChain para la estrategia de marketing, ahora con historial."""
    llm = ChatVertexAI(
        project=PROJECT_ID, 
        location=REGION, 
        model=model_name,
        temperature=temperature
    )

    retriever = vectorstore.as_retriever(k=5) if vectorstore else RunnableLambda(lambda query: [])

    # El prompt ahora incluye el historial de conversación
    prompt_template = ChatPromptTemplate.from_template(
        """
        Eres un "Estratega de Campañas Senior" en AMARETIS. Tu tarea es responder a preguntas de alto nivel con un marco estratégico holístico, manteniendo el contexto de la conversación.

        **CONTEXTO DE CAMPAÑAS SIMILARES (si está disponible):**
        {context}

        **HISTORIAL DE CONVERSACIÓN:**
        {history}

        **TAREA:**
        Basándote en el contexto, el historial y la siguiente pregunta, desarrolla o refina una estrategia de marketing de alto nivel.
        La estrategia debe enfocarse en los siguientes pilares:
        1.  **Análisis Estratégico**: ¿Cuál es el problema principal a resolver y la oportunidad de mercado?
        2.  **Público Objetivo Clave**: Descripción del perfil principal al que nos dirigimos.
        3.  **Concepto Creativo Central**: ¿Cuál es la gran idea o el mensaje principal de la campaña?
        4.  **Pilares Tácticos**: Sugerencia de las principales áreas de acción.
        5.  **KPIs Estratégicos**: ¿Cuáles son las 2-3 métricas más importantes para medir el éxito general?

        **PREGUNTA DEL USUARIO:**
        {question}

        **RESPUESTA:**
        Proporciona una respuesta clara, profesional y estratégica. Si la pregunta es un seguimiento, úsala para refinar o expandir tu respuesta anterior.
        """
    )

    # El pipeline ahora necesita 'history' como input
    pipeline = (
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "history": RunnablePassthrough() # Se pasa el historial
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return pipeline

class IntegratedMarketingAgent:
    """
    Clase wrapper que adapta la cadena de marketing para la interfaz
    requerida por el supervisor de LangGraph, ahora con manejo de historial.
    """
    name = "integrated_marketing_agent"

    def __init__(self, vectorstore: Optional[Any], model_name: str, temperature: float):
        # El pipeline se crea una vez, pero se invoca con el historial en cada llamada
        self.pipeline: Runnable = create_marketing_pipeline(vectorstore, model_name, temperature)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Formatea el historial de LangGraph a una cadena legible para el prompt."""
        if not history:
            return "No hay conversación previa."
        
        formatted_history = []
        for message in history:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            if role == "user":
                formatted_history.append(f"Usuario: {content}")
            elif role == "assistant":
                # Solo incluimos las respuestas del propio agente para no confundirlo
                if message.get("name") == self.name:
                    formatted_history.append(f"Asistente: {content}")

        return "\n".join(formatted_history)

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Punto de entrada para LangGraph que ahora pasa correctamente el historial."""
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: La solicitud para el agente de marketing integrado está vacía."}
            
        try:
            # Extrae y formatea el historial
            history = input_dict.get("history", [])
            formatted_history = self._format_history(history)
            
            # Invoca el pipeline con el input del usuario y el historial formateado
            response = self.pipeline.invoke({
                "question": user_input,
                "history": formatted_history
            })
            
            return {"output": response}
        except Exception as e:
            logger.error(f"Error en la invocación del Integrated Marketing Agent: {e}", exc_info=True)
            return {"output": f"Error técnico en el agente de marketing integrado: {e}"}

def create_integrated_marketing_agent(vectorstore: Optional[Any], model_name: str, temperature: float) -> IntegratedMarketingAgent:
    """Función de fábrica para crear una instancia del agente."""
    return IntegratedMarketingAgent(vectorstore, model_name, temperature)