# integrated_marketing_agent.py (Refactorizado para configuración centralizada)

import os
import logging
from typing import Dict, Any, Optional
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
    """Crea la cadena de LangChain para la estrategia de marketing."""
    llm = ChatVertexAI(
        project=PROJECT_ID, 
        location=REGION, 
        model=model_name, # Usar configuración centralizada
        temperature=temperature
    )

    if not vectorstore:
        def dummy_retriever_func(query: str):
            logger.warning("Vectorstore no proporcionado, el contexto de búsqueda estará vacío.")
            return []
        
        retriever = RunnableLambda(dummy_retriever_func)

    else:
        retriever = vectorstore.as_retriever(k=5)

    prompt_template = ChatPromptTemplate.from_template(
        """
        Eres un "Estratega de Campañas Senior" en AMARETIS. Tu tarea es responder a preguntas de alto nivel con un marco estratégico holístico.

        **CONTEXTO DE CAMPAÑAS SIMILARES (si está disponible):**
        {context}

        **TAREA:**
        Basándote en el contexto y la siguiente pregunta, desarrolla una estrategia de marketing de alto nivel.
        La estrategia debe enfocarse en los siguientes pilares:
        1.  **Análisis Estratégico**: ¿Cuál es el problema principal a resolver y la oportunidad de mercado?
        2.  **Público Objetivo Clave**: Descripción del perfil principal al que nos dirigimos.
        3.  **Concepto Creativo Central**: ¿Cuál es la gran idea o el mensaje principal de la campaña?
        4.  **Pilares Tácticos**: Sugerencia de las principales áreas de acción (ej. Marketing de Contenidos, Eventos Exclusivos, Campaña Digital).
        5.  **KPIs Estratégicos**: ¿Cuáles son las 2-3 métricas más importantes para medir el éxito general?

        **PREGUNTA DEL USUARIO:**
        {question}

        **RESPUESTA:**
        Proporciona una respuesta clara, profesional y estratégica. No entres en detalles de ejecución, enfócate en el "qué" y el "porqué".
        """
    )

    pipeline = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return pipeline

class IntegratedMarketingAgent:
    """
    Clase wrapper que adapta la cadena de marketing para la interfaz
    requerida por el supervisor de LangGraph.
    """
    name = "integrated_marketing_agent"

    def __init__(self, vectorstore: Optional[Any], model_name: str, temperature: float):
        self.pipeline: Runnable = create_marketing_pipeline(vectorstore, model_name, temperature)

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Punto de entrada para LangGraph que pasa correctamente el historial."""
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: La solicitud para el agente de marketing integrado está vacía."}
            
        try:
            history = input_dict.get("history", [])
            response = self.pipeline.invoke(user_input)
            return {"output": response}
        except Exception as e:
            logger.error(f"Error en la invocación del Integrated Marketing Agent: {e}")
            return {"output": f"Error técnico en el agente de marketing integrado: {e}"}

def create_integrated_marketing_agent(vectorstore: Optional[Any], model_name: str, temperature: float) -> IntegratedMarketingAgent:
    """Función de fábrica para crear una instancia del agente."""
    return IntegratedMarketingAgent(vectorstore, model_name, temperature)
