# integrated_marketing_agent.py (Versión refactorizada con Wrapper)

import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

def create_marketing_pipeline(vectorstore) -> Runnable:
    """Crea la cadena de LangChain para la estrategia de marketing."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.8)

    # Asegúrate de que el vectorstore no es None antes de usarlo
    if not vectorstore:
        # Esto devuelve un retriever "falso" que no hará nada si no hay vectorstore
        # para evitar que el programa se caiga al iniciar.
        class DummyRetriever:
            def invoke(self, input): return []
            def __call__(self, input): return []
        retriever = DummyRetriever()
        logger.warning("Vectorstore no proporcionado a IntegratedMarketingAgent, el contexto estará vacío.")
    else:
        retriever = vectorstore.as_retriever(k=5)

    prompt_template = ChatPromptTemplate.from_template(
        """
        Du bist ein Senior Marketing Strategist bei AMARETIS. Deine Aufgabe ist es, eine integrierte Marketing-Strategie zu entwickeln.

        KONTEXT (Ähnliche Kampagnen):
        {context}

        AUFGABE:
        Basierend auf dem Kontext und der folgenden Anfrage, entwickle eine ganzheitliche Marketing-Strategie.
        Die Strategie sollte folgende Punkte umfassen:
        1.  **Zielgruppen-Analyse**: Wer sind die primären und sekundären Zielgruppen?
        2.  **Kernbotschaft**: Was ist die zentrale Botschaft der Kampagne?
        3.  **Kanal-Mix**: Welche Kanäle (Online/Offline) sollten genutzt werden und warum? (z.B. Social Media, SEO, Content Marketing, PR, Events)
        4.  **Phasen & Timing**: Skizziere einen groben Zeitplan für die Kampagne (z.B. Teasing, Launch, Sustain).
        5.  **KPIs**: Welche 3-5 Kennzahlen sind entscheidend für den Erfolg?

        ANFRAGE DES BENUTZERS:
        {question}

        ANTWORTE mit einer klaren, strukturierten und professionellen Strategie.
        """
    )

    # La cadena de LangChain que define la lógica
    pipeline = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )
    return pipeline

# ⚙️ WRAPPER-CLASE: La solución para la compatibilidad con el Supervisor
class IntegratedMarketingAgent:
    """
    Clase wrapper que adapta la cadena de marketing para la interfaz
    requerida por el supervisor de LangGraph.
    """
    name = "integrated_marketing_agent"

    def __init__(self, vectorstore):
        # La lógica real del agente es la cadena de LangChain
        self.pipeline: Runnable = create_marketing_pipeline(vectorstore)

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Punto de entrada estándar para LangGraph."""
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: La solicitud para el agente de marketing integrado está vacía."}
            
        try:
            # Invocamos la cadena de LangChain interna
            response = self.pipeline.invoke(user_input)
            
            # Extraemos el contenido del AIMessage o del string
            final_output = response.content if hasattr(response, 'content') else str(response)
            
            return {"output": final_output}
        except Exception as e:
            logger.error(f"Error en la invocación del Integrated Marketing Agent: {e}")
            return {"output": f"Error técnico en el agente de marketing integrado: {e}"}

# La función que el supervisor importará para crear el agente
def create_integrated_marketing_agent(vectorstore):
    return IntegratedMarketingAgent(vectorstore)