# rag_agent.py

# 1. Importaciones necesarias
from langgraph.prebuilt import create_react_agent
# Clase de LangChain para interactuar con Gemini
from langchain_google_genai import ChatGoogleGenerativeAI 
# Importa tus herramientas (estas funciones deben estar definidas en tools.py)
from tools import get_rag_documents, calculate_budget 

# 2. Función principal para crear el agente
def create_amaretis_rag_agent(debug: bool = False):
    """
    Inicializa y devuelve un agente ReAct impulsado por Gemini 2.5 Flash, 
    equipado con herramientas RAG.
    """
    try:
        # Define el LLM (Gemini 2.5 Flash)
        # Asegúrate de que la variable de entorno GOOGLE_API_KEY esté configurada.
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            # Añadir verbosity para propósitos de debug si es necesario
            verbose=debug 
        )

        # Define la lista de herramientas que el agente puede usar
        tools = [get_rag_documents, calculate_budget]

        # Crea el Agente ReAct
        rag_agent = create_react_agent(
            model=model,
            tools=tools,
            prompt="""Eres un asistente financiero RAG experto. 
            Utiliza la herramienta 'get_rag_documents' para encontrar información en el plan 
            de marketing antes de responder cualquier pregunta sobre la empresa. 
            Si el usuario pregunta sobre cálculos financieros o presupuestos, 
            debes usar la herramienta 'calculate_budget'. Sé conciso y profesional.
            """,
        )
        
        return rag_agent

    except Exception as e:
        print(f"Error al crear el agente Gemini: {e}")
        print("Asegúrate de tener instalado 'langchain-google-genai' y de que tu GOOGLE_API_KEY esté configurada.")
        return None