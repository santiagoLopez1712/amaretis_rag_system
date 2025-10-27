# llm_config.py
import os
import logging
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()
logger = logging.getLogger(__name__)

# --- MODELOS DISPONIBLES ---
MODEL_LOCAL = "llama3:8b"
MODEL_CLOUD = "gemini-1.5-flash-001" # Modelo rápido y potente de Vertex AI

def get_llm_instance(temperature: float = 0.7) -> BaseChatModel:
    """
    Fábrica de Modelos de Lenguaje (LLM Factory).
    Lee la variable de entorno LLM_PROVIDER y devuelve la instancia
    del modelo correspondiente (local o en la nube).
    """
    
    provider = os.getenv("LLM_PROVIDER", "local").lower()
    
    if provider == "cloud":
        logger.info(f"🚀 Cargando modelo de NUBE (Vertex AI): {MODEL_CLOUD}")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("LLM_PROVIDER='cloud' pero GOOGLE_CLOUD_PROJECT no está configurado.")
        
        return ChatVertexAI(
            project=project_id,
            model=MODEL_CLOUD,
            temperature=temperature
        )
        
    elif provider == "local":
        logger.info(f"🏠 Cargando modelo LOCAL (Ollama): {MODEL_LOCAL}")
        return ChatOllama(
            model=MODEL_LOCAL,
            temperature=temperature
        )
        
    else:
        raise ValueError(f"Proveedor de LLM no válido: '{provider}'. Elige 'local' o 'cloud'.")

# --- Exportamos una instancia genérica para los agentes que no son ReAct ---
# (Como integrated_marketing_agent)
llm_default = get_llm_instance()