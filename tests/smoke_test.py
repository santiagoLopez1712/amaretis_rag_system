import sys
from pathlib import Path
# Asegurar que la raíz del proyecto esté en sys.path cuando se ejecuta el script de tests directamente
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from time import time
from supervisor import SupervisorManager
from rag_agent import create_amaretis_rag_agent
from langchain.schema import HumanMessage  # usar la ubicación estándar en esta instalación
# Import explícito de la clase de mensajes que usa internamente langchain_core
try:
    from langchain_core.schema import HumanMessage as CoreHumanMessage
except Exception:
    # Dejar el error claro para que instales la dependencia correcta
    raise ImportError("Instala 'langchain-core' en el venv: pip install -U langchain-core")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_test")

def test_supervisor_startup():
    t0 = time()
    s = SupervisorManager()
    logger.info(f"Agentes cargados: {s.agent_names}")
    elapsed = time() - t0
    print(f"Supervisor inicializado en {elapsed:.1f}s")

def test_process_simple_question():
    s = SupervisorManager()
    q = "Hola, ¿qué puedes hacer?"
    answer, source, image = s.process_question(q, [])
    print("Pregunta:", q)
    print("Respuesta:", answer)
    print("Origen:", source)
    print("Imagen devuelta:", image)

def test_rag_agent_direct():
    agent_executor, retriever, rag = create_amaretis_rag_agent(debug=True)
    if rag is None:
        print("RAG Agent no pudo inicializarse.")
        return
    try:
        # pasar una lista de mensajes del tipo que espera el prompt
        scratch = [CoreHumanMessage(content="")]  # lista válida
        out = rag.invoke({"input": "Hola, dame un ejemplo de respuesta breve.", "history": [], "agent_scratchpad": scratch})
    except Exception as e:
        print("Error invoking rag:", e)
        return
    print("RAG-Agent output:", out)

if __name__ == "__main__":
    test_supervisor_startup()
    test_process_simple_question()
    test_rag_agent_direct()