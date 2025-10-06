# run_demo_test.py
import os
from dotenv import load_dotenv

# Cargar variables de entorno (incluida la GOOGLE_API_KEY)
load_dotenv()

# Configurar variables de entorno para gRPC (para logs más limpios)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"

# Importar solo lo necesario para el agente y las herramientas
from rag_agent import create_amaretis_rag_agent 
from tools import get_rag_documents, calculate_budget 

# Definición de la función principal
def run_agent_test():
    if 'GOOGLE_API_KEY' not in os.environ:
        print("❌ ERROR: La variable de entorno GOOGLE_API_KEY no está configurada.")
        return
        
    print("--- 🚀 Iniciando Test Rápido del Agente AMARETIS RAG ---") 
    
    # ⚠️ ASUNCIÓN: La DB 'chroma_amaretis_db' ya existe.
    
    # 5. Inicialización del agente RAG
    print("🔹 5. Inicialización del agente RAG...")
    
    # NOTA: Usamos debug=True para ver el Thought/Action/Observation del agente
    rag_agent = create_amaretis_rag_agent(debug=True) 
    
    if not rag_agent:
        print("❌ Fallo al inicializar el agente RAG. Abortando.")
        return
    
    print("✅ Agente RAG inicializado.")

    # 6. Prueba de consulta al agente RAG
    print("\n🔹 6. Prueba de consulta al agente RAG...")
    
    # --- Consulta de Prueba para forzar la búsqueda ---
    # Usaremos una pregunta que sabemos que tiene datos tabulares para forzar la búsqueda RAG
    query = "Welche spezifischen Gruppen werden im Rahmen der MASSNAHME 02 (Lokale Netzwerke aktivieren) in PHASE 1 für die Kooperationsformate identifiziert, insbesondere im Hinblick auf kirchliche Träger und Wohnungsunternehmen?" 
    print(f"❓ Consulta: {query}")
    
    try:
        # La función invoke iniciará la cadena de pensamiento del agente
        response = rag_agent.invoke({
            "input": query,
            "history": [], # Siempre vacío para un inicio de conversación
        })
    
        # La respuesta final en LangChain AgentExecutor está en la clave 'output'
        final_response_content = response.get('output', 'No se pudo obtener la respuesta final.')
    
        # Imprimir el resultado final
        print(f"\n💡 Resultado del agente:\n{final_response_content}") 
    
    except Exception as e:
        print(f"\n❌ Error al ejecutar la consulta de prueba: {e}")

if __name__ == "__main__":
    run_agent_test()