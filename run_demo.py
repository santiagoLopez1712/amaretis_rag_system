# 1. Importaciones necesarias
import os
# Configurar variables de entorno para gRPC (opcional, pero recomendado para evitar demasiados logs)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"
# Importar funciones de los otros scripts
from data_loader import extract_tables_from_directory_to_json
from data_chunkieren import load_structured_data, chunk_documents, embed_and_store, CHROMA_DIR
from rag_agent import create_amaretis_rag_agent # Suponiendo que tienes este archivo
from tools import get_rag_documents # Importa tus herramientas (get_rag_documents, calculate_budget)
# No necesitas importar calculate_budget aquí, pero asegúrate de que esté en tools.py

# Rutas de los archivos
DATA_DIR = "data"
JSON_PATH = "structured_data.json"

def main():
    # Establece la API Key. Debe estar en tu entorno.
    if 'GOOGLE_API_KEY' not in os.environ:
        print("❌ ERROR: La variable de entorno GOOGLE_API_KEY no está configurada.")
        return
        
    print("--- 🚀 Iniciando Ejecución de AMARETIS RAG System ---") 
    
    # 1. Extracción de datos
    print("🔹 1. Extracción de PDFs a JSON...")
    extract_tables_from_directory_to_json(DATA_DIR, JSON_PATH)
    
    # 2. Carga, Chunking y Embeddings (Pasos 2-4)
    print("🔹 2. Carga de datos estructurados...")
    documents = load_structured_data(JSON_PATH)
    print(f"📄 {len(documents)} documentos encontrados.")

    print("🔹 3. Chunking de documentos...")
    chunks = chunk_documents(documents)
    print(f"✂️ {len(chunks)} chunks creados.")

    print("🔹 4. Creación de embeddings y almacenamiento en ChromaDB...")
    embed_and_store(chunks)
    print(f"✅ Datos embebidos y almacenados en {CHROMA_DIR}.")

    # 5. Inicialización del agente RAG
    print("🔹 5. Inicialización del agente RAG...")
    # Asegúrate de que rag_agent.py contenga la función create_amaretis_rag_agent
    rag_agent = create_amaretis_rag_agent(debug=True) # Pasa debug=True para más detalles
    
    if not rag_agent:
        print("❌ Fallo al inicializar el agente RAG. Abortando.")
        return
    
    print("✅ Agente RAG inicializado.")

    # 6. Prueba de consulta al agente RAG
    print("🔹 6. Prueba de consulta al agente RAG...")
    # Pregunta de prueba
    query = "Welche Positionen sind in der Tabelle 'Marketing-Budget 2025' für Bioventure auf Seite 57 aufgeführt?"
    
    # El agente debe usar la herramienta RAG y potencialmente la de budget
    try:
        response = rag_agent.invoke({
            "input": query,
            "history": [],  # <-- ¡ESTO RESUELVE EL ERROR!
        })
    
        # La respuesta final en LangGraph ReAct está en la clave 'output'
        final_response_content = response.get('output', 'No se pudo obtener la respuesta final.')
    
        # Imprimir el resultado final
        print(query)
        print(f"💡 Resultado del agente: {final_response_content}") 
    
    except Exception as e:
        print(f"❌ Error al ejecutar la consulta de prueba: {e}")

if __name__ == "__main__":
    main()