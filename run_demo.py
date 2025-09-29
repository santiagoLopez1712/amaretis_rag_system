# run_demo.py
import os
from data_loader import extract_tables_from_directory_to_json
from data_chunkieren import load_structured_data, chunk_documents, embed_and_store, CHROMA_DIR
from rag_agent import create_amaretis_rag_agent

DATA_DIR = "data"
JSON_PATH = "structured_data.json"

def main():
    print("🔹 1. Extracción de PDFs a JSON...")
    extract_tables_from_directory_to_json(DATA_DIR, JSON_PATH)

    print("🔹 2. Carga de datos estructurados...")
    documents = load_structured_data(JSON_PATH)
    print(f"📄 {len(documents)} documentos encontrados.")

    print("🔹 3. Chunking de documentos...")
    chunks = chunk_documents(documents)
    print(f"✂️ {len(chunks)} chunks creados.")

    print("🔹 4. Creación de embeddings y almacenamiento en ChromaDB...")
    embed_and_store(chunks)
    print(f"✅ Datos embebidos y almacenados en {CHROMA_DIR}.")

    print("🔹 5. Inicialización del agente RAG...")
    rag_agent = create_amaretis_rag_agent(debug=True)
    if not rag_agent:
        print("❌ Error inicializando el agente RAG.")
        return

    print("🔹 6. Prueba de consulta al agente RAG...")
    query = "Muéstrame los proyectos de marketing de AMARETIS en 2024"
    result = rag_agent.invoke({"input": query})
    print("💡 Resultado del agente:")
    print(result)

if __name__ == "__main__":
    main()
