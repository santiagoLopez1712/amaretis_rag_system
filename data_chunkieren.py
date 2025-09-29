# data_chunkieren.py
import os
import json
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Cargar variables de entorno
load_dotenv()

# Rutas
DATA_PATH = "structured_data.json"
CHROMA_DIR = "chroma_amaretis_db"  # Base de datos Chroma

def load_structured_data(json_path):
    """Carga datos estructurados desde JSON"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=item["content"], metadata=item) for item in data]

def chunk_documents(documents, chunk_size=800, chunk_overlap=100):
    """Divide documentos en chunks más pequeños para evitar crashes"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def embed_and_store(chunks):
    """Crea embeddings y los almacena en ChromaDB"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},  # ← fuerza CPU
        encode_kwargs={"normalize_embeddings": True}
    )

    # Borrar base de datos antigua si existe
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # Crear nueva base de datos Chroma con telemetry desactivada
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="amaretis_knowledge",
        anonymized_telemetry=True  # ← evita warnings
    )
    return db

def main():
    """Función principal"""
    print("🔄 Cargando datos...")
    documents = load_structured_data(DATA_PATH)
    print(f"📄 {len(documents)} documentos encontrados.")

    print("✂️ Chunking de documentos...")
    chunks = chunk_documents(documents)
    print(f"✅ {len(chunks)} chunks creados.")

    print("📦 Creando embeddings y guardando en ChromaDB...")
    embed_and_store(chunks)
    print(f"✅ Todos los datos guardados en '{CHROMA_DIR}'.")

if __name__ == "__main__":
    main()
