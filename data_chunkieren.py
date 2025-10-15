"""
=============================================================================
AMARETIS RAG System - Data Ingestion Pipeline (v3.1 - Fixes)
=============================================================================

Mejoras de esta versión:
1.  SimpleFileSystemStore: Implementación manual de un almacén de documentos
    persistente para evitar ImportErrors con las versiones de LangChain.
2.  Corrección de AttributeError: Se elimina la llamada a `.persist()` que es
    obsoleta en las nuevas versiones de ChromaDB.
3.  Limpieza de Logs: Se eliminan los emojis para evitar UnicodeEncodeError
    en terminales de Windows.
=============================================================================
"""
import os
import json
import shutil
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Sequence
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import BaseStore

# --- CONFIGURACIÓN DE LOGGING ---
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
LOG_DIR = "rag_logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"rag_processing_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN GENERAL ---
load_dotenv()
DATA_PATH = "structured_data.json"
CHROMA_DIR = "chroma_amaretis_db"
DOCSTORE_DIR = "docstore"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"

# ====================================================================
# NUESTRA PROPIA IMPLEMENTACIÓN DE FileSystemStore
# ====================================================================
class SimpleFileSystemStore(BaseStore[str, Document]):
    """
    Una implementación simple de un almacén de documentos (docstore) basado en el
    sistema de archivos. Almacena cada documento como un archivo pickle.
    """
    def __init__(self, root_path: str):
        self._root_path = Path(root_path)
        self._root_path.mkdir(exist_ok=True, parents=True)

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        for key, value in key_value_pairs:
            with open(self._root_path / key, "wb") as f:
                pickle.dump(value, f)

    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        values = []
        for key in keys:
            try:
                with open(self._root_path / key, "rb") as f:
                    values.append(pickle.load(f))
            except FileNotFoundError:
                values.append(None)
        return values

    def mdelete(self, keys: List[str]) -> None:
        for key in keys:
            try:
                os.remove(self._root_path / key)
            except FileNotFoundError:
                pass

    def yield_keys(self, prefix: str = None) -> List[str]:
        return [p.name for p in self._root_path.iterdir()]

# ====================================================================

def load_structured_data(json_path: str) -> list[Document]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Archivo no encontrado: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [Document(page_content=item["content"], metadata=item) for item in data]

def setup_parent_document_retriever(documents: list[Document], clear_existing: bool = True):
    logger.info("="*70)
    logger.info("AMARETIS RAG - Pipeline de Ingesta (v3.1 - Custom Docstore)")
    logger.info("="*70)

    if clear_existing:
        if os.path.exists(CHROMA_DIR):
            logger.warning(f"Eliminando base de datos (ChromaDB) antigua en '{CHROMA_DIR}'...")
            shutil.rmtree(CHROMA_DIR)
        if os.path.exists(DOCSTORE_DIR):
            logger.warning(f"Eliminando almacén de documentos (Docstore) antiguo en '{DOCSTORE_DIR}'...")
            shutil.rmtree(DOCSTORE_DIR)
    
    logger.info("Paso 1: Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": DEVICE}
    )
    logger.info(f"   Modelo '{MODEL_NAME}' cargado en {DEVICE.upper()}\n")

    logger.info("Paso 2: Definiendo splitters para 'padres' e 'hijos'...")
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    logger.info("   Splitters definidos\n")

    logger.info("Paso 3: Configurando Vectorstore y nuestro Docstore custom...")
    vectorstore = Chroma(
        collection_name="amaretis_split_parents",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    store = SimpleFileSystemStore(root_path=DOCSTORE_DIR)
    logger.info("   Almacenes configurados\n")

    logger.info("Paso 4: Creando y poblando el ParentDocumentRetriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    retriever.add_documents(documents, ids=None)
    logger.info(f"   Retriever poblado con {len(documents)} documentos.\n")
    
    logger.info("Paso 5: Cambios persistidos en disco.")
    logger.info(f"   - Vectorstore en: '{CHROMA_DIR}'")
    logger.info(f"   - Docstore en:    '{DOCSTORE_DIR}'\n")

    logger.info("="*70)
    logger.info("Pipeline de ingesta completado exitosamente")
    logger.info("="*70)

def main():
    """Función principal para ejecutar el pipeline de ingesta."""
    try:
        documents = load_structured_data(DATA_PATH)
        logger.info(f"Cargados {len(documents)} documentos desde '{DATA_PATH}'.")
        setup_parent_document_retriever(documents, clear_existing=True)
        
    except Exception as e:
        logger.error(f"Error catastrófico durante la ejecución: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
