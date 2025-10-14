import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# CONFIGURACIÓN DE LOGGING - SUPRIMIR WARNINGS

# Desactiva telemetría de ChromaDB
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# Suprimir logs de ChromaDB
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('chromadb.telemetry').setLevel(logging.ERROR)
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.ERROR)

# Suprimir otros logs innecesarios
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Suprimir warnings de gRPC (ALTS credentials)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

# Cargar variables de entorno
load_dotenv()


# CONFIGURACIÓN

DATA_PATH = "structured_data.json"
CHROMA_DIR = "chroma_amaretis_db"
BATCH_SIZE = 50  # Procesa 50 PDFs a la vez
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Modelo más ligero y rápido
USE_GPU = False  # Cambiar a True si tienes GPU (NVIDIA)
DEVICE = "cuda" if USE_GPU else "cpu"

# Crear carpeta de logs
LOG_DIR = "rag_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configurar logging en archivo
log_file = os.path.join(LOG_DIR, f"rag_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# FUNCIONES PRINCIPALES


def load_structured_data(json_path):
    """Carga datos estructurados desde JSON"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [
        Document(page_content=item["content"], metadata=item) 
        for item in data
    ]

def chunk_documents(documents, chunk_size=800, chunk_overlap=100):
    """Divide documentos en chunks más pequeños"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def get_embeddings_model():
    """Crea y retorna el modelo de embeddings"""
    logger.info(f"🔗 Cargando modelo: {MODEL_NAME} en {DEVICE.upper()}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": DEVICE}
    )
    
    logger.info(f"✅ Modelo cargado exitosamente")
    return embeddings

def create_chroma_db(chunks, embeddings, clear_existing=False):
    """Crea una nueva base de datos ChromaDB"""
    if clear_existing and os.path.exists(CHROMA_DIR):
        logger.warning(f"🗑️ Eliminando base de datos antigua en '{CHROMA_DIR}'...")
        shutil.rmtree(CHROMA_DIR)
    
    logger.info(f"💾 Creando ChromaDB con {len(chunks)} chunks...")
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="amaretis_knowledge"
    )
    
    logger.info(f"✅ ChromaDB creada con éxito")
    return db

def add_to_existing_chroma(chunks, embeddings):
    """Añade chunks a una ChromaDB existente (INCREMENTAL)"""
    if not os.path.exists(CHROMA_DIR):
        logger.warning("📦 ChromaDB no existe. Creando nueva...")
        return create_chroma_db(chunks, embeddings, clear_existing=False)
    
    logger.info(f"📦 Conectando a ChromaDB existente...")
    
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="amaretis_knowledge"
    )
    
    logger.info(f"➕ Añadiendo {len(chunks)} chunks a ChromaDB existente...")
    db.add_documents(chunks)
    
    logger.info(f"✅ {len(chunks)} chunks añadidos exitosamente")
    return db

def process_with_batching(json_path, incremental=True, clear_db=False):
    """
    Procesa datos con batching para optimizar memoria.
    
    Args:
        json_path: Ruta del archivo JSON
        incremental: Si True, añade a ChromaDB existente. Si False, crea nueva.
        clear_db: Si True, borra ChromaDB anterior (solo si incremental=False)
    """
    logger.info("="*70)
    logger.info("🚀 AMARETIS RAG System - Pipeline de Ingesta de Datos (OPTIMIZADO)")
    logger.info("="*70)
    logger.info(f"⚙️ Configuración:")
    logger.info(f"   - Modelo: {MODEL_NAME}")
    logger.info(f"   - Dispositivo: {DEVICE.upper()}")
    logger.info(f"   - Modo: {'INCREMENTAL' if incremental else 'RECREAR'}")
    logger.info(f"   - Batch size: {BATCH_SIZE}")
    logger.info("")
    
    try:
        # Paso 1: Cargar datos
        logger.info("🔹 Paso 1: Cargando datos estructurados desde JSON...")
        documents = load_structured_data(json_path)
        logger.info(f"   ✅ {len(documents)} documentos cargados\n")

        # Paso 2: Chunking
        logger.info("🔹 Paso 2: Dividiendo documentos en chunks...")
        chunks = chunk_documents(documents)
        logger.info(f"   ✅ {len(chunks)} chunks creados\n")

        # Paso 3: Cargar modelo de embeddings una sola vez
        logger.info("🔹 Paso 3: Cargando modelo de embeddings...")
        embeddings = get_embeddings_model()
        logger.info("")

        # Paso 4: Procesar embeddings con batching
        logger.info("🔹 Paso 4: Procesando embeddings y almacenando en ChromaDB...\n")
        
        if incremental and os.path.exists(CHROMA_DIR):
            # Modo incremental: añadir a DB existente
            logger.info(f"📊 Modo INCREMENTAL activado")
            logger.info(f"   Chunks a procesar: {len(chunks)}")
            
            db = None
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
                
                logger.info(f"   📦 Procesando lote {batch_num}/{total_batches} ({len(batch)} chunks)...")
                
                if db is None:
                    db = add_to_existing_chroma(batch, embeddings)
                else:
                    db.add_documents(batch)
                    logger.info(f"      ✅ Lote {batch_num} añadido")
        else:
            # Modo recrear: crear nueva DB desde cero
            logger.info(f"🔄 Modo RECREAR activado (se borrará DB anterior)")
            logger.info(f"   Chunks a procesar: {len(chunks)}")
            
            db = None
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
                
                logger.info(f"   📦 Procesando lote {batch_num}/{total_batches} ({len(batch)} chunks)...")
                
                if i == 0:
                    # Primer lote: crear DB
                    db = create_chroma_db(batch, embeddings, clear_existing=clear_db)
                else:
                    # Lotes siguientes: añadir a DB
                    db.add_documents(batch)
                    logger.info(f"      ✅ Lote {batch_num} añadido")
        
        logger.info("")
        logger.info("="*70)
        logger.info("✨ Pipeline completado exitosamente")
        logger.info("="*70)
        logger.info(f"📊 Resumen:")
        logger.info(f"   - Total de documentos procesados: {len(documents)}")
        logger.info(f"   - Total de chunks creados: {len(chunks)}")
        logger.info(f"   - Base de datos guardada en: '{CHROMA_DIR}'")
        logger.info(f"   - Log guardado en: '{log_file}'")
        logger.info("")
        
        return db
        
    except Exception as e:
        logger.error(f"\n❌ Error durante la ejecución: {e}", exc_info=True)
        raise

def main():
    """Función principal"""
    
    # Opciones de ejecución
    
    # Opción 1: Procesar desde cero (borra DB anterior)
    # process_with_batching(DATA_PATH, incremental=False, clear_db=True)
    
    # Opción 2: Procesar incrementalmente (añade a DB existente)
    # process_with_batching(DATA_PATH, incremental=True)
    
    # Opción 3: Recrear sin borrar otros datos (para la primera vez)
    process_with_batching(DATA_PATH, incremental=False, clear_db=False)

if __name__ == "__main__":
    main()