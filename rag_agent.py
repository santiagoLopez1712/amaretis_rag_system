"""
=============================================================================
AMARETIS RAG Agent - v4.1 (FASE 1: Prompts corregidos + k aumentado)
=============================================================================

Cambios en esta versión:
1.  Prompts corregidos: ChatPromptTemplate.from_messages() en lugar de from_template()
2.  k aumentado de 3 a 8 documentos para mejor contexto
3.  Estructura explícita en prompts para respuestas claras
4.  Query Analysis mejorado
=============================================================================
"""

import os
import logging
import pickle
import pdfplumber
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Sequence
from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import BaseStore

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada.")

REGION = os.getenv("GOOGLE_CLOUD_REGION")
if not REGION:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_REGION no está configurada.")

# ===========================================
# CONFIGURACIÓN CONSISTENTE (FASE 1)
# ===========================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "chroma_amaretis_db"
DOCSTORE_DIR = "docstore"
CHROMA_COLLECTION_NAME = "amaretis_split_parents"

# FASE 1: Aumentar k de 3 a 8
RETRIEVAL_K = 8  # ← CAMBIO CRÍTICO

# ===========================================
# SIMPLE FILE SYSTEM STORE (sin cambios)
# ===========================================
class SimpleFileSystemStore(BaseStore[str, Document]):
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

# ===========================================
# QUERY ANALYSIS (sin cambios)
# ===========================================
class QueryAnalyzer:
    @staticmethod
    def is_valid_query(query: str) -> Tuple[bool, str]:
        if not query or not query.strip():
            return False, "La pregunta está vacía."
        if len(query.strip()) < 3:
            return False, "La pregunta es demasiado corta (mínimo 3 caracteres)."
        if len(query.strip()) > 2000:
            return False, "La pregunta es demasiado larga (máximo 2000 caracteres)."
        non_answerable_keywords = [
            "clima hoy", "weather", "bitcoin", "stock", "precio actual",
            "noticias", "news", "tiempo real", "real-time", "hoy es", "qué hora"
        ]
        if any(keyword in query.lower() for keyword in non_answerable_keywords):
            return False, "Esta pregunta requiere información en tiempo real que no está en nuestra base de datos."
        return True, "OK"

# ===========================================
# RAG AGENT v4.1 (FASE 1)
# ===========================================
class RAGAgent:
    name = "rag_agent"
    
    def __init__(self, debug: bool = False, model_name: str = "gemini-2.5-pro", temperature: float = 0.7, **kwargs):
        self.debug = debug
        self.model_name = model_name
        self.temperature = temperature
        self.persist_directory = kwargs.get("persist_directory", CHROMA_PERSIST_DIR)
        self.docstore_directory = kwargs.get("docstore_directory", DOCSTORE_DIR)
        self.collection_name = kwargs.get("collection_name", CHROMA_COLLECTION_NAME)
        self.embedding_model = kwargs.get("embedding_model", EMBEDDING_MODEL)
        self.retrieval_k = kwargs.get("retrieval_k", RETRIEVAL_K)  # ← FASE 1: Usar RETRIEVAL_K
        
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[ChatVertexAI] = None
        self.retriever: Optional[ParentDocumentRetriever] = None
        self.tools: List[Tool] = []
        self.agent: Optional[AgentExecutor] = None
        self.query_analyzer = QueryAnalyzer()

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info(f"Cargando modelo de embeddings: {self.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embeddings

    def _load_retriever(self) -> Optional[ParentDocumentRetriever]:
        try:
            if not Path(self.persist_directory).exists() or not Path(self.docstore_directory).exists():
                logger.error(f"Directorios de base de datos no encontrados. Ejecuta 'data_chunkieren.py' primero.")
                return None

            logger.info("Cargando Vectorstore (ChromaDB)...")
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            logger.info("Cargando Docstore (SimpleFileSystemStore)...")
            store = SimpleFileSystemStore(root_path=self.docstore_directory)

            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
                parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000)
            )
            logger.info(f"✅ ParentDocumentRetriever cargado. Recuperando top {self.retrieval_k} documentos.")
            return retriever
            
        except Exception as e:
            logger.error(f"Error crítico al cargar el retriever: {e}", exc_info=True)
            return None

    def _tool_document_search(self, query: str) -> str:
        """
        Busca documentos usando ParentDocumentRetriever con k aumentado.
        FASE 1: Aumentar k de 3 a 8
        """
        if not self.retriever:
            return "Error: El sistema de recuperación de documentos no está disponible."
        
        try:
            logger.info(f"Buscando documentos (k={self.retrieval_k}) para: '{query}'")
            
            # FASE 1.1: Usar .invoke() en lugar de .get_relevant_documents()
            retrieved_docs = self.retriever.invoke(query)
            
            # Tomar los primeros k documentos
            retrieved_docs = retrieved_docs[:self.retrieval_k]
            
            if not retrieved_docs:
                logger.warning("No se encontraron documentos relevantes.")
                return "No se encontraron documentos relevantes en la base de conocimiento para esta consulta."
            
            # Formatear documentos para el LLM
            context_str = ""
            for i, doc in enumerate(retrieved_docs, 1):
                metadata_str = f"Fuente: {doc.metadata.get('file', 'N/A')}, Página: {doc.metadata.get('page', 'N/A')}"
                context_str += f'--- DOCUMENTO {i} ({metadata_str}) ---\n\n'
                context_str += doc.page_content
                context_str += f'\n\n--- FIN DOCUMENTO {i} ---\n\n'
            
            logger.info(f"Se encontraron {len(retrieved_docs)} documentos padre relevantes.")
            return context_str

        except Exception as e:
            logger.error(f"Error en la herramienta de búsqueda: {e}", exc_info=True)
            return f"Error técnico al buscar en la base de datos: {e}"

    def _tool_query_uploaded_pdf(self, query_and_path: str) -> str:
        """Procesa preguntas sobre PDFs subidos."""
        try:
            parts = query_and_path.split('|')
            if len(parts) != 2:
                return "Error: Usa el formato 'ruta/archivo.pdf|tu pregunta'"
            
            file_path_str, query = parts
            file_path = Path(file_path_str)
            if not file_path.exists():
                return f"Error: Archivo no encontrado: {file_path_str}"

            logger.info(f"Procesando PDF temporal: {file_path_str}")
            with pdfplumber.open(file_path) as pdf:
                docs = [
                    Document(
                        page_content=page.extract_text() or "",
                        metadata={"file": file_path.name, "page": i+1}
                    )
                    for i, page in enumerate(pdf.pages)
                ]

            if not docs:
                return "Error: No se pudo extraer texto del PDF."

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)
            
            temp_vectorstore = Chroma.from_documents(split_docs, self.embeddings)
            
            from langchain.chains import RetrievalQA
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=temp_vectorstore.as_retriever()
            )
            result = qa_chain.invoke({"query": query})
            return result.get("result", "No se pudo obtener respuesta del PDF.")

        except Exception as e:
            logger.error(f"Error procesando PDF temporal: {e}", exc_info=True)
            return f"Error al procesar el archivo PDF: {e}"

    def _setup_tools(self) -> List[Tool]:
        """Configura las herramientas del agente."""
        return [
            Tool(
                name="knowledge_base_search",
                func=self._tool_document_search,
                description=(
                    "Búsqueda en la base de datos de documentos internos de AMARETIS. "
                    "Devuelve fragmentos relevantes de documentos para responder preguntas sobre proyectos, "
                    "conceptos, políticas y conocimiento corporativo."
                )
            ),
            Tool(
                name="uploaded_file_search",
                func=self._tool_query_uploaded_pdf,
                description=(
                    "Procesa preguntas sobre archivos PDF que el usuario acaba de subir. "
                    "Formato requerido: 'ruta/del/archivo.pdf|pregunta'"
                )
            )
        ]

    def _create_analyst_agent(self) -> Optional[AgentExecutor]:
        """
        FASE 1.2: Re-añadir placeholders de herramientas al prompt del sistema.
        - `create_react_agent` requiere que {tools} y {tool_names} estén en el prompt.
        - Se mantiene la estructura `from_messages` para compatibilidad con `agent_scratchpad`.
        """
        try:
            system_prompt = """Eres un "Analista Senior de Inteligencia Corporativa" en AMARETIS. 
Tu misión es proporcionar respuestas profundas, bien estructuradas y basadas ÚNICAMENTE 
en los documentos de la base de conocimiento.

**ESTRUCTURA OBLIGATORIA DE RESPUESTA:**

1. **RESUMEN EJECUTIVO** (máximo 2 frases)
   - Qué es el tema, por qué es importante
   - Directo al punto

2. **COMPONENTES PRINCIPALES** (máximo 3-4 puntos)
   Para cada componente:
   - Nombre claro
   - Descripción (2-3 frases)
   - Relación con AMARETIS

3. **ANÁLISIS ESTRATÉGICO** (1 párrafo)
   - Implicaciones para nuestro contexto
   - Conexiones con otros elementos si aplica

4. **CONCLUSIÓN** (1-2 frases)
   - Síntesis final

5. **FUENTES CONSULTADAS**
   - Lista exacta de archivos y páginas usadas
   - Formato: "Archivo.pdf, página X"

**INSTRUCCIONES CRÍTICAS:**

1. Usa la herramienta `knowledge_base_search` para obtener contexto
2. LEE TODO EL CONTENIDO de la observación antes de responder
3. NO inventes información. Si no está en los documentos, dilo claramente
4. Estructura tu respuesta EXACTAMENTE como arriba
5. Cita todas las fuentes al final. Tu respuesta final debe ser en español.

**HERRAMIENTAS DISPONIBLES:**
{tools}

**NOMBRES DE HERRAMIENTAS (úsalos exactamente):**
{tool_names}
"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            if not self.llm:
                self.llm = ChatVertexAI(
                    project=PROJECT_ID,
                    location=REGION,
                    model=self.model_name,
                    temperature=self.temperature
                )
            
            agent_runnable = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
            
            executor = AgentExecutor(
                agent=agent_runnable,
                tools=self.tools,
                verbose=self.debug,
                handle_parsing_errors="Por favor, reintenta con un formato válido.",
                max_iterations=5,
                max_execution_time=300  # 5 minutos
            )
            executor.name = "rag_agent"
            return executor
            
        except Exception as e:
            logger.error(f"Error creando el agente: {e}", exc_info=True)
            return None
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.agent:
            self.agent = self.initialize_complete_agent()
            if not self.agent:
                return {"output": "Error: No se pudo inicializar el RAG Agent"}
        
        user_input = input_dict.get("input", "").strip()
        history = input_dict.get("history", [])
        
        is_valid, reason = self.query_analyzer.is_valid_query(user_input)
        if not is_valid:
            logger.warning(f"Pregunta rechazada: {reason}")
            return {"output": f"Error: {reason}"}
        
        try:
            result = self.agent.invoke({"input": user_input, "history": history})
            return {"output": result.get("output", str(result))}
        except Exception as e:
            logger.error(f"Error en invoke: {e}", exc_info=True)
            return {"output": f"Error técnico: {str(e)}"}

    def initialize_complete_agent(self) -> Optional[AgentExecutor]:
        logger.info("Inicializando RAG Agent v4.1 (FASE 1)...")
        self.retriever = self._load_retriever()
        
        if not self.retriever:
            logger.error("No se pudo cargar el ParentDocumentRetriever")
            return None
        
        self.tools = self._setup_tools()
        self.agent = self._create_analyst_agent()
        
        if self.agent:
            logger.info("✅ RAG Agent v4.1 inicializado correctamente")
        else:
            logger.error("Error inicializando RAG Agent v4.1")
        
        return self.agent

def create_amaretis_rag_agent(debug: bool = False, **kwargs) -> Tuple[Optional[AgentExecutor], Optional[Any]]:
    try:
        rag = RAGAgent(debug=debug, **kwargs)
        agent = rag.initialize_complete_agent()
        return agent, rag.retriever
    except Exception as e:
        logger.error(f"Error creando AMARETIS RAG Agent: {e}", exc_info=True)
        return None, None