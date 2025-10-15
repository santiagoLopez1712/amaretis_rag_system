"""
=============================================================================
AMARETIS RAG Agent - v4.0 (Parent Document Retriever Integration)
=============================================================================

Mejoras de esta versión:
1.  Integración del ParentDocumentRetriever: El agente ahora utiliza el
    retriever configurado en `data_chunkieren.py`, que busca chunks pequeños
    pero devuelve los documentos "padre" completos para un contexto más rico.
2.  SimpleFileSystemStore: Se incluye la implementación del docstore custom
    para que el agente sea autocontenido y pueda reconstruir el retriever.
3.  Prompt de "Analista Senior": El prompt del agente ha sido actualizado
    para instruir al LLM a actuar como un analista experto, sintetizando
    respuestas detalladas y bien estructuradas a partir del contexto amplio
    que recibe.
4.  Refactorización de Herramientas: La herramienta de bósqueda principal
    ahora usa el ParentDocumentRetriever directamente, simplificando la lógica
    y eliminando la necesidad de una cadena `RetrievalQA` separada para la
    bósqueda en la base de conocimientos principal.
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import BaseStore
from data_chunkieren import SimpleFileSystemStore

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada.")

REGION = os.getenv("GOOGLE_CLOUD_REGION")
if not REGION:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_REGION no está configurada.")

# ===========================================
# CONFIGURACIÓN CONSISTENTE
# ===========================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "chroma_amaretis_db"
DOCSTORE_DIR = "docstore" # Directorio para nuestro FileSystemStore
CHROMA_COLLECTION_NAME = "amaretis_split_parents" # Colección usada en data_chunkieren

# ===========================================
# QUERY ANALYSIS (sin cambios)
# ===========================================
class QueryAnalyzer:
    @staticmethod
    def is_valid_query(query: str) -> Tuple[bool, str]:
        if not query or not query.strip(): return False, "La pregunta está vacía."
        if len(query.strip()) < 3: return False, "La pregunta es demasiado corta (mínimo 3 caracteres)."
        if len(query.strip()) > 2000: return False, "La pregunta es demasiado larga (máximo 2000 caracteres)."
        non_answerable_keywords = ["clima hoy", "weather", "bitcoin", "stock", "precio actual", "noticias", "news", "tiempo real", "real-time", "hoy es", "qué hora", "zona horaria"]
        if any(keyword in query.lower() for keyword in non_answerable_keywords):
            return False, "Esta pregunta requiere información en tiempo real que no está en nuestra base de datos."
        return True, "OK"

# ===========================================
# RAG AGENT v4.0
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
        """
        Carga y reconstruye el ParentDocumentRetriever a partir de los almacenes
        persistidos (ChromaDB y SimpleFileSystemStore).
        """
        try:
            if not Path(self.persist_directory).exists() or not Path(self.docstore_directory).exists():
                logger.error(f"No se encontraron los directorios de la base de datos. Ejecuta 'data_chunkieren.py' primero.")
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
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=400), # Debe coincidir con data_chunkieren
                parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000) # Debe coincidir con data_chunkieren
            )
            logger.info("✅ ParentDocumentRetriever cargado y listo.")
            return retriever
        except Exception as e:
            logger.error(f"❌ Error crítico al cargar el retriever: {e}", exc_info=True)
            return None

    def _tool_document_search(self, query: str) -> str:
        """
        Herramienta que usa el ParentDocumentRetriever para buscar documentos
        relevantes y devolver su contenido formateado para el prompt del agente.
        """
        if not self.retriever:
            return "Error: El sistema de recuperación de documentos no está disponible."
        
        try:
            logger.info(f"Buscando documentos para la consulta: '{query}'")
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            if not retrieved_docs:
                logger.warning("No se encontraron documentos relevantes.")
                return "No se encontraron documentos relevantes en la base de conocimiento para esta consulta."
            
            # Formatear los documentos "padre" para que el LLM los analice
            context_str = ""
            for i, doc in enumerate(retrieved_docs):
                metadata_str = f"Fuente: {doc.metadata.get('file', 'N/A')}, Página: {doc.metadata.get('page', 'N/A')}"
                context_str += f'--- INICIO DOCUMENTO {i+1} ({metadata_str}) ---\n\n'
                context_str += doc.page_content
                context_str += f'\n\n--- FIN DOCUMENTO {i+1} ---\n\n'
            
            logger.info(f"Se encontraron {len(retrieved_docs)} documentos padre relevantes.")
            return context_str

        except Exception as e:
            logger.error(f"Error en la herramienta de bósqueda de documentos: {e}", exc_info=True)
            return f"Error técnico al buscar en la base de datos: {e}"

    def _tool_query_uploaded_pdf(self, query_and_path: str) -> str:
        """
        Procesa preguntas sobre un PDF subido por el usuario.
        Esta función ahora es secundaria y solo para archivos temporales.
        """
        # (Esta función se mantiene sin cambios importantes por ahora,
        # ya que su lógica de crear un vectorstore temporal sigue siendo válida)
        try:
            parts = query_and_path.split('|')
            if len(parts) != 2: return "❌ Formato incorrecto. Usa: 'ruta/archivo.pdf|tu pregunta'"
            
            file_path_str, query = parts
            file_path = Path(file_path_str)
            if not file_path.exists(): return f"❌ Archivo no encontrado: {file_path_str}"

            logger.info(f"Procesando PDF temporal: {file_path_str}")
            with pdfplumber.open(file_path) as pdf:
                docs = [Document(page_content=page.extract_text() or "", metadata={"file": file_path.name, "page": i+1}) for i, page in enumerate(pdf.pages)]

            if not docs: return "❌ No se pudo extraer texto del PDF."

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)
            
            temp_vectorstore = Chroma.from_documents(split_docs, self.embeddings)
            
            # Usamos una cadena simple de QA para esta tarea específica
            from langchain.chains import RetrievalQA
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=temp_vectorstore.as_retriever()
            )
            result = qa_chain.invoke({"query": query})
            return result.get("result", "No se pudo obtener una respuesta del PDF.")

        except Exception as e:
            logger.error(f"Error procesando PDF temporal: {e}", exc_info=True)
            return f"❌ Error al procesar el archivo PDF: {e}"

    def _setup_tools(self) -> List[Tool]:
        """Configura las herramientas del agente."""
        tools = [
            Tool(
                name="knowledge_base_search",
                func=self._tool_document_search,
                description=(
                    "\u00dasla para responder CUALQUIER pregunta sobre AMARETIS, sus proyectos, clientes o conocimiento interno. "
                    "Esta herramienta busca en la base de datos de documentos de la empresa y devuelve el contenido completo de los documentos relevantes."
                )
            ),
            Tool(
                name="uploaded_file_search",
                func=self._tool_query_uploaded_pdf,
                description=(
                    "\u00dasla SOLAMENTE si el usuario menciona explícitamente un 'archivo subido' o 'PDF cargado' en su pregunta MÁS RECIENTE. "
                    "El input DEBE tener el formato: 'ruta/del/archivo.pdf|pregunta del usuario'"
                )
            )
        ]
        logger.info(f"✅ Herramientas configuradas: {[tool.name for tool in tools]}")
        return tools

    def _create_analyst_agent(self) -> Optional[AgentExecutor]:
        """
        Crea el agente con el nuevo prompt de "Analista Senior".
        """
        try:
            prompt = ChatPromptTemplate.from_template('''
Eres un "Analista Senior de Inteligencia de Marketing" en AMARETIS. Tu misión es proporcionar respuestas detalladas, bien estructuradas y perspicaces, basadas EXCLUSIVAMENTE en el contexto de los documentos proporcionados.

**PROCESO OBLIGATORIO:**

1.  **Analiza la Pregunta:** Comprende profundamente la solicitud del usuario.
2.  **Busca en la Base de Conocimiento:** Usa la herramienta `knowledge_base_search` para obtener los documentos relevantes. La observación de esta herramienta contendrá el texto completo de uno o más documentos internos.
3.  **Sintetiza la Respuesta:** Lee CUIDADOSAMENTE todo el contexto proporcionado en la observación. Tu respuesta final debe ser una síntesis experta de esta información. NO inventes nada que no esté en el texto.
4.  **Estructura y Cita:**
    *   Comienza con un resumen ejecutivo (1-2 frases).
    *   Desarrolla la respuesta con párrafos claros y, si es apropiado, usa listas con viñetas.
    *   Al final de tu respuesta, AÑADE SIEMPRE una sección de "Fuentes" citando los documentos que usaste (ej. "Fuente: nombre_del_archivo.pdf, Página: 5"). La información de la fuente se encuentra en la línea `--- INICIO DOCUMENTO ... ---`.

**HERRAMIENTAS DISPONIBLES:**
Nombres de Herramientas: {tool_names}
{tools}

**FORMATO DE PENSAMIENTO Y RESPUESTA:**

Thought: El usuario pregunta sobre [tema]. Necesito obtener el contexto completo de los documentos internos. Usaré la herramienta `knowledge_base_search`.
Action: knowledge_base_search
Action Input: [La pregunta original del usuario]
Observation: [Recibirás el contenido completo de varios documentos aquí]
Thought: He recibido el contexto de los documentos. Ahora voy a leerlo detenidamente, sintetizar la información clave, estructurar la respuesta como un analista senior y añadir las citas al final.
Final Answer: [Aquí va tu respuesta final, completa, bien estructurada y con la sección de fuentes al final.]

**IMPORTANTE:**
- Tu respuesta final debe ser completa y no solo una repetición de la observación. Debes procesar la información.
- Si la observación indica que "No se encontraron documentos relevantes", tu respuesta final debe ser exactamente esa frase.

Historial previo: {history}
Pregunta del usuario: {input}

Comienza tu análisis:
{agent_scratchpad}
''')

            if not self.llm:
                self.llm = ChatVertexAI(project=PROJECT_ID, location=REGION, model=self.model_name, temperature=self.temperature)
            
            agent_runnable = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
            
            executor = AgentExecutor(
                agent=agent_runnable,
                tools=self.tools,
                verbose=self.debug,
                handle_parsing_errors="Por favor, reintenta con un formato de acción válido.",
                max_iterations=5,
                max_execution_time=90
            )
            executor.name = "rag_agent"
            return executor
            
        except Exception as e:
            logger.error(f"Error creando el agente analista: {e}", exc_info=True)
            return None
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.agent:
            self.agent = self.initialize_complete_agent()
            if not self.agent:
                return {"output": "❌ Error: No se pudo inicializar el RAG Agent"}
        
        user_input = input_dict.get("input", "").strip()
        history = input_dict.get("history", [])
        
        is_valid, reason = self.query_analyzer.is_valid_query(user_input)
        if not is_valid:
            logger.warning(f"Pregunta rechazada: {reason}")
            return {"output": f"❌ {reason}"}
        
        try:
            result = self.agent.invoke({"input": user_input, "history": history})
            return {"output": result.get("output", str(result))}
        except Exception as e:
            logger.error(f"Error en invoke del RAG Agent: {e}", exc_info=True)
            return {"output": f"❌ Error técnico en el agente RAG: {str(e)}"}

    def initialize_complete_agent(self) -> Optional[AgentExecutor]:
        """Inicializa y devuelve el agente completamente configurado."""
        logger.info("Inicializando RAG Agent v4.0...")
        self.retriever = self._load_retriever()
        
        if not self.retriever:
            logger.error("Fallo al cargar el ParentDocumentRetriever. El agente no puede funcionar.")
            return None
        
        self.tools = self._setup_tools()
        self.agent = self._create_analyst_agent()
        
        if self.agent:
            logger.info("✅ RAG Agent v4.0 inicializado correctamente.")
        else:
            logger.error("❌ Error fatal al inicializar el RAG Agent v4.0.")
        
        return self.agent

def create_amaretis_rag_agent(debug: bool = False, **kwargs) -> Tuple[Optional[AgentExecutor], Optional[Any]]:
    """
    Función de fábrica para crear e inicializar el RAG Agent.
    Devuelve el agente y, como segundo elemento, el retriever (o None).
    """
    try:
        rag = RAGAgent(debug=debug, **kwargs)
        agent = rag.initialize_complete_agent()
        return agent, rag.retriever
    except Exception as e:
        logger.error(f"Error creando AMARETIS RAG Agent: {e}", exc_info=True)
        return None, None