"""
=============================================================================
AMARETIS RAG Agent - Versión Mejorada con Query Analysis
=============================================================================

Mejoras implementadas:
1. Query Analysis - Detecta preguntas válidas vs inválidas
2. Validación robusta de respuestas - Verifica que hay resultados en ChromaDB
3. Modelo de embeddings consistente - Usa all-MiniLM-L6-v2 como data_chunkieren
4. Caché de embeddings - Carga el modelo una sola vez
5. Logging detallado - Rastrea búsquedas fallidas
6. Mejor manejo de errores - Sin alucinaciones
=============================================================================
"""

import os
import logging
import pdfplumber
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada.")

REGION = os.getenv("GOOGLE_CLOUD_REGION")
if not REGION:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_REGION no está configurada.")

# ============================================
# CONFIGURACIÓN CONSISTENTE
# ============================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Mismo que data_chunkieren.py
CHROMA_PERSIST_DIR = "./chroma_amaretis_db"
CHROMA_COLLECTION = "amaretis_knowledge"
RETRIEVAL_K = 3  # Número de documentos a recuperar
MIN_RELEVANCE_SCORE = 0.3  # Score mínimo de relevancia (0-1)

# ============================================
# QUERY ANALYSIS
# ============================================

class QueryAnalyzer:
    """Analiza la calidad de la pregunta antes de procesarla"""
    
    @staticmethod
    def is_valid_query(query: str) -> Tuple[bool, str]:
        """
        Valida si la pregunta es procesable.
        Retorna (es_válida, razón)
        """
        if not query or not query.strip():
            return False, "La pregunta está vacía."
        
        if len(query.strip()) < 3:
            return False, "La pregunta es demasiado corta (mínimo 3 caracteres)."
        
        if len(query.strip()) > 2000:
            return False, "La pregunta es demasiado larga (máximo 2000 caracteres)."
        
        # Detectar preguntas que no se pueden responder con la BD
        non_answerable_keywords = [
            "clima hoy", "weather", "bitcoin", "stock", "precio actual",
            "noticias", "news", "tiempo real", "real-time", "hoy es",
            "qué hora", "zona horaria"
        ]
        
        query_lower = query.lower()
        for keyword in non_answerable_keywords:
            if keyword in query_lower:
                return False, f"Esta pregunta requiere información en tiempo real que no está en nuestra base de datos."
        
        return True, "OK"
    
    @staticmethod
    def is_metadata_query(query: str) -> bool:
        """Detecta si es una pregunta sobre metadata (para no buscar en BD)"""
        metadata_keywords = [
            "cuántos documentos", "cuántos pdfs", "qué archivos",
            "cuál es tu nombre", "quién eres", "cómo funciona"
        ]
        return any(keyword in query.lower() for keyword in metadata_keywords)
    
    @staticmethod
    def is_pdf_upload_query(query: str) -> bool:
        """Detecta si el usuario quiere cargar un PDF específico"""
        upload_keywords = ["archivo subido", "pdf subido", "archivo cargado"]
        return any(keyword in query.lower() for keyword in upload_keywords)


class RAGAgent:
    name = "rag_agent"
    
    def __init__(self, debug: bool = False, model_name: str = "gemini-2.5-pro", temperature: float = 0.7, **kwargs):
        self.debug = debug
        self.model_name = model_name
        self.temperature = temperature
        self.collection_name = kwargs.get("collection_name", CHROMA_COLLECTION)
        self.persist_directory = kwargs.get("persist_directory", CHROMA_PERSIST_DIR)
        self.embedding_model = kwargs.get("embedding_model", EMBEDDING_MODEL)
        self.retrieval_k = kwargs.get("retrieval_k", RETRIEVAL_K)
        
        # Caché de embeddings (cargada una sola vez)
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        
        self.llm: Optional[ChatVertexAI] = None
        self.vectorstore: Optional[Chroma] = None
        self.tools: List[Tool] = []
        self.agent: Optional[AgentExecutor] = None
        self.query_analyzer = QueryAnalyzer()

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy loading de embeddings (carga una sola vez)"""
        if self._embeddings is None:
            logger.info(f"Cargando modelo de embeddings: {self.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embeddings

    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """Carga el vectorstore de ChromaDB de forma robusta."""
        try:
            if not Path(self.persist_directory).exists():
                logger.error(f"El directorio de ChromaDB no existe: {self.persist_directory}")
                logger.error("Por favor, ejecuta primero 'python data_chunkieren.py'")
                return None
            
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Validar que la BD no está vacía
            collection_count = vectorstore._collection.count()
            if collection_count > 0:
                logger.info(
                    f"✅ ChromaDB '{self.collection_name}' cargado exitosamente\n"
                    f"   Total de documentos: {collection_count}"
                )
                return vectorstore
            else:
                logger.warning("⚠️ ChromaDB existe pero está vacío. Ejecuta 'python data_chunkieren.py'")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error al cargar ChromaDB: {e}", exc_info=True)
            return None
    
    def _create_qa_chain(self, vectorstore: Chroma) -> Optional[RetrievalQA]:
        """Crea la cadena de Q&A con configuración optimizada"""
        try:
            if not self.llm:
                self.llm = ChatVertexAI(
                    project=PROJECT_ID,
                    location=REGION,
                    model=self.model_name, # Usar configuración centralizada
                    temperature=self.temperature # Usar configuración centralizada
                )
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retrieval_k}
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                verbose=self.debug,
                return_source_documents=True
            )
            
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creando QA chain: {e}")
            return None
    
    def _safe_qa_invoke(self, qa_chain: RetrievalQA, query: str) -> str:
        """Invoca Q&A con validaciones robustas"""
        try:
            result = qa_chain.invoke({"query": query})
            answer = result.get("result", "").strip()
            source_documents = result.get("source_documents", [])
            
            # Validación 1: Verificar que hay respuesta
            if not answer:
                logger.warning(f"No se obtuvo respuesta para: {query}")
                return (
                    "No se encontró información relevante en la base de datos para tu pregunta.\n"
                    "Por favor, intenta reformular tu pregunta o verifica que los documentos "
                    "necesarios estén en la base de datos."
                )
            
            # Validación 2: Verificar que hay documentos fuente
            if not source_documents:
                logger.warning(f"Sin documentos fuente para: {query}")
                return (
                    "La búsqueda no encontró documentos relevantes. "
                    "Por favor, intenta con otros términos."
                )
            
            # Validación 3: Generar citas
            citations = []
            seen_sources = set()
            
            for doc in source_documents:
                file_name = doc.metadata.get("file", "N/A")
                page_num = doc.metadata.get("page", "N/A")
                company = doc.metadata.get("company", "")
                
                source_key = f"{file_name} (página {page_num})" if page_num != "N/A" else file_name
                
                if source_key not in seen_sources:
                    citations.append(f"• {source_key}")
                    seen_sources.add(source_key)
            
            # Construir respuesta final
            if citations:
                citations_text = "\n\n📚 **Fuentes utilizadas:**\n" + "\n".join(citations)
                return answer + citations_text
            else:
                return answer
                
        except Exception as e:
            logger.error(f"Error en QA invoke: {e}", exc_info=True)
            return (
                f"Error al procesar tu pregunta: {str(e)}\n"
                f"Por favor, intenta de nuevo o reformula tu pregunta."
            )

    def _tool_query_uploaded_pdf(self, query_and_path: str) -> str:
        """Procesa preguntas sobre PDFs subidos"""
        try:
            parts = query_and_path.split('|')
            if len(parts) != 2:
                return "❌ Formato incorrecto. Usa: 'ruta/archivo.pdf|tu pregunta'"
            
            file_path_str, query = parts
            file_path = Path(file_path_str)

            if not file_path.exists():
                return f"❌ Archivo no encontrado: {file_path_str}"

            logger.info(f"Procesando PDF: {file_path_str}")
            
            # Extraer texto
            pages_content = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        pages_content.append({
                            "text": page_text,
                            "metadata": {"file": file_path.name, "page": i + 1}
                        })
            
            if not pages_content:
                return "❌ No se pudo extraer texto del PDF. ¿Es una imagen escaneada?"

            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            docs = []
            for page in pages_content:
                chunks = text_splitter.split_text(page["text"])
                for chunk in chunks:
                    docs.append(Document(
                        page_content=chunk,
                        metadata=page["metadata"]
                    ))

            if not docs:
                return "❌ El PDF no contiene texto procesable."

            # Crear vectorstore temporal
            temp_vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            qa_chain = self._create_qa_chain(temp_vectorstore)
            if not qa_chain:
                return "❌ Error al procesar el PDF."

            return self._safe_qa_invoke(qa_chain, query)
            
        except Exception as e:
            logger.error(f"Error procesando PDF: {e}", exc_info=True)
            return (
                f"❌ Error al procesar el archivo PDF.\n"
                f"Causa: {str(e)}\n"
                f"Por favor, verifica que el archivo no esté corrupto."
            )

    def _tool_get_raw_documents(self, query: str) -> str:
        """Una herramienta simple que solo recupera documentos crudos del vectorstore."""
        try:
            if not self.vectorstore:
                return "Error: La base de conocimiento no está disponible."
            
            retrieved_docs = self.vectorstore.similarity_search(query, k=self.retrieval_k)
            
            if not retrieved_docs:
                return "No se encontraron documentos relevantes para la consulta."
                
            # Formatear los documentos para la observación del agente
            context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            return context_str
            
        except Exception as e:
            logger.error(f"Error en la recuperación de documentos: {e}", exc_info=True)
            return f"Error al buscar en la base de datos: {e}"

    def setup_tools(self, vectorstore: Optional[Chroma] = None) -> List[Tool]:
        """Configura las herramientas del agente"""
        tools = []
        
        if vectorstore is None:
            vectorstore = self.load_existing_vectorstore()
        
        # Herramienta 1: Búsqueda en documentos
        if vectorstore:
            tools.append(Tool(
                name="document_search",
                func=self._tool_get_raw_documents,
                description=(
                    "Busca y devuelve fragmentos de documentos de la base de conocimiento de AMARETIS. "
                    "Usa esta herramienta para obtener el contexto necesario para responder preguntas sobre la empresa."
                )
            ))
        
        # Herramienta 2: Búsqueda en PDF subidos
        tools.append(Tool(
            name="uploaded_file_search",
            func=self._tool_query_uploaded_pdf,
            description=(
                "Busca información en un PDF específico que el usuario acaba de subir. "
                "El input DEBE tener el formato: 'ruta/del/archivo.pdf|pregunta del usuario'"
            )
        ))

        logger.info(f"✅ Herramientas configuradas: {[tool.name for tool in tools]}")
        return tools
    
    def create_marketing_agent(self, tools: List[Tool]) -> Optional[AgentExecutor]:
        """Crea el agente con prompt mejorado"""
        try:
            prompt = ChatPromptTemplate.from_template("""
Eres el "Asistente AMARETIS" - un experto en búsqueda de información corporativa.

Tu ÚNICA misión es responder preguntas basándote en la información de tus herramientas.
NUNCA inventes información. Si no encuentras algo, dilo claramente.

**REGLAS DE ENRUTAMIENTO:**
1. Si la pregunta menciona un "archivo subido", usa `uploaded_file_search`
2. Para todas las otras preguntas sobre la empresa, usa `document_search`
3. SOLO responde basándote en lo que tus herramientas retornen

**HERRAMIENTAS DISPONIBLES:**
{tools}

**NOMBRE DE HERRAMIENTAS (úsalos exactamente):**
{tool_names}

**INSTRUCCIONES DE RESPUESTA:**
Thought: [Analiza si esta pregunta se puede responder con tus herramientas]
Action: [Nombre exacto de la herramienta a usar]
Action Input: [La pregunta o entrada formateada correctamente]
Observation: [Lo que la herramienta retorna]
Thought: [Resumen de la información recibida]
Final Answer: [Tu respuesta basada ÚNICAMENTE en la Observation]

**IMPORTANTE:**
- Si la herramienta retorna "no se encontró", dile al usuario que no hay esa información
- No hagas suposiciones
- Sé preciso y conciso

Historial previo: {history}
Pregunta del usuario: {input}

Comienza:
{agent_scratchpad}""")

            if not self.llm:
                self.llm = ChatVertexAI(
                    project=PROJECT_ID,
                    location=REGION,
                    model=self.model_name, # Usar configuración centralizada
                    temperature=self.temperature # Usar configuración centralizada
                )
            
            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
            
            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                max_execution_time=60
            )
            executor.name = "rag_agent"
            
            return executor
            
        except Exception as e:
            logger.error(f"Error creando agente: {e}")
            return None
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Invoca el agente con validaciones"""
        if not self.agent:
            self.agent, _ = self.initialize_complete_agent()
            if not self.agent:
                return {"output": "❌ Error: No se pudo inicializar el RAG Agent"}
        
        user_input = input_dict.get("input", "").strip()
        history = input_dict.get("history", [])
        
        # Analizar la pregunta
        is_valid, reason = self.query_analyzer.is_valid_query(user_input)
        if not is_valid:
            logger.warning(f"Pregunta rechazada: {reason}")
            return {"output": f"❌ {reason}"}
        
        try:
            result = self.agent.invoke({
                "input": user_input,
                "history": history
            })
            return {"output": result.get("output", str(result))}
            
        except Exception as e:
            logger.error(f"Error en invoke: {e}", exc_info=True)
            return {"output": f"❌ Error técnico: {str(e)}"}

    def initialize_complete_agent(self) -> Tuple[Optional[AgentExecutor], Optional[Chroma]]:
        """Inicializa el agente completo"""
        logger.info("Inicializando RAG Agent...")
        self.vectorstore = self.load_existing_vectorstore()
        
        if not self.vectorstore:
            logger.error("No se pudo cargar ChromaDB")
            return None, None
        
        self.tools = self.setup_tools(self.vectorstore)
        if not self.tools:
            logger.error("No se pudieron configurar las herramientas")
            return None, self.vectorstore
        
        self.agent = self.create_marketing_agent(self.tools)
        
        if self.agent:
            logger.info("✅ RAG Agent inicializado correctamente")
        else:
            logger.error("Error inicializando RAG Agent")
        
        return self.agent, self.vectorstore


def create_amaretis_rag_agent(debug: bool = False, **kwargs) -> Tuple[Optional[AgentExecutor], Optional[Chroma]]:
    """Factory function para crear el RAG Agent"""
    try:
        rag = RAGAgent(debug=debug, **kwargs)
        return rag.initialize_complete_agent()
    except Exception as e:
        logger.error(f"Error creando AMARETIS RAG Agent: {e}")
        return None, None
