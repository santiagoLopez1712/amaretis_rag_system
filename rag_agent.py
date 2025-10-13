# rag_agent.py (Versión con carga de DB corregida)

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

class RAGAgent:
    name = "rag_agent"
    
    def __init__(self, debug: bool = False, temperature: float = 0.7, **kwargs):
        self.debug = debug
        self.temperature = temperature
        self.collection_name = kwargs.get("collection_name", "amaretis_knowledge")
        self.persist_directory = kwargs.get("persist_directory", "./chroma_amaretis_db")
        self.embedding_model = kwargs.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
        self.retrieval_k = kwargs.get("retrieval_k", 3)
        self.llm: Optional[ChatVertexAI] = None
        self.vectorstore: Optional[Chroma] = None
        self.tools: List[Tool] = []
        self.agent: Optional[AgentExecutor] = None

    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """Carga el vectorstore de ChromaDB de forma robusta."""
        try:
            if not Path(self.persist_directory).exists():
                logger.error(f"El directorio de la base de datos Chroma no existe: {self.persist_directory}")
                logger.error("Por favor, ejecuta primero el script 'data_chunkieren.py' para crear la base de datos.")
                return None
            
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
            
            vectorstore = Chroma(
                collection_name=self.collection_name, 
                embedding_function=embeddings, 
                persist_directory=self.persist_directory
            )
            
            # --- CORRECCIÓN CLAVE ---
            # Verificamos si la colección tiene elementos de una forma más robusta.
            # El método .get() puede causar errores internos en ChromaDB con colecciones vacías.
            # Usar ._collection.count() es más seguro.
            if vectorstore._collection.count() > 0:
                logger.info(f"Vectorstore '{self.collection_name}' cargado exitosamente desde {self.persist_directory}")
            else:
                logger.warning("El vectorstore existe pero está vacío.")

            return vectorstore
        except Exception as e:
            logger.error(f"Error crítico al cargar ChromaDB: {e}", exc_info=True)
            return None
    
    def _create_qa_chain(self, vectorstore: Chroma) -> Optional[RetrievalQA]:
        try:
            if not self.llm: 
                self.llm = ChatVertexAI(project=PROJECT_ID, location=REGION, model="gemini-2.5-pro", temperature=self.temperature)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.retrieval_k})
            return RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, verbose=self.debug, return_source_documents=True)
        except Exception as e:
            logger.error(f"Error creando QA chain: {e}")
            return None
    
    def _safe_qa_invoke(self, qa_chain: RetrievalQA, query: str) -> str:
        try:
            result = qa_chain.invoke({"query": query})
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])
            if not answer or not source_documents:
                return "In der Wissensdatenbank wurden keine relevanten Informationen für diese Anfrage gefunden."
            citations = []
            seen_sources = set()
            for doc in source_documents:
                file_name = doc.metadata.get("file", "N/A")
                page_num = doc.metadata.get("page", "N/A")
                source_key = f"{file_name} (página {page_num})" if page_num != "N/A" else file_name
                if source_key not in seen_sources:
                    citations.append(f"- {source_key}")
                    seen_sources.add(source_key)
            if citations:
                citations_text = "\n\n**Fuentes:**\n" + "\n".join(citations)
                return answer + citations_text
            else:
                return answer
        except Exception as e:
            logger.error(f"Error en QA invoke: {e}")
            return f"Fehler bei der Dokumentensuche: {e}"

    def _tool_query_uploaded_pdf(self, query_and_path: str) -> str:
        try:
            parts = query_and_path.split('|')
            if len(parts) != 2:
                return "Error: Formato de entrada incorrecto. Se esperaba 'ruta|pregunta'."
            file_path_str, query = parts
            file_path = Path(file_path_str)

            if not file_path.exists():
                return f"Error: El archivo no se encuentra en: {file_path_str}"

            logger.info(f"Procesando PDF subido: {file_path_str} para la pregunta: '{query}'")
            
            pages_content = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        pages_content.append({"text": page_text, "metadata": {"file": file_path.name, "page": i + 1}})
            
            if not pages_content:
                return "No se pudo extraer texto del PDF. Es posible que sea una imagen escaneada."

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            
            docs = []
            for page in pages_content:
                chunks = text_splitter.split_text(page["text"])
                for chunk in chunks:
                    docs.append(Document(page_content=chunk, metadata=page["metadata"]))

            if not docs:
                return "El texto extraído del PDF estaba vacío después del procesamiento."

            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            temp_vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
            
            qa_chain = self._create_qa_chain(temp_vectorstore)
            if not qa_chain: return "Error al crear la cadena de Q&A para el archivo subido."

            return self._safe_qa_invoke(qa_chain, query)
        except Exception as e:
            logger.error(f"Error en _tool_query_uploaded_pdf: {e}", exc_info=True)
            try:
                file_name = Path(query_and_path.split('|')[0]).name
            except:
                file_name = "desconocido"
            return (
                f"Lo siento, ocurrió un error al procesar el archivo PDF '{file_name}'.\n"
                f"Causa: {type(e).__name__} - {e}.\n\n"
                "Por favor, verifica que el archivo no esté corrupto, protegido por contraseña o sea una imagen escaneada."
            )

    def setup_tools(self, vectorstore: Optional[Chroma] = None) -> List[Tool]:
        tools = []
        if vectorstore is None: vectorstore = self.load_existing_vectorstore()
        
        if vectorstore:
            qa_chain = self._create_qa_chain(vectorstore)
            if qa_chain:
                tools.append(Tool(
                    name="document_search",
                    func=lambda query: self._safe_qa_invoke(qa_chain, query),
                    description="Útil para responder preguntas usando la base de conocimiento interna y permanente de la empresa."
                ))
        
        tools.append(Tool(
            name="uploaded_file_search",
            func=self._tool_query_uploaded_pdf,
            description="Útil para responder preguntas sobre un archivo PDF específico que el usuario acaba de subir. El input DEBE ser una cadena con el formato 'ruta/del/archivo.pdf|pregunta del usuario'."
        ))

        logger.info(f"Tools configurados para RAG Agent: { [tool.name for tool in tools] }")
        return tools
    
    def create_marketing_agent(self, tools: List[Tool]) -> Optional[AgentExecutor]:
        try:
            prompt = ChatPromptTemplate.from_template("""
            Eres el "Bibliotecario Corporativo" de AMARETIS. Tu única misión es responder preguntas usando tus herramientas.

            **REGLAS DE ENRUTAMIENTO DE HERRAMIENTAS:**
            1.  Si la pregunta del usuario menciona explícitamente un "archivo subido" o contiene una instrucción del sistema sobre un archivo específico, DEBES usar la herramienta `uploaded_file_search`.
            2.  Para todas las demás preguntas sobre conocimiento general de la empresa, usa la herramienta `document_search`.
            3.  **ANTWORTE AUSSCHLIESSLICH BASIEREND AUF DER INFORMATION, DIE DU DURCH DEINE WERKZEUGE ERHÄLTST (DER "OBSERVATION").** Si la herramienta no encuentra nada, informa de ello. No inventes.

            **HERRAMIENTAS (TOOLS):**
            {tools}

            **FORMATO DE RESPUESTA OBLIGATORIO**
            Thought: [Tu análisis de la pregunta y tu decisión sobre qué herramienta usar.]
            Action: [El nombre exacto de la herramienta. Debe ser uno de [{tool_names}]]
            Action Input: [La entrada para la herramienta. Si usas `uploaded_file_search`, recuerda el formato 'ruta/del/archivo.pdf|pregunta del usuario'.]
            Observation: [El resultado de la herramienta.]
            Thought: [Tu resumen de la información obtenida.]
            Final Answer: [La respuesta final para el usuario.]

            **ANFORDERUNG**
            Beginne jetzt!

            Bisheriger Verlauf: {history}
            Aktuelle Frage: {input}
            Dein Gedankengang:
            {agent_scratchpad}""")

            if not self.llm: 
                self.llm = ChatVertexAI(project=PROJECT_ID, location=REGION, model="gemini-2.5-pro", temperature=self.temperature)
            agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=self.debug, handle_parsing_errors=True, max_iterations=5, max_execution_time=60)
            executor.name = "rag_agent"
            return executor
        except Exception as e:
            logger.error(f"Error creando agente: {e}")
            return None
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.agent:
            self.agent, _ = self.initialize_complete_agent()
            if not self.agent: return {"output": "Fehler: RAG Agent konnte nicht initialisiert werden."}
        
        user_input = input_dict.get("input", "")
        history = input_dict.get("history", [])
        
        try:
            result = self.agent.invoke({"input": user_input, "history": history})
            return {"output": result.get("output", str(result))}
        except Exception as e:
            logger.error(f"Fehler bei RAG Agent Invoke: {e}")
            return {"output": f"Technischer Fehler im RAG Agenten: {e}"}

    def initialize_complete_agent(self) -> Tuple[Optional[AgentExecutor], Optional[Chroma]]:
        logger.info("Inicializando RAG Agent completo...")
        self.vectorstore = self.load_existing_vectorstore()
        self.tools = self.setup_tools(self.vectorstore)
        if not self.tools:
            logger.error("No se pudieron configurar las herramientas para RAG Agent")
            return None, self.vectorstore
        self.agent = self.create_marketing_agent(self.tools)
        if self.agent:
            logger.info("RAG Agent inicializado exitosamente")
        else:
            logger.error("Error inicializando RAG Agent")
        return self.agent, self.vectorstore

def create_amaretis_rag_agent(debug: bool = False, **kwargs) -> Tuple[Optional[AgentExecutor], Optional[Chroma]]:
    try:
        rag = RAGAgent(debug=debug, **kwargs)
        return rag.initialize_complete_agent()
    except Exception as e:
        logger.error(f"Error creando agente AMARETIS: {e}")
        return None, None