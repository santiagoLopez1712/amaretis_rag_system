# rag_agent.py 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List, Tuple, Any, Dict
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class RAGAgent:
    name = "rag_agent"
    
    def __init__(self, debug: bool = False, temperature: float = 0.7, **kwargs):
        self.debug = debug
        self.temperature = temperature
        self.collection_name = kwargs.get("collection_name", "amaretis_knowledge")
        self.persist_directory = kwargs.get("persist_directory", "./chroma_amaretis_db")
        self.embedding_model = kwargs.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
        self.retrieval_k = kwargs.get("retrieval_k", 5)
        self.llm = None
        self.vectorstore = None
        self.tools = None
        self.agent: Optional[AgentExecutor] = None

    def load_existing_vectorstore(self) -> Optional[Chroma]:
        try:
            if not Path(self.persist_directory).exists():
                logger.warning(f"Directorio {self.persist_directory} no existe")
                return None
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
            vectorstore = Chroma(collection_name=self.collection_name, embedding_function=embeddings, persist_directory=self.persist_directory)
            count = vectorstore._collection.count()
            logger.info(f"Vectorstore cargado con {count} documentos")
            if count == 0:
                logger.warning("Vectorstore está vacío")
                return None
            return vectorstore
        except Exception as e:
            logger.error(f"Error cargando vectorstore: {e}")
            return None
    
    def _safe_llm_invoke(self, query: str) -> str:
        try:
            if not self.llm: self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=self.temperature)
            chain = self.llm | StrOutputParser()
            return chain.invoke(f"Antworte natürlich und hilfreich auf: {query}")
        except Exception as e:
            logger.error(f"Error en LLM invoke: {e}")
            return f"Entschuldigung, technisches Problem: {query}"
    
    def _create_qa_chain(self, vectorstore: Chroma) -> Optional[RetrievalQA]:
        try:
            if not self.llm: self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=self.temperature)
            
            # --- MEJORA DE CALIDAD: Usar 'similarity' para ser menos restrictivo ---
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retrieval_k}
            )
            
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

            # Formatear las fuentes para la cita
            citations = []
            seen_sources = set() # Para evitar duplicados

            for doc in source_documents:
                # Extraemos los metadatos que nos interesan
                file_name = doc.metadata.get("file", "N/A")
                page_num = doc.metadata.get("page", "N/A")
                
                # Creamos una clave única para esta fuente (archivo + página)
                source_key = f"{file_name} (página {page_num})"
                
                if source_key not in seen_sources:
                    citations.append(f"- {source_key}")
                    seen_sources.add(source_key)

            if citations:
                # Añadimos la sección de fuentes a la respuesta
                citations_text = "\n\n**Fuentes:**\n" + "\n".join(citations)
                return answer + citations_text
            else:
                return answer

        except Exception as e:
            logger.error(f"Error en QA invoke: {e}")
            return f"Fehler bei der Dokumentensuche: {e}"
    
    def setup_tools(self, vectorstore: Optional[Chroma] = None) -> List[Tool]:
        tools = []
        if vectorstore is None: vectorstore = self.load_existing_vectorstore()
        if vectorstore:
            qa_chain = self._create_qa_chain(vectorstore)
            if qa_chain:
                tools.append(Tool(
                    name="document_search",
                    func=lambda query: self._safe_qa_invoke(qa_chain, query),
                    description="Suche nach spezifischen Informationen in der Wissensdatenbank über Kampagnen, Projekte, Kunden, Strategien und historische Daten."
                ))
        tools.append(Tool(
            name="general_chat",
            func=self._safe_llm_invoke,
            description="Für allgemeine Fragen, Smalltalk oder Fragen die nicht dokumentenbasiert sind."
        ))
        logger.info(f"Tools configurados: {[tool.name for tool in tools]}")
        return tools
    
    def create_marketing_agent(self, tools: List[Tool]) -> Optional[AgentExecutor]:
        try:
            prompt = ChatPromptTemplate.from_template("""
            Du bist ein präziser und faktenbasierter Marketing-Assistent für AMARETIS.

            **DEINE WICHTIGSTE REGEL:**
            **ANTWORTE AUSSCHLIESSLICH BASIEREND AUF DER INFORMATION, DIE DU DURCH DEINE WERKZEUGE ERHÄLTST (DER "OBSERVATION").**
            **ERFINDE KEINE INFORMATIONEN UND NUTZE NICHT DEIN ALLGEMEINES WISSEN.**
            Wenn die "Observation" die Antwort nicht enthält, antworte, dass du die Information in den bereitgestellten Dokumenten nicht finden konntest.

            **WERKZEUGE**
            Du hast Zugriff auf die folgenden Werkzeuge:
            {tools}

            **FORMATO DE RESPUESTA OBLIGATORIO**
            Du musst IMMER das folgende Format verwenden.

            Thought: [Deine detaillierte Analyse der Frage und dein Plan, welches Werkzeug du nutzen wirst.]
            Action: [Der exakte Name des Werkzeugs. Muss einer aus [{tool_names}] sein.]
            Action Input: [Die präzise Eingabe für das Werkzeug.]
            Observation: [Das Ergebnis des Werkzeugs. Dies wird vom System ausgefüllt.]
            Thought: [Deine Zusammenfassung der gesammelten Informationen und die Formulierung der endgültigen Antwort BASIEREND AUF DER OBSERVATION.]
            Final Answer: [Die finale, umfassende Antwort auf die ursprüngliche Frage des Benutzers.]

            **ANFORDERUNG**
            Beginne jetzt! Beantworte die folgende Frage des Benutzers und halte dich strikt an die oben beschriebenen Regeln.

            Bisheriger Verlauf: {history}
            Aktuelle Frage: {input}
            Dein Gedankengang:
            {agent_scratchpad}""")

            if not self.llm: self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=self.temperature)
            agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=self.debug, handle_parsing_errors=True, max_iterations=5, max_execution_time=30)
            executor.name = "rag_agent"
            return executor
        except Exception as e:
            logger.error(f"Error creando agente: {e}")
            return None
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.agent:
            self.agent, _ = self.initialize_complete_agent()
            if not self.agent: return {"output": "Fehler: RAG Agent konnte nicht initialisiert werden."}
        input_data = {"input": input_dict.get("input", ""), "history": input_dict.get("history", [])}
        try:
            result = self.agent.invoke(input_data)
            return {"output": result.get("output", str(result))}
        except Exception as e:
            logger.error(f"Fehler bei RAG Agent Invoke: {e}")
            return {"output": f"Technischer Fehler im RAG Agenten: {e}"}

    def initialize_complete_agent(self) -> Tuple[Optional[AgentExecutor], Optional[Chroma]]:
        logger.info("Inicializando RAG Agent completo...")
        self.vectorstore = self.load_existing_vectorstore()
        self.tools = self.setup_tools(self.vectorstore)
        if not self.tools:
            logger.error("No se pudieron configurar las herramientas")
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