# rag_agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List, Tuple, Any, Dict # << Dict añadido
import logging
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# === Configuración de logging ===
logger = logging.getLogger(__name__)

class RAGAgent:
    """
    RAG Agent mejorado con manejo de errores y configuración flexible
    """
    
    # Añadimos 'name' para ser consistentes con la filosofía de nodos de LangGraph
    name = "rag_agent"
    
    def __init__(
        self,
        collection_name: str = "amaretis_knowledge",
        persist_directory: str = "./chroma_amaretis_db",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        retrieval_k: int = 5,
        temperature: float = 0.7,
        debug: bool = False
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.retrieval_k = retrieval_k
        self.temperature = temperature
        self.debug = debug
        
        # Inicializar componentes
        self.llm = None
        self.vectorstore = None
        self.tools = None
        self.agent: Optional[AgentExecutor] = None # Tipado para claridad
        
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        try:
            if not Path(self.persist_directory).exists():
                logger.warning(f"Directorio {self.persist_directory} no existe")
                return None
                
            logger.info(f"Cargando embeddings: {self.embedding_model}")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"Cargando vectorstore: {self.collection_name}")
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory
            )
            
            try:
                count = vectorstore._collection.count()
                logger.info(f"Vectorstore cargado con {count} documentos")
                if count == 0:
                    logger.warning("Vectorstore está vacío")
                    return None
            except Exception as e:
                logger.warning(f"No se pudo verificar contenido del vectorstore: {e}")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error cargando vectorstore: {e}")
            return None
    
    def _safe_llm_invoke(self, query: str) -> str:
        """Invocación segura del LLM con manejo de errores"""
        try:
            if not self.llm:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=self.temperature
                )

            response = self.llm.invoke(f"Antworte natürlich und hilfreich auf: {query}")

            if hasattr(response, 'content') and isinstance(response.content, str):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
            
        except Exception as e:
            logger.error(f"Error en LLM invoke: {e}")
            return f"Entschuldigung, ich hatte ein technisches Problem bei der Verarbeitung Ihrer Frage: {query}"
    
    def _create_qa_chain(self, vectorstore: Chroma) -> Optional[RetrievalQA]:
        try:
            if not self.llm:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=self.temperature
                )
            
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": 0.5
                }
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
        try:
            result = qa_chain.invoke({"query": query})
            
            if isinstance(result, dict):
                answer = result.get("result", "")
                sources = result.get("source_documents", [])
                
                if self.debug and sources:
                    logger.info(f"Fuentes encontradas: {len(sources)}")
                    for i, doc in enumerate(sources[:3]):
                        logger.info(f"Fuente {i+1}: {doc.page_content[:100]}...")
                
                return answer if answer else "Keine relevanten Informationen gefunden."
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Error en QA invoke: {e}")
            return f"Fehler bei der Dokumentensuche: {e}"
    
    def setup_tools(self, vectorstore: Optional[Chroma] = None) -> List[Tool]:
        tools = []
        
        if vectorstore is None:
            vectorstore = self.load_existing_vectorstore()
        
        if vectorstore:
            qa_chain = self._create_qa_chain(vectorstore)
            if qa_chain:
                tools.append(
                    Tool(
                        name="document_search",
                        func=lambda query: self._safe_qa_invoke(qa_chain, query),
                        description=(
                            "Suche nach spezifischen Informationen in der Wissensdatenbank. "
                            "Verwende dieses Tool für Fragen über:\n"
                            "- Kampagnen und Projekte\n"
                            "- Kunden und Branchen\n"
                            "- Strategien und Best Practices\n"
                            "- Historische Daten und Berichte"
                        )
                    )
                )
            else:
                logger.warning("QA Chain konnte nicht erstellt werden")
        else:
            logger.warning("Kein Vectorstore verfügbar - document_search Tool nicht verfügbar")
        
        tools.append(
            Tool(
                name="general_chat",
                func=self._safe_llm_invoke,
                description=(
                    "Für allgemeine Fragen, Smalltalk oder Fragen die nicht "
                    "dokumentenbasiert sind. Verwende für Begrüßungen, "
                    "Erklärungen und allgemeine Beratung."
                )
            )
        )
        
        logger.info(f"Tools configurados: {[tool.name for tool in tools]}")
        return tools
    
    def create_marketing_agent(self, tools: List[Tool]) -> Optional[AgentExecutor]:
        try:
            prompt = ChatPromptTemplate.from_template("""
Du bist ein intelligenter Marketing- und Kommunikations-Assistent für AMARETIS, 
eine Full-Service-Werbeagentur in Göttingen.

Deine Expertise umfasst:
- Kampagnenentwicklung und -analyse
- Strategische Kommunikation
- Marktforschung und Trends
- Kreative Lösungsansätze
- Branchenspezifische Insights

WICHTIGE REGELN:
1. Nutze **document_search** für alle Fragen zu: 
    - Vergangenen Kampagnen und Projekten
    - Kunden- oder branchenspezifischen Informationen
    - Best Practices und bewährten Strategien
    - Konkreten Daten und Fallstudien
    
2. Nutze **general_chat** nur für:
    - Begrüßungen und Smalltalk  
    - Allgemeine Marketingberatung ohne Dokumentenbezug
    - Erklärungen von Konzepten
    - Kreative Brainstorming-Anfragen
    
3. Antworte immer:
    - Professionell aber zugänglich
    - Mit konkreten, umsetzbaren Empfehlungen
    - **VERWENDE Spanisch, wenn die Frage des Benutzers auf Spanisch ist, und Deutsch, 
      wenn die Frage auf Deutsch ist (oder wenn du nur deutsche Quellen zitierst).**
    - Mit Verweis auf relevante Quellen wenn verfügbar
    - Im Kontext der deutschen Marketinglandschaft

Verfügbare Tools: {tools}
Tool-Namen: {tool_names}

Bisheriger Verlauf: {history}
Aktuelle Frage: {input}

Nutze dieses Format:
Thought: [Deine Analyse der Frage]
Action: [Tool-Name]
Action Input: [Eingabe für das Tool]
Observation: [Ergebnis des Tools]
... (wiederhole bei Bedarf)
Thought: Ich habe alle nötigen Informationen gesammelt.
Final Answer: [Deine finale, hilfreiche Antwort]

{agent_scratchpad}
""")

            if not self.llm:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=self.temperature
                )

            agent = create_react_agent(
                llm=self.llm, 
                tools=tools, 
                prompt=prompt
            )

            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=self.debug,
                handle_parsing_errors=True,
                max_iterations=5,
                max_execution_time=30,
            )
            
            executor.name = "marketing_rag_agent"
            return executor
            
        except Exception as e:
            logger.error(f"Error creando agente: {e}")
            return None
    
    # 🌟🌟🌟 MÉTODO CLAVE AÑADIDO PARA COMPATIBILIDAD CON LANGGRAPH 🌟🌟🌟
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método de compatibilidad para LangGraph. 
        Delega la llamada al AgentExecutor interno (self.agent).
        """
        if not self.agent:
            # Si el agente aún no está inicializado, lo hacemos ahora
            self.agent, _ = self.initialize_complete_agent()
            if not self.agent:
                 return {"output": "Fehler: RAG Agent konnte nicht initialisiert werden."}

        # El AgentExecutor espera la entrada en la clave 'input'
        input_data = {
            "input": input_dict.get("input", "Leere Anfrage."),
            # La historia se puede pasar desde el estado, si es necesario para el prompt
            "history": input_dict.get("history", []) 
        }

        try:
            # Llamar al AgentExecutor interno
            result = self.agent.invoke(input_data)
            
            # Formatear la salida del AgentExecutor para el estado de LangGraph
            final_output = result.get("output", str(result))
            
            return {"output": final_output}
        except Exception as e:
            logger.error(f"Fehler bei RAG Agent Invoke: {e}")
            return {"output": f"Technischer Fehler im RAG Agenten: {e}"}

    # <<< MÉTODO CORREGIDO PARA DEVOLVER VECTORSTORE Y AGENTE >>>
    def initialize_complete_agent(self) -> Tuple[Optional[AgentExecutor], Optional[Chroma]]:
        logger.info("Inicializando RAG Agent completo...")
        
        self.vectorstore = self.load_existing_vectorstore()
        self.tools = self.setup_tools(self.vectorstore)
        
        if not self.tools:
            logger.error("No se pudieron configurar las herramientas")
            # Devolver None para el AgentExecutor, pero el vectorstore (si existe)
            return None, self.vectorstore 
        
        self.agent = self.create_marketing_agent(self.tools)
        
        if self.agent:
            logger.info("RAG Agent inicializado exitosamente")
        else:
            logger.error("Error inicializando RAG Agent")
            
        # RETORNO FINAL: Agente y Vectorstore
        # self.agent es un AgentExecutor (Runnable)
        return self.agent, self.vectorstore

# === Funciones de compatibilidad hacia atrás ===
def load_existing_vectorstore():
    logger.warning("Usando función deprecated. Considera usar RAGAgent class.")
    rag = RAGAgent()
    return rag.load_existing_vectorstore()

def setup_tools(vectorstore: Optional[Chroma] = None):
    logger.warning("Usando función deprecated. Considera usar RAGAgent class.")
    rag = RAGAgent()
    return rag.setup_tools(vectorstore)

def create_agent(tools: list):
    logger.warning("Usando función deprecated. Considera usar RAGAgent class.")
    rag = RAGAgent()
    return rag.create_marketing_agent(tools)

# <<< FUNCIÓN PRINCIPAL CORREGIDA PARA DEVOLVER VECTORSTORE Y AGENTE >>>
def create_amaretis_rag_agent(
    collection_name: str = "amaretis_knowledge",
    persist_directory: str = "./chroma_amaretis_db",
    debug: bool = False
) -> Tuple[Optional[AgentExecutor], Optional[Chroma]]:
    try:
        rag = RAGAgent(
            collection_name=collection_name,
            persist_directory=persist_directory,
            debug=debug
        )
        # Llama al método corregido que devuelve la tupla
        return rag.initialize_complete_agent() 
    except Exception as e:
        logger.error(f"Error creando agente AMARETIS: {e}")
        # En caso de error fatal, devuelve None para ambos
        return None, None