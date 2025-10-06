# tools.py 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_DIR = "chroma_amaretis_db" 

def get_rag_documents(query: str) -> str:
    """
    Search the vector database (ChromaDB) for documents relevant to the user's query.
    """
    try:
        # 1. Definir Embeddings (debe coincidir con data_chunkieren.py)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2" # ¡Usamos 768D para evitar conflictos!
        )
        
        # 2. Cargar la colección existente (¡CORRECCIÓN CLAVE!)
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR, 
            embedding_function=embeddings,
            collection_name="amaretis_knowledge" # Nombre de colección de data_chunkieren.py
        )
        
        # 3. Retrieve the top N most relevant documents
        docs: list[Document] = vectorstore.similarity_search(query, k=4)
        
        # 4. Format the documents into a single, clean string for the agent (con metadatos)
        context_parts = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            source = f"Fuente: {metadata.get('file', 'N/A')} - Pág. {metadata.get('page', 'N/A')}"
            context_parts.append(f"[Documento {i+1}] ({source})\n{doc.page_content}")
            
        context = "\n---\n".join(context_parts)
        
        if not context:
            return "No relevant documents found in the database."
        
        return f"Contexto Recuperado:\n{context}"
        
    except Exception as e:
        # Esto atrapará el error si la DB no existe o la API falla
        return f"Error accessing RAG database: {e}"

def calculate_budget(data: str) -> str:
    # ... (tu código placeholder)
    return "Based on the input data, the calculated budget allocation is $50,000 for digital and $20,000 for print media."