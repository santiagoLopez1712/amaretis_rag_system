# tools.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# NOTE: Must match the directory defined in data_chunkieren.py
CHROMA_DIR = "chroma_amaretis_db" 

def get_rag_documents(query: str) -> str:
    """
    Search the vector database (ChromaDB) for documents relevant to the user's query.
    Returns the text content of the retrieved documents as a single string.
    """
    try:
        # NOTE: Use the SAME embeddings model you used to create the index!
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load the existing Chroma collection
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR, 
            embedding_function=embeddings
        )
        
        # Retrieve the top N most relevant documents
        docs: list[Document] = vectorstore.similarity_search(query, k=4)
        
        # Format the documents into a single, clean string for the agent
        context = "\n---\n".join([doc.page_content for doc in docs])
        
        if not context:
             return "No relevant documents found in the database."
        
        return f"Retrieved Context:\n{context}"
        
    except Exception as e:
        # In case the directory hasn't been created yet (step 4), this handles it gracefully.
        return f"Error accessing RAG database (or DB not yet created): {e}"

def calculate_budget(data: str) -> str:
    """
    Calculates the marketing budget based on financial data provided in the input string.
    This is a placeholder function for the agent to use.
    """
    # Placeholder implementation
    return "Based on the input data, the calculated budget allocation is $50,000 for digital and $20,000 for print media."