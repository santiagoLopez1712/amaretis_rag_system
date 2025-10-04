# test_apis.py - TEST COMPLETO DE TODAS LAS APIS
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

print("=" * 60)
print("TEST DE APIs - AMARETIS RAG SYSTEM")
print("=" * 60)

# ===== TEST 1: VERIFICAR VARIABLES DE ENTORNO =====
print("\n📋 PASO 1: Verificando variables de entorno...")
google_key = os.getenv("GOOGLE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(f"Google API Key: {'✅ Set' if google_key else '❌ Missing'}")
print(f"Tavily API Key: {'✅ Set' if tavily_key else '❌ Missing'}")
print(f"LangChain API Key: {'✅ Set' if langchain_key else '❌ Missing'}")
print(f"HuggingFace Token: {'✅ Set' if hf_token else '❌ Missing'}")

# ===== TEST 2: GOOGLE GEMINI =====
print("\n🤖 PASO 2: Probando Google Gemini...")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
    response = llm.invoke("Hola, esto es un test")
    print("✅ Google Gemini: Funcionando")
    print(f"   Respuesta: {response.content[:100]}...")
except Exception as e:
    print(f"❌ Google Gemini Error: {e}")

# ===== TEST 3: TAVILY SEARCH =====
print("\n🔍 PASO 3: Probando Tavily Search...")
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    search = TavilySearchResults(max_results=1)
    result = search.invoke("test search")
    print("✅ Tavily: Funcionando")
    print(f"   Resultados encontrados: {len(result) if isinstance(result, list) else 1}")
except Exception as e:
    print(f"❌ Tavily Error: {e}")

# ===== TEST 4: HUGGINGFACE EMBEDDINGS =====
print("\n🧮 PASO 4: Probando HuggingFace Embeddings...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    # Test con texto simple
    test_text = "Este es un test de embeddings"
    embedded = embeddings.embed_query(test_text)
    print("✅ HuggingFace Embeddings: Funcionando")
    print(f"   Dimensión del vector: {len(embedded)}")
except Exception as e:
    print(f"❌ HuggingFace Embeddings Error: {e}")

# ===== TEST 5: CHROMADB =====
print("\n💾 PASO 5: Probando ChromaDB...")
try:
    import chromadb
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Crear vector store temporal en memoria
    vector_store = Chroma(
        collection_name="test_collection",
        embedding_function=embeddings
    )
    print("✅ ChromaDB: Funcionando")
    print(f"   Versión: {chromadb.__version__}")
except Exception as e:
    print(f"❌ ChromaDB Error: {e}")

# ===== TEST 6: LANGCHAIN CORE =====
print("\n⛓️ PASO 6: Probando LangChain Core...")
try:
    import langchain_core
    from langchain_core.documents import Document
    
    # Crear documento de prueba
    doc = Document(page_content="Test content", metadata={"source": "test"})
    print("✅ LangChain Core: Funcionando")
    print(f"   Versión: {langchain_core.__version__}")
except Exception as e:
    print(f"❌ LangChain Core Error: {e}")

# ===== TEST 7: LANGGRAPH =====
print("\n📊 PASO 7: Probando LangGraph...")
try:
    import langgraph
    from langgraph.graph import StateGraph
    from typing_extensions import TypedDict
    
    class TestState(TypedDict):
        test: str
    
    graph = StateGraph(TestState)
    print("✅ LangGraph: Funcionando")
    print(f"   Versión: {langgraph.__version__}")
except Exception as e:
    print(f"❌ LangGraph Error: {e}")

# ===== TEST 8: PDF PROCESSING =====
print("\n📄 PASO 8: Probando PDF Processing...")
try:
    import pdfplumber
    print("✅ PDFPlumber: Funcionando")
    print(f"   Versión: {pdfplumber.__version__}")
except Exception as e:
    print(f"❌ PDFPlumber Error: {e}")

# ===== TEST 9: GRADIO =====
print("\n🎨 PASO 9: Probando Gradio...")
try:
    import gradio as gr
    print("✅ Gradio: Funcionando")
    print(f"   Versión: {gr.__version__}")
except Exception as e:
    print(f"❌ Gradio Error: {e}")

# ===== RESUMEN FINAL =====
print("\n" + "=" * 60)
print("✨ TEST COMPLETADO")
print("=" * 60)
print("\nSi todos los tests pasaron, el sistema está listo para usar.")
print("Si alguno falló, revisa el error específico arriba.\n")