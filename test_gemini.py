# test_gemini.py (Versión final con la corrección del diccionario)

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

print("--- INICIANDO PRUEBA DE CONEXIÓN DEFINITIVA ---")

# 1. Cargar la clave de API desde .env
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERROR FATAL: No se encontró la GOOGLE_API_KEY en el archivo .env")
    exit()
print("✅ Clave de API encontrada.")

try:
    # 2. Intentar inicializar el modelo 'gemini-pro'
    print("\n🔄 Inicializando el modelo 'gemini-pro'...")
    
    # --- LA CORRECCIÓN FINAL ---
    # Pasamos un diccionario directamente, que es lo que el error de validación pedía.
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        client_options={"api_endpoint": "generativelanguage.googleapis.com"}
    )
    
    print("✅ Modelo inicializado correctamente.")

    # 3. Realizar una invocación simple
    print("🔄 Enviando una pregunta de prueba...")
    response = llm.invoke("Hola, responde solo con 'OK'")
    
    # 4. Imprimir la respuesta
    print("\n" + "="*50)
    print("🎉 ¡ÉXITO! LA CONEXIÓN CON LA API DE GEMINI FUNCIONA.")
    print(f"   Respuesta del modelo: {response.content}")
    print("="*50)

except Exception as e:
    print("\n" + "!"*50)
    print("💥 FRACASO. La conexión con la API de Gemini sigue fallando.")
    print("   Este es el error:")
    print(f"   Tipo de error: {type(e).__name__}")
    print(f"   Mensaje: {e}")
    print("!"*50)