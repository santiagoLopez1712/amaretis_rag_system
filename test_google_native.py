# test_google_native.py

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

print("--- INICIANDO PRUEBA NATIVA DIRECTA CON GOOGLE ---")

# 1. Cargar la clave de API desde .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ ERROR FATAL: No se encontró GOOGLE_API_KEY en el archivo .env.")
    exit()
print("✅ Clave de API encontrada.")

try:
    # 2. Configurar la librería nativa de Google con tu clave
    genai.configure(api_key=api_key)
    print("✅ Librería de Google configurada.")

    # 3. Inicializar el modelo 'gemini-pro'
    print("\n🔄 Inicializando el modelo 'gemini-pro'...")
    model = genai.GenerativeModel('gemini-pro')
    print("✅ Modelo inicializado correctamente.")

    # 4. Realizar una petición simple
    print("🔄 Enviando una pregunta de prueba...")
    response = model.generate_content("Hola, responde solo con 'OK'")
    
    # 5. Imprimir la respuesta
    print("\n" + "="*50)
    print("🎉 ¡ÉXITO! LA CONEXIÓN NATIVA CON LA API DE GEMINI FUNCIONA.")
    print(f"   Respuesta del modelo: {response.text}")
    print("="*50)

except Exception as e:
    print("\n" + "!"*50)
    print("💥 FRACASO. La conexión directa con la API de Gemini también ha fallado.")
    print("   Este es el error final:")
    print(f"   Tipo de error: {type(e).__name__}")
    print(f"   Mensaje: {e}")
    print("!"*50)