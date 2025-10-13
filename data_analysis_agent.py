# data_analysis_agent.py (Versión con capacidad de análisis de tablas en PDF)

import os
import re
import io
import logging
import pandas as pd
import pdfplumber
from typing import Dict, Any, Optional

from smolagents import InferenceClientModel, CodeAgent
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- OPTIMIZACIÓN 1: Instrucciones Generales y Robustas ---
# Ahora incluye la nueva habilidad de leer tablas de PDFs.
GENERAL_NOTES = """
### ROL Y DIRECTIVAS
Eres un experto "Científico de Datos" que escribe código en Python para analizar datos y generar visualizaciones.

### REGLAS DE EJECUCIÓN
1.  **Analiza la Petición**: Lee la petición del usuario para entender qué análisis o visualización necesita.
2.  **Identifica la Fuente de Datos**: La petición mencionará un nombre de archivo (ej. `uploads/mi_archivo.pdf` o `uploads/mis_datos.csv`).
3.  **PROCESA EL ARCHIVO**:
    * Si el archivo es un **PDF**, DEBES usar primero el código `extract_tables_from_pdf(file_path)` para convertir las tablas en DataFrames de pandas. Luego, selecciona el DataFrame más relevante para el análisis.
    * Si el archivo es un **CSV o XLSX**, puedes leerlo directamente con `pd.read_csv(file_path)` o `pd.read_excel(file_path)`.
4.  **Escribe el Código de Análisis**: Usa los DataFrames obtenidos para manipular los datos y `matplotlib.pyplot` o `seaborn` para graficar.
5.  **GUARDA LAS GRÁFICAS (Regla Crítica)**: Para cualquier visualización, DEBES guardarla en un archivo. **NUNCA uses `plt.show()`**. Usa siempre `plt.savefig('figures/nombre_descriptivo.png')`.
6.  **Responde con un Resumen**: Tu respuesta final debe ser un texto que describa el análisis realizado o la gráfica generada.
"""

# --- NUEVA HABILIDAD: Función para extraer tablas de un PDF ---
def extract_tables_from_pdf(file_path: str) -> list:
    """Extrae todas las tablas de un archivo PDF y las devuelve como una lista de DataFrames de pandas."""
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("El archivo proporcionado no es un PDF.")
    
    tables = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            extracted_tables = page.extract_tables()
            for table_data in extracted_tables:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                tables.append(df)
    logging.info(f"Se extrajeron {len(tables)} tablas del archivo {file_path}")
    return tables

# Inicialización del modelo y del agente base
try:
    model = InferenceClientModel("meta-llama/Llama-3.1-70B-Instruct")
    smol_agent_instance = CodeAgent(
        tools=[extract_tables_from_pdf], # <-- Se añade la nueva función como una herramienta
        model=model,
        additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn", "io", "pdfplumber"]
    )
    os.makedirs("figures", exist_ok=True)
except Exception as e:
    logging.error(f"Error al inicializar smolagents: {e}")
    model = None
    smol_agent_instance = None

class DataAnalysisAgentRunnable:
    """
    Wrapper que adapta smol_agent_instance para ser un "Científico de Datos Bajo Demanda",
    compatible con la interfaz de LangGraph.
    """
    name = "data_analysis_agent"

    def __init__(self, smol_agent: Optional[CodeAgent], notes: str):
        self.smol_agent = smol_agent
        self.general_notes = notes

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta el tipo de archivo mencionado en el prompt y ejecuta el análisis.
        """
        if not self.smol_agent:
            return {"output": "Error: El agente de análisis de datos no pudo inicializarse."}

        user_prompt = input_dict.get("input", "").strip()
        if not user_prompt:
            return {"output": "Error: La solicitud para el análisis de datos está vacía."}

        # El prompt ya viene aumentado desde app.py con la ruta del archivo.
        # La lógica del agente se encargará de decidir cómo leerlo.
        augmented_prompt = user_prompt
        
        try:
            history = input_dict.get("history", [])
            response = self.smol_agent.run(
                augmented_prompt,
                additional_notes=self.general_notes
            )
            return {"output": response}
        except Exception as e:
            logging.error(f"Error en la ejecución de smolagents: {e}")
            return {"output": f"Fehler bei der smolagents-Ausführung: {e}"}

# Exportamos la instancia del Wrapper que el Supervisor espera
agent = DataAnalysisAgentRunnable(smol_agent=smol_agent_instance, notes=GENERAL_NOTES)

if __name__ == "__main__":
    # Bloque de prueba para ejecución directa del archivo (opcional)
    print("🔍 Data Analysis Agent Test (Científico de Datos Bajo Demanda)")
    # ... (código de prueba puede ser añadido aquí)