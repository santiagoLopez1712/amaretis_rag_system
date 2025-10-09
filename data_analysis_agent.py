# data_analysis_agent.py (Versión optimizada como "Científico de Datos Bajo Demanda")

import os
import re
import logging
from typing import Dict, Any

from smolagents import InferenceClientModel, CodeAgent
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

GENERAL_NOTES = """
### ROL Y DIRECTIVAS
Eres un experto "Científico de Datos" que escribe código en Python para analizar datos y generar visualizaciones.

### REGLAS DE EJECUCIÓN
1.  **Analiza la Petición**: Lee la petición del usuario para entender qué análisis o visualización necesita.
2.  **Identifica la Fuente de Datos**: La petición contendrá los datos, ya sea como texto/tabla o mencionando un nombre de archivo (ej. `uploads/mi_archivo.csv`).
3.  **Escribe el Código**: Usa `pandas` para manipular los datos y `matplotlib.pyplot` o `seaborn` para graficar.
4.  **GUARDA LAS GRÁFICAS (Regla Crítica)**: Para cualquier visualización, DEBES guardarla en un archivo. **NUNCA uses `plt.show()`**. Usa siempre `plt.savefig('figures/nombre_descriptivo.png')`.
5.  **Responde con un Resumen**: Tu respuesta final debe ser un texto que describa el análisis realizado o la gráfica generada.
"""

try:
    model = InferenceClientModel("meta-llama/Llama-3.1-70B-Instruct")
    smol_agent_instance = CodeAgent(
        tools=[],
        model=model,
        additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn", "io"]
    )
    os.makedirs("figures", exist_ok=True)
except Exception as e:
    logging.error(f"Error al inicializar smolagents: {e}")
    model = None
    smol_agent_instance = None

class DataAnalysisAgentRunnable:
    name = "data_analysis_agent"

    def __init__(self, smol_agent: CodeAgent, notes: str):
        self.smol_agent = smol_agent
        self.general_notes = notes

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.smol_agent:
            return {"output": "Error: El agente de análisis de datos no pudo inicializarse."}

        user_prompt = input_dict.get("input", "").strip()
        if not user_prompt:
            return {"output": "Error: La solicitud para el análisis de datos está vacía."}
            
        # El 'augmented_prompt' es el mismo 'user_prompt' ya que app.py ya lo enriquece
        augmented_prompt = user_prompt
        
        try:
            response = self.smol_agent.run(
                augmented_prompt,
                additional_notes=self.general_notes
            )
            return {"output": response}

        except Exception as e:
            logging.error(f"Error en la ejecución de smolagents: {e}")
            return {"output": f"Fehler bei der smolagents-Ausführung: {e}"}

agent = DataAnalysisAgentRunnable(smol_agent=smol_agent_instance, notes=GENERAL_NOTES)

if __name__ == "__main__":
    print("🔍 Data Analysis Agent Test (Científico de Datos Bajo Demanda)")
    print("Introduce tu petición. Puedes mencionar un archivo o pegar los datos.")
    
    test_prompt_pegado = """
    Analiza los siguientes datos de ventas y crea un gráfico de barras guardado como 'figures/ventas_por_producto.png':
    
    Producto,Ventas
    Laptop,150
    Mouse,300
    Teclado,220
    Monitor,120
    """
    print(f"\n--- Probando con datos pegados ---\n{test_prompt_pegado}")
    respuesta_dict = agent.invoke({"input": test_prompt_pegado})
    print("\n📊 Resultado del Análisis:\n")
    print(respuesta_dict.get("output", "Error al obtener la respuesta."))