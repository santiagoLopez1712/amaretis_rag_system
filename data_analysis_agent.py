# data_analysis_agent.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import InferenceClientModel, CodeAgent 
from dotenv import load_dotenv
# from langchain_core.runnables import RunnableLambda # <--- Ya no es necesario
from typing import Dict, Any, List

# 🔑 HuggingFace-Token laden und anmelden
load_dotenv()

# 📓 Zusätzliche Notizen (z.B. Beschreibung der Spalten)
additional_notes = """
### Variablenbeschreibung:
- 'company': Name des Unternehmens
- 'concept': Finanzkennzahl (z.B. Umsatz, Ausgaben, Eigenkapital)
- Spalten im Format '2024-03-31': stellen Quartalsdaten dar
- Diese Datei enthält Finanzergebnisse verschiedener Unternehmen über mehrere Quartale hinweg.
"""

# 🧠 LLM-Modell mit Inference API initialisieren
# Nota: La librería smolagents maneja internamente la autenticación si HF_TOKEN está en .env
model = InferenceClientModel("meta-llama/Llama-3.1-70B-Instruct")

# 🤖 Interne smolagents-Instanz definieren
smol_agent_instance = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=[
        "numpy",
        "pandas",
        "matplotlib.pyplot",
        "seaborn"
    ]
)
# smol_agent_instance.name = "data_analysis_agent" # El nombre se define en la clase wrapper

# 📁 Sicherstellen, dass der Ausgabeordner existiert
os.makedirs("figures", exist_ok=True)


# ⚙️ WRAPPER-CLASE: Imita la interfaz de AgentExecutor
# ESTA CLASE ES LA SOLUCIÓN CLAVE para que LangGraph lo compile.
class DataAnalysisAgentRunnable:
    """
    Clase wrapper que adapta smol_agent_instance para la interfaz
    Runnable/AgentExecutor, permitiendo su uso en LangGraph.
    """
    def __init__(self, smol_agent: CodeAgent, notes: str):
        self.smol_agent = smol_agent
        self.additional_notes = notes
        # Definición del nombre para el supervisor
        self.name = "data_analysis_agent" 

    # Método clave que LangChain/LangGraph invoca
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Función adaptada para LangChain/LangGraph AgentState, recibe {'input': str}
        y devuelve {'output': str}.
        """
        user_prompt = input_dict.get("input", "")

        if not user_prompt:
            return {"output": "Fehler: Keine Eingabe im Diktat gefunden."}

        # Ausführung des smolagents.CodeAgent mit der internen Methode .run()
        try:
            # La invocación del agente es directa al método .run() de smolagents
            antwort = self.smol_agent.run(
                user_prompt,
                additional_args={
                    "source_file": "all_company_financials.csv",
                    "additional_notes": self.additional_notes
                }
            )
        except Exception as e:
            return {"output": f"Fehler bei der smolagents-Ausführung: {e}"}
        
        # Rückgabe del resultado en el formato de AgentExecutor/LangGraph
        return {"output": antwort}

# 🚀 Exportar la instancia de la Clase Wrapper que el Supervisor espera
# El objeto 'agent' es ahora una instancia de la clase wrapper
agent = DataAnalysisAgentRunnable(smol_agent=smol_agent_instance, notes=additional_notes) 


# --- La función generate_apple_profit_plot() no es relevante para el Supervisor ---
def generate_apple_profit_plot():
    """Erstellt ein Diagramm für den Gewinn von Apple in den letzten 3 Jahren aus all_company_financials.csv und speichert es unter figures/apple_profit_last3years.png. Gibt den Bildpfad zurück."""
    # ... (Ihre Implementierung bleibt hier erhalten)
    return "figures/apple_profit_last3years.png" # Platzhalter

if __name__ == "__main__":
    # Test de Consola
    print("🔍 Data Analysis Agent Test")
    print("Bitte gib deinen Analyseauftrag ein (z.B. 'Vergleiche die Verbindlichkeiten von Apple und Microsoft im Jahr 2024.'):\n")
    user_prompt = input("> ")

    # Usamos el wrapper 'agent' para probar la interfaz del Supervisor
    respuesta_dict = agent.invoke({"input": user_prompt})

    print("\n📊 Analyseergebnis:\n")
    print(respuesta_dict.get("output", "Error al obtener la respuesta."))