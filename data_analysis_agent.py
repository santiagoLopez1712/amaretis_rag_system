# data_analysis_agent.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import InferenceClientModel, CodeAgent 
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda # <--- NEUER IMPORT
from typing import Dict, Any
import re # Notwendig für die Beispiel-Funktion

# 🔑 HuggingFace-Token laden und anmelden
load_dotenv()

# 🧠 LLM-Modell mit Inference API initialisieren
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
smol_agent_instance.name = "data_analysis_agent"

# 📁 Sicherstellen, dass der Ausgabeordner existiert
os.makedirs("figures", exist_ok=True)

# 📓 Zusätzliche Notizen (z.B. Beschreibung der Spalten)
additional_notes = """
### Variablenbeschreibung:
- 'company': Name des Unternehmens
- 'concept': Finanzkennzahl (z.B. Umsatz, Ausgaben, Eigenkapital)
- Spalten im Format '2024-03-31': stellen Quartalsdaten dar
- Diese Datei enthält Finanzergebnisse verschiedener Unternehmen über mehrere Quartale hinweg.
"""

# ⚙️ WRAPPER-FUNKTION: Führt smolagents.run aus
def smol_agent_runner(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Funktion adaptiert den smol_agent für die LangGraph/LangChain AgentState.
    """
    # LangChain/LangGraph AgentExecutor sendet die Eingabe unter der 'input'-Key
    user_prompt = input_dict.get("input", "")

    if not user_prompt:
        return {"output": "Fehler: Keine Eingabe im Diktat gefunden."}

    # Ausführung des smolagents.CodeAgent mit der internen Methode .run()
    antwort = smol_agent_instance.run(
        user_prompt,
        additional_args={
            "source_file": "all_company_financials.csv",
            "additional_notes": additional_notes
        }
    )
    
    # Rückgabe des Ergebnisses im LangChain-AgentExecutor-Format (mit 'output')
    return {"output": antwort}

# 🚀 Exportieren Sie die LangChain RunnableLambda-Instanz, die der Supervisor erwartet
agent = RunnableLambda(smol_agent_runner)
agent.name = "data_analysis_agent" 

def generate_apple_profit_plot():
    """Erstellt ein Diagramm für den Gewinn von Apple in den letzten 3 Jahren aus all_company_financials.csv und speichert es unter figures/apple_profit_last3years.png. Gibt den Bildpfad zurück."""
    # ... (Ihre Implementierung bleibt hier erhalten)
    return "figures/apple_profit_last3years.png" # Platzhalter

if __name__ == "__main__":
    # 📣 Benutzerinteraktion – Eingabeaufforderung
    print("🔍 Bitte gib deinen Analyseauftrag ein (z.B. 'Vergleiche die Verbindlichkeiten von Apple und Microsoft im Jahr 2024.'):\n")
    user_prompt = input("> ")

    # 🏃 Hier verwenden wir weiterhin die smol_agent_instance für die direkte Konsolen-Ausführung
    antwort = smol_agent_instance.run(
        user_prompt,
        additional_args={
            "source_file": "all_company_financials.csv",
            "additional_notes": additional_notes
        }
    )

    # 🖨 Ergebnis anzeigen
    print("\n📊 Analyseergebnis:\n")
    print(antwort)