# 📦 Notwendige Bibliotheken importieren
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import InferenceClientModel, CodeAgent 
from dotenv import load_dotenv
from huggingface_hub import login

# 🔑 HuggingFace-Token laden und anmelden
load_dotenv()

# 🧠 LLM-Modell mit Inference API initialisieren
model = InferenceClientModel("meta-llama/Llama-3.1-70B-Instruct")

# 🤖 Agent definieren mit erlaubten Bibliotheken
agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=[
        "numpy",
        "pandas",
        "matplotlib.pyplot",
        "seaborn"
    ]
)

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

def generate_apple_profit_plot():
    """Erstellt ein Diagramm für den Gewinn von Apple in den letzten 3 Jahren aus all_company_financials.csv und speichert es unter figures/apple_profit_last3years.png. Gibt den Bildpfad zurück."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    # CSV laden
    df = pd.read_csv("all_company_financials.csv")
    # Apple, concept=Gewinn filtern
    df_apple = df[(df['company'].str.lower() == 'apple') & (df['concept'].str.lower().str.contains('gewinn'))]
    # Nur Spalten mit Jahreszahlen extrahieren
    year_cols = [col for col in df_apple.columns if re.match(r"^20\\d{2}", col)]
    # Letzte 3 Jahre finden
    years = sorted(year_cols)[-3:]
    values = [float(df_apple[y].values[0]) if not df_apple[y].isnull().all() else 0 for y in years]
    # Plot
    plt.figure(figsize=(6,4))
    plt.bar(years, values, color="#0071c5")
    plt.title("Apple Gewinn (letzte 3 Jahre)")
    plt.ylabel("Gewinn (Mrd. USD)")
    plt.xlabel("Jahr")
    plt.tight_layout()
    # Dateiname: Nur Buchstaben/Zahlen, alles lowercase
    safe_name = re.sub(r'[^a-z0-9]', '', 'Apple')
    fname = f"{safe_name}_profit_last3years.png"
    out_path = os.path.join("figures", fname)
    plt.savefig(out_path)
    plt.close()
    return out_path

if __name__ == "__main__":
    # 📣 Benutzerinteraktion – Eingabeaufforderung
    print("🔍 Bitte gib deinen Analyseauftrag ein (z.B. 'Vergleiche die Verbindlichkeiten von Apple und Microsoft im Jahr 2024.'):\n")
    user_prompt = input("> ")

    # 🏃 Agent ausführen mit Analyseauftrag
    antwort = agent.run(
        user_prompt,
        additional_args={
            "source_file": "all_company_financials.csv",
            "additional_notes": additional_notes
        }
    )

    # 🖨 Ergebnis anzeigen
    print("\n📊 Analyseergebnis:\n")
    print(antwort)
