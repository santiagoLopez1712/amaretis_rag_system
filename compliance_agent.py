# compliance_agent.py (Refactorizado para leer reglas desde YAML)

import os
import re
import logging
import yaml
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada.")

REGION = os.getenv("GOOGLE_CLOUD_REGION")
if not REGION:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_REGION no está configurada.")

class ComplianceAgent:
    """
    Agente especializado en compliance, governance y tratamiento de datos
    para AMARETIS según regulaciones alemanas y europeas.
    """
    name = "compliance_agent" 
    
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.3, rules_path: str = "compliance_rules.yaml"):
        self.llm = ChatVertexAI(
            project=PROJECT_ID,
            location=REGION,
            model=model_name,
            temperature=temperature
        )
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                self.rules = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Archivo de reglas de compliance no encontrado en: {rules_path}")
            self.rules = {}
        except Exception as e:
            logger.error(f"Error cargando o parseando el archivo YAML de reglas: {e}")
            self.rules = {}

        self.tools = self._setup_tools()
        self.agent: Optional[AgentExecutor] = self._create_agent()
    
    def _tool_check_gdpr_compliance(self, content: str) -> str:
        """Verifica compliance con DSGVO/GDPR."""
        personal_data_patterns = self.rules.get('gdpr_patterns', {})
        findings = []
        for data_type, pattern in personal_data_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                findings.append({
                    "type": data_type,
                    "count": len(matches),
                    "examples": matches[:2],
                    "recommendation": self._get_gdpr_recommendation(data_type)
                })
        
        if not findings:
            return "✅ Keine offensichtlichen personenbezogenen Daten gefunden."
        
        report = "🚨 DSGVO-COMPLIANCE PRÜFUNG:\n\n"
        for finding in findings:
            report += f"- {finding['type'].upper()}: {finding['count']} gefunden\n"
            report += f"  Empfehlung: {finding['recommendation']}\n\n"
        
        return report

    def _tool_check_marketing_compliance(self, campaign_content: str) -> str:
        """Prüft Marketing-Compliance nach deutschem Recht."""
        compliance_issues = self.rules.get('marketing_compliance', {})
        issues = []
        for issue_type, details in compliance_issues.items():
            if not isinstance(details, dict): continue
            pattern = details.get('pattern')
            if not pattern: continue

            matches = re.findall(pattern, campaign_content.lower())
            if matches:
                issues.append({
                    "type": issue_type,
                    "matches": list(set(matches)),
                    "risk": details.get('risk', 'UNBEKANNT'),
                    "recommendation": details.get('recommendation', self.rules.get('marketing_compliance', {}).get('default_recommendation', 'Rechtsberatung einholen.'))
                })
        
        if not issues:
            return "✅ Keine offensichtlichen Compliance-Probleme gefunden."
        
        report = "⚖️ MARKETING-COMPLIANCE PRÜFUNG:\n\n"
        for issue in issues:
            report += f"🚨 {issue['type'].upper()} - Risiko: {issue['risk']}\n"
            report += f"  Gefunden: {', '.join(issue['matches'])}\n"
            report += f"  Empfehlung: {issue['recommendation']}\n\n"
        
        return report

    def _tool_check_data_retention(self, data_info: str) -> str:
        """Prüft und empfiehlt Datenaufbewahrungsrichtlinien."""
        retention_guidelines = self.rules.get('data_retention', {}).get('guidelines', {})
        recommendations = []
        for data_type, retention in retention_guidelines.items():
            if data_type.replace(" ", " ") in data_info.lower():
                recommendations.append(f"📋 {data_type.title()}: {retention}")
        
        if not recommendations:
            default_rec = self.rules.get('data_retention', {}).get('default_recommendation', "Keine spezifische Empfehlung gefunden.")
            recommendations.append(default_rec)
        
        report = "🗂️ DATENAUFBEWAHRUNG EMPFEHLUNGEN:\n\n"
        report += "\n".join(recommendations)
        report += "\n\n💡 Hinweis: Implementieren Sie einen Prozess zur regelmäßigen Löschung alter Daten!"
        
        return report

    def _setup_tools(self) -> List[Tool]:
        """Herramientas específicas para compliance"""
        return [
            Tool(
                name="check_gdpr_compliance",
                func=self._tool_check_gdpr_compliance,
                description="Prüft Inhalte auf DSGVO/GDPR-Compliance und identifiziert personenbezogene Daten die geschützt werden müssen."
            ),
            Tool(
                name="check_marketing_compliance",
                func=self._tool_check_marketing_compliance,
                description="Prüft Marketing-Inhalte auf Compliance mit deutschem Werberecht (UWG) und identifiziert potenzielle Probleme."
            ),
            Tool(
                name="check_data_retention",
                func=self._tool_check_data_retention,
                description="Gibt Empfehlungen für Datenaufbewahrungszeiten basierend auf deutschen und EU-Gesetzen."
            )
        ]
    
    def _get_gdpr_recommendation(self, data_type: str) -> str:
        """Empfehlungen für DSGVO-Compliance aus den Regeln."""
        recommendations = self.rules.get('gdpr_recommendations', {})
        return recommendations.get(data_type, recommendations.get('default', "DSGVO-konforme Verarbeitung sicherstellen."))
    
    # Las funciones _get_compliance_risk y _get_compliance_recommendation ya no son necesarias
    # porque su lógica ahora está dentro de _tool_check_marketing_compliance

    def _create_agent(self) -> AgentExecutor:
        """Erstellt Compliance-Agent"""
        prompt = ChatPromptTemplate.from_template("""
        Du bist ein "Compliance-Experte" bei AMARETIS, spezialisiert auf deutsches und EU-Recht.
        Deine Aufgabe ist es, Marketing-Inhalte und Datenprozesse auf rechtliche Risiken zu prüfen und klare, umsetzbare Handlungsempfehlungen zu geben.

        **PRINZIPIEN:**
        1.  **Vorsicht**: Im Zweifel immer den konservativeren, sichereren Weg empfehlen.
        2.  **Praktikabilität**: Gib Empfehlungen, die ein Marketing-Team verstehen und umsetzen kann.
        3.  **Genauigkeit**: Nutze deine Werkzeuge, um eine detaillierte und genaue Analyse zu erstellen.

        **VERFÜGBARE TOOLS:**
        {tools}

        **FORMAT:**
        Thought: [Deine Analyse der Anfrage und welches Werkzeug am besten passt.]
        Action: [Der Name des Werkzeugs aus [{tool_names}]]
        Action Input: [Der Inhalt, der geprüft werden soll.]  
        Observation: [Das Ergebnis des Werkzeugs.]
        Thought: Die Prüfung ist abgeschlossen. Ich fasse die Ergebnisse und Empfehlungen zusammen.
        Final Answer: [Eine klare Zusammenfassung der gefundenen Risiken und konkrete nächste Schritte.]

        **BEGINNE JETZT**

        Anfrage: {input}
        Bisherige Prüfungen: {history}
        Dein Gedankengang:
        {agent_scratchpad}
        """)
        
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=45
        )
        return executor
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Kompatibilitätsmethode für LangGraph, die den Verlauf korrekt weitergibt."""
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: Eingabe für Compliance-Prüfung ist leer."}
        try:
            history = input_dict.get("history", [])
            result = self.agent.invoke({"input": user_input, "history": history})
            final_output = result.get("output", str(result))
            return {"output": final_output}
        except Exception as e:
            logger.error(f"Error bei der Ausführung des Compliance Agent: {e}")
            return {"output": f"Fehler bei der Compliance-Prüfung: {e}"}