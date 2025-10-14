# compliance_agent.py (Refactorizado para configuración centralizada)

import os
import re
import logging
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
    
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.3):
        self.llm = ChatVertexAI(
            project=PROJECT_ID,
            location=REGION,
            model=model_name, # Usar configuración centralizada
            temperature=temperature
        )
        self.tools = self._setup_tools()
        self.agent: Optional[AgentExecutor] = self._create_agent()
    
    def _tool_check_gdpr_compliance(self, content: str) -> str:
        """Verifica compliance con DSGVO/GDPR."""
        personal_data_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'(\+49|0)\d{2,4}[-\s]?\d{6,8}',
            "name": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            "address": r'\d+\s+[A-Za-z\s]+,\s*\d{5}\s+[A-Za-z]+',
        }
        
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
        compliance_issues = {
            "superlative": r'\b(beste[rn]?|einzig|weltweit führend|revolutionär)\b',
            "medical_claims": r'\b(heilt|therapiert|medizinisch bewiesen)\b', 
            "financial_promises": r'\b(garantiert|risikofrei|sicher verdienen)\b',
            "urgency_pressure": r'\b(nur heute|letzte chance|sofort)\b',
        }
        
        issues = []
        for issue_type, pattern in compliance_issues.items():
            matches = re.findall(pattern, campaign_content.lower())
            if matches:
                issues.append({
                    "type": issue_type,
                    "matches": list(set(matches)),
                    "risk": self._get_compliance_risk(issue_type),
                    "recommendation": self._get_compliance_recommendation(issue_type)
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
        retention_guidelines = {
            "customer_data": "3 Jahre nach letztem Kontakt (DSGVO Art. 5)",
            "campaign_data": "5 Jahre für Reporting-Zwecke", 
            "financial_data": "10 Jahre (HGB §257)",
            "web_analytics": "14 Monate (TTDSG)",
            "email_marketing": "Bis Widerruf + 3 Jahre Nachweis"
        }
        
        recommendations = []
        for data_type, retention in retention_guidelines.items():
            if data_type.replace("_", " ") in data_info.lower():
                recommendations.append(f"📋 {data_type.title()}: {retention}")
        
        if not recommendations:
            recommendations = ["📋 Allgemeine Empfehlung: 3 Jahre Aufbewahrung für Marketing-Daten, sofern keine gesetzlichen Ausnahmen gelten."]
        
        report = "🗂️ DATENAUFBEWAHRUNG EMPFEHLUNGEN:\n\n"
        report += "\n".join(recommendations)
        report += "\n\n💡 Hinweis: Implementieren Sie einen Prozess zur regelmäßigen Löschung alter Daten!"
        
        return report

    def _setup_tools(self) -> List[Tool]:
        """Herramientas específicas para compliance"""
        tools = [
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
        return tools
    
    def _get_gdpr_recommendation(self, data_type: str) -> str:
        """Empfehlungen für DSGVO-Compliance"""
        recommendations = {
            "email": "Einverständniserklärung einholen, Opt-out ermöglichen.",
            "phone": "Explizite Einwilligung für Telefonmarketing erforderlich (Double-Opt-In empfohlen).", 
            "name": "Datenminimierung beachten, nur notwendige Namen speichern.",
            "address": "Zweckbindung beachten, regelmäßig aktualisieren und nur für den vereinbarten Zweck verwenden."
        }
        return recommendations.get(data_type, "DSGVO-konforme Verarbeitung sicherstellen.")
    
    def _get_compliance_risk(self, issue_type: str) -> str:
        """Risikobewertung für Compliance-Issues"""
        risk_levels = {
            "superlative": "MITTEL",
            "medical_claims": "HOCH", 
            "financial_promises": "HOCH",
            "urgency_pressure": "NIEDRIG"
        }
        return risk_levels.get(issue_type, "MITTEL")
    
    def _get_compliance_recommendation(self, issue_type: str) -> str:
        """Empfehlungen für Compliance-Issues"""
        recommendations = {
            "superlative": "Aussagen belegen (z.B. mit Studien) oder abschwächen (z.B. 'einer der führenden...').",
            "medical_claims": "Strengstens vermeiden, es sei denn, es handelt sich um ein zugelassenes medizinisches Produkt mit Belegen.",
            "financial_promises": "Vermeiden. Stattdessen über potenzielle Vorteile sprechen und Risiken erwähnen.",
            "urgency_pressure": "Zeitdruck reduzieren und transparent kommunizieren (z.B. 'Angebot gültig bis...')."
        }
        return recommendations.get(issue_type, "Rechtsberatung einholen.")
    
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

