# compliance_agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent  
from langchain_core.prompts import ChatPromptTemplate
import re
from datetime import datetime
from typing import List, Dict, Any

class ComplianceAgent:
    """
    Agente especializado en compliance, governance y tratamiento de datos
    para AMARETIS según regulaciones alemanas y europeas
    """
    
    def __init__(self, temperature: float = 0.3):  # Menor temperatura para compliance
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=temperature
        )
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Herramientas específicas para compliance"""
        
        tools = []
        
        # Tool 1: DSGVO/GDPR Compliance Check
        def check_gdpr_compliance(content: str) -> str:
            """Verifica compliance con DSGVO/GDPR"""
            
            # Patrones de datos personales
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
                        "examples": matches[:2],  # Solo primeros 2 ejemplos
                        "recommendation": self._get_gdpr_recommendation(data_type)
                    })
            
            if not findings:
                return "✅ Keine offensichtlichen personenbezogenen Daten gefunden."
            
            report = "🚨 DSGVO-COMPLIANCE PRÜFUNG:\n\n"
            for finding in findings:
                report += f"- {finding['type'].upper()}: {finding['count']} gefunden\n"
                report += f"  Empfehlung: {finding['recommendation']}\n\n"
            
            return report
        
        tools.append(Tool(
            name="check_gdpr_compliance",
            func=check_gdpr_compliance,
            description=(
                "Prüft Inhalte auf DSGVO/GDPR-Compliance und identifiziert "
                "personenbezogene Daten die geschützt werden müssen."
            )
        ))
        
        # Tool 2: Marketing Compliance (UWG, Werbereglementierung)
        def check_marketing_compliance(campaign_content: str) -> str:
            """Prüft Marketing-Compliance nach deutschem Recht"""
            
            # Problematische Begriffe/Praktiken
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
                        "matches": matches,
                        "risk": self._get_compliance_risk(issue_type),
                        "recommendation": self._get_compliance_recommendation(issue_type)
                    })
            
            if not issues:
                return "✅ Keine offensichtlichen Compliance-Probleme gefunden."
            
            report = "⚖️ MARKETING-COMPLIANCE PRÜFUNG:\n\n"
            for issue in issues:
                report += f"🚨 {issue['type'].upper()} - Risiko: {issue['risk']}\n"
                report += f"   Gefunden: {', '.join(issue['matches'])}\n"
                report += f"   Empfehlung: {issue['recommendation']}\n\n"
            
            return report
        
        tools.append(Tool(
            name="check_marketing_compliance",
            func=check_marketing_compliance,
            description=(
                "Prüft Marketing-Inhalte auf Compliance mit deutschem "
                "Werberecht (UWG) und identifiziert potenzielle Probleme."
            )
        ))
        
        # Tool 3: Data Retention Policy
        def check_data_retention(data_info: str) -> str:
            """Prüft und empfiehlt Datenaufbewahrungsrichtlinien"""
            
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
                recommendations = ["📋 Allgemeine Empfehlung: 3 Jahre Aufbewahrung für Marketing-Daten"]
            
            report = "🗂️ DATENAUFBEWAHRUNG EMPFEHLUNGEN:\n\n"
            report += "\n".join(recommendations)
            report += "\n\n💡 Hinweis: Regelmäßige Löschung alter Daten implementieren!"
            
            return report
        
        tools.append(Tool(
            name="check_data_retention",
            func=check_data_retention,
            description=(
                "Gibt Empfehlungen für Datenaufbewahrungszeiten "
                "basierend auf deutschen und EU-Gesetzen."
            )
        ))
        
        return tools
    
    def _get_gdpr_recommendation(self, data_type: str) -> str:
        """Empfehlungen für DSGVO-Compliance"""
        recommendations = {
            "email": "Einverständniserklärung einholen, Opt-out ermöglichen",
            "phone": "Explizite Einwilligung für Telefonmarketing erforderlich", 
            "name": "Datenminimierung beachten, nur notwendige Namen speichern",
            "address": "Zweckbindung beachten, regelmäßig aktualisieren"
        }
        return recommendations.get(data_type, "DSGVO-konforme Verarbeitung sicherstellen")
    
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
            "superlative": "Belege/Studien hinzufügen oder abschwächen",
            "medical_claims": "Nur mit medizinischen Belegen, Disclaimer hinzufügen",
            "financial_promises": "Risiken erwähnen, keine Garantien geben",
            "urgency_pressure": "Zeitdruck reduzieren, ehrlich kommunizieren"
        }
        return recommendations.get(issue_type, "Rechtsberatung einholen")
    
    def _create_agent(self) -> AgentExecutor:
        """Erstellt Compliance-Agent"""
        
        prompt = ChatPromptTemplate.from_template("""
Du bist ein Compliance-Experte bei AMARETIS mit Spezialisierung auf:
- DSGVO/GDPR Datenschutz
- Deutsches Werberecht (UWG)
- Data Governance Best Practices  
- Marketing-Compliance

Deine Aufgabe ist es, alle Marketing-Inhalte und Datenverarbeitungsprozesse 
auf rechtliche Compliance zu prüfen und konkrete Handlungsempfehlungen zu geben.

PRINZIPIEN:
1. Vorsichtsprinzip: Im Zweifel konservativ bewerten
2. Praktikabilität: Umsetzbare Empfehlungen geben
3. Dokumentation: Alle Prüfungen dokumentieren
4. Aktualität: Deutsche/EU-Gesetze berücksichtigen

Verfügbare Tools: {tools}
Tool-Namen: {tool_names}

Anfrage: {input}
Bisherige Prüfungen: {agent_scratchpad}

Format:
Thought: [Analyse der Compliance-Anfrage]
Action: [Tool-Name]
Action Input: [Zu prüfender Inhalt]  
Observation: [Prüfungsergebnis]
... (weitere Prüfungen)
Thought: Compliance-Prüfung abgeschlossen.
Final Answer: [Zusammenfassung + Empfehlungen]
""")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=30
        )
        
        executor.name = "compliance_agent"
        return executor
    
    def audit_content(self, content: str, content_type: str = "marketing") -> str:
        """Führt vollständige Compliance-Prüfung durch"""
        
        audit_request = f"""
        Bitte führe eine vollständige Compliance-Prüfung durch für:
        
        CONTENT-TYP: {content_type}
        INHALT: {content}
        
        Prüfe auf:
        1. DSGVO/GDPR Compliance
        2. Marketing-Compliance (UWG)
        3. Datenaufbewahrung (falls relevant)
        
        Gib konkrete, umsetzbare Empfehlungen.
        """
        
        try:
            result = self.agent.invoke({"input": audit_request})
            return result.get("output", result)
        except Exception as e:
            return f"Fehler bei Compliance-Prüfung: {e}"

# Convenience function
def create_compliance_agent() -> ComplianceAgent:
    """Erstellt Compliance Agent Instanz"""
    return ComplianceAgent()