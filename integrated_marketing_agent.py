
# integrated_marketing_agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
import re
from datetime import datetime
import json
from typing import List, Dict, Any, Optional

# ==============================
# 1️⃣ Compliance Agent Optimizado
# ==============================
class ComplianceAgent:
    """Agente de compliance (DSGVO/GDPR, UWG, Data Retention)"""

    def __init__(self, temperature: float = 0.3):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temperature)
        self.tools = self._setup_tools()
        self.agent = self._create_agent()

    def _setup_tools(self) -> List[Tool]:
        tools = []

        # DSGVO/GDPR Check
        def check_gdpr_compliance(content: str) -> str:
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
            return findings if findings else []

        tools.append(Tool(
            name="check_gdpr_compliance",
            func=check_gdpr_compliance,
            description="Detecta datos personales y sugiere cumplimiento DSGVO/GDPR"
        ))

        # Marketing Compliance (UWG)
        def check_marketing_compliance(content: str) -> str:
            compliance_issues = {
                "superlative": r'\b(beste[rn]?|einzig|weltweit führend|revolutionär)\b',
                "medical_claims": r'\b(heilt|therapiert|medizinisch bewiesen)\b',
                "financial_promises": r'\b(garantiert|risikofrei|sicher verdienen)\b',
                "urgency_pressure": r'\b(nur heute|letzte chance|sofort)\b',
            }
            issues = []
            for issue_type, pattern in compliance_issues.items():
                matches = re.findall(pattern, content.lower())
                if matches:
                    issues.append({
                        "type": issue_type,
                        "matches": matches,
                        "risk": self._get_compliance_risk(issue_type),
                        "recommendation": self._get_compliance_recommendation(issue_type)
                    })
            return issues

        tools.append(Tool(
            name="check_marketing_compliance",
            func=check_marketing_compliance,
            description="Analiza contenido de marketing para riesgos legales"
        ))

        # Data Retention
        def check_data_retention(content: str) -> str:
            retention_guidelines = {
                "customer_data": "3 años tras último contacto (DSGVO Art. 5)",
                "campaign_data": "5 años para reporting",
                "financial_data": "10 años (HGB §257)",
                "web_analytics": "14 meses (TTDSG)",
                "email_marketing": "Hasta revocación + 3 años prueba"
            }
            recommendations = []
            for data_type, retention in retention_guidelines.items():
                if data_type.replace("_", " ") in content.lower():
                    recommendations.append({"type": data_type, "retention": retention})
            if not recommendations:
                recommendations.append({"type": "general", "retention": "3 años para datos de marketing"})
            return recommendations

        tools.append(Tool(
            name="check_data_retention",
            func=check_data_retention,
            description="Sugerencias de retención de datos según leyes alemanas y EU"
        ))

        return tools

    def _get_gdpr_recommendation(self, data_type: str) -> str:
        recommendations = {
            "email": "Solicitar consentimiento, ofrecer opt-out",
            "phone": "Consentimiento explícito requerido",
            "name": "Minimizar datos, solo almacenar necesarios",
            "address": "Respetar finalidad, mantener actualizado"
        }
        return recommendations.get(data_type, "Asegurar procesamiento conforme DSGVO")

    def _get_compliance_risk(self, issue_type: str) -> str:
        risk_levels = {
            "superlative": "MEDIO",
            "medical_claims": "ALTO",
            "financial_promises": "ALTO",
            "urgency_pressure": "BAJO"
        }
        return risk_levels.get(issue_type, "MEDIO")

    def _get_compliance_recommendation(self, issue_type: str) -> str:
        recommendations = {
            "superlative": "Añadir evidencia o moderar afirmación",
            "medical_claims": "Solo con respaldo médico, añadir disclaimer",
            "financial_promises": "Mencionar riesgos, no dar garantías",
            "urgency_pressure": "Reducir presión temporal"
        }
        return recommendations.get(issue_type, "Consultar asesoría legal")

    def audit_content(self, content: str) -> Dict[str, Any]:
        """Ejecuta auditoría completa y retorna JSON"""
        return {
            "gdpr_findings": self.tools[0].func(content),
            "marketing_findings": self.tools[1].func(content),
            "data_retention": self.tools[2].func(content),
            "audit_timestamp": datetime.utcnow().isoformat()
        }


# =====================================
# 2️⃣ Brief Generator Agent Optimizado
# =====================================
class BriefGeneratorAgent:
    """Genera briefs de marketing estructurados"""

    def __init__(self, vectorstore, temperature: float = 0.7):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temperature)
        self.vectorstore = vectorstore

    def generate_brief(self, client_info: str, campaign_requirements: str, additional_context: str = "") -> Dict[str, Any]:
        """Genera un brief y lo retorna como JSON"""
        input_text = f"""
        CLIENT INFORMATION:
        {client_info}
        
        CAMPAIGN REQUIREMENTS:
        {campaign_requirements}
        
        ADDITIONAL CONTEXT:
        {additional_context}
        """
        try:
            response = self.llm.invoke(input_text)
            brief_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            brief_text = f"Error generando brief: {e}"

        # Aquí puedes hacer parsing adicional si quieres extraer secciones
        return {
            "brief_text": brief_text,
            "generated_at": datetime.utcnow().isoformat()
        }


# =====================================
# 3️⃣ Pipeline Integrado
# =====================================
class MarketingPipeline:
    """Pipeline que genera briefs y audita compliance"""

    def __init__(self, vectorstore):
        self.brief_agent = BriefGeneratorAgent(vectorstore)
        self.compliance_agent = ComplianceAgent()

    def generate_and_audit(self, client_info: str, campaign_requirements: str, additional_context: str = "") -> Dict[str, Any]:
        # 1. Generar brief
        brief = self.brief_agent.generate_brief(client_info, campaign_requirements, additional_context)

        # 2. Auditoría de compliance
        audit = self.compliance_agent.audit_content(brief["brief_text"])

        # 3. Resultado integrado
        return {
            "brief": brief,
            "compliance_audit": audit
        }


# ===============================
# Función de conveniencia
# ===============================
def create_marketing_pipeline(vectorstore) -> MarketingPipeline:
    return MarketingPipeline(vectorstore)
