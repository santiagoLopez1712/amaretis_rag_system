# 🎯 AMARETIS Marketing Intelligence System  
*Multi-Agent RAG Platform for Strategic Marketing – Göttingen, Germany*  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Integrated-orange)
![Gemini](https://img.shields.io/badge/LLM-Gemini--1.5--Pro-yellow)
![Gradio](https://img.shields.io/badge/UI-Gradio-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Overview | Descripción General

**AMARETIS Marketing Intelligence System** es una plataforma de inteligencia artificial **multi-agente** que combina razonamiento estratégico, recuperación aumentada por generación (RAG) y análisis de datos automatizado.  
Desarrollada en **Göttingen, Alemania**, su objetivo es ofrecer soporte inteligente para campañas de marketing, compliance legal y análisis de información empresarial.  

> The system is designed as a modular AI architecture where each agent specializes in a different cognitive function — from research and compliance to RAG-based question answering.

---

## 🧩 System Architecture | Arquitectura del Sistema

┌──────────────────────────┐
│ Gradio Frontend │ ← (User Interface)
└────────────┬─────────────┘
│
┌────────────▼────────────┐
│ 🧭 SupervisorManager │ ← Orquestador de agentes
└────────────┬────────────┘
│
┌───────────┼────────────────────────────────────────┐
│ │
▼ ▼
🧠 RAG Agent (rag_agent) 📊 Data Analysis Agent

Respuestas con fuentes - Análisis y visualización segura

Búsqueda en vectorstore (Chroma / FAISS) - Genera gráficos con UUID únicos
│
▼
💡 Integrated Marketing Agent

Estratega de campañas

Toma contexto y historial
│
▼
🧾 Brief Generator Agent

Planificador táctico de briefings
│
▼
⚖️ Compliance Agent

Analiza riesgos legales con reglas + LLM
│
▼
🌐 Web Search Agent

Investigación online con citas verificables

yaml
Code kopieren

---

## 🚀 Key Features | Características Principales

- **Multi-Agente Orquestado:** coordinación automática por el `SupervisorManager`.
- **Arquitectura RAG Completa:** integración con Chroma y embeddings de HuggingFace.
- **Análisis de Datos Seguro:** ejecución aislada (sandbox o REPL) sin vulnerabilidades.
- **Compliance Inteligente:** reglas externas y razonamiento jurídico basado en IA.
- **Interfaz Amigable (Gradio):** chat interactivo con carga de PDF, CSV y Excel.
- **Citas y Transparencia:** cada respuesta incluye referencias a sus fuentes.

---

## ⚙️ Installation | Instalación

```bash
git clone https://github.com/santiagoLopez1712/amaretis_rag_system.git
cd amaretis_rag_system
pip install -r requirements.txt
Crea un archivo .env en la raíz del proyecto con tus claves de API:

bash
Code kopieren
GOOGLE_API_KEY=tu_clave_gemini
TAVILY_API_KEY=tu_clave_tavily
▶️ Running the App | Ejecución de la Aplicación
bash
Code kopieren
python app.py --port 7860
Luego abre en tu navegador:

arduino
Code kopieren
http://localhost:7860
🔐 Security & Robustness | Seguridad y Robustez
✅ Sandboxed Execution: el agente de análisis ejecuta código en entorno seguro (sin exec).
✅ Unique Filenames: generación de gráficos con nombres UUID, evitando conflictos concurrentes.
✅ External Configs: reglas de compliance y parámetros de LLM se gestionan desde archivos JSON/YAML.
✅ Centralized Model Setup: todos los agentes usan la configuración unificada del modelo gemini-1.5-pro.

🧠 Agents Overview | Descripción de Agentes
Agente	Rol Principal	Tecnologías Clave
rag_agent.py	Recuperación y respuestas con contexto + citas	LangChain, Chroma, HuggingFace
integrated_marketing_agent.py	Estrategia de marketing contextual	LCEL, Prompt Design
brief_generator_agent.py	Generación de briefings estructurados	ReAct Agent, Tools
compliance_agent.py	Análisis legal con reglas + IA	Regex, LLM, YAML Config
data_analysis_agent.py	Análisis de datos y visualización segura	Pandas, Matplotlib, Sandbox
web_such_agent.py	Búsqueda y síntesis web con fuentes	TavilySearch, Web Scraping
supervisor_main.py	Orquestador principal de agentes	LangGraph, Routing dinámico
data_loader.py / data_chunkieren.py	Procesamiento de documentos	PyPDF, Text Chunking
app.py	Interfaz con Gradio	UI + File Upload

💾 Data Flow | Flujo de Datos
📥 Carga y procesamiento: data_loader extrae y limpia los documentos.

🔍 Segmentación: data_chunkieren divide textos para embeddings.

🧬 Vectorización: creación de Chroma o FAISS en memoria.

💡 Consulta: el rag_agent busca la respuesta más relevante.

🧩 Supervisor: decide qué agente manejará la solicitud.

🖼️ Salida: app.py muestra texto, imagen o gráfico en la interfaz Gradio.

🧭 Example Interaction | Ejemplo de Interacción
Usuario:

“Analiza este PDF y dime qué estrategias de marketing aplicaron para fidelizar clientes.”

Asistente:

“El documento muestra tres enfoques clave: programas de puntos, comunicación omnicanal y personalización dinámica.

📚 Fuente: kampagnenbericht_q2.pdf, páginas 3–5”

🔮 Future Enhancements | Mejoras Futuras
Integración de FAISS en memoria para análisis de PDFs más rápidos.

Persistencia del historial de conversación entre sesiones.

Modo multiusuario concurrente con gestión de sesiones aisladas.

Dashboard analítico web para visualización de resultados agregados.

🏢 About AMARETIS | Sobre AMARETIS
AMARETIS es una agencia creativa en Göttingen, Alemania, especializada en diseño, comunicación estratégica y soluciones de inteligencia artificial aplicadas al marketing.

Este sistema fue desarrollado como parte del laboratorio interno de IA para optimizar procesos de análisis, planificación y cumplimiento normativo.

👤 Author | Autor
Santiago López Otálvaro
Web & AI Specialist · AMARETIS
📍 Göttingen, Germany
🌐 github.com/santiagoLopez1712

📄 License
Este proyecto está licenciado bajo la MIT License.
Puedes usarlo, modificarlo y adaptarlo libremente con atribución al autor original.

⭐ If you like this project, consider giving it a star on GitHub!

yaml
Code kopieren

---

¿Quieres que te genere también la **versión en formato `README_ES.md` (solo español)** para incluir como segunda opción en el repo (por ejemplo, `README_EN.md` y `README_ES.md`)?  
Así tendrías una versión técnica bilingüe y otra local para presentaciones en Alemania.