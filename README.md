# 🎯 AMARETIS Marketing Intelligence System

## 🌍 Overview
**AMARETIS Marketing Intelligence System** is a multi-agent AI system designed in Göttingen (Germany) to support marketing teams and decision-makers in tasks such as campaign planning, compliance validation, data analysis, and RAG-based knowledge management.

The system combines advanced LLM orchestration (via **LangGraph** and **LangChain**) with specialized AI agents — each trained for a distinct domain — all coordinated by a central **SupervisorManager**.  
It integrates **retrieval-augmented generation (RAG)**, **compliance auditing**, and **automated brief generation** into a unified intelligent assistant.

---

## 🧠 Architecture

### Agent Ecosystem

| Agent | Role | Description |
|--------|------|-------------|
| 🧩 **SupervisorManager** | Orchestrator | Routes user queries intelligently to the appropriate specialized agent via LangGraph. |
| 📚 **RAG Agent** | Knowledge Retrieval | Executes context-aware QA over stored knowledge bases and uploaded documents using Chroma or FAISS. |
| 📊 **Data Analysis Agent** | Data Scientist | Analyzes CSV/XLSX/PDF data, generating safe Python visualizations in sandboxed environments. |
| ⚖️ **Compliance Agent** | Legal & Governance Expert | Checks GDPR, UWG, and data retention compliance using hybrid rules + LLM reasoning. |
| 📝 **Brief Generator Agent** | Marketing Planner | Produces structured, strategic briefs with segmentation, SMART goals, and creative guidance. |
| 🌐 **research_agent** | Research Analyst | Performs multi-step online research with source citations using TavilySearch and web scraping. |
| 💼 **Integrated Marketing Agent** | Strategic Synthesizer | Provides high-level marketing strategy and synthesis across all agents’ outputs. |

---

## 🧩 System Flow

1. **User Interaction via Gradio UI**
   - Users upload files (PDF, CSV, Excel) or ask questions directly.
   - Input is processed and augmented with file-based instructions.

2. **SupervisorManager**
   - Routes the question to the appropriate agent (e.g., Compliance, Data Analysis, RAG).
   - Maintains conversational context and executes the selected agent’s chain.

3. **Specialized Agents**
   - Each agent operates autonomously and returns structured outputs with citations or sources.

4. **Output Rendering**
   - Results (text, charts, insights) are displayed in the **Gradio Interface**, including document references.

---

## 🛡️ Security & Reliability

### ✅ Addressed Issues
| Category | Problem | Resolution |
|-----------|----------|-------------|
| **Security** | Remote code execution (use of `exec`) | Replaced with **sandboxed PythonREPL** execution. |
| **Concurrency** | Overwriting charts (`plot.png`) | Added **directory clearing** before each visualization. |
| **Consistency** | Model names hardcoded | Centralized LLM configuration (`gemini-2.5-pro`) in Supervisor. |
| **Compliance Rules** | Hardcoded regex/recommendations | Moved to external YAML configuration. |
| **Conversation Context** | Ignored by some agents | Integrated history context into ReAct and LCEL chains. |

---

## 🧰 Tech Stack    

| Layer | Technology |
|-------|-------------|
| **Frontend** | Gradio |
| **Core Framework** | LangChain, LangGraph |
| **Vector Databases** | ChromaDB, FAISS (in-memory) |
| **LLMs** | Google Gemini 2.5 Pro |
| **Backend** | Python 3.10+ |
| **Visualization** | Matplotlib, Pandas |
| **Search APIs** | TavilySearch, WebScraper Tools |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/santiagoLopez1712/amaretis_rag_system.git
cd amaretis_rag_system
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file with your API keys:
```bash
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 5️⃣ Launch the App
```bash
python app.py --port 7860
```
The web app will open automatically at [http://localhost:7860](http://localhost:7860)

---

## 📈 Example Workflow

1. Upload a **marketing report** (`.pdf`) → The **RAG Agent** extracts and summarizes key findings.  
2. Ask: *“¿Cumple esta campaña con el DSGVO y la UWG?”* → The **Compliance Agent** evaluates it and generates recommendations.  
3. Request: *“Crea un brief estratégico basado en este contenido.”* → The **Brief Generator** builds a complete marketing brief with SMART goals.  
4. Analyze uploaded **CSV** → The **Data Analysis Agent** creates charts and insights safely.  

---

## 🧩 Modular Expansion

You can easily extend AMARETIS by adding new agents following the pattern:
```python
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate

class NewAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [Tool(name="my_tool", func=self.my_func, description="...")]
```

Then register it in `supervisor.py` for routing.

---

## 🧪 Testing

---

## 🧩 Folder Structure
```
amaretis_rag_system/
│
├── app.py                          # Gradio web interface
├── supervisor.py                   # LangGraph orchestration
├── rag_agent.py                    # RAG retrieval agent
├── compliance_agent.py             # Compliance & GDPR checker
├── data_analysis_agent.py          # Data visualization and analysis
├── brief_generator_agent.py        # Brief generation agent
├── integrated_marketing_agent.py   # Strategic synthesis
├── web_such_agent.py               # Web research agent
├── data_loader.py                  # Dataset preparation
├── data_chunkieren.py              # Text chunking logic
├── requirements.txt                # Dependencies 
└── .gitignore                      # Git ignore   
```

---

## 📚 References
- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Gradio Interface](https://www.gradio.app/)
- [Google Gemini API](https://ai.google.dev/)
- [Tavily Search API](https://tavily.com/)

---

## 🧑‍💻 Author
**Santiago López Otálvaro**  
Developer & AI Specialist — AMARETIS Agentur für Kommunikation 
Göttingen, Germany  

[GitHub Profile](https://github.com/santiagoLopez1712)
[LinkedIn Profile](https://www.linkedin.com/in/santiago-lopez-otalvaro-a129ba336/)
[Xing Profile](https://www.xing.com/profile/Santiago_LopezOtalvaro/web_profiles/cv)
 

---

## 🪪 License
MIT License © 2025 Santiago López Otálvaro

---

⭐ *If you like this project, give it a star on GitHub!* ⭐
