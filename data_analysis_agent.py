# data_analysis_agent.py (Refactorizado con LangChain y Vertex AI)

import os
import re
import logging
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import uuid
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonAstREPLTool

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada.")

REGION = os.getenv("GOOGLE_CLOUD_REGION")
if not REGION:
    raise ValueError("La variable de entorno GOOGLE_CLOUD_REGION no está configurada.")

class DataAnalysisAgent:
    name = "data_analysis_agent"

    def __init__(self, temperature: float = 0.5):
        self.llm = ChatVertexAI(project=PROJECT_ID, location=REGION, model="gemini-2.5-pro", temperature=temperature)
        self.tools = self._setup_tools()
        self.agent: Optional[AgentExecutor] = self._create_agent()

    def _load_dataframe(self, file_path: str) -> Optional[pd.DataFrame]:
        """Carga un archivo (CSV, XLSX, o tabla de PDF) en un DataFrame de pandas."""
        try:
            file_path = file_path.strip().strip("'\"") # Limpiar la ruta
            if file_path.lower().endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.lower().endswith('.xlsx'):
                return pd.read_excel(file_path)
            elif file_path.lower().endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        if tables:
                            table_data = tables[0]
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            logger.info(f"Tabla extraída de PDF {file_path} con {len(df)} filas.")
                            return df
                logger.warning(f"No se encontraron tablas en el PDF: {file_path}")
                return None
            else:
                logger.error(f"Tipo de archivo no soportado: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error cargando el archivo {file_path}: {e}")
            return None

    def _tool_analyze_dataframe(self, query: str) -> str:
        """
        Analiza un archivo de datos (CSV, XLSX, PDF) para responder una pregunta.
        Input: Una cadena que contiene la ruta del archivo y la pregunta, separados por '|'.
        Ejemplo: 'uploads/datos.csv|¿Cuál es el promedio de ventas?'
        """
        try:
            code_to_execute = "" # Inicializar para un manejo de errores robusto
            file_path, user_question = query.split('|', 1)
            df = self._load_dataframe(file_path.strip())
            if df is None:
                return f"Error: No se pudo cargar o encontrar datos tabulares en el archivo {file_path}."

            code_prompt = ChatPromptTemplate.from_template(
                "Dada la siguiente pregunta y las primeras 5 filas de un DataFrame de pandas llamado 'df', escribe un script de Python que imprima el resultado de la pregunta. "
                "Solo responde con el código. El código será ejecutado en un REPL de Python seguro.\n\n"
                "Pregunta: {question}\n\nDataFrame (df.head()):\n{head}\n\nCódigo Python:"
            )
            chain = code_prompt | self.llm | StrOutputParser()
            code_to_execute = chain.invoke({"question": user_question, "head": df.head().to_string()}).strip('`python \n`').strip()
            
            logger.info(f"Ejecutando código de análisis: {code_to_execute}")
            
            # Usamos PythonAstREPLTool para una ejecución segura
            repl = PythonAstREPLTool(locals={"df": df})
            result = repl.run(code_to_execute)
            
            return f"Análisis completado. Resultado:\n{str(result)}"
        except Exception as e:
            error_message = f"Error durante el análisis: {e}. El código que falló fue: '{code_to_execute}'"
            logger.error(error_message)
            return error_message

    def _tool_create_visualization(self, query: str) -> str:
        """
        Crea una visualización a partir de un archivo de datos.
        Input: Una cadena que contiene la ruta del archivo y la descripción del gráfico, separados por '|'.
        Ejemplo: 'uploads/datos.csv|Crea un gráfico de barras de ventas por categoría'
        """
        try:
            code_to_execute = "" # Inicializar para un manejo de errores robusto
            file_path, user_question = query.split('|', 1)
            df = self._load_dataframe(file_path.strip())
            if df is None:
                return f"Error: No se pudo cargar o encontrar datos tabulares en el archivo {file_path}."

            # Generar un nombre de archivo único para el gráfico
            os.makedirs("figures", exist_ok=True)
            plot_filename = f"figures/plot_{uuid.uuid4()}.png"

            code_prompt = ChatPromptTemplate.from_template(
                "Escribe código Python usando matplotlib.pyplot (como plt) y un DataFrame de pandas llamado 'df' para crear la visualización solicitada. "
                " GUARDA la figura en la ruta '{plot_path}'. NO uses plt.show().\n\n"
                "Petición: {question}\n\nColumnas del DataFrame (df.columns): {columns}\n\nCódigo Python:"
            )
            chain = code_prompt | self.llm | StrOutputParser()
            code_to_execute = chain.invoke({
                "question": user_question, 
                "columns": df.columns.tolist(),
                "plot_path": plot_filename
            }).strip('`python \n`').strip()

            logger.info(f"Ejecutando código de visualización: {code_to_execute}")
            
            # Usamos PythonAstREPLTool para una ejecución segura
            repl = PythonAstREPLTool(locals={"df": df, "plt": plt})
            repl.run(code_to_execute)

            return f"Visualización creada y guardada en '{plot_filename}'."
        except Exception as e:
            error_message = f"Error durante la visualización: {e}. El código que falló fue: '{code_to_execute}'"
            logger.error(error_message)
            return error_message

    def _setup_tools(self) -> List[Tool]:
        return [
            Tool(name="analyze_dataframe", func=self._tool_analyze_dataframe, description="Útil para realizar cálculos, estadísticas o análisis sobre datos de un archivo."),
            Tool(name="create_visualization", func=self._tool_create_visualization, description="Útil para crear un gráfico o visualización a partir de los datos de un archivo.")
        ]

    def _create_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_template("""
        Eres un "Científico de Datos". Tu trabajo es analizar datos de archivos y crear visualizaciones.
        La pregunta del usuario contendrá una instrucción del sistema con la ruta del archivo, como '[Instrucción del sistema: Para esta tarea, utiliza el archivo 'uploads/archivo.csv'.]'.
        
        **PROCESO OBLIGATORIO:**
        1. Extrae la ruta del archivo de la instrucción del sistema.
        2. Extrae la pregunta real del usuario.
        3. Elige la herramienta correcta: `analyze_dataframe` para cálculos, `create_visualization` para gráficos.
        4. Construye el input de la herramienta en el formato 'ruta/del/archivo|pregunta del usuario'.

        Herramientas: {tools}
        
        Thought: [Tu razonamiento para extraer la ruta, la pregunta y elegir la herramienta.]
        Action: [Herramienta a usar de [{tool_names}]]
        Action Input: [La cadena combinada 'ruta/del/archivo|pregunta del usuario']
        Observation: [Resultado de la herramienta.]
        Thought: Ya tengo la respuesta.
        Final Answer: [Un resumen claro del resultado para el usuario. Si se creó una visualización, asegúrate de mencionar la ruta del archivo donde se guardó.]

        Pregunta: {input}
        Historial: {history}
        Tu Gedankengang: {agent_scratchpad}
        """)
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=False, handle_parsing_errors=True, max_iterations=5, max_execution_time=60)

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.agent:
            return {"output": "Error: El agente de análisis de datos no pudo inicializarse."}
        user_input = input_dict.get("input", "")
        if not user_input:
            return {"output": "Error: La solicitud para el análisis de datos está vacía."}
        
        try:
            history = input_dict.get("history", [])
            result = self.agent.invoke({"input": user_input, "history": history})
            return {"output": result.get("output", str(result))}
        except Exception as e:
            logger.error(f"Error en la invocación del Data Analysis Agent: {e}")
            return {"output": f"Error en el agente de análisis de datos: {e}"}

agent = DataAnalysisAgent()
