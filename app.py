# app.py (Versión con corrección de Gradio)

import gradio as gr
import logging
import argparse
import shutil
import webbrowser
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from supervisor import SupervisorManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

print("🚀 Inicializando SupervisorManager (esto puede tardar un momento)...")
try:
    SUPERVISOR_INSTANCE = SupervisorManager()
    print("✅ SupervisorManager inicializado y listo.")
except Exception as e:
    SUPERVISOR_INSTANCE = None
    logger.critical(f"❌ ERROR FATAL al inicializar SupervisorManager: {e}", exc_info=True)
    print(f"\n" + "="*80)
    print("❌ ERROR CRÍTICO: No se pudo inicializar el SupervisorManager. La aplicación no puede continuar.")
    print(f"   CAUSA: {e}")
    print("   Por favor, revisa el Traceback en los logs o en la terminal para más detalles.")
    print("="*80 + f"\n")

class AmaretisWebApp:
    def __init__(self, supervisor: Optional[SupervisorManager]):
        self.supervisor = supervisor

    def process_message(self, message: str, history: List[Tuple[Optional[str], Optional[str]]], uploaded_file: Optional[Any]) -> Tuple[List[Tuple[Optional[str], Optional[str]]], str, Optional[str], Optional[Any]]:
        if not self.supervisor:
            error_msg = "El sistema de IA no está disponible debido a un error de inicialización."
            history.append((message, error_msg))
            return history, "", None, None

        user_input = message.strip()
        augmented_input = user_input
        
        if uploaded_file is not None:
            temp_path = Path(uploaded_file.name)
            permanent_path = UPLOADS_DIR / temp_path.name
            shutil.copy(temp_path.as_posix(), permanent_path.as_posix())
            logger.info(f"Archivo subido y guardado en: {permanent_path}")
            
            file_extension = permanent_path.suffix.lower()
            
            if file_extension in ['.csv', '.xlsx']:
                instruction = (f"[Instrucción del sistema: Para esta tarea, utiliza el archivo '{permanent_path.as_posix()}'.]")
            elif file_extension == '.pdf':
                instruction = (
                    f"[Instrucción del sistema: El usuario ha subido el archivo '{permanent_path.as_posix()}'. "
                    f"Para responder la pregunta, DEBES usar la herramienta `uploaded_file_search` con el input: "
                    f"'{permanent_path.as_posix()}|{user_input}' ")
            else:
                instruction = "[Instrucción del sistema: Se ha subido un archivo de tipo no soportado.]"

            augmented_input = f"{user_input}\n\n{instruction}"
            logger.info(f"Prompt aumentado: {augmented_input}")

        if not augmented_input:
            return history, "", None, None

        try:
            # [ADAPTADOR] Convertir el historial de tuplas (formato Gradio) a diccionarios (formato Agente)
            agent_history = []
            for user_msg, assistant_msg in history:
                if user_msg is not None:
                    agent_history.append({"role": "user", "content": user_msg})
                if assistant_msg is not None:
                    agent_history.append({"role": "assistant", "content": assistant_msg})
            
            # Ahora pasamos el historial convertido al supervisor
            answer_text, source, image_path = self.supervisor.process_question(augmented_input, agent_history)
            formatted_answer = f"{answer_text}\n\n📚 *Fuente: {source}*"
            
            # Gradio maneja el historial implícitamente
            history.append((user_input, formatted_answer))
            return history, "", image_path, None
            
        except Exception as e:
            logger.error(f"Error procesando el mensaje: {e}", exc_info=True)
            error_msg = "Lo siento, ocurrió un error inesperado al procesar tu pregunta."
            history.append((user_input, error_msg))
            return history, "", None, None

def create_interface(supervisor_instance: Optional[SupervisorManager]) -> gr.Blocks:
    if supervisor_instance is None:
        with gr.Blocks(title="Error - AMARETIS") as interface:
            gr.Markdown("# ❌ Error Crítico del Sistema\nEl backend de IA no pudo iniciarse. Por favor, revisa los logs de la terminal para más detalles.")
        return interface

    app = AmaretisWebApp(supervisor=supervisor_instance)

    css = """ .gradio-container { max-width: 1200px; margin: auto; } .chat-container { box-shadow: 0 4px 6px rgba(0,0,0,0.1); } """
    
    with gr.Blocks(css=css, title="AMARETIS Marketing Intelligence") as interface:
        gr.HTML("<h1>🎯 AMARETIS Marketing Intelligence System</h1>")
        
        with gr.Row():
            with gr.Column(scale=2):
                # --- CORRECCIÓN: Se elimina el argumento 'type' obsoleto ---
                chatbot = gr.Chatbot(label="💬 Marketing Assistant", height=600)
                msg_input = gr.Textbox(label="Tu Pregunta", placeholder="Sube un archivo y haz una pregunta sobre él...")
                with gr.Row():
                    send_btn = gr.Button("📤 Enviar", variant="primary")
                    clear_btn = gr.Button("🗑️ Limpiar Chat", variant="secondary")
            
            with gr.Column(scale=1):
                image_output = gr.Image(label="📈 Visualización", height=400)
                file_uploader = gr.File(label="Subir Archivo (PDF, CSV, XLSX)", file_types=[".pdf", ".csv", ".xlsx"])
                gr.HTML("<h3>💡 Consejos</h3><ul><li>Sube un PDF y haz preguntas sobre su contenido.</li><li>Sube un CSV y pide un análisis.</li></ul>")
        
        event_args = {
            "fn": app.process_message,
            "inputs": [msg_input, chatbot, file_uploader],
            "outputs": [chatbot, msg_input, image_output, file_uploader]
        }
        send_btn.click(**event_args)
        msg_input.submit(**event_args)
        
        clear_btn.click(lambda: (None, "", None, None), outputs=[chatbot, msg_input, image_output, file_uploader])
        
    return interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMARETIS Marketing Intelligence App")
    parser.add_argument("--port", type=int, default=7860, help="Puerto del servidor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host del servidor")
    args = parser.parse_args()

    if SUPERVISOR_INSTANCE is not None:
        print("🚀 Lanzando la interfaz de AMARETIS...")
        interface = create_interface(supervisor_instance=SUPERVISOR_INSTANCE)

        interface.launch(server_name=args.host, server_port=args.port, inbrowser=True)
    else:
        print("🔴 La aplicación no se lanzará debido a un error fatal en la inicialización.")
