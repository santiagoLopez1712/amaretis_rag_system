# app.py (Versión final con subida de archivos, caching y lanzamiento automático)

import gradio as gr
import logging
import argparse
import shutil
import webbrowser
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Asegúrate de que el supervisor se importa correctamente
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
    """Clase que encapsula la lógica de la aplicación web Gradio."""
    def __init__(self, supervisor: Optional[SupervisorManager]):
        self.supervisor = supervisor

    def process_message(self, message: str, history: List[Dict[str, str]], uploaded_file: Optional[Any]) -> Tuple[List[Dict[str, str]], str, Optional[str], Optional[Any]]:
        """Procesa la pregunta, maneja el archivo subido y actualiza el historial."""
        if not self.supervisor:
            error_msg = "El sistema de IA no está disponible debido a un error de inicialización."
            if message: history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
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
                instruction = (
                    f"[Instrucción del sistema: Para esta tarea, utiliza el archivo '{permanent_path.as_posix()}'.]"
                )
            elif file_extension == '.pdf':
                instruction = (
                    f"[Instrucción del sistema: El usuario ha subido el archivo '{permanent_path.as_posix()}'. "
                    f"Para responder la pregunta, DEBES usar la herramienta `uploaded_file_search` con el input: "
                    f"'{permanent_path.as_posix()}|{user_input}' ]"
                )
            else:
                instruction = "[Instrucción del sistema: Se ha subido un archivo de tipo no soportado.]"

            augmented_input = f"{user_input}\n\n{instruction}"
            logger.info(f"Prompt aumentado: {augmented_input}")

        if not augmented_input:
            # Evita procesar si no hay ni texto ni archivo
            return history, "", None, None

        history.append({"role": "user", "content": user_input})

        try:
            answer_text, source, image_path = self.supervisor.process_question(augmented_input)
            formatted_answer = f"{answer_text}\n\n📚 *Fuente: {source}*"
            history.append({"role": "assistant", "content": formatted_answer})
            return history, "", image_path, None
            
        except Exception as e:
            logger.error(f"Error procesando el mensaje: {e}", exc_info=True)
            error_msg = "Lo siento, ocurrió un error inesperado al procesar tu pregunta."
            history.append({"role": "assistant", "content": error_msg})
            return history, "", None, None

def create_interface(supervisor_instance: Optional[SupervisorManager]) -> gr.Blocks:
    """Crea y configura la interfaz de usuario de Gradio."""
    
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
                chatbot = gr.Chatbot(label="💬 Marketing Assistant", height=600, elem_classes=["chat-container"], type='messages')
                msg_input = gr.Textbox(label="Tu Pregunta", placeholder="Sube un archivo y haz una pregunta sobre él...")
                with gr.Row():
                    send_btn = gr.Button("📤 Enviar", variant="primary")
                    clear_btn = gr.Button("🗑️ Limpiar Chat", variant="secondary")
            
            with gr.Column(scale=1):
                image_output = gr.Image(label="📈 Visualización", height=400)
                file_uploader = gr.File(label="Subir Archivo (PDF, CSV, XLSX)", file_types=[".pdf", ".csv", ".xlsx"])
                gr.HTML("<h3>💡 Consejos</h3><ul><li>Sube un PDF y haz preguntas sobre su contenido.</li><li>Sube un CSV y pide un análisis.</li><li>Pide la creación de un brief.</li></ul>")
        
        event_args = {
            "fn": app.process_message,
            "inputs": [msg_input, chatbot, file_uploader],
            "outputs": [chatbot, msg_input, image_output, file_uploader]
        }
        send_btn.click(**event_args)
        msg_input.submit(**event_args)
        
        clear_btn.click(lambda: ([], "", None, None), outputs=[chatbot, msg_input, image_output, file_uploader])
        
    return interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMARETIS Marketing Intelligence App")
    parser.add_argument("--port", type=int, default=7860, help="Puerto del servidor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host del servidor")
    args = parser.parse_args()

    if SUPERVISOR_INSTANCE is not None:
        print("🚀 Lanzando la interfaz de AMARETIS...")
        interface = create_interface(supervisor_instance=SUPERVISOR_INSTANCE)
        
        # --- NUEVA FUNCIONALIDAD: Abrir el navegador automáticamente ---
        # Gradio > 4 usa el argumento 'inbrowser' directamente en launch()
        # pero para asegurar compatibilidad, lo manejamos así.
        # En versiones más nuevas, simplemente `inbrowser=True` podría funcionar.
        def open_url():
            webbrowser.open(f"http://localhost:{args.port}")
        
        # Abrimos el navegador un segundo después de lanzar el servidor
        # para darle tiempo a iniciarse.
        interface.load(fn=open_url, inputs=None, outputs=None, _js="(async () => { await new Promise(r => setTimeout(r, 1000)); })")
        
        interface.launch(server_name=args.host, server_port=args.port)
    else:
        print("🔴 La aplicación no se lanzará debido a un error fatal en la inicialización.")