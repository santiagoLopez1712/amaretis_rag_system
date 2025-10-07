# app.py 

import gradio as gr
import logging
import argparse
from typing import List, Optional, Tuple, Dict

from supervisor import SupervisorManager

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- INICIALIZACIÓN ÚNICA (Singleton/Caching) ---
print("🚀 Inicializando SupervisorManager (esto puede tardar un momento)...")
try:
    SUPERVISOR_INSTANCE = SupervisorManager()
    print("✅ SupervisorManager inicializado y listo.")
except Exception as e:
    SUPERVISOR_INSTANCE = None
    logger.critical(f"❌ ERROR FATAL al inicializar SupervisorManager: {e}", exc_info=True)
    # Imprimimos el error fatal aquí para que sea visible inmediatamente
    print(f"\n" + "="*80)
    print("❌ ERROR CRÍTICO: No se pudo inicializar el SupervisorManager. La aplicación no puede continuar.")
    print(f"   CAUSA: {e}")
    print("   Por favor, revisa el Traceback en los logs o en la terminal para más detalles.")
    print("="*80 + f"\n")


class AmaretisWebApp:
    """Clase que encapsula la lógica de la aplicación web Gradio."""
    def __init__(self, supervisor: Optional[SupervisorManager]):
        self.supervisor = supervisor

    def process_message(self, message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str, Optional[str]]:
        """Procesa la pregunta del usuario y actualiza el historial."""
        if not self.supervisor:
            error_msg = "El sistema de IA no está disponible debido a un error de inicialización."
            history.append({"role": "assistant", "content": error_msg})
            return history, "", None

        user_input = history[-1]["content"] if history and history[-1]["role"] == "user" else message

        if not user_input or not user_input.strip():
            return history, "", None

        try:
            answer_text, source, image_path = self.supervisor.process_question(user_input)
            formatted_answer = f"{answer_text}\n\n📚 *Fuente: {source}*"
            history.append({"role": "assistant", "content": formatted_answer})
            return history, "", image_path
            
        except Exception as e:
            logger.error(f"Error procesando el mensaje: {e}", exc_info=True)
            error_msg = "Lo siento, ocurrió un error inesperado al procesar tu pregunta."
            history.append({"role": "assistant", "content": error_msg})
            return history, "", None

def create_interface(supervisor_instance: Optional[SupervisorManager]) -> gr.Blocks:
    """Crea y configura la interfaz de usuario de Gradio."""
    
    # Si el supervisor es None, mostramos un mensaje de error en la UI
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
                msg_input = gr.Textbox(label="Tu Pregunta", placeholder="Ej: 'Crea un brief para un nuevo cliente de bebidas energéticas'")
                with gr.Row():
                    send_btn = gr.Button("📤 Enviar", variant="primary")
                    clear_btn = gr.Button("🗑️ Limpiar Chat", variant="secondary")
            
            with gr.Column(scale=1):
                image_output = gr.Image(label="📈 Visualización", height=400)
                gr.HTML("<h3>💡 Consejos</h3><ul><li>Pide resúmenes de campañas.</li><li>Solicita la creación de un brief.</li><li>Pide análisis de datos para generar gráficos.</li></ul>")

        send_btn.click(fn=app.process_message, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input, image_output])
        msg_input.submit(fn=app.process_message, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input, image_output])
        clear_btn.click(lambda: ([], "", None), outputs=[chatbot, msg_input, image_output])
        
    return interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMARETIS Marketing Intelligence App")
    parser.add_argument("--port", type=int, default=7860, help="Puerto del servidor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host del servidor")
    args = parser.parse_args()

    # --- CORRECCIÓN CLAVE: Verificamos si la inicialización falló ANTES de lanzar la UI ---
    if SUPERVISOR_INSTANCE is not None:
        print("🚀 Lanzando la interfaz de AMARETIS...")
        interface = create_interface(supervisor_instance=SUPERVISOR_INSTANCE)
        interface.launch(server_name=args.host, server_port=args.port)
    else:
        print("🔴 La aplicación no se lanzará debido a un error fatal en la inicialización.")
        # Opcional: lanzar una excepción para detener el script si se ejecuta en un entorno automatizado
        # raise RuntimeError("Supervisor failed to initialize.")