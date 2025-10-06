import gradio as gr
import os
import re
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatSession:
    """Maneja el estado de una sesión de chat individual"""
    history: List[Dict[str, str]] = field(default_factory=list)
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_interaction(self, user_input: str, assistant_output: str):
        """Añade interacción al historial de forma thread-safe"""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": assistant_output})
        
        # Limitar historial para evitar memory leaks
        if len(self.history) > 20:
            self.history = self.history[-20:]

class AmaretisMultiAgentApp:
    """
    Aplicación principal para el sistema multi-agente de AMARETIS
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.sessions: Dict[str, ChatSession] = {}
        self.session_lock = threading.Lock()
        
        # Inicializar componentes
        self.initialize_agents()
        
    def initialize_agents(self):
        """Inicializa agentes de forma segura"""
        try:
            # Import dinámico para evitar circular dependencies
            from supervisor_main import SupervisorManager
            self.supervisor = SupervisorManager()
            logger.info("✅ Supervisor inicializado correctamente")
            
        except ImportError as e:
            logger.error(f"Error importando SupervisorManager: {e}")
            # Fallback a imports individuales
            try:
                import supervisor_main
                # Verificar que los objetos existen
                if hasattr(supervisor_main, 'rag_agent') and supervisor_main.rag_agent:
                    self.rag_agent = supervisor_main.rag_agent
                    self.tools = getattr(supervisor_main, 'tools', [])
                    self.qa_ethics_agent = getattr(supervisor_main, 'qa_ethics_agent', None)
                    logger.info("✅ Agentes cargados como fallback")
                else:
                    raise ValueError("Agentes no están inicializados en supervisor_main")
            except Exception as fallback_error:
                logger.error(f"Error en fallback: {fallback_error}")
                self.supervisor = None
                
        except Exception as e:
            logger.error(f"Error inicializando agentes: {e}")
            self.supervisor = None
    
    def get_or_create_session(self, session_id: str) -> ChatSession:
        """Obtiene o crea sesión de chat de forma thread-safe"""
        with self.session_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = ChatSession(session_id=session_id)
            return self.sessions[session_id]
    
    def is_data_analysis_request(self, user_input: str) -> bool:
        """Detecta solicitudes de análisis de datos mejorada"""
        analysis_keywords = [
            "analysiere", "analyse", "plot", "diagramm", "visualisiere",
            "statistik", "vergleich", "vergleiche", "csv", "datenanalyse",
            "korrelation", "trend", "zeitreihe", "daten", "tabelle",
            "chart", "graph", "visualización"
        ]
        
        # Para AMARETIS - keywords de marketing
        marketing_keywords = [
            "kampagne", "performance", "roi", "conversion", "engagement",
            "reach", "impressions", "clicks", "budget", "zielgruppe"
        ]
        
        # Combinar keywords
        all_keywords = analysis_keywords + marketing_keywords
        return any(keyword in user_input.lower() for keyword in all_keywords)
    
    def get_latest_figure(self) -> Optional[str]:
        """Busca la figura más reciente con manejo de errores robusto"""
        try:
            figures_path = Path("figures")
            if not figures_path.exists():
                logger.debug("Directorio 'figures' no existe")
                return None
                
            # Buscar archivos de imagen
            image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf"]
            figures = []
            
            for extension in image_extensions:
                figures.extend(figures_path.glob(extension))
            
            if not figures:
                logger.debug("No se encontraron figuras")
                return None
                
            # Encontrar la más reciente
            try:
                latest_figure = max(figures, key=lambda f: f.stat().st_mtime)
                logger.info(f"Figura más reciente: {latest_figure}")
                return str(latest_figure)
            except (OSError, ValueError) as e:
                logger.error(f"Error obteniendo figura más reciente: {e}")
                return str(figures[0])  # Fallback a la primera
                
        except Exception as e:
            logger.error(f"Error en get_latest_figure: {e}")
            return None
    
    def log_interaction(self, user_input: str, answer: str, source: str):
        """Log de interacciones con manejo de errores"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"\n⏰ {timestamp}\n"
                f"📝 Pregunta: {user_input}\n"
                f"🤖 Respuesta: {answer[:200]}{'...' if len(answer) > 200 else ''}\n"
                f"📚 Fuente: {source}\n"
                f"{'-' * 60}\n"
            )
            
            with open("chat_log.txt", "a", encoding="utf-8") as f:
                f.write(log_entry)
                
        except Exception as e:
            logger.error(f"Error escribiendo log: {e}")
    
    def process_message(
        self, 
        message: str, 
        history: List[List[str]], 
        session_id: str = "default"
    ) -> Tuple[List[List[str]], Optional[str]]:
        """
        Procesa mensaje y retorna historial actualizado + imagen opcional
        """
        if not message or not message.strip():
            return history, None
            
        user_input = message.strip()
        session = self.get_or_create_session(session_id)
        
        try:
            # Usar supervisor si está disponible
            if self.supervisor:
                answer, source = self.supervisor.process_question(user_input)
                image_path = None
                
                # Verificar si es análisis de datos para buscar imagen
                if self.is_data_analysis_request(user_input):
                    image_path = self.get_latest_figure()
                    
            else:
                # Fallback a lógica manual (versión simplificada)
                answer = "Sistema no disponible temporalmente. Intenta más tarde."
                source = "Sistema"
                image_path = None
            
            # QA/Ethics check si está disponible
            qa_result = ""
            try:
                if hasattr(self, 'qa_ethics_agent') and self.qa_ethics_agent:
                    qa_result = self.qa_ethics_agent.run(answer, [source])
                    qa_result = f"\n⚖️ QA: {qa_result}" if qa_result else ""
            except Exception as e:
                logger.warning(f"QA agent error: {e}")
            
            # Formato de respuesta
            formatted_answer = f"{answer}\n\n📚 Fuente: {source}{qa_result}"
            
            # Actualizar historial de Gradio
            history.append([user_input, formatted_answer])
            
            # Actualizar sesión
            session.add_interaction(user_input, answer)
            
            # Log
            self.log_interaction(user_input, answer, source)
            
            return history, image_path
            
        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
            error_msg = f"Error procesando tu pregunta: {e}"
            history.append([user_input, error_msg])
            return history, None

def create_amaretis_interface(debug: bool = False) -> gr.Blocks:
    """
    Crea interfaz de Gradio optimizada para AMARETIS
    """
    
    # Inicializar aplicación
    app = AmaretisMultiAgentApp(debug=debug)
    
    # Custom CSS para AMARETIS
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
        max-width: 1200px;
        margin: auto;
    }
    
    .chat-container {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header {
        background: linear-gradient(90deg, #2196F3, #21CBF3);
        color: white;
        padding: 20px;
        border-radius: 10px 10px 0 0;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="AMARETIS Marketing Intelligence") as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎯 AMARETIS Marketing Intelligence System</h1>
            <p>KI-gestütztes Knowledge Management für Marketing & Kommunikation</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat principal
                chatbot = gr.Chatbot(
                    label="💬 Marketing Assistant",
                    height=500,
                    container=True,
                    elem_classes=["chat-container"]
                )
                
                # Input
                msg_input = gr.Textbox(
                    label="Deine Frage",
                    placeholder="z.B. 'Zeige mir erfolgreiche B2B Kampagnen' oder 'Analysiere Performance der letzten Quartal'",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("📤 Senden", variant="primary")
                    clear_btn = gr.Button("🗑️ Chat löschen", variant="secondary")
            
            with gr.Column(scale=1):
                # Panel lateral para imágenes y controles
                image_output = gr.Image(
                    label="📈 Visualisierung",
                    height=300,
                    container=True
                )
                
                # Info panel
                gr.HTML("""
                <div style="background: #f0f0f0; padding: 15px; border-radius: 10px; margin-top: 20px;">
                    <h3>💡 Tipps</h3>
                    <ul>
                        <li>Frage nach Kampagnen und Strategien</li>
                        <li>Bitte um Datenvisualisierung</li>
                        <li>Suche nach Best Practices</li>
                        <li>Analysiere Markttrends</li>
                    </ul>
                </div>
                """)
        
        # Estado para sesión
        session_state = gr.State(value="default")
        
        # Event handlers
        def process_and_display(message, history, session_id):
            """Wrapper para procesar mensaje y actualizar interfaz"""
            if not message.strip():
                return history, "", None
                
            updated_history, image_path = app.process_message(message, history, session_id)
            return updated_history, "", image_path
        
        # Conectar eventos
        send_btn.click(
            fn=process_and_display,
            inputs=[msg_input, chatbot, session_state],
            outputs=[chatbot, msg_input, image_output],
            queue=True
        )
        
        msg_input.submit(
            fn=process_and_display,
            inputs=[msg_input, chatbot, session_state],
            outputs=[chatbot, msg_input, image_output],
            queue=True
        )
        
        clear_btn.click(
            fn=lambda: ([], None),
            outputs=[chatbot, image_output],
            queue=False
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>🏢 AMARETIS Agentur für Kommunikation | 📍 Göttingen</p>
            <p>Powered by Multi-Agent AI System</p>
        </div>
        """)
    
    return interface

def main():
    """Función principal mejorada"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AMARETIS Marketing Intelligence App")
    parser.add_argument("--debug", action="store_true", help="Activar modo debug")
    parser.add_argument("--port", type=int, default=7860, help="Puerto del servidor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host del servidor")
    
    args = parser.parse_args()
    
    try:
        # Crear interfaz
        interface = create_amaretis_interface(debug=args.debug)
        
        # Lanzar aplicación
        print(f"\n🚀 Iniciando AMARETIS Marketing Intelligence...")
        print(f"📍 URL: http://localhost:{args.port}")
        print(f"🔧 Debug mode: {'ON' if args.debug else 'OFF'}")
        
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,  # Cambiar a True para URL pública
            debug=args.debug,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Error iniciando aplicación: {e}")
        print(f"❌ Error crítico: {e}")

if __name__ == "__main__":
    main()