import os
import pdfplumber
import json
import sys
import logging
from io import StringIO


# CONFIGURACIÓN DE LOGGING

# Suprimir advertencias de pdfminer (Cannot set gray...)
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('pdfminer.pdfinterp').setLevel(logging.ERROR)

# Suprimir otros logs innecesarios
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

def extract_tables_from_directory_to_json(directory, output_path):
    """Extrae tablas y texto de PDFs en un directorio y los guarda en JSON"""
    extracted_data = []

    def process_pdf(file_path, company_name):
        """Procesa un único archivo PDF"""
        file_extracted_data = []
        
        # Guardar stderr original para suprimir advertencias de pdfminer
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Intentar extraer tablas
                    tables = page.extract_tables()
                    
                    if tables:
                        # Restaurar stderr temporalmente para mostrar progreso
                        sys.stderr = original_stderr
                        print(f"📄 {os.path.basename(file_path)} - Seite {page_num + 1}: {len(tables)} Tabellen gefunden")
                        sys.stderr = StringIO()
                        
                        for table_idx, table in enumerate(tables):
                            # Formatear la tabla
                            text = "\n".join([
                                " | ".join([cell if cell else "" for cell in row]) 
                                for row in table
                            ])
                            file_extracted_data.append({
                                "type": "table",
                                "company": company_name,
                                "file": os.path.basename(file_path),
                                "page": page_num + 1,
                                "table_index": table_idx,
                                "content": text
                            })
                    else:
                        # Extraer texto si no hay tablas
                        text = page.extract_text()
                        if text:
                            file_extracted_data.append({
                                "type": "text",
                                "company": company_name,
                                "file": os.path.basename(file_path),
                                "page": page_num + 1,
                                "content": text
                            })
            
            return file_extracted_data
            
        except Exception as e:
            sys.stderr = original_stderr
            print(f"❌ Fehler beim Verarbeiten von {os.path.basename(file_path)}: {e}")
            return []
            
        finally:
            sys.stderr = original_stderr

    # Iterar sobre todos los archivos y subdirectorios
    for root, dirs, files in os.walk(directory):
        company_name = os.path.basename(root) if root != directory else "root"
        
        for filename in files:
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(root, filename)
                print(f"🔄 Verarbeitungsdatei: {filename}...")
                data = process_pdf(file_path, company_name)
                extracted_data.extend(data)

    # Guardar el resultado final
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ {len(extracted_data)} Datensätze wurden in '{output_path}' gespeichert")


if __name__ == "__main__":
    extract_tables_from_directory_to_json("data", "structured_data.json")