# data_loader.py (Código Corregido)
import os
import pdfplumber
import json

def extract_tables_from_directory_to_json(directory, output_path):
    extracted_data = []

    # 1. Función auxiliar para procesar un solo archivo PDF
    def process_pdf(file_path, company_name):
        file_extracted_data = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Intentar extraer tablas
                    tables = page.extract_tables()
                    if tables:
                        print(f"📄 {file_path} - Seite {page_num + 1}: {len(tables)} Tabellen gefunden")
                        for table_idx, table in enumerate(tables):
                            # Formatear la tabla
                            text = "\n".join([
                                " | ".join([cell if cell else "" for cell in row]) for row in table
                            ])
                            file_extracted_data.append({
                                "type": "table",
                                "company": company_name,
                                "file": os.path.basename(file_path),
                                "page": page_num + 1,
                                "table_index": table_idx,
                                "content": text
                            })
                    # Extraer texto si no hay tablas o si el documento es solo texto
                    else:
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
            print(f"❌ Fehler: {file_path} konnte nicht gelesen werden. {e}")
            return []

    # 2. Iterar sobre todos los archivos y subdirectorios
    for root, dirs, files in os.walk(directory):
        # Determinar el nombre de la compañía/carpeta si no es la raíz
        company_name = os.path.basename(root) if root != directory else "root"
        
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)
                
                # Procesar el PDF y añadir los resultados
                data = process_pdf(file_path, company_name)
                extracted_data.extend(data)


    # 3. Guardar el resultado final
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(extracted_data)} Datensätze wurden in die JSON-Datei gespeichert: {output_path}")


if __name__ == "__main__":
    # Asegúrate de que la carpeta 'data' exista y contenga los PDFs
    extract_tables_from_directory_to_json("data", "structured_data.json")