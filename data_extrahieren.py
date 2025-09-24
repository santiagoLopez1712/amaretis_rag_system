# data_loader.py
import os
import pdfplumber
import json

def extract_tables_from_directory_to_json(directory, output_path):
    extracted_data = []

    for company in os.listdir(directory):
        company_path = os.path.join(directory, company)

        # Wenn es sich direkt um eine PDF-Datei handelt
        if os.path.isfile(company_path) and company_path.endswith(".pdf"):
            company = "root"
            file_path = company_path
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        if tables:
                            print(f"📄 {file_path} - Seite {page_num+1}: {len(tables)} Tabellen gefunden")
                            for table_idx, table in enumerate(tables):
                                text = "\n".join([
                                    " | ".join([cell if cell else "" for cell in row]) for row in table
                                ])
                                extracted_data.append({
                                    "type": "table",
                                    "company": company,
                                    "file": os.path.basename(file_path),
                                    "page": page_num + 1,
                                    "table_index": table_idx,
                                    "content": text
                                })
                        else:
                            text = page.extract_text()
                            if text:
                                extracted_data.append({
                                    "type": "text",
                                    "company": company,
                                    "file": os.path.basename(file_path),
                                    "page": page_num + 1,
                                    "content": text
                                })
            except Exception as e:
                print(f"❌ Fehler: {file_path} konnte nicht gelesen werden. {e}")

        # Wenn es sich um ein Verzeichnis handelt
        elif os.path.isdir(company_path):
            for filename in os.listdir(company_path):
                if not filename.endswith(".pdf"):
                    continue

                file_path = os.path.join(company_path, filename)
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            tables = page.extract_tables()
                            if tables:
                                print(f"📄 {file_path} - Seite {page_num+1}: {len(tables)} Tabellen gefunden")
                                for table_idx, table in enumerate(tables):
                                    text = "\n".join([
                                        " | ".join([cell if cell else "" for cell in row]) for row in table
                                    ])
                                    extracted_data.append({
                                        "type": "table",
                                        "company": company,
                                        "file": filename,
                                        "page": page_num + 1,
                                        "table_index": table_idx,
                                        "content": text
                                    })
                            else:
                                text = page.extract_text()
                                if text:
                                    extracted_data.append({
                                        "type": "text",
                                        "company": company,
                                        "file": filename,
                                        "page": page_num + 1,
                                        "content": text
                                    })
                except Exception as e:
                    print(f"❌ Fehler: {file_path} konnte nicht gelesen werden. {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(extracted_data)} Datensätze wurden in die JSON-Datei gespeichert: {output_path}")


if __name__ == "__main__":
    extract_tables_from_directory_to_json("data", "structured_data.json")