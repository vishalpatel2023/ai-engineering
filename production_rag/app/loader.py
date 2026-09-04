import os
from pypdf import PdfReader


def load_documents(folder_path):

    documents = []

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        # TXT
        if filename.endswith(".txt"):

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append({
                "source": filename,
                "text": text,
                "page": None
            })

        # PDF
        elif filename.endswith(".pdf"):

            reader = PdfReader(file_path)

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if text:
                    documents.append({
                        "source": filename,
                        "text": text,
                        "page": page_number
                    })

    return documents