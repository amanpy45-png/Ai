# loader func

def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        documents = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return documents