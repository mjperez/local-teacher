# esta capa no debe saber nada de embeddings ni de prompts.

import json
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def cargar_archivos(ruta_carpeta: Path | str) -> list[Document]:
    documentos: list[Document] = []
    ruta_carpeta = Path(ruta_carpeta)
    try:
        for item in ruta_carpeta.rglob("*"):
            if item.is_file():
                if item.suffix in [".txt", ".md"]:
                    contenido = TextLoader(str(item), encoding="utf-8")
                    documentos.extend(contenido.load())
                elif item.suffix == ".jsonl":
                    with open(item, "r", encoding="utf-8") as f:
                        for idx, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            data = json.loads(line)
                            
                            # Formatear el JSONL específicamente para Natural Questions
                            if "question" in data and "answer" in data:
                                resp = ", ".join(data["answer"]) if isinstance(data["answer"], list) else str(data["answer"])
                                text = f"Q: {data['question']}\nA: {resp}"
                            else:
                                text = json.dumps(data, ensure_ascii=False)
                                
                            doc = Document(
                                page_content=text, 
                                metadata={"source": str(item), "line": idx}
                            )
                            documentos.append(doc)
    except Exception as e:
        print(f"Error cargando archivos: {e}")
    return documentos
