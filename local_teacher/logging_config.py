import os
import logging
from pathlib import Path

def setup_logging():
    PROJECT_ROOT = Path(__file__).parent.parent
    LOGS_DIR = PROJECT_ROOT / "logs"
    LOGS_DIR.mkdir(exist_ok=True)
    LOG_FILE = LOGS_DIR / "local_teacher.log"

    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("docling").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("fastembed").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
