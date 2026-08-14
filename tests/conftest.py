import sys
from pathlib import Path

# Añade la raíz del proyecto al sys.path para que pytest resuelva los módulos de local_teacher
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
