import json
from pathlib import Path

analysis = json.loads(Path('graphify-out/.graphify_analysis.json').read_text(encoding='utf-8'))
for cid, nodes in analysis['communities'].items():
    print(f"Community {cid}: {nodes}")
