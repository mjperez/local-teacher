import sys, json
from graphify.build import build_from_json
from graphify.cluster import score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json
from pathlib import Path

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text(encoding="utf-8"))
detection  = json.loads(Path('graphify-out/.graphify_detect.json').read_text(encoding="utf-8"))
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text(encoding="utf-8"))

G = build_from_json(extraction, root=Path('.'), directed=False)
communities = {int(k): v for k, v in analysis['communities'].items()}
cohesion = {int(k): v for k, v in analysis['cohesion'].items()}
tokens = {'input': extraction.get('input_tokens', 0), 'output': extraction.get('output_tokens', 0)}

labels = {
    0: "Core Graph Config",
    1: "Data Ingestion Pipeline",
    2: "Vector Retrieval",
    3: "App UI Config",
    4: "Query Criticism",
    5: "Chunking and Cache",
    6: "Query Optimization",
    7: "Media and Text Loaders",
    8: "Document Extraction Loaders",
    9: "CLI and Logging",
    10: "Vector Store Testing",
    11: "Metadata Types",
    12: "Module Init",
    13: "Detect Scripts",
    14: "Storage Init",
    15: "Branch Scripts",
    16: "Pytest Conftest"
}

questions = suggest_questions(G, communities, labels)

report = generate(G, communities, cohesion, labels, analysis['gods'], analysis['surprises'], detection, tokens, Path('.'), suggested_questions=questions)
Path('graphify-out/GRAPH_REPORT.md').write_text(report, encoding="utf-8")
Path('graphify-out/.graphify_labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False), encoding="utf-8")
wrote = to_json(G, communities, 'graphify-out/graph.json', community_labels=labels, force=True)
if not wrote:
    print('ERROR: refused to shrink graphify-out/graph.json')
print('Report updated with community labels')
