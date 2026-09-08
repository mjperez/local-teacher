import json
from pathlib import Path

result = json.loads(Path('graphify-out/.graphify_detect.json').read_text(encoding='utf-8'))
print(f"Corpus: {result['total_files']} files · ~{result['total_words']} words")
for k, v in result['files'].items():
    if v:
        print(f"  {k}: {len(v)} files")
