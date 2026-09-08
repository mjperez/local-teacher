import sys, json
from graphify.detect import detect
from pathlib import Path

open('graphify-out/.graphify_python', 'w', encoding='utf-8').write(sys.executable)
Path('graphify-out/.graphify_root').write_text(str(Path('.').resolve()), encoding='utf-8')

result = detect(Path('.'))
Path('graphify-out/.graphify_detect.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
print(f'Detected {result["total_files"]} files')
