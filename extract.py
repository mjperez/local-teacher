import json, os
from pathlib import Path
from graphify.extract import collect_files, extract
from graphify.llm import extract_corpus_parallel

def main():
    detect = json.loads(Path('graphify-out/.graphify_detect.json').read_text(encoding="utf-8"))

    # Part A: AST
    code_files = []
    for f in detect.get('files', {}).get('code', []):
        code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])

    if code_files:
        ast_result = extract(code_files, cache_root=Path('.'))
        Path('graphify-out/.graphify_ast.json').write_text(json.dumps(ast_result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f'AST: {len(ast_result["nodes"])} nodes, {len(ast_result["edges"])} edges')
    else:
        ast_result = {'nodes':[],'edges':[],'input_tokens':0,'output_tokens':0}
        Path('graphify-out/.graphify_ast.json').write_text(json.dumps(ast_result, ensure_ascii=False), encoding="utf-8")
        print('No code files - skipping AST extraction')

    # Part B: Semantic
    all_files = [Path(f) for cat in ('document', 'paper', 'image') for f in detect.get('files', {}).get(cat, [])]
    if all_files:
        print(f"Running semantic extraction on {len(all_files)} files...")
        key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        sem_result = extract_corpus_parallel(all_files, backend="gemini", api_key=key, root=Path('.'), cache_root=Path('.'))
        Path('graphify-out/.graphify_semantic.json').write_text(json.dumps(sem_result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f'Semantic: {len(sem_result["nodes"])} nodes, {len(sem_result["edges"])} edges')
    else:
        sem_result = {'nodes':[],'edges':[],'hyperedges':[],'input_tokens':0,'output_tokens':0}
        Path('graphify-out/.graphify_semantic.json').write_text(json.dumps(sem_result, ensure_ascii=False), encoding="utf-8")
        print('No semantic files - skipping semantic extraction')

    # Part C: Merge
    seen = {n['id'] for n in ast_result['nodes']}
    merged_nodes = list(ast_result['nodes'])
    for n in sem_result.get('nodes', []):
        if n['id'] not in seen:
            merged_nodes.append(n)
            seen.add(n['id'])

    merged_edges = ast_result['edges'] + sem_result.get('edges', [])
    merged_hyperedges = sem_result.get('hyperedges', [])
    merged = {
        'nodes': merged_nodes,
        'edges': merged_edges,
        'hyperedges': merged_hyperedges,
        'input_tokens': sem_result.get('input_tokens', 0),
        'output_tokens': sem_result.get('output_tokens', 0),
    }
    Path('graphify-out/.graphify_extract.json').write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f'Merged: {len(merged_nodes)} nodes, {len(merged_edges)} edges')

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
