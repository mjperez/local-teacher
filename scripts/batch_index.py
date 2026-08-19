import json
import os
import re
import ssl
import subprocess
import sys
import urllib.request

# --- Local Ollama Configuration ---
LOCAL_EMBEDDING_URL = "http://localhost:11434/v1"
LOCAL_EMBEDDING_MODEL = "openai:nomic-embed-text"

REQUIREMENTS_PATH = "requirements.txt"
MANIFEST_PATH = "indexed_packages.json"

# Skip boilerplate packages without standalone docs
IGNORED_PACKAGES = {
    "pip", "setuptools", "wheel", "certifi", "charset-normalizer",
    "idna", "urllib3", "six", "typing_extensions", "colorama",
    "attrs", "iniconfig", "packaging", "filelock", "frozenlist"
}

# Matches: name, name==1.2.3, name>=1.0, name~=1.0, name[extra]==1.0
REQ_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?:==|>=|<=|~=|!=)?\s*([A-Za-z0-9.\-+_]*)"
)


def parse_requirements(path):
    """Lee requirements.txt y devuelve [{'name': ..., 'version': ...}, ...].
    Ignora comentarios, líneas vacías, -r/-e/git/URLs."""
    packages = []
    skipped = []

    if not os.path.exists(path):
        print(f"No encontré {path}")
        return packages

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r", "-e", "--", "git+", "http://", "https://")):
                skipped.append(line)
                continue

            match = REQ_LINE_RE.match(line)
            if not match:
                skipped.append(line)
                continue

            name, version = match.group(1), match.group(2)
            packages.append({"name": name, "version": version or ""})

    if skipped:
        print(f"Ignoré {len(skipped)} línea(s) que no pude interpretar:")
        for s in skipped:
            print(f"  - {s}")

    return packages


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def parse_force_arg():
    """Lee --force nombre1,nombre2 de sys.argv y devuelve un set en minúsculas."""
    force_list = set()
    if "--force" in sys.argv:
        idx = sys.argv.index("--force")
        if idx + 1 < len(sys.argv):
            force_list = {n.strip().lower() for n in sys.argv[idx + 1].split(",") if n.strip()}
        else:
            print("⚠️  Usaste --force sin especificar paquetes. Ejemplo: --force kuzu,redis")
    return force_list


def main():
    force_list = parse_force_arg()

    # 1. Provide isolated variables strictly to child processes
    local_env = os.environ.copy()
    local_env["OPENAI_API_KEY"] = "ollama"
    local_env["OPENAI_BASE_URL"] = LOCAL_EMBEDDING_URL
    local_env["OPENAI_API_BASE"] = LOCAL_EMBEDDING_URL
    local_env["DOCS_MCP_EMBEDDING_MODEL"] = LOCAL_EMBEDDING_MODEL

    ssl_context = ssl._create_unverified_context()

    # 2. Get list of packages from requirements.txt
    packages = parse_requirements(REQUIREMENTS_PATH)
    if not packages:
        print("No se encontraron paquetes para indexar.")
        return

    manifest = load_manifest()

    filtered = [
        p for p in packages
        if p["name"].lower() not in IGNORED_PACKAGES
        and (p["name"].lower() in force_list or p["name"] not in manifest)
    ]

    already_done = len(packages) - len(filtered)
    if force_list:
        print(f"Forzando re-scrape de: {', '.join(sorted(force_list))}")
    print(f"{already_done} paquete(s) ya indexados, saltados.")
    print(f"Indexing {len(filtered)} packages locally...\n")

    if not filtered:
        print("Nada para hacer — todo ya está indexado.")
        return

    # 3. Iterate and scrape
    for pkg in filtered:
        name = pkg["name"]
        version = pkg.get("version", "")

        pypi_api = f"https://pypi.org/pypi/{name}/json"
        req = urllib.request.Request(pypi_api, headers={"User-Agent": "Mozilla/5.0"})

        doc_url = None
        try:
            with urllib.request.urlopen(req, timeout=5, context=ssl_context) as resp:
                data = json.loads(resp.read().decode())
                info = data.get("info", {})
                urls = info.get("project_urls") or {}

                doc_url = (
                    urls.get("Documentation")
                    or urls.get("Docs")
                    or urls.get("Homepage")
                    or info.get("home_page")
                    or urls.get("Source")
                    or urls.get("Repository")
                    or f"https://pypi.org/project/{name}/"
                )
        except Exception:
            doc_url = f"https://pypi.org/project/{name}/"

        # Ensure trailing slash and valid URL structure
        if not doc_url.startswith("http"):
            doc_url = f"https://pypi.org/project/{name}/"

        print("\n==========================================")
        print(f"==> [{name}@{version}] Scraping from: {doc_url}")
        print("==========================================")

        cmd = [
            "npx", "-y", "@arabold/docs-mcp-server", "scrape",
            name,
            doc_url,
            "--embedding-model", LOCAL_EMBEDDING_MODEL,
            "--max-pages", "15",
            "--scope", "subpages"
        ]
        if version:
            cmd.extend(["--version", version])

        try:
            result = subprocess.run(cmd, shell=True, env=local_env, check=False)
        except Exception as err:
            print(f"Error scraping {name}: {err}")
            continue

        # Solo marcamos como indexado si terminó sin error.
        # Si falla, queda pendiente para la próxima corrida automáticamente.
        if result.returncode == 0:
            manifest[name] = version
            save_manifest(manifest)
        else:
            print(f"⚠️  {name} falló (returncode={result.returncode}), no se marca como indexado. Reintentará la próxima corrida.")


if __name__ == "__main__":
    main()