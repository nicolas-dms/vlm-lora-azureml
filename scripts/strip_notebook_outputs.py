"""Strip every output and execution count from the notebooks in this repo.

Run this before every commit. It is not cosmetic hygiene: stored outputs are
where this repository actually leaks. Cleaning the *source* of a cell is not
enough, because the cell's recorded output holds the subscription GUID, the
tenant GUID, the resource group and workspace names, the storage account name,
blob URLs, Studio run links and the local `C:\\Users\\...` path of whoever ran it.
None of that is visible when you read the notebook in VS Code.

Verify with a grep over the whole tree afterwards, not by eye:

    Get-ChildItem -Recurse -File | Select-String "<your subscription id>"

(PowerShell 5.1 has no `-Recurse` on Select-String - pipe Get-ChildItem into it.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def strip(path: Path) -> bool:
    """Empty every output in `path`. Returns True if the file changed."""
    original = path.read_text(encoding="utf-8")
    nb = json.loads(original)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        # Leaves behind a run timestamp and, on some kernels, the interpreter
        # path. Harmless-looking, and one more identifier than this repo wants.
        cell.get("metadata", {}).pop("execution", None)

    # indent=1 plus a trailing newline is what Jupyter itself writes. Matching it
    # keeps the diff to the outputs instead of reformatting the whole file.
    cleaned = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if cleaned == original:
        return False
    path.write_text(cleaned, encoding="utf-8")
    return True


def main() -> int:
    notebooks = sorted(REPO.rglob("*.ipynb"))
    notebooks = [n for n in notebooks if ".ipynb_checkpoints" not in n.parts and ".venv" not in n.parts]
    if not notebooks:
        print("no notebooks found")
        return 0

    for nb_path in notebooks:
        changed = strip(nb_path)
        print(f"{'stripped' if changed else 'clean   '}  {nb_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
