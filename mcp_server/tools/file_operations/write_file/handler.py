"""Handler for fs-write-file tool"""
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _normalize_path(relative_path: str) -> Path:
    candidate = (WORKSPACE_ROOT / relative_path).resolve()
    if not candidate.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Path escapes workspace root: {relative_path}")
    return candidate


def handler(path: str, content: str, create_parents: bool = False) -> str:
    """Create or overwrite a file within the workspace"""
    try:
        target = _normalize_path(path)
        if target.exists() and target.is_dir():
            return f"Path is a directory: {path}"

        parent = target.parent
        if not parent.exists():
            if create_parents:
                parent.mkdir(parents=True, exist_ok=True)
            else:
                return f"Parent directory does not exist: {parent.relative_to(WORKSPACE_ROOT)}"

        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {target.relative_to(WORKSPACE_ROOT)}"
    except Exception as exc:
        return f"Error writing file {path}: {exc}"
