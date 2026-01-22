"""Handler for fs-list-directory tool"""
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _normalize_path(relative_path: str) -> Path:
    candidate = (WORKSPACE_ROOT / relative_path).resolve()
    if not candidate.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Path escapes workspace root: {relative_path}")
    return candidate


def handler(path: str = ".", max_entries: int = 200, show_hidden: bool = False) -> str:
    """List entries in a workspace directory"""
    try:
        target = _normalize_path(path)
        if not target.exists():
            return f"Not found: {path}"
        if not target.is_dir():
            return f"Path is not a directory: {path}"

        entries = []
        for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if not show_hidden and entry.name.startswith('.'):
                continue
            marker = "/" if entry.is_dir() else ""
            entries.append(f"{entry.name}{marker}")
            if len(entries) >= max_entries:
                break

        relative = target.relative_to(WORKSPACE_ROOT)
        joined = "\n".join(entries) if entries else "<empty>"
        return f"Listing for {relative or '.'}:\n{joined}"
    except Exception as exc:
        return f"Error listing {path}: {exc}"
