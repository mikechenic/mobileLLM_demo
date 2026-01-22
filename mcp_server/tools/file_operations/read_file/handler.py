"""Handler for fs-read-file tool"""
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _normalize_path(relative_path: str) -> Path:
    candidate = (WORKSPACE_ROOT / relative_path).resolve()
    if not candidate.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Path escapes workspace root: {relative_path}")
    return candidate


def handler(path: str, max_bytes: int | None = None) -> str:
    """Read a file within the workspace with optional truncation"""
    try:
        target = _normalize_path(path)
        if not target.exists():
            return f"Not found: {path}"
        if target.is_dir():
            return f"Path is a directory: {path}"

        data = target.read_bytes()
        truncated = False
        if max_bytes is not None and max_bytes > 0 and len(data) > max_bytes:
            data = data[:max_bytes]
            truncated = True

        text = data.decode("utf-8", errors="replace")
        suffix = " (truncated)" if truncated else ""
        return f"Contents of {target.relative_to(WORKSPACE_ROOT)}{suffix}:\n{text}"
    except Exception as exc:
        return f"Error reading file {path}: {exc}"
