"""Handler for fs-delete-path tool"""
import shutil
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _normalize_path(relative_path: str) -> Path:
    candidate = (WORKSPACE_ROOT / relative_path).resolve()
    if not candidate.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Path escapes workspace root: {relative_path}")
    return candidate


def handler(path: str, recursive: bool = False) -> str:
    """Delete a file or directory within the workspace"""
    try:
        target = _normalize_path(path)
        if target == WORKSPACE_ROOT:
            return "Refusing to delete the workspace root"
        if not target.exists():
            return f"Not found: {path}"

        if target.is_dir():
            if recursive:
                shutil.rmtree(target)
                return f"Recursively deleted directory {target.relative_to(WORKSPACE_ROOT)}"
            if any(target.iterdir()):
                return "Directory is not empty; set recursive=true to remove"
            target.rmdir()
            return f"Deleted empty directory {target.relative_to(WORKSPACE_ROOT)}"

        target.unlink()
        return f"Deleted file {target.relative_to(WORKSPACE_ROOT)}"
    except Exception as exc:
        return f"Error deleting {path}: {exc}"
