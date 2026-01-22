"""Handler for fs-run-command tool"""
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _normalize_path(relative_path: str) -> Path:
    candidate = (WORKSPACE_ROOT / relative_path).resolve()
    if not candidate.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Path escapes workspace root: {relative_path}")
    return candidate


def _trim(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def handler(command: str, cwd: str = ".", timeout_seconds: int = 30, max_output: int = 4000) -> str:
    """Run a shell command from within the workspace"""
    try:
        workdir = _normalize_path(cwd)
        if not workdir.is_dir():
            return f"Working directory is not a folder: {cwd}"

        completed = subprocess.run(
            command,
            cwd=workdir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        stdout = _trim(completed.stdout or "", max_output)
        stderr = _trim(completed.stderr or "", max_output)

        parts = [f"exit_code: {completed.returncode}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_seconds} seconds"
    except Exception as exc:
        return f"Error running command '{command}': {exc}"
