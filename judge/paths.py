"""
Canonical paths for shipped benchmark data and repo root resolution.

The MedBench-Agent-95 gold JSONL lives under references/ (skill-style layout).
CLI accepts relative paths from the repository root or absolute paths.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Shipped gold standard: 13 tasks × JSONL (used by batch_runner, gold_retrieval, calibration).
DEFAULT_BENCHMARK_REL = "references/medbench-agent-95"


def repo_root() -> Path:
    """Repository root directory."""
    return _REPO_ROOT


def default_benchmark_dir() -> Path:
    """Absolute path to shipped references/medbench-agent-95."""
    return (_REPO_ROOT / DEFAULT_BENCHMARK_REL).resolve()


def resolve_benchmark_dir(path: str | Path | None) -> str:
    """
    Resolve a benchmark directory to an absolute path string.

    Relative paths are interpreted from the repository root (not cwd),
    so batch jobs work regardless of current working directory.

    If ``medbench-agent-95`` at repo root no longer exists, redirects to
    ``references/medbench-agent-95`` (layout migration).
    """
    if path is None or str(path).strip() == "":
        return str(default_benchmark_dir())
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    cand = (_REPO_ROOT / p).resolve()
    if cand.exists():
        return str(cand)
    # Back-compat: scripts that still pass the old top-level folder name
    parts = p.parts
    if parts == ("medbench-agent-95",) or parts == (".", "medbench-agent-95"):
        alt = default_benchmark_dir()
        if alt.exists():
            return str(alt)
    return str(cand)
