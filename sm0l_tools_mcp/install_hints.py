from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Dict, List, Optional, Set, Tuple

# Common module -> PyPI package mapping (best-effort).
# For higher accuracy, we prefer manifest deps_pypi when available.
MOD_TO_PKG: Dict[str, str] = {
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "fitz": "PyMuPDF",
    "sentence_transformers": "sentence-transformers",
    "plotly_resampler": "plotly-resampler",
    "youtubesearchpython": "youtube-search-python",
}


def load_extras_map() -> Dict[str, Set[str]]:
    """Load extras->packages mapping bundled in package data."""
    with resources.files("sm0l_tools_mcp.data").joinpath("extras_map.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: set(v) for k, v in raw.items()}


def missing_module_from_stderr(stderr: str) -> Optional[str]:
    """Parse a missing module name from Python import errors."""
    m = re.search(r"No module named ['\"]([A-Za-z0-9_\.]+)['\"]", stderr)
    if not m:
        return None
    return m.group(1).split(".")[0]


def choose_extras_for_packages(extras_map: Dict[str, Set[str]], pkgs: Set[str]) -> List[str]:
    """Pick extras that cover as many pkgs as possible (greedy set cover)."""
    remaining = set(pkgs)
    chosen: List[str] = []

    # Don't auto-suggest "all" unless user explicitly wants it.
    candidates = {k: v for k, v in extras_map.items() if k != "all"}

    while remaining:
        best = None
        best_cover = set()
        for extra, e_pkgs in candidates.items():
            cover = remaining & e_pkgs
            if len(cover) > len(best_cover):
                best = extra
                best_cover = cover
        if not best or not best_cover:
            break
        chosen.append(best)
        remaining -= best_cover
        # keep candidate extras; no need to remove
    return sorted(set(chosen))


def install_hints_for_tool(
    tool_name: str,
    missing_module: Optional[str],
    deps_pypi: List[str],
    extras_map: Dict[str, Set[str]],
) -> Dict:
    """Generate install hints payload for a tool execution error."""
    deps_set = set(deps_pypi)
    suggested_extras = choose_extras_for_packages(extras_map, deps_set) if deps_set else []

    hints: Dict = {
        "tool": tool_name,
        "missing_module": missing_module,
        "deps_pypi": deps_pypi,
        "suggested_extras": suggested_extras,
        "pip_commands": [],
        "notes": [],
    }

    # Prefer extras if we can infer them from the deps list.
    if suggested_extras:
        hints["pip_commands"].append(f"pip install -e .[{','.join(suggested_extras)}]")
        hints["notes"].append("Preferred: install via extras for reproducible capability bundles.")
    elif missing_module:
        pkg = MOD_TO_PKG.get(missing_module, missing_module)
        hints["pip_commands"].append(f"pip install {pkg}")
        hints["notes"].append("Fallback: install the missing package directly.")
    else:
        hints["pip_commands"].append("pip install -e .")
        hints["notes"].append("Install the base package first, then add extras as needed.")

    # Extra system-deps reminders
    if "pytesseract" in deps_set or missing_module == "pytesseract":
        hints["notes"].append("OCR also needs the system 'tesseract' binary installed and on PATH.")
    if "pydub" in deps_set or missing_module == "pydub":
        hints["notes"].append("Audio tools often need ffmpeg installed and on PATH.")
    if "playwright" in deps_set or missing_module == "playwright":
        hints["pip_commands"].append("playwright install")
        hints["notes"].append("Playwright needs browser binaries installed via 'playwright install'.")

    return hints

def system_dependency_status() -> Dict[str, Optional[bool]]:
    """Best-effort checks for common system dependencies."""
    import shutil

    def has(exe: str) -> Optional[bool]:
        try:
            return shutil.which(exe) is not None
        except Exception:
            return None

    return {
        "tesseract": has("tesseract"),
        "ffmpeg": has("ffmpeg"),
    }
