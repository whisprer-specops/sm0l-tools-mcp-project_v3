from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolSpec:
    id: str
    tool_name: str
    source_path: str
    description: str
    domain: str
    subdomain: str
    tags: List[str]
    deps_pypi: List[str]
    imports_unknown: List[str]


def load_manifest() -> Dict[str, Any]:
    with resources.files("sm0l_tools_mcp.data").joinpath("sm0l_tools_manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tool_specs() -> List[ToolSpec]:
    m = load_manifest()
    out: List[ToolSpec] = []
    for t in m.get("tools", []):
        out.append(ToolSpec(
            id=t["id"],
            tool_name=t["tool_name"],
            source_path=t["source_path"],
            description=t.get("description",""),
            domain=t.get("domain","MISC"),
            subdomain=t.get("subdomain","MISC"),
            tags=t.get("tags", []),
            deps_pypi=t.get("deps_pypi", []),
            imports_unknown=t.get("imports_unknown", []),
        ))
    return out
