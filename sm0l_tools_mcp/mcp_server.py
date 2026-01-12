from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .manifest_loader import load_tool_specs, ToolSpec
from .toolchains import ChainStep, build_recipe_candidates, classify_goal
from .install_hints import load_extras_map, missing_module_from_stderr, install_hints_for_tool

def _resolve_script_path(tools_dir: Path, source_path: str) -> Path:
    """Resolve a script path robustly even if the extracted folder layout differs.

    Resolution order:
      1) exact match tools_dir/source_path
      2) match by suffix (ending with source_path)
      3) match by basename (unique)
      4) if multiple, prefer shallowest path; else raise with candidates
    """
    # 1) exact
    p = (tools_dir / source_path).resolve()
    if p.exists():
        return p

    # normalize to forward slashes
    sp_norm = source_path.replace("\\", "/")
    sp_parts = sp_norm.split("/")
    basename = sp_parts[-1]

    # 2) suffix match
    suffix_matches = []
    for cand in tools_dir.rglob(basename):
        try:
            rel = cand.relative_to(tools_dir).as_posix()
        except Exception:
            rel = str(cand)
        if rel.endswith(sp_norm):
            suffix_matches.append(cand)
    if len(suffix_matches) == 1:
        return suffix_matches[0].resolve()
    if len(suffix_matches) > 1:
        # pick shallowest
        suffix_matches.sort(key=lambda x: len(x.relative_to(tools_dir).parts))
        return suffix_matches[0].resolve()

    # 3) basename unique
    base_matches = list(tools_dir.rglob(basename))
    if len(base_matches) == 1:
        return base_matches[0].resolve()
    if len(base_matches) > 1:
        # 4) choose shallowest, but also provide candidates in error if ambiguous
        base_matches.sort(key=lambda x: len(x.relative_to(tools_dir).parts))
        # If the shallowest is strictly shallower than the next, pick it.
        if len(base_matches) >= 2:
            d0 = len(base_matches[0].relative_to(tools_dir).parts)
            d1 = len(base_matches[1].relative_to(tools_dir).parts)
            if d0 < d1:
                return base_matches[0].resolve()
        cands = [str(x.relative_to(tools_dir)) for x in base_matches[:25]]
        raise FileNotFoundError(
            "Ambiguous script location for "
            f"{source_path}. Found multiple candidates under tools_dir. "
            f"Set SM0L_TOOLS_DIR to the exact extracted root or de-duplicate filenames. "
            f"Candidates (first 25): {cands}"
        )

    raise FileNotFoundError(
        f"Script not found for source_path={source_path}. "
        f"Tried exact {p} and basename search under {tools_dir}."
    )



def _run_script(
    script_path: Path,
    args: List[str],
    timeout_s: int,
    workdir: Optional[str],
    env: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    merged_env = dict(os.environ)
    if env:
        for k, v in env.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("env must be a dict[str,str]")
            merged_env[k] = v

    cmd = [sys.executable, str(script_path), *args]

    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            cwd=workdir or None,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "exit_code": int(p.returncode),
            "stdout": p.stdout,
            "stderr": p.stderr,
            "duration_ms": dt_ms,
            "cmd": cmd,
            "workdir": workdir,
        }
    except subprocess.TimeoutExpired as e:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "exit_code": -1,
            "stdout": (e.stdout or ""),
            "stderr": (e.stderr or "") + f"\nTIMEOUT after {timeout_s}s",
            "duration_ms": dt_ms,
            "cmd": cmd,
            "workdir": workdir,
        }


def build_server(tools_dir: Path) -> FastMCP:
    mcp = FastMCP("sm0l-tools-mcp")

    specs = load_tool_specs()
    extras_map = load_extras_map()

    # Quick lookup by tool_name
    spec_by_tool = {s.tool_name: s for s in specs}

    @mcp.tool
    def list_sm0l_tools() -> Dict[str, Any]:
        """List all tools from the bundled manifest (grouped by domain/subdomain)."""
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for s in specs:
            grouped.setdefault(s.domain, {}).setdefault(s.subdomain, []).append({
                "id": s.id,
                "tool_name": s.tool_name,
                "source_path": s.source_path,
                "description": s.description,
                "tags": s.tags,
                "deps_pypi": s.deps_pypi,
            })
        return {"count": len(specs), "grouped": grouped}

    @mcp.tool
    def tool_info(tool_name: str) -> Dict[str, Any]:
        """Return detailed metadata for a specific tool."""
        s = spec_by_tool.get(tool_name)
        if not s:
            raise ValueError(f"Unknown tool_name: {tool_name}")
        return {
            "id": s.id,
            "tool_name": s.tool_name,
            "source_path": s.source_path,
            "description": s.description,
            "domain": s.domain,
            "subdomain": s.subdomain,
            "tags": s.tags,
            "deps_pypi": s.deps_pypi,
            "imports_unknown": s.imports_unknown,
        }

    # Create one MCP tool per manifest entry, but route execution to the actual script path.
    for s in specs:
        script_path = _resolve_script_path(tools_dir, s.source_path)

        def _make_tool(spec: ToolSpec, sp: Path):
            def _tool(
                args: List[str] = [],
                timeout_s: int = 60,
                workdir: Optional[str] = None,
                env: Optional[Dict[str, str]] = None,
            ) -> Dict[str, Any]:
                if not isinstance(args, list) or any(not isinstance(x, str) for x in args):
                    raise ValueError("args must be a list[str]")
                result = _run_script(sp, args=args, timeout_s=timeout_s, workdir=workdir, env=env)

                # Attach manifest + install hints on failure.
                if result.get("exit_code", 0) != 0:
                    missing = missing_module_from_stderr(result.get("stderr", "")) or None
                    result["tool_meta"] = {
                        "id": spec.id,
                        "tool_name": spec.tool_name,
                        "domain": spec.domain,
                        "subdomain": spec.subdomain,
                        "deps_pypi": spec.deps_pypi,
                        "source_path": spec.source_path,
                    }
                    result["install_hints"] = install_hints_for_tool(
                        tool_name=spec.tool_name,
                        missing_module=missing,
                        deps_pypi=spec.deps_pypi,
                        extras_map=extras_map,
                    )
                return result

            _tool.__name__ = spec.tool_name
            _tool.__doc__ = f"""{spec.description}

Manifest ID: {spec.id}
Domain: {spec.domain}/{spec.subdomain}
Tags: {", ".join(spec.tags[:24])}{(" ..." if len(spec.tags) > 24 else "")}
"""
            return _tool

        mcp.tool(_make_tool(s, script_path))

    return mcp


def main() -> None:
    # Expect your extracted scripts to live in ./tools next to this repo, or override via SM0L_TOOLS_DIR
    here = Path(__file__).resolve().parent.parent
    tools_dir = Path(os.environ.get("SM0L_TOOLS_DIR", str(here / "tools"))).resolve()

    if not tools_dir.exists():
        raise SystemExit(
            "Missing tools directory (or path is wrong).\n"
            f"Expected: {tools_dir}\n"
            "Set SM0L_TOOLS_DIR to your extracted scripts folder."
        )

    mcp = build_server(tools_dir)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
