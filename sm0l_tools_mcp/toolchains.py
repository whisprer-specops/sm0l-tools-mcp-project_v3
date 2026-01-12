from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import re


@dataclass(frozen=True)
class ChainStep:
    tool_name: str
    id: str
    domain: str
    subdomain: str
    tags: List[str]
    deps_pypi: List[str]
    description: str


def _goal_keywords(goal: str) -> Set[str]:
    goal = (goal or "").lower()
    # split on non-word; keep short tokens too
    toks = set(re.findall(r"[a-z0-9_]+", goal))
    return toks


def classify_goal(goal: str) -> Dict[str, Any]:
    """Very small heuristic classifier from natural language goal to desired capabilities."""
    toks = _goal_keywords(goal)

    wants = {
        "web": bool(toks & {"web","http","scrape","crawler","crawl","api","fetch","request","requests","html","site","url"}),
        "browser": bool(toks & {"selenium","playwright","browser","headless","login","render","js","javascript"}),
        "pdf": bool(toks & {"pdf","paper","papers","document","docs","extract"}),
        "office": bool(toks & {"excel","xlsx","spreadsheet","workbook","sheet","openpyxl"}),
        "nlp": bool(toks & {"nlp","summarize","summary","sentiment","entities","entity","ner","language","text"}),
        "embeddings": bool(toks & {"embedding","embeddings","vector","semantic","similarity","search"}),
        "ml": bool(toks & {"model","train","classification","regression","sklearn","torch","pytorch"}),
        "viz": bool(toks & {"plot","chart","visualize","viz","graph","dashboard","map","plotly","matplotlib"}),
        "system": bool(toks & {"monitor","cpu","ram","disk","process","watch","watchdog","filesystem"}),
        "schedule": bool(toks & {"schedule","cron","timer","apscheduler","jobs","workflow","prefect","dagster","airflow"}),
    }

    # Default: if nothing matches, assume "misc helpers"
    if not any(wants.values()):
        wants["misc"] = True
    else:
        wants["misc"] = False

    return wants


def phases_for_goal(wants: Dict[str, Any]) -> List[str]:
    """Ordered phases; chain builder selects one tool per phase if possible."""
    phases: List[str] = []
    if wants.get("web") or wants.get("browser"):
        phases.append("ingest_web")
    if wants.get("pdf") or wants.get("office"):
        phases.append("ingest_docs")
    if wants.get("system"):
        phases.append("observe")
    # transformation is nearly always useful if dataframes are present
    phases.append("transform")
    if wants.get("nlp") or wants.get("embeddings") or wants.get("ml"):
        phases.append("analyze")
    if wants.get("viz"):
        phases.append("visualize")
    phases.append("report")
    return phases


def pick_best(steps: List[ChainStep], preferred_domains: List[Tuple[str,str]], required_tags: Set[str] = set()) -> Optional[ChainStep]:
    """Pick the best step by preferred domain order, then tag overlap, then shortest deps list."""
    if required_tags:
        cand = [s for s in steps if required_tags & set(s.tags)]
        if cand:
            steps = cand

    for d, sd in preferred_domains:
        cand = [s for s in steps if s.domain == d and (sd == "*" or s.subdomain == sd)]
        if cand:
            # rank by (tag overlap, deps count, description length)
            cand.sort(key=lambda s: (-len(required_tags & set(s.tags)), len(s.deps_pypi), len(s.description)))
            return cand[0]

    # fallback: any
    if not steps:
        return None
    steps.sort(key=lambda s: (len(s.deps_pypi), len(s.description)))
    return steps[0]


def build_recipe_candidates(all_steps: List[ChainStep], goal: str, max_candidates: int = 5) -> List[Dict[str, Any]]:
    wants = classify_goal(goal)
    phases = phases_for_goal(wants)

    # phase preferences
    phase_prefs: Dict[str, List[Tuple[str,str]]] = {
        "ingest_web": [("WEB","SCRAPE"), ("WEB","BROWSER"), ("WEB","HTTP"), ("WEB","API"), ("MISC","MISC")],
        "ingest_docs": [("DOC","PDF"), ("DOC","OFFICE"), ("DOC","MISC"), ("MISC","MISC")],
        "observe": [("SYS","MON"), ("MISC","MISC")],
        "transform": [("DATA","DF"), ("MISC","MISC")],
        "analyze": [("NLP","EMBED"), ("NLP","SPACY"), ("NLP","BASIC"), ("ML","TORCH"), ("ML","SKLEARN"), ("MISC","MISC")],
        "visualize": [("VIZ","PLOTS"), ("VIZ","MAPS"), ("VIZ","GPU"), ("VIZ","TERM"), ("MISC","MISC")],
        "report": [("DOC","OFFICE"), ("DOC","PDF"), ("VIZ","TERM"), ("MISC","MISC")],
    }

    phase_tags: Dict[str, Set[str]] = {
        "ingest_web": {"requests","httpx","urllib3","bs4","mechanicalsoup","selenium","playwright","pyppeteer","scrape","browser","api","http"},
        "ingest_docs": {"pdf","pypdf2","pymupdf","fitz","pdfplumber","openpyxl","xlsx","excel"},
        "observe": {"psutil","watchdog","system","filesystem"},
        "transform": {"pandas","polars","numpy","arrow","dataframe"},
        "analyze": {"nlp","spacy","textblob","sentence_transformers","embedding","torch","sklearn"},
        "visualize": {"plotly","matplotlib","seaborn","datashader","holoviews","pydeck","pyecharts","plotext","chart","plot","map"},
        "report": {"openpyxl","pdf","rich","pandas","polars"},
    }

    # Candidate generation: for now, one canonical chain and a few alternates by swapping within phase families.
    # We'll create up to max_candidates by taking top-N options per phase and combining lightly (cartesian with cap).
    per_phase_options: List[List[ChainStep]] = []
    for ph in phases:
        pref = phase_prefs.get(ph, [("MISC","MISC")])
        tags = phase_tags.get(ph, set())
        # gather plausible steps (cap to 8 per phase for combinatorics)
        plausible = [s for s in all_steps if (set(s.tags) & tags) or (s.domain in {p[0] for p in pref})]
        # rank and keep top options
        ranked: List[ChainStep] = []
        used = set()
        for _ in range(12):
            pick = pick_best([s for s in plausible if s.tool_name not in used], pref, required_tags=tags)
            if not pick:
                break
            ranked.append(pick)
            used.add(pick.tool_name)
        if not ranked:
            ranked = plausible[:1]
        per_phase_options.append(ranked[:6])

    # Build candidate chains with controlled branching
    candidates: List[List[ChainStep]] = [[]]
    for opts in per_phase_options:
        new_cands = []
        for c in candidates:
            for o in opts[:3]:  # keep branching small
                new_cands.append(c + [o])
        candidates = new_cands[:max_candidates]

    # Render candidate recipes
    out: List[Dict[str, Any]] = []
    for idx, chain in enumerate(candidates[:max_candidates], start=1):
        deps = sorted({p for s in chain for p in s.deps_pypi})
        out.append({
            "candidate": idx,
            "phases": phases,
            "deps_union": deps,
            "steps": [
                {
                    "tool_name": s.tool_name,
                    "id": s.id,
                    "domain": s.domain,
                    "subdomain": s.subdomain,
                    "tags": s.tags,
                    "deps_pypi": s.deps_pypi,
                    "description": s.description,
                }
                for s in chain
            ],
        })
    return out
