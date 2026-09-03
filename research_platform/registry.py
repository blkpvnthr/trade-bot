"""Immutable, content-addressed experiment registry — reproducibility substrate.

A spec_hash is computed over {name, config, search_space, promotion_criteria, git_commit,
data_snapshot}. Same inputs → same hash → same experiment (idempotent registration). Any change
to config / code / data → new hash → new experiment. created_utc is metadata, NOT in the hash.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def data_snapshot_hash(paths: List[str]) -> str:
    """Fast manifest hash over (path, size, mtime_ns) for the given files/dirs."""
    h = hashlib.sha256()
    for p in sorted(map(str, paths)):
        pp = Path(p)
        files = sorted(pp.rglob("*")) if pp.is_dir() else [pp]
        for f in files:
            try:
                st = f.stat()
                h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
            except Exception:
                h.update(f"{f}:missing".encode())
    return h.hexdigest()[:16]


@dataclass
class ExperimentSpec:
    name: str
    description: str
    config: Dict[str, Any]
    search_space: Dict[str, Any]          # PREREGISTERED: what is being tested (anti p-hacking)
    promotion_criteria: Dict[str, Any]    # PREREGISTERED: thresholds for promote/reject
    data_paths: List[str] = field(default_factory=list)
    git_commit: str = field(default_factory=git_commit)
    data_snapshot: str = ""
    created_utc: Optional[str] = None      # metadata, excluded from hash

    def __post_init__(self):
        if not self.data_snapshot and self.data_paths:
            self.data_snapshot = data_snapshot_hash(self.data_paths)
        if self.created_utc is None:
            self.created_utc = datetime.now(timezone.utc).isoformat()

    @property
    def spec_hash(self) -> str:
        payload = {"name": self.name, "config": self.config, "search_space": self.search_space,
                   "promotion_criteria": self.promotion_criteria, "git_commit": self.git_commit,
                   "data_snapshot": self.data_snapshot}
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["spec_hash"] = self.spec_hash
        return d


class Registry:
    """Append-only spec store. Registration is idempotent and refuses to mutate an existing spec."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, spec: ExperimentSpec) -> str:
        p = self.root / f"{spec.spec_hash}.json"
        if p.exists():
            return spec.spec_hash                      # already registered → reproducible no-op
        p.write_text(json.dumps(spec.to_dict(), indent=2, default=str))
        return spec.spec_hash

    def get(self, spec_hash: str) -> Optional[Dict[str, Any]]:
        p = self.root / f"{spec_hash}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def list(self) -> List[Dict[str, Any]]:
        return [json.loads(p.read_text()) for p in sorted(self.root.glob("*.json"))]
