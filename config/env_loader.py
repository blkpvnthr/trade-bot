"""
config/env_loader.py

Instance-aware environment loader for QuantumQuant.

Load order (later values override earlier):
  1. Root .env            — shared defaults across all instances
  2. instances/<INSTANCE>/.env — instance-specific overrides

Usage
-----
    from config.env_loader import load_instance_env
    load_instance_env()                      # reads INSTANCE from shell env
    load_instance_env("paper")               # explicit instance name
    path = load_instance_env()               # returns Path to instance .env

Instance selection
------------------
    INSTANCE=paper python run.py
    INSTANCE=pivot_engine python run.py
    INSTANCE=rl_breakout python run.py

    If INSTANCE is not set, defaults to "default".

Idempotency
-----------
    Safe to call multiple times — the same instance is only loaded once.
    If called with a different instance name it will load that env on top of
    whatever is already in os.environ, with override=True.

Precedence
----------
    Shell environment > instance .env > root .env
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Set

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_loaded_instances: Set[str] = set()


def load_instance_env(instance: str | None = None) -> Path:
    """
    Load shared root .env then instance-specific .env into os.environ.

    Parameters
    ----------
    instance : str | None
        Instance name (e.g. "paper", "live", "pivot_engine").
        Defaults to the INSTANCE shell variable, then "default".

    Returns
    -------
    Path
        Path to the instance .env file (may not exist if using root .env only).
    """
    from dotenv import load_dotenv  # imported lazily so the package is optional at import time

    if instance is None:
        instance = os.environ.get("INSTANCE", "default")

    instance_env = _PROJECT_ROOT / "instances" / instance / ".env"

    if instance in _loaded_instances:
        return instance_env

    # Step 1: shared root .env — override=False means shell env always wins
    root_env = _PROJECT_ROOT / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)

    # Step 2: instance .env — override=True so instance values beat root defaults
    if instance_env.exists():
        load_dotenv(instance_env, override=True)

    _loaded_instances.add(instance)
    return instance_env


def instance_name() -> str:
    """Return the active instance name (reads INSTANCE from os.environ)."""
    return os.environ.get("INSTANCE", "default")


def instance_root() -> Path:
    """Return the Path to the active instance directory."""
    return _PROJECT_ROOT / "instances" / instance_name()
