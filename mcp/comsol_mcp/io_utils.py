"""
io_utils.py
Small utilities for comsol_mcp.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

def ensure_run_dirs(run_path: Path) -> None:
    (run_path / "inputs").mkdir(parents=True, exist_ok=True)
    (run_path / "outputs").mkdir(parents=True, exist_ok=True)
    (run_path / "plots").mkdir(parents=True, exist_ok=True)
    (run_path / "logs").mkdir(parents=True, exist_ok=True)
    (run_path / "models").mkdir(parents=True, exist_ok=True)

def load_yaml_params(inputs_dir: Path) -> Dict[str, Any]:
    """Load YAML params from the four canonical input files and merge into one dict."""
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML required for comsol.apply_inputs. Install: pip install pyyaml") from e

    params: Dict[str, Any] = {}
    for fn in ["geometry.yaml", "ops.yaml", "materials.yaml", "chemistry.yaml"]:
        p = inputs_dir / fn
        if not p.exists():
            continue
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"{fn} must parse to a mapping/dict")
        params.update(data)
    return params
