#!/usr/bin/env python3
"""
comsol_mcp_server.py
Stdio MCP server skeleton for driving COMSOL runs for PFR.

IMPORTANT:
- This is a skeleton with a backend adapter (comsol_api.py).
- Implement the adapter using either COMSOL Java API (mphserver) or an available Python client.
- The server enforces run-directory I/O and produces deterministic outputs.
"""

import json
import sys
import traceback
from pathlib import Path

from comsol_api import ComsolBackend
from io_utils import ensure_run_dirs, load_yaml_params

ROOT = Path.cwd()
RUNS_DIR = ROOT / "results" / "runs"

_backend = ComsolBackend()

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def run_path(run_id: str) -> Path:
    return RUNS_DIR / run_id

def handle_open_or_create_model(args):
    rp = run_path(args["run_id"])
    ensure_run_dirs(rp)
    model_file = _backend.open_or_create(
        rp,
        template_path=args.get("template_path"),
        model_path=args.get("model_path"),
    )
    return {"status": "ok", "model_file": str(model_file)}

def handle_apply_inputs(args):
    rp = run_path(args["run_id"])
    inputs_dir = Path(args.get("inputs_dir") or (rp / "inputs"))
    params = load_yaml_params(inputs_dir)
    strict = bool(args.get("strict", True))
    _backend.apply_parameters(rp, params, strict=strict)
    return {"status": "ok", "applied": list(params.keys())}

def handle_build_geometry(args):
    _backend.build_geometry(run_path(args["run_id"]))
    return {"status": "ok"}

def handle_mesh(args):
    _backend.mesh(run_path(args["run_id"]), mesh_id=args.get("mesh_id"))
    return {"status": "ok"}

def handle_run_pipeline(args):
    pipeline = args.get("pipeline") or ["A", "B", "C", "D"]
    _backend.run_pipeline(run_path(args["run_id"]), pipeline)
    return {"status": "ok", "pipeline": pipeline}

def handle_run_study(args):
    _backend.run_study(run_path(args["run_id"]), args["study_id"])
    return {"status": "ok", "study_id": args["study_id"]}

def handle_export_fields(args):
    rp = run_path(args["run_id"])
    fmt = args.get("format") or "h5"
    fields = args.get("fields")
    out = _backend.export_fields(rp, fmt=fmt, fields=fields)
    return {"status": "ok", "outputs": out}

def handle_export_kpis(args):
    out = _backend.export_kpis(run_path(args["run_id"]))
    return {"status": "ok", "kpis_path": out}

def handle_render_png(args):
    out = _backend.render_png(run_path(args["run_id"]), plot_id=args["plot_id"])
    return {"status": "ok", "png_path": out}

def handle_close_model(args):
    _backend.close(run_path(args["run_id"]))
    return {"status": "ok"}

HANDLERS = {
    "comsol.open_or_create_model": handle_open_or_create_model,
    "comsol.apply_inputs": handle_apply_inputs,
    "comsol.build_geometry": handle_build_geometry,
    "comsol.mesh": handle_mesh,
    "comsol.run_pipeline": handle_run_pipeline,
    "comsol.run_study": handle_run_study,
    "comsol.export_fields": handle_export_fields,
    "comsol.export_kpis": handle_export_kpis,
    "comsol.render_png": handle_render_png,
    "comsol.close_model": handle_close_model,
}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            name = req.get("name")
            args = req.get("arguments", {})
            if name not in HANDLERS:
                send({"error": f"Unknown method {name}"})
                continue
            send(HANDLERS[name](args))
        except Exception as e:
            send({"error": str(e), "traceback": traceback.format_exc()})

if __name__ == "__main__":
    main()
