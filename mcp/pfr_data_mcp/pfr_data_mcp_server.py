#!/usr/bin/env python3
"""
pfr_data_mcp server.py
Stdio-based MCP server enforcing PFR_Data_Schema.md
"""

import json
import sys
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
RUNS_DIR = ROOT / "results" / "runs"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def send(response):
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()

def handle_create_run(params):
    run_id = params.get("run_id")
    if not run_id:
        run_id = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    run_path = RUNS_DIR / run_id
    run_path.mkdir(parents=True, exist_ok=False)

    run_card = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "status": "created",
        "model_version": params["model_version"],
        "physics_engine": params["physics_engine"],
        "discovery_engine": params["discovery_engine"],
        "notes": params.get("notes", "")
    }

    with open(run_path / "run_card.json", "w") as f:
        json.dump(run_card, f, indent=2)

    send({"run_id": run_id, "path": str(run_path)})

def handle_snapshot_inputs(params):
    run_id = params["run_id"]
    params_dir = Path(params["params_dir"])
    run_path = RUNS_DIR / run_id
    inputs_dir = run_path / "inputs"
    inputs_dir.mkdir(exist_ok=False)

    hashes = {}
    for fname in ["geometry.yaml", "ops.yaml", "materials.yaml", "chemistry.yaml"]:
        src = params_dir / fname
        dst = inputs_dir / fname
        shutil.copy2(src, dst)
        hashes[fname] = sha256_file(dst)

    run_card_path = run_path / "run_card.json"
    run_card = json.load(open(run_card_path))
    run_card["input_hashes"] = hashes
    run_card["status"] = "inputs_snapshotted"

    json.dump(run_card, open(run_card_path, "w"), indent=2)
    send({"status": "ok"})


def handle_validate_outputs(params):
    """Validate run outputs against PFR_Data_Schema.md"""
    run_id = params["run_id"]
    run_path = RUNS_DIR / run_id
    outputs_dir = run_path / "outputs"
    
    issues = []
    
    # Check required files exist
    fields_file = outputs_dir / "fields.h5"
    kpis_file = outputs_dir / "kpis.json"
    
    if not fields_file.exists():
        issues.append("Missing outputs/fields.h5")
    if not kpis_file.exists():
        issues.append("Missing outputs/kpis.json")
    
    # If fields.h5 exists, check for required datasets
    if fields_file.exists():
        try:
            import h5py
            with h5py.File(fields_file, 'r') as f:
                # Check for required Flash physics fields
                if 'flash/DeltaB' not in f:
                    issues.append("Missing flash/DeltaB in fields.h5")
                if 'flash/chi' not in f:
                    issues.append("Missing flash/chi in fields.h5")
                
                # Check chi bounds if present
                if 'flash/chi' in f:
                    chi = f['flash/chi'][:]
                    if chi.min() < 0 or chi.max() > 1:
                        issues.append(f"chi out of bounds [0,1]: min={chi.min()}, max={chi.max()}")
                    
                # Check for NaNs/infs
                for key in f.keys():
                    if hasattr(f[key], 'shape'):
                        import numpy as np
                        data = f[key][:]
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            issues.append(f"NaN or inf values in {key}")
        except ImportError:
            # h5py not available - check if it's a placeholder
            content = fields_file.read_text(errors='ignore')
            if 'TODO' in content or len(content) < 100:
                issues.append("fields.h5 is a placeholder (h5py not available for validation)")
        except Exception as e:
            issues.append(f"Error reading fields.h5: {str(e)}")
    
    # Validate kpis.json structure
    if kpis_file.exists():
        try:
            kpis = json.load(open(kpis_file))
            if kpis.get("status") == "TODO":
                issues.append("kpis.json is a placeholder")
            # Check for required KPI sections
            required_sections = ["flash", "reduction", "power", "thermal", "plasma"]
            for section in required_sections:
                if section not in kpis:
                    issues.append(f"Missing KPI section: {section}")
        except Exception as e:
            issues.append(f"Error reading kpis.json: {str(e)}")
    
    # Update run_card with validation result
    run_card_path = run_path / "run_card.json"
    run_card = json.load(open(run_card_path))
    
    valid = len(issues) == 0
    run_card["validation"] = {
        "valid": valid,
        "issues": issues,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }
    run_card["status"] = "validated" if valid else "invalid"
    
    json.dump(run_card, open(run_card_path, "w"), indent=2)
    send({"valid": valid, "issues": issues})


def handle_register_run(params):
    """Register a validated run as immutable and queryable"""
    run_id = params["run_id"]
    run_path = RUNS_DIR / run_id
    run_card_path = run_path / "run_card.json"
    
    run_card = json.load(open(run_card_path))
    
    # Only allow registration if validation passed
    if run_card.get("status") != "validated":
        send({"error": f"Cannot register run with status '{run_card.get('status')}'. Must be 'validated'."})
        return
    
    if not run_card.get("validation", {}).get("valid", False):
        send({"error": "Cannot register invalid run. Fix validation issues first."})
        return
    
    # Mark as registered
    run_card["status"] = "registered"
    run_card["registered_utc"] = datetime.utcnow().isoformat() + "Z"
    
    json.dump(run_card, open(run_card_path, "w"), indent=2)
    send({"status": "ok", "run_id": run_id})


def handle_list_runs(params):
    """List runs with optional filters"""
    filters = params.get("filters", {})
    
    runs = []
    if RUNS_DIR.exists():
        for run_dir in RUNS_DIR.iterdir():
            if run_dir.is_dir():
                run_card_path = run_dir / "run_card.json"
                if run_card_path.exists():
                    run_card = json.load(open(run_card_path))
                    
                    # Apply filters
                    match = True
                    for key, value in filters.items():
                        if run_card.get(key) != value:
                            match = False
                            break
                    
                    if match:
                        runs.append({
                            "run_id": run_card.get("run_id"),
                            "status": run_card.get("status"),
                            "model_version": run_card.get("model_version"),
                            "timestamp_utc": run_card.get("timestamp_utc")
                        })
    
    # Sort by timestamp descending
    runs.sort(key=lambda x: x.get("timestamp_utc", ""), reverse=True)
    send({"runs": runs})


def handle_get_run(params):
    """Get full metadata for a run"""
    run_id = params["run_id"]
    run_path = RUNS_DIR / run_id
    run_card_path = run_path / "run_card.json"
    
    if not run_card_path.exists():
        send({"error": f"Run {run_id} not found"})
        return
    
    run_card = json.load(open(run_card_path))
    send(run_card)


def main():
    for line in sys.stdin:
        req = json.loads(line)
        method = req.get("name")
        params = req.get("arguments", {})

        try:
            if method == "pfr.create_run":
                handle_create_run(params)
            elif method == "pfr.snapshot_inputs":
                handle_snapshot_inputs(params)
            elif method == "pfr.validate_outputs":
                handle_validate_outputs(params)
            elif method == "pfr.register_run":
                handle_register_run(params)
            elif method == "pfr.list_runs":
                handle_list_runs(params)
            elif method == "pfr.get_run":
                handle_get_run(params)
            else:
                send({"error": f"Method {method} not implemented yet"})
        except Exception as e:
            send({"error": str(e)})

if __name__ == "__main__":
    main()
