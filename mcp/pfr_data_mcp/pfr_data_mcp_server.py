#!/usr/bin/env python3
"""
pfr_data_mcp_server.py
MCP server enforcing PFR_Data_Schema.md

Implements the MCP JSON-RPC 2.0 protocol over stdio.
"""

import json
import sys
import shutil
import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Use absolute path for runs directory
_THIS_DIR = Path(__file__).parent.resolve()
ROOT = _THIS_DIR.parent.parent
RUNS_DIR = ROOT / "results" / "runs"


# MCP Protocol Implementation

def send_response(id, result=None, error=None):
    """Send a JSON-RPC 2.0 response."""
    response = {"jsonrpc": "2.0", "id": id}
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result if result is not None else {}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def send_error(id, code, message, data=None):
    """Send a JSON-RPC 2.0 error response."""
    error = {"code": code, "message": message}
    if data:
        error["data"] = data
    send_response(id, error=error)


# Utility functions

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Tool handlers

def handle_create_run(params):
    run_id = params.get("run_id")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_path = RUNS_DIR / run_id
    run_path.mkdir(parents=True, exist_ok=False)

    run_card = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "created",
        "model_version": params["model_version"],
        "physics_engine": params["physics_engine"],
        "discovery_engine": params["discovery_engine"],
        "notes": params.get("notes", "")
    }

    with open(run_path / "run_card.json", "w") as f:
        json.dump(run_card, f, indent=2)

    return {"run_id": run_id, "path": str(run_path)}


def handle_snapshot_inputs(params):
    run_id = params["run_id"]
    params_dir = Path(params["params_dir"])
    run_path = RUNS_DIR / run_id
    inputs_dir = run_path / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    hashes = {}
    for fname in ["geometry.yaml", "ops.yaml", "materials.yaml", "chemistry.yaml"]:
        src = params_dir / fname
        dst = inputs_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            hashes[fname] = sha256_file(dst)

    run_card_path = run_path / "run_card.json"
    run_card = json.load(open(run_card_path))
    run_card["input_hashes"] = hashes
    run_card["status"] = "inputs_snapshotted"

    json.dump(run_card, open(run_card_path, "w"), indent=2)
    return {"status": "ok", "hashes": hashes}


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
            import numpy as np
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
                    
                # Check for NaNs/infs in all datasets
                def check_group(group, prefix=""):
                    for key in group.keys():
                        item = group[key]
                        path = f"{prefix}/{key}" if prefix else key
                        if hasattr(item, 'shape'):  # It's a dataset
                            data = item[:]
                            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                                issues.append(f"NaN or inf values in {path}")
                        elif hasattr(item, 'keys'):  # It's a group
                            check_group(item, path)
                
                check_group(f)
                
        except ImportError:
            issues.append("h5py not available for validation")
        except Exception as e:
            issues.append(f"Error reading fields.h5: {str(e)}")
    
    # Validate kpis.json structure
    if kpis_file.exists():
        try:
            kpis = json.load(open(kpis_file))
            if kpis.get("status") == "TODO":
                issues.append("kpis.json is a placeholder")
            # Check for required KPI sections
            required_sections = ["flash", "thermal", "power"]
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
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }
    run_card["status"] = "validated" if valid else "invalid"
    
    json.dump(run_card, open(run_card_path, "w"), indent=2)
    return {"valid": valid, "issues": issues}


def handle_register_run(params):
    """Register a validated run as immutable and queryable"""
    run_id = params["run_id"]
    run_path = RUNS_DIR / run_id
    run_card_path = run_path / "run_card.json"
    
    run_card = json.load(open(run_card_path))
    
    # Only allow registration if validation passed
    if run_card.get("status") != "validated":
        raise ValueError(f"Cannot register run with status '{run_card.get('status')}'. Must be 'validated'.")
    
    if not run_card.get("validation", {}).get("valid", False):
        raise ValueError("Cannot register invalid run. Fix validation issues first.")
    
    # Mark as registered
    run_card["status"] = "registered"
    run_card["registered_utc"] = datetime.now(timezone.utc).isoformat()
    
    json.dump(run_card, open(run_card_path, "w"), indent=2)
    return {"status": "ok", "run_id": run_id}


def handle_list_runs(params):
    """List runs with optional filters"""
    filters = params.get("filters", {})
    
    runs = []
    if RUNS_DIR.exists():
        for run_dir in RUNS_DIR.iterdir():
            if run_dir.is_dir():
                run_card_path = run_dir / "run_card.json"
                if run_card_path.exists():
                    try:
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
                    except:
                        pass
    
    # Sort by timestamp descending
    runs.sort(key=lambda x: x.get("timestamp_utc", ""), reverse=True)
    return {"runs": runs}


def handle_get_run(params):
    """Get full metadata for a run"""
    run_id = params["run_id"]
    run_path = RUNS_DIR / run_id
    run_card_path = run_path / "run_card.json"
    
    if not run_card_path.exists():
        raise ValueError(f"Run {run_id} not found")
    
    return json.load(open(run_card_path))


# Tool definitions

TOOLS = {
    "pfr.create_run": {
        "handler": handle_create_run,
        "description": "Create a new PFR simulation run",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Optional run ID (auto-generated if not provided)"},
                "model_version": {"type": "string", "description": "Model version string"},
                "physics_engine": {"type": "string", "description": "Physics engine (e.g., COMSOL)"},
                "discovery_engine": {"type": "string", "description": "Discovery engine (e.g., NVIDIA-PhysicsNeMo)"},
                "notes": {"type": "string", "description": "Optional notes"}
            },
            "required": ["model_version", "physics_engine", "discovery_engine"]
        }
    },
    "pfr.snapshot_inputs": {
        "handler": handle_snapshot_inputs,
        "description": "Snapshot input parameter files into the run directory",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "params_dir": {"type": "string", "description": "Path to params directory"}
            },
            "required": ["run_id", "params_dir"]
        }
    },
    "pfr.validate_outputs": {
        "handler": handle_validate_outputs,
        "description": "Validate run outputs against PFR_Data_Schema.md",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"]
        }
    },
    "pfr.register_run": {
        "handler": handle_register_run,
        "description": "Register a validated run as immutable",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"]
        }
    },
    "pfr.list_runs": {
        "handler": handle_list_runs,
        "description": "List runs with optional filters",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filters": {"type": "object", "description": "Key-value filters"}
            }
        }
    },
    "pfr.get_run": {
        "handler": handle_get_run,
        "description": "Get full metadata for a run",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"]
        }
    },
}


def handle_initialize(params):
    """Handle MCP initialize request."""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "pfr_data_mcp",
            "version": "0.3.0"
        }
    }


def handle_tools_list():
    """Handle tools/list request."""
    tools = []
    for name, tool in TOOLS.items():
        tools.append({
            "name": name,
            "description": tool["description"],
            "inputSchema": tool["inputSchema"]
        })
    return {"tools": tools}


def handle_tools_call(params):
    """Handle tools/call request."""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    result = TOOLS[tool_name]["handler"](arguments)
    return {"content": [{"type": "text", "text": json.dumps(result)}]}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})
            
            if method == "initialize":
                result = handle_initialize(params)
                send_response(req_id, result)
            elif method == "notifications/initialized":
                # Client notification, no response needed
                pass
            elif method == "tools/list":
                result = handle_tools_list()
                send_response(req_id, result)
            elif method == "tools/call":
                result = handle_tools_call(params)
                send_response(req_id, result)
            else:
                send_error(req_id, -32601, f"Method not found: {method}")
                
        except json.JSONDecodeError as e:
            send_error(None, -32700, f"Parse error: {e}")
        except Exception as e:
            req_id = req.get("id") if 'req' in dir() else None
            send_error(req_id, -32000, str(e), {"traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
