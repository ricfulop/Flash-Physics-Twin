#!/usr/bin/env python3
"""
comsol_mcp_server.py
MCP server for driving COMSOL runs for PFR.

Implements the MCP JSON-RPC 2.0 protocol over stdio.
"""

import json
import sys
import traceback
from pathlib import Path

# Fix imports to work when run from different cwd
_THIS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_THIS_DIR))

from comsol_api import ComsolBackend
from io_utils import ensure_run_dirs, load_yaml_params

# Use absolute path for runs directory
ROOT = _THIS_DIR.parent.parent
RUNS_DIR = ROOT / "results" / "runs"

_backend = ComsolBackend()

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


def run_path(run_id: str) -> Path:
    return RUNS_DIR / run_id


# Tool handlers

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


TOOLS = {
    "comsol.open_or_create_model": {
        "handler": handle_open_or_create_model,
        "description": "Open an existing COMSOL model or create from template",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Run identifier"},
                "template_path": {"type": "string", "description": "Path to .mph template"},
                "model_path": {"type": "string", "description": "Path to existing .mph model"}
            },
            "required": ["run_id"]
        }
    },
    "comsol.apply_inputs": {
        "handler": handle_apply_inputs,
        "description": "Apply input parameters from YAML files to the COMSOL model",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "inputs_dir": {"type": "string"},
                "strict": {"type": "boolean", "default": True}
            },
            "required": ["run_id"]
        }
    },
    "comsol.build_geometry": {
        "handler": handle_build_geometry,
        "description": "Build the model geometry",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"]
        }
    },
    "comsol.mesh": {
        "handler": handle_mesh,
        "description": "Generate mesh for the model",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "mesh_id": {"type": "string"}
            },
            "required": ["run_id"]
        }
    },
    "comsol.run_pipeline": {
        "handler": handle_run_pipeline,
        "description": "Run the full A→B→C→D simulation pipeline",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "pipeline": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["run_id"]
        }
    },
    "comsol.run_study": {
        "handler": handle_run_study,
        "description": "Run a specific study",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "study_id": {"type": "string"}
            },
            "required": ["run_id", "study_id"]
        }
    },
    "comsol.export_fields": {
        "handler": handle_export_fields,
        "description": "Export field data to HDF5 format",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "format": {"type": "string", "default": "h5"},
                "fields": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["run_id"]
        }
    },
    "comsol.export_kpis": {
        "handler": handle_export_kpis,
        "description": "Export KPIs to JSON",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"]
        }
    },
    "comsol.render_png": {
        "handler": handle_render_png,
        "description": "Render a plot to PNG",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "plot_id": {"type": "string"}
            },
            "required": ["run_id", "plot_id"]
        }
    },
    "comsol.close_model": {
        "handler": handle_close_model,
        "description": "Close the model and release resources",
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
            "name": "comsol_mcp",
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
