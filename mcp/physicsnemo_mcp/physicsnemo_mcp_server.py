#!/usr/bin/env python3
"""
physicsnemo_mcp_server.py
MCP server for PhysicsNeMo integration - parameter inversion and surrogate training.

Provides tools for:
- Listing registered PFR runs
- Building datasets from run outputs
- Training inverse models to infer Flash physics parameters
- Generating inversion reports

Usage:
    python physicsnemo_mcp_server.py
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("physicsnemo_mcp")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.physicsnemo_mcp.trainer_inverse import InverseTrainer, chi_model

# Global trainer instance
_trainer: Optional[InverseTrainer] = None

def get_trainer() -> InverseTrainer:
    """Get or create the global trainer instance."""
    global _trainer
    if _trainer is None:
        _trainer = InverseTrainer(results_dir=PROJECT_ROOT / "results" / "inversions")
    return _trainer


# ============================================================================
# Tool Handlers
# ============================================================================

def handle_list_registered_runs(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all registered PFR runs with optional filters.
    
    Args:
        params: {filters: {status, synthetic_mode, notes_contains, min_bias_V, max_bias_V}}
    
    Returns:
        {runs: [{run_id, status, notes, bias_V, synthetic_mode, ...}, ...]}
    """
    filters = params.get("filters", {})
    runs_dir = PROJECT_ROOT / "results" / "runs"
    
    if not runs_dir.exists():
        return {"runs": [], "error": "Runs directory not found"}
    
    runs = []
    
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        
        run_card_file = run_path / "run_card.json"
        if not run_card_file.exists():
            continue
        
        try:
            with open(run_card_file) as f:
                run_card = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load run card for {run_path.name}: {e}")
            continue
        
        # Apply filters
        if filters:
            # Status filter
            if "status" in filters:
                if run_card.get("status") != filters["status"]:
                    continue
            
            # Synthetic mode filter
            if "synthetic_mode" in filters:
                if run_card.get("synthetic_mode") != filters["synthetic_mode"]:
                    continue
            
            # Notes contains filter
            if "notes_contains" in filters:
                notes = run_card.get("notes", "")
                if filters["notes_contains"].lower() not in notes.lower():
                    continue
            
            # Bias voltage filters (need to extract from ops.yaml)
            if "min_bias_V" in filters or "max_bias_V" in filters:
                ops_file = run_path / "inputs" / "ops.yaml"
                if ops_file.exists():
                    try:
                        import yaml
                        with open(ops_file) as f:
                            ops = yaml.safe_load(f)
                        bias_v = abs(ops.get("bias", {}).get("voltage", 0))
                        
                        if "min_bias_V" in filters and bias_v < filters["min_bias_V"]:
                            continue
                        if "max_bias_V" in filters and bias_v > filters["max_bias_V"]:
                            continue
                        
                        run_card["bias_V"] = bias_v
                    except Exception:
                        pass
        
        # Add to results
        runs.append({
            "run_id": run_path.name,
            "status": run_card.get("status"),
            "notes": run_card.get("notes"),
            "timestamp_utc": run_card.get("timestamp_utc"),
            "model_version": run_card.get("model_version"),
            "synthetic_mode": run_card.get("synthetic_mode"),
            "bias_V": run_card.get("bias_V"),
        })
    
    # Sort by timestamp
    runs.sort(key=lambda x: x.get("timestamp_utc", ""), reverse=True)
    
    return {"runs": runs, "count": len(runs)}


def handle_build_dataset(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a dataset from registered runs.
    
    Args:
        params: {run_ids, targets, features, aggregation}
    
    Returns:
        {dataset: {features: [...], targets: [...], n_points, run_ids}}
    """
    run_ids = params.get("run_ids", [])
    targets = params.get("targets", ["flash/chi"])
    features = params.get("features", ["em/E_bias"])
    aggregation = params.get("aggregation", "mean")
    
    if not run_ids:
        return {"error": "No run_ids provided"}
    
    trainer = get_trainer()
    
    # For now, support single target/feature
    target = targets[0] if targets else "flash/chi"
    feature = features[0] if features else "em/E_bias"
    
    try:
        feature_vals, target_vals = trainer.load_dataset(
            run_ids,
            runs_dir=PROJECT_ROOT / "results" / "runs",
            target=target,
            feature=feature,
            aggregation=aggregation
        )
        
        return {
            "dataset": {
                "features": feature_vals.tolist(),
                "targets": target_vals.tolist(),
                "feature_name": feature,
                "target_name": target,
                "n_points": len(feature_vals),
                "run_ids": run_ids,
                "aggregation": aggregation,
            }
        }
    except Exception as e:
        return {"error": str(e)}


def handle_train_inverse(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train an inverse model to infer Flash physics parameters.
    
    Args:
        params: {problem_id, run_ids, infer, known, output_dir, n_bootstrap}
    
    Returns:
        {result: {fitted_params, uncertainties, mse, r_squared, ...}}
    """
    problem_id = params.get("problem_id")
    run_ids = params.get("run_ids", [])
    infer = params.get("infer", [])
    known = params.get("known", {})
    output_dir = params.get("output_dir")
    n_bootstrap = params.get("n_bootstrap", 100)
    
    if not problem_id:
        return {"error": "problem_id is required"}
    if not run_ids:
        return {"error": "run_ids is required"}
    if not infer:
        return {"error": "infer list is required (e.g., ['r_act', 'B_s'])"}
    
    trainer = get_trainer()
    
    try:
        result = trainer.train(
            problem_id=problem_id,
            run_ids=run_ids,
            infer=infer,
            known=known,
            output_dir=output_dir,
            n_bootstrap=n_bootstrap
        )
        
        return {
            "result": {
                "problem_id": result.problem_id,
                "status": result.status,
                "fitted_params": result.fitted_params,
                "uncertainties": result.param_uncertainties,
                "mse": result.mse,
                "rmse": result.rmse,
                "r_squared": result.r_squared,
                "n_points": result.n_points,
                "output_dir": result.output_dir,
            }
        }
    except Exception as e:
        logger.exception(f"Inversion failed: {e}")
        return {"error": str(e)}


def handle_report(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a report for a completed inversion.
    
    Args:
        params: {problem_id, include_plots}
    
    Returns:
        {report: {fitted_params, uncertainties, fit_quality, plots, ...}}
    """
    problem_id = params.get("problem_id")
    include_plots = params.get("include_plots", True)
    
    if not problem_id:
        return {"error": "problem_id is required"}
    
    trainer = get_trainer()
    
    try:
        report = trainer.generate_report(problem_id, include_plots=include_plots)
        return {"report": report}
    except Exception as e:
        logger.exception(f"Report generation failed: {e}")
        return {"error": str(e)}


# ============================================================================
# MCP Server Main
# ============================================================================

def dispatch(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch tool calls to handlers."""
    handlers = {
        "physicsnemo.list_registered_runs": handle_list_registered_runs,
        "physicsnemo.build_dataset": handle_build_dataset,
        "physicsnemo.train_inverse": handle_train_inverse,
        "physicsnemo.report": handle_report,
    }
    
    handler = handlers.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}
    
    return handler(params)


def main():
    """MCP server main loop using stdio transport."""
    logger.info("PhysicsNeMo MCP server starting...")
    
    # Load tool definitions
    tools_file = Path(__file__).parent / "physicsnemo_mcp_tools.json"
    with open(tools_file) as f:
        tools_def = json.load(f)
    
    logger.info(f"Loaded {len(tools_def['tools'])} tools")
    
    # Simple stdio protocol
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = {"error": f"Invalid JSON: {e}"}
            print(json.dumps(response), flush=True)
            continue
        
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        if method == "tools/list":
            # Return tool definitions
            response = {
                "id": request_id,
                "result": {"tools": tools_def["tools"]}
            }
        elif method == "tools/call":
            # Execute tool
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            
            result = dispatch(tool_name, tool_args)
            
            response = {
                "id": request_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            }
        else:
            response = {
                "id": request_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"}
            }
        
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
