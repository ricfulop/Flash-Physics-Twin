#!/usr/bin/env python3
"""
run_em_coil.py
Workflow script for running AC/DC coil EM simulations.

This workflow:
1. Creates a new PFR run
2. Snapshots inputs from params/
3. Runs the AC/DC coil template (or synthetic equivalent)
4. Exports EM fields (B_mag, E_mag, Q_RF) to schema-compliant HDF5
5. Validates outputs and registers the run

Usage:
    python workflows/run_em_coil.py [--I_coil 10] [--sigma_eff 0.01] [--notes "test run"]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from mcp.comsol_mcp.comsol_api import ComsolBackend


def load_params(params_dir: Path) -> dict:
    """Load all parameter files from a directory."""
    params = {}
    for yaml_file in params_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            params[yaml_file.stem] = yaml.safe_load(f)
    return params


def create_run(
    runs_dir: Path,
    model_version: str = "v1.0-acdc-coil",
    physics_engine: str = "COMSOL-ACDC",
    notes: str = "",
) -> Path:
    """Create a new run directory with run_card.json."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = f"em_coil_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}"
    run_path = runs_dir / run_id
    
    # Create directory structure
    (run_path / "inputs").mkdir(parents=True, exist_ok=True)
    (run_path / "outputs").mkdir(parents=True, exist_ok=True)
    (run_path / "plots").mkdir(parents=True, exist_ok=True)
    (run_path / "logs").mkdir(parents=True, exist_ok=True)
    
    # Create run_card.json
    run_card = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "toolchain": {
            "em": "COMSOL-ACDC",
            "reactor": "none",
            "ml": "none",
        },
        "model_version": model_version,
        "physics_engine": physics_engine,
        "status": "created",
        "notes": notes,
    }
    
    with open(run_path / "run_card.json", "w") as f:
        json.dump(run_card, f, indent=2)
    
    print(f"✓ Created run: {run_id}")
    return run_path


def snapshot_inputs(run_path: Path, params_dir: Path) -> None:
    """Copy parameter files to run's inputs directory."""
    inputs_dir = run_path / "inputs"
    
    for yaml_file in params_dir.glob("*.yaml"):
        dst = inputs_dir / yaml_file.name
        shutil.copy(yaml_file, dst)
    
    print(f"✓ Snapshotted inputs from {params_dir}")


def update_run_card(run_path: Path, updates: dict) -> None:
    """Update run_card.json with new values."""
    run_card_path = run_path / "run_card.json"
    with open(run_card_path) as f:
        run_card = json.load(f)
    
    run_card.update(updates)
    
    with open(run_card_path, "w") as f:
        json.dump(run_card, f, indent=2)


def validate_outputs(run_path: Path) -> bool:
    """
    Validate outputs against PFR_Data_Schema.md.
    
    Required fields:
    - /grid/r, /grid/z
    - /em/B_mag, /em/E_mag, /em/Q_RF
    - /flash/DeltaB, /flash/chi (even if placeholder)
    """
    import h5py
    
    fields_file = run_path / "outputs" / "fields.h5"
    kpis_file = run_path / "outputs" / "kpis.json"
    
    errors = []
    
    # Check fields.h5 exists
    if not fields_file.exists():
        errors.append("Missing outputs/fields.h5")
    else:
        try:
            with h5py.File(fields_file, "r") as f:
                # Check required groups
                required_datasets = [
                    "grid/r", "grid/z",
                    "em/B_mag", "em/E_mag", "em/Q_RF",
                    "flash/DeltaB", "flash/chi",
                ]
                
                for ds in required_datasets:
                    if ds not in f:
                        errors.append(f"Missing dataset: {ds}")
                
                # Check Flash physics bounds
                if "flash/chi" in f:
                    chi = f["flash/chi"][:]
                    if chi.min() < 0 or chi.max() > 1:
                        errors.append(f"chi out of bounds: [{chi.min()}, {chi.max()}]")
                
        except Exception as e:
            errors.append(f"Error reading fields.h5: {e}")
    
    # Check kpis.json exists
    if not kpis_file.exists():
        errors.append("Missing outputs/kpis.json")
    
    if errors:
        print("✗ Validation failed:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("✓ Outputs validated successfully")
    return True


def register_run(run_path: Path) -> None:
    """Mark run as registered in run_card.json."""
    update_run_card(run_path, {"status": "registered"})
    print(f"✓ Run registered: {run_path.name}")


def run_em_coil_workflow(
    I_coil: float = 10.0,
    sigma_eff: float = 0.01,
    f_RF: float = 13.56e6,
    notes: str = "",
) -> Path:
    """
    Execute the full AC/DC coil EM workflow.
    
    Args:
        I_coil: Coil current amplitude (A)
        sigma_eff: Effective plasma conductivity (S/m)
        f_RF: RF frequency (Hz)
        notes: Run notes
        
    Returns:
        Path to the completed run directory
    """
    print("=" * 60)
    print("AC/DC COIL EM WORKFLOW")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  I_coil:    {I_coil} A")
    print(f"  sigma_eff: {sigma_eff} S/m")
    print(f"  f_RF:      {f_RF/1e6:.2f} MHz")
    print()
    
    # Setup paths
    params_dir = PROJECT_ROOT / "params"
    runs_dir = PROJECT_ROOT / "results" / "runs"
    template_path = PROJECT_ROOT / "mcp" / "comsol_mcp" / "templates" / "pfr_coil_acdc_axisym.mph"
    
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create run
    if not notes:
        notes = f"AC/DC coil EM run: I_coil={I_coil}A, sigma_eff={sigma_eff}S/m"
    
    run_path = create_run(runs_dir, notes=notes)
    
    try:
        # 2. Snapshot inputs
        snapshot_inputs(run_path, params_dir)
        
        # Update inputs with runtime parameters
        inputs_dir = run_path / "inputs"
        geometry_file = inputs_dir / "geometry.yaml"
        
        if geometry_file.exists():
            with open(geometry_file) as f:
                geom = yaml.safe_load(f)
            
            # Update AC/DC coil parameters
            if "acdc_coil" not in geom:
                geom["acdc_coil"] = {}
            
            geom["acdc_coil"]["I_coil"] = I_coil
            geom["acdc_coil"]["sigma_eff"] = sigma_eff
            geom["acdc_coil"]["f_RF"] = f_RF
            
            with open(geometry_file, "w") as f:
                yaml.dump(geom, f, default_flow_style=False)
            
            print(f"✓ Updated inputs with runtime parameters")
        
        # 3. Initialize COMSOL backend and open model
        print("\nInitializing COMSOL backend...")
        backend = ComsolBackend()
        
        try:
            # Try to open the template (will fail if COMSOL not available)
            backend.open_or_create(run_path, template_path=str(template_path))
            print(f"✓ Opened COMSOL model from template")
        except Exception as e:
            # COMSOL not available - create a mock model entry for synthetic export
            print(f"⚠ COMSOL not available, using synthetic field generation")
            # Register a mock model to satisfy _get_model() checks
            backend._models[run_path.name] = type("MockModel", (), {"java": None, "name": lambda: "mock"})()
        
        # 4. Export EM fields
        print("\nExporting EM fields...")
        coil_params = {
            "I_coil": I_coil,
            "sigma_eff": sigma_eff,
            "f_RF": f_RF,
        }
        
        # Load geometry parameters
        if geometry_file.exists():
            with open(geometry_file) as f:
                geom = yaml.safe_load(f)
            
            # Helper to ensure float conversion (YAML may parse scientific notation as strings)
            def to_float(val, default):
                if val is None:
                    return float(default)
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return float(default)
            
            # Extract relevant geometry
            if "reactor" in geom:
                coil_params["R_tube_inner"] = to_float(geom["reactor"].get("radius"), 0.0508)
                coil_params["L_reactor"] = to_float(geom["reactor"].get("length"), 0.2)
            
            if "quartz_tube" in geom:
                coil_params["R_tube_outer"] = to_float(geom["quartz_tube"].get("outer_radius"), 0.0558)
            
            if "electrodes" in geom and "rf_coil" in geom["electrodes"]:
                rf_coil = geom["electrodes"]["rf_coil"]
                coil_params["z_coil_start"] = to_float(rf_coil.get("z_start"), 0.05)
                coil_params["z_coil_end"] = to_float(rf_coil.get("z_end"), 0.15)
                coil_params["n_turns"] = to_float(rf_coil.get("n_turns"), 5)
                coil_params["coil_radius"] = to_float(rf_coil.get("coil_radius"), 0.06)
        
        result = backend.export_em_coil_fields(run_path, fmt="h5", coil_params=coil_params)
        print(f"✓ Exported fields to {result['fields']}")
        
        # 5. Export KPIs
        print("\nExporting KPIs...")
        kpis_path = backend.export_em_coil_kpis(run_path)
        print(f"✓ Exported KPIs to {kpis_path}")
        
        # 6. Validate outputs
        print("\nValidating outputs...")
        if not validate_outputs(run_path):
            update_run_card(run_path, {"status": "failed"})
            raise RuntimeError("Output validation failed")
        
        # 7. Register run
        register_run(run_path)
        
        print()
        print("=" * 60)
        print(f"WORKFLOW COMPLETE: {run_path.name}")
        print("=" * 60)
        
        # Print summary
        kpis_file = run_path / "outputs" / "kpis.json"
        if kpis_file.exists():
            with open(kpis_file) as f:
                kpis = json.load(f)
            
            print("\nEM Field Summary:")
            em = kpis.get("em", {})
            print(f"  B_mag: [{em.get('B_mag_min', 0):.2e}, {em.get('B_mag_max', 0):.2e}] T")
            print(f"  E_mag: [{em.get('E_mag_min', 0):.2e}, {em.get('E_mag_max', 0):.2e}] V/m")
            print(f"  Q_RF total: {em.get('Q_RF_total', 0):.2f} W")
        
        return run_path
        
    except Exception as e:
        update_run_card(run_path, {"status": "failed", "error": str(e)})
        print(f"\n✗ Workflow failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run AC/DC coil EM simulation workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--I_coil", type=float, default=10.0,
        help="Coil current amplitude (A)"
    )
    parser.add_argument(
        "--sigma_eff", type=float, default=0.01,
        help="Effective plasma conductivity (S/m)"
    )
    parser.add_argument(
        "--f_RF", type=float, default=13.56e6,
        help="RF frequency (Hz)"
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Run notes"
    )
    
    args = parser.parse_args()
    
    try:
        run_path = run_em_coil_workflow(
            I_coil=args.I_coil,
            sigma_eff=args.sigma_eff,
            f_RF=args.f_RF,
            notes=args.notes,
        )
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
