#!/usr/bin/env python3
"""
run_pfr_with_em.py
Workflow script for running coupled EM + Thermal + Flash PFR simulations.

This workflow:
1. Creates a new PFR run
2. Snapshots inputs from params/
3. Generates EM fields (Q_RF) using surrogate or COMSOL
4. Runs reactor solve (heat/flow/species/bias/Flash) with Q_RF coupling
5. Exports & validates & registers

Usage:
    python workflows/run_pfr_with_em.py [options]
    
Examples:
    # Basic run with defaults
    python workflows/run_pfr_with_em.py

    # Sweep sigma_eff
    python workflows/run_pfr_with_em.py --sigma_eff 0.1 --bias_voltage 500

    # Science mode with higher Q_RF
    python workflows/run_pfr_with_em.py --sigma_eff 1.0 --synthetic_mode science
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


def to_float(val, default):
    """Safely convert value to float."""
    if val is None:
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def create_run(
    runs_dir: Path,
    model_version: str = "v1.0-coupled-em",
    physics_engine: str = "COMSOL-coupled",
    notes: str = "",
) -> Path:
    """Create a new run directory with run_card.json."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = f"pfr_em_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}"
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
            "em": "surrogate",
            "reactor": "COMSOL-coupled",
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
    - /em/Q_RF, /em/B_mag, /em/E_mag, /em/E_bias
    - /thermal/T_gas, /thermal/T_wall
    - /flash/DeltaB, /flash/chi
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
                # Check required groups/datasets
                required_datasets = [
                    "grid/r", "grid/z",
                    "em/Q_RF", "em/B_mag", "em/E_mag", "em/E_bias",
                    "thermal/T_gas", "thermal/T_wall",
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
                
                # Check em_mode attribute
                if "em_mode" not in f.attrs:
                    errors.append("Missing em_mode attribute in fields.h5")
                
        except Exception as e:
            errors.append(f"Error reading fields.h5: {e}")
    
    # Check kpis.json exists and has em.mode
    if not kpis_file.exists():
        errors.append("Missing outputs/kpis.json")
    else:
        try:
            with open(kpis_file) as f:
                kpis = json.load(f)
            if "em" not in kpis or "mode" not in kpis.get("em", {}):
                errors.append("Missing em.mode in kpis.json")
        except Exception as e:
            errors.append(f"Error reading kpis.json: {e}")
    
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


def run_pfr_with_em_workflow(
    em_mode: str = "surrogate",
    bias_voltage: float = 200.0,
    I_coil: float = 10.0,
    sigma_eff: float = 0.01,
    synthetic_mode: str = "pipeline",
    flash_enabled: bool = True,
    notes: str = "",
) -> Path:
    """
    Execute the full coupled PFR + EM workflow.
    
    Pipeline steps:
    A. EM generation (compute Q_RF)
    B. Thermal solve (T_gas from Q_RF heating)
    C. Flow/species (placeholder)
    D. Flash physics (DeltaB, chi)
    
    Args:
        em_mode: "surrogate" (internal model) or "comsol" (load from AC/DC)
        bias_voltage: DC bias voltage (V)
        I_coil: Coil current amplitude (A)
        sigma_eff: Effective plasma conductivity (S/m)
        synthetic_mode: "pipeline" or "science" for Flash parameters
        flash_enabled: Whether Flash mechanism is active
        notes: Run notes
        
    Returns:
        Path to the completed run directory
    """
    print("=" * 60)
    print("COUPLED PFR + EM WORKFLOW")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  em_mode:        {em_mode}")
    print(f"  bias_voltage:   {bias_voltage} V")
    print(f"  I_coil:         {I_coil} A")
    print(f"  sigma_eff:      {sigma_eff} S/m")
    print(f"  synthetic_mode: {synthetic_mode}")
    print(f"  flash_enabled:  {flash_enabled}")
    print()
    
    # Setup paths
    params_dir = PROJECT_ROOT / "params"
    runs_dir = PROJECT_ROOT / "results" / "runs"
    template_path = PROJECT_ROOT / "templates" / "pfr_v1_template.mph"
    
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create run
    if not notes:
        notes = f"Coupled PFR+EM: V_bias={bias_voltage}V, σ_eff={sigma_eff}S/m"
    
    run_path = create_run(runs_dir, notes=notes)
    
    try:
        # 2. Snapshot inputs
        snapshot_inputs(run_path, params_dir)
        
        # Update inputs with runtime parameters
        inputs_dir = run_path / "inputs"
        
        # Update ops.yaml
        ops_file = inputs_dir / "ops.yaml"
        if ops_file.exists():
            with open(ops_file) as f:
                ops = yaml.safe_load(f)
            
            ops["bias"]["voltage"] = float(bias_voltage)
            
            if "em" not in ops:
                ops["em"] = {}
            ops["em"]["mode"] = em_mode
            
            if "surrogate" not in ops["em"]:
                ops["em"]["surrogate"] = {}
            ops["em"]["surrogate"]["I_coil"] = float(I_coil)
            ops["em"]["surrogate"]["sigma_eff"] = float(sigma_eff)
            
            with open(ops_file, "w") as f:
                yaml.dump(ops, f, default_flow_style=False)
        
        # Load geometry parameters
        geometry_file = inputs_dir / "geometry.yaml"
        em_params = {
            "I_coil": I_coil,
            "sigma_eff": sigma_eff,
        }
        
        if geometry_file.exists():
            with open(geometry_file) as f:
                geom = yaml.safe_load(f)
            
            if "reactor" in geom:
                em_params["R_tube_inner"] = to_float(geom["reactor"].get("radius"), 0.0508)
                em_params["L_reactor"] = to_float(geom["reactor"].get("length"), 0.2)
            
            if "quartz_tube" in geom:
                em_params["R_tube_outer"] = to_float(geom["quartz_tube"].get("outer_radius"), 0.0558)
            
            if "electrodes" in geom and "rf_coil" in geom["electrodes"]:
                rf_coil = geom["electrodes"]["rf_coil"]
                em_params["z_coil_start"] = to_float(rf_coil.get("z_start"), 0.05)
                em_params["z_coil_end"] = to_float(rf_coil.get("z_end"), 0.15)
                em_params["n_turns"] = to_float(rf_coil.get("n_turns"), 5)
                em_params["coil_radius"] = to_float(rf_coil.get("coil_radius"), 0.06)
            
            if "acdc_coil" in geom:
                em_params["f_RF"] = to_float(geom["acdc_coil"].get("f_RF"), 13.56e6)
        
        print(f"✓ Updated inputs with runtime parameters")
        
        # 3. Initialize COMSOL backend
        print("\nInitializing backend...")
        backend = ComsolBackend()
        
        try:
            backend.open_or_create(run_path, template_path=str(template_path))
            print(f"✓ Opened COMSOL model from template")
        except Exception as e:
            print(f"⚠ COMSOL not available, using synthetic field generation")
            backend._models[run_path.name] = type("MockModel", (), {"java": None, "name": lambda: "mock"})()
        
        # Update run_card with em_mode
        update_run_card(run_path, {"toolchain": {"em": em_mode, "reactor": "COMSOL-coupled", "ml": "none"}})
        
        # 4. Run coupled simulation pipeline
        print("\n--- Pipeline A: EM Generation ---")
        print(f"Computing Q_RF using {em_mode} model...")
        
        print("\n--- Pipeline B: Thermal Solve ---")
        print("Computing T_gas with Q_RF heating...")
        
        print("\n--- Pipeline C: Flow/Species ---")
        print("Computing flow and species fields...")
        
        print("\n--- Pipeline D: Flash Physics ---")
        print("Computing DeltaB and chi with bias and thermal effects...")
        
        # Execute coupled export
        result = backend.export_coupled_fields(
            run_path,
            fmt="h5",
            em_mode=em_mode,
            bias_voltage=bias_voltage,
            flash_enabled=flash_enabled,
            flash_params=None,
            em_params=em_params,
            thermal_params=None,
            synthetic_mode=synthetic_mode,
        )
        print(f"\n✓ Exported coupled fields to {result['fields']}")
        
        # 5. Export KPIs (with thermal params for energy balance)
        print("\nExporting KPIs...")
        kpis_path = backend.export_coupled_kpis(run_path, em_mode=em_mode, thermal_params=None)
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
            
            print("\nResults Summary:")
            em = kpis.get("em", {})
            power = kpis.get("power", {})
            thermal = kpis.get("thermal", {})
            flash = kpis.get("flash", {})
            
            print(f"  EM Mode: {em.get('mode', 'N/A')}")
            print(f"  Q_RF total: {power.get('Q_RF_total', 0):.2f} W")
            
            print(f"  T_gas: [{thermal.get('T_min', 0):.1f}, {thermal.get('T_max', 0):.1f}] K (mean: {thermal.get('T_mean', 0):.1f} K)")
            
            if thermal.get('unphysical_temperature_flag', False):
                print(f"  ⚠ WARNING: Unphysical temperature detected (T_max > 1500 K)")
            
            # Energy balance
            conv = power.get('convective_removal_estimate')
            wall = power.get('wall_loss_estimate')
            if conv is not None or wall is not None:
                print(f"  Energy balance:")
                print(f"    - Q_RF input:     {power.get('Q_RF_total', 0):.2f} W")
                if conv is not None:
                    print(f"    - Convective out: {conv:.2f} W")
                if wall is not None:
                    print(f"    - Wall loss:      {wall:.2f} W")
            
            print(f"  chi_avg: {flash.get('chi_volume_avg', 0):.4f}")
            print(f"  chi>0.5 fraction: {flash.get('chi_volume_fraction_gt_0p5', 0):.2%}")
        
        return run_path
        
    except Exception as e:
        update_run_card(run_path, {"status": "failed", "error": str(e)})
        print(f"\n✗ Workflow failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run coupled PFR + EM simulation workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--em_mode", type=str, default="surrogate",
        choices=["surrogate", "comsol"],
        help="EM field source: surrogate (internal model) or comsol (load from AC/DC)"
    )
    parser.add_argument(
        "--bias_voltage", type=float, default=200.0,
        help="DC bias voltage (V)"
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
        "--synthetic_mode", type=str, default="pipeline",
        choices=["pipeline", "science"],
        help="Flash parameter mode: pipeline (conservative) or science (onset-visible)"
    )
    parser.add_argument(
        "--flash_enabled", type=bool, default=True,
        help="Enable Flash mechanism"
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Run notes"
    )
    
    args = parser.parse_args()
    
    try:
        run_path = run_pfr_with_em_workflow(
            em_mode=args.em_mode,
            bias_voltage=args.bias_voltage,
            I_coil=args.I_coil,
            sigma_eff=args.sigma_eff,
            synthetic_mode=args.synthetic_mode,
            flash_enabled=args.flash_enabled,
            notes=args.notes,
        )
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
