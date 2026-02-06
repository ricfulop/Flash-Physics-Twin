#!/usr/bin/env python3
"""
rf_sweep_powder_region.py

RF-assisted activation sweep focused on the powder fall region.

Computes powder-region-specific KPIs:
- flash.powder_chi_fraction_gt_0p5
- flash.powder_RF_assisted_activation_fraction  
- em.powder_E_RF_max, em.powder_E_eff_over_lambda_fraction

Generates:
1. V/V_onset × gamma_RF matrix for powder region
2. Required bias for target activation curves
3. Coil optimization analysis to increase E_RF in powder region
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp" / "comsol_mcp"))
from comsol_api import ComsolBackend

# Constants
LAMBDA_ONSET = 60443.0  # V/m (from PhysicsNeMo inversion)
RESULTS_DIR = Path(__file__).parent.parent / "results" / "rf_sweep_powder"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default geometry parameters
DEFAULT_GAP = 0.01  # 10mm gap
DEFAULT_T_PREHEAT = 1100.0  # K
DEFAULT_I_COIL = 50.0  # A
DEFAULT_SIGMA_EFF = 0.1  # S/m

# Powder region definition (matching ops.yaml)
POWDER_REGION = {
    'r_min': 0.005,      # 5mm from centerline
    'r_max': 0.035,      # 35mm from centerline
    'z_start': 0.06,     # 60mm from inlet
    'z_end': 0.14,       # 140mm from inlet
}


def run_single_case(
    backend: ComsolBackend,
    run_id: str,
    bias_voltage: float,
    gamma_RF: float,
    gap_distance: float = DEFAULT_GAP,
    T_preheat: float = DEFAULT_T_PREHEAT,
    I_coil: float = DEFAULT_I_COIL,
    coil_params: Optional[Dict] = None,
) -> Dict:
    """Run a single RF-assisted case and return KPIs including powder region."""
    
    run_path = RESULTS_DIR / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Create mock model entry
    models_dir = run_path / "models"
    models_dir.mkdir(exist_ok=True)
    model_file = models_dir / "rf_sweep.mph"
    model_file.write_text("# Placeholder")
    backend._models[run_id] = type('MockModel', (), {'java': None, 'name': lambda: 'rf_sweep'})()
    
    # Configure EM parameters
    em_params = {
        "I_coil": I_coil,
        "sigma_eff": DEFAULT_SIGMA_EFF,
        "n_turns": 8,
        "coil_radius": 0.055,
        "z_coil_start": 0.04,
        "z_coil_end": 0.16,
    }
    
    # Override with custom coil params if provided
    if coil_params:
        em_params.update(coil_params)
    
    thermal_params = {
        "T_wall": T_preheat,
        "T_inlet": T_preheat,
        "T_ref": T_preheat,
        "gamma_RF": gamma_RF,
        "lambda_onset": LAMBDA_ONSET,
    }
    
    flash_params = {
        "gap_distance": gap_distance,
    }
    
    # Run coupled simulation
    backend.export_coupled_fields(
        run_path,
        fmt="h5",
        em_mode="surrogate",
        bias_voltage=bias_voltage,
        flash_enabled=True,
        flash_params=flash_params,
        em_params=em_params,
        thermal_params=thermal_params,
        synthetic_mode="science",
    )
    
    # Export KPIs with powder region
    backend.export_coupled_kpis(
        run_path,
        em_mode="surrogate",
        thermal_params=thermal_params,
        powder_region=POWDER_REGION,
    )
    
    # Load KPIs
    kpis_file = run_path / "outputs" / "kpis.json"
    with open(kpis_file, 'r') as f:
        kpis = json.load(f)
    
    # Extract relevant metrics
    result = {
        "run_id": run_id,
        "bias_voltage": bias_voltage,
        "gamma_RF": gamma_RF,
        "I_coil": I_coil,
        # Whole volume KPIs
        "chi_fraction_gt_0p5": kpis["flash"]["chi_volume_fraction_gt_0p5"],
        "chi_mean": kpis["flash"]["chi_volume_avg"],
        "RF_assisted_activation_fraction": kpis["flash"].get("RF_assisted_activation_fraction", 0),
        "E_eff_over_lambda_fraction": kpis["flash"].get("E_eff_over_lambda_fraction", 0),
        "E_RF_induced_max": kpis["em"].get("E_RF_induced_max", 0),
        "E_eff_max": kpis["em"].get("E_eff_max", 0),
        # Powder region KPIs
        "powder_chi_fraction_gt_0p5": kpis["flash"].get("powder_chi_fraction_gt_0p5", 0),
        "powder_chi_avg": kpis["flash"].get("powder_chi_avg", 0),
        "powder_RF_assisted_activation_fraction": kpis["flash"].get("powder_RF_assisted_activation_fraction", 0),
        "powder_E_eff_over_lambda_fraction": kpis["flash"].get("powder_E_eff_over_lambda_fraction", 0),
        "powder_E_RF_max": kpis["em"].get("powder_E_RF_max", 0),
        "powder_E_RF_mean": kpis["em"].get("powder_E_RF_mean", 0),
        "powder_E_eff_max": kpis["em"].get("powder_E_eff_max", 0),
        "powder_E_eff_mean": kpis["em"].get("powder_E_eff_mean", 0),
        "powder_T_mean": kpis["thermal"].get("powder_T_mean", 0),
        "powder_T_max": kpis["thermal"].get("powder_T_max", 0),
        # Thermal
        "T_max": kpis["thermal"]["T_max"],
        "unphysical": kpis["thermal"].get("unphysical_temperature_flag", False),
    }
    
    return result


def find_onset_voltage(
    backend: ComsolBackend,
    target_chi_mean: float = 0.5,
    V_min: float = 300,
    V_max: float = 800,
    use_powder_region: bool = True,
) -> float:
    """
    Find the bias voltage where chi_mean ≈ target (with gamma_RF = 0).
    If use_powder_region is True, uses powder region chi.
    """
    print(f"\n{'='*60}")
    print(f"Finding V_onset where {'powder ' if use_powder_region else ''}chi_mean ≈ {target_chi_mean} (gamma_RF=0)")
    print(f"{'='*60}")
    
    iterations = 0
    max_iterations = 15
    
    while (V_max - V_min) > 5 and iterations < max_iterations:
        V_mid = (V_min + V_max) / 2
        
        run_id = f"onset_search_{iterations}"
        result = run_single_case(backend, run_id, V_mid, gamma_RF=0.0)
        
        chi_mean = result["powder_chi_avg"] if use_powder_region else result["chi_mean"]
        
        print(f"  V={V_mid:.0f}V: {'powder_' if use_powder_region else ''}chi_mean={chi_mean:.4f}")
        
        if abs(chi_mean - target_chi_mean) < 0.02:
            print(f"  → Found V_onset ≈ {V_mid:.0f}V")
            return V_mid
        elif chi_mean < target_chi_mean:
            V_min = V_mid
        else:
            V_max = V_mid
        
        iterations += 1
    
    V_onset = (V_min + V_max) / 2
    print(f"  → V_onset ≈ {V_onset:.0f}V (after {iterations} iterations)")
    return V_onset


def run_rf_sweep(
    backend: ComsolBackend,
    V_onset: float,
    V_fractions: List[float] = [0.6, 0.7, 0.8, 0.9, 1.0],
    gamma_values: List[float] = [0, 0.05, 0.1, 0.2, 0.4],
) -> List[Dict]:
    """Run the full RF-assisted sweep."""
    print(f"\n{'='*60}")
    print(f"Running RF-assisted sweep (powder region focus)")
    print(f"V_onset = {V_onset:.0f}V")
    print(f"V fractions: {V_fractions}")
    print(f"gamma_RF values: {gamma_values}")
    print(f"{'='*60}")
    
    all_results = []
    total = len(V_fractions) * len(gamma_values)
    count = 0
    
    for V_frac in V_fractions:
        V = V_onset * V_frac
        
        for gamma in gamma_values:
            count += 1
            run_id = f"rf_sweep_V{int(V_frac*100)}_g{int(gamma*100)}"
            
            print(f"\n[{count}/{total}] V={V:.0f}V ({V_frac:.0%} of V_onset), γ_RF={gamma}")
            
            result = run_single_case(backend, run_id, V, gamma)
            result["V_fraction"] = V_frac
            
            print(f"  Whole: chi>0.5: {result['chi_fraction_gt_0p5']:.1%}, RF_assist: {result['RF_assisted_activation_fraction']:.1%}")
            print(f"  Powder: chi>0.5: {result['powder_chi_fraction_gt_0p5']:.1%}, RF_assist: {result['powder_RF_assisted_activation_fraction']:.1%}, E_RF={result['powder_E_RF_max']/1000:.1f}kV/m")
            
            all_results.append(result)
    
    return all_results


def analyze_required_bias_curves(
    backend: ComsolBackend,
    gamma_values: List[float] = [0, 0.1, 0.2, 0.4],
    targets: List[float] = [0.2, 0.3, 0.4],
    V_range: Tuple[float, float] = (300, 700),
    V_step: float = 25,
) -> Dict:
    """
    For each gamma_RF, find the voltage required to achieve target powder activation.
    """
    print(f"\n{'='*60}")
    print(f"Computing required bias for target activation curves")
    print(f"Targets: {[f'{t*100:.0f}%' for t in targets]}")
    print(f"gamma_RF values: {gamma_values}")
    print(f"{'='*60}")
    
    curves = {gamma: {} for gamma in gamma_values}
    
    for gamma in gamma_values:
        print(f"\n--- γ_RF = {gamma} ---")
        
        # Scan voltage range
        V_values = np.arange(V_range[0], V_range[1] + V_step, V_step)
        chi_values = []
        
        for V in V_values:
            run_id = f"bias_scan_g{int(gamma*100)}_V{int(V)}"
            result = run_single_case(backend, run_id, V, gamma)
            chi_values.append(result["powder_chi_fraction_gt_0p5"])
        
        chi_values = np.array(chi_values)
        
        # Find V required for each target
        for target in targets:
            V_req = None
            for i in range(len(V_values) - 1):
                if chi_values[i] < target <= chi_values[i+1]:
                    # Linear interpolation
                    slope = (chi_values[i+1] - chi_values[i]) / V_step
                    if slope > 0:
                        V_req = V_values[i] + (target - chi_values[i]) / slope
                        break
            
            if V_req is None:
                if chi_values[-1] >= target:
                    V_req = V_values[np.argmax(chi_values >= target)]
                else:
                    V_req = float('inf')
            
            curves[gamma][target] = V_req
            print(f"  {target*100:.0f}% activation: V = {V_req:.0f}V" if V_req < float('inf') else f"  {target*100:.0f}% activation: NOT ACHIEVED")
        
        curves[gamma]["V_values"] = V_values.tolist()
        curves[gamma]["chi_values"] = chi_values.tolist()
    
    return curves


def analyze_coil_optimization(
    backend: ComsolBackend,
    V_bias: float = 450,
    gamma_RF: float = 0.2,
) -> Dict:
    """
    Analyze coil parameter adjustments to increase E_RF in powder region.
    
    Strategies:
    1. Increase coil current (I_coil)
    2. Reduce coil radius to bring field closer
    3. Adjust coil z-position to center on powder region
    4. Modify turn density
    """
    print(f"\n{'='*60}")
    print(f"Coil Optimization Analysis")
    print(f"Goal: Increase powder E_RF by 2× while keeping wall heating acceptable")
    print(f"Baseline: V={V_bias}V, γ_RF={gamma_RF}")
    print(f"{'='*60}")
    
    # Baseline run
    baseline = run_single_case(backend, "coil_opt_baseline", V_bias, gamma_RF)
    baseline_E_RF = baseline["powder_E_RF_max"]
    baseline_T_wall = baseline["powder_T_max"]
    
    print(f"\n--- Baseline ---")
    print(f"  powder_E_RF_max: {baseline_E_RF/1000:.2f} kV/m")
    print(f"  powder_T_max: {baseline_T_wall:.0f} K")
    print(f"  powder_chi>0.5: {baseline['powder_chi_fraction_gt_0p5']*100:.1f}%")
    
    target_E_RF = baseline_E_RF * 2.0
    print(f"\nTarget: powder_E_RF_max ≥ {target_E_RF/1000:.2f} kV/m")
    
    optimization_results = {"baseline": baseline, "strategies": []}
    
    # Strategy 1: Increase coil current
    print(f"\n--- Strategy 1: Increase coil current ---")
    # E_RF scales linearly with I_coil, Q_RF scales with I_coil^2
    # To double E_RF, need 2× current, but Q_RF increases 4×
    for I_mult in [1.5, 2.0, 2.5, 3.0]:
        I_coil = DEFAULT_I_COIL * I_mult
        run_id = f"coil_opt_I{int(I_mult*100)}"
        result = run_single_case(backend, run_id, V_bias, gamma_RF, I_coil=I_coil)
        
        E_RF_ratio = result["powder_E_RF_max"] / baseline_E_RF
        T_increase = result["powder_T_max"] - baseline_T_wall
        
        status = "✓" if result["powder_E_RF_max"] >= target_E_RF and result["powder_T_max"] < 1500 else "✗"
        print(f"  I_coil={I_coil:.0f}A ({I_mult}×): E_RF={result['powder_E_RF_max']/1000:.2f}kV/m ({E_RF_ratio:.2f}×), T+{T_increase:.0f}K → {result['powder_T_max']:.0f}K {status}")
        
        optimization_results["strategies"].append({
            "strategy": "increase_I_coil",
            "I_coil": I_coil,
            "I_multiplier": I_mult,
            "powder_E_RF_max": result["powder_E_RF_max"],
            "E_RF_ratio": E_RF_ratio,
            "powder_T_max": result["powder_T_max"],
            "powder_chi_fraction": result["powder_chi_fraction_gt_0p5"],
            "acceptable": result["powder_T_max"] < 1500,
        })
    
    # Strategy 2: Reduce coil radius (closer to powder region)
    print(f"\n--- Strategy 2: Reduce coil radius ---")
    # Smaller coil radius increases field in inner region
    # but also increases local heating
    for r_coil in [0.050, 0.045, 0.040]:  # Default is 0.055
        coil_params = {"coil_radius": r_coil}
        run_id = f"coil_opt_R{int(r_coil*1000)}"
        result = run_single_case(backend, run_id, V_bias, gamma_RF, coil_params=coil_params)
        
        E_RF_ratio = result["powder_E_RF_max"] / baseline_E_RF
        T_increase = result["powder_T_max"] - baseline_T_wall
        
        status = "✓" if result["powder_E_RF_max"] >= target_E_RF and result["powder_T_max"] < 1500 else "✗"
        print(f"  R_coil={r_coil*1000:.0f}mm: E_RF={result['powder_E_RF_max']/1000:.2f}kV/m ({E_RF_ratio:.2f}×), T+{T_increase:.0f}K → {result['powder_T_max']:.0f}K {status}")
        
        optimization_results["strategies"].append({
            "strategy": "reduce_coil_radius",
            "coil_radius": r_coil,
            "powder_E_RF_max": result["powder_E_RF_max"],
            "E_RF_ratio": E_RF_ratio,
            "powder_T_max": result["powder_T_max"],
            "powder_chi_fraction": result["powder_chi_fraction_gt_0p5"],
            "acceptable": result["powder_T_max"] < 1500,
        })
    
    # Strategy 3: Adjust coil z-position to center on powder region
    print(f"\n--- Strategy 3: Center coil on powder region ---")
    # Powder region: z = [0.06, 0.14], center = 0.10
    # Default coil: z = [0.04, 0.16], center = 0.10 (already centered)
    # Try tighter coverage
    for z_start, z_end in [(0.05, 0.15), (0.055, 0.145), (0.06, 0.14)]:
        coil_params = {"z_coil_start": z_start, "z_coil_end": z_end}
        run_id = f"coil_opt_Z{int(z_start*1000)}_{int(z_end*1000)}"
        result = run_single_case(backend, run_id, V_bias, gamma_RF, coil_params=coil_params)
        
        E_RF_ratio = result["powder_E_RF_max"] / baseline_E_RF
        T_increase = result["powder_T_max"] - baseline_T_wall
        
        status = "✓" if result["powder_E_RF_max"] >= target_E_RF and result["powder_T_max"] < 1500 else "✗"
        print(f"  z=[{z_start*1000:.0f},{z_end*1000:.0f}]mm: E_RF={result['powder_E_RF_max']/1000:.2f}kV/m ({E_RF_ratio:.2f}×), T+{T_increase:.0f}K → {result['powder_T_max']:.0f}K {status}")
        
        optimization_results["strategies"].append({
            "strategy": "adjust_coil_z",
            "z_coil_start": z_start,
            "z_coil_end": z_end,
            "powder_E_RF_max": result["powder_E_RF_max"],
            "E_RF_ratio": E_RF_ratio,
            "powder_T_max": result["powder_T_max"],
            "powder_chi_fraction": result["powder_chi_fraction_gt_0p5"],
            "acceptable": result["powder_T_max"] < 1500,
        })
    
    # Strategy 4: Combined - increase I and reduce radius
    print(f"\n--- Strategy 4: Combined (I_coil + R_coil) ---")
    for I_mult, r_coil in [(1.5, 0.050), (1.5, 0.045), (2.0, 0.050)]:
        I_coil = DEFAULT_I_COIL * I_mult
        coil_params = {"coil_radius": r_coil}
        run_id = f"coil_opt_comb_I{int(I_mult*100)}_R{int(r_coil*1000)}"
        result = run_single_case(backend, run_id, V_bias, gamma_RF, I_coil=I_coil, coil_params=coil_params)
        
        E_RF_ratio = result["powder_E_RF_max"] / baseline_E_RF
        T_increase = result["powder_T_max"] - baseline_T_wall
        
        status = "✓" if result["powder_E_RF_max"] >= target_E_RF and result["powder_T_max"] < 1500 else "✗"
        print(f"  I={I_coil:.0f}A, R={r_coil*1000:.0f}mm: E_RF={result['powder_E_RF_max']/1000:.2f}kV/m ({E_RF_ratio:.2f}×), T+{T_increase:.0f}K → {result['powder_T_max']:.0f}K {status}")
        
        optimization_results["strategies"].append({
            "strategy": "combined",
            "I_coil": I_coil,
            "coil_radius": r_coil,
            "powder_E_RF_max": result["powder_E_RF_max"],
            "E_RF_ratio": E_RF_ratio,
            "powder_T_max": result["powder_T_max"],
            "powder_chi_fraction": result["powder_chi_fraction_gt_0p5"],
            "acceptable": result["powder_T_max"] < 1500,
        })
    
    # Find best acceptable strategy
    acceptable = [s for s in optimization_results["strategies"] if s["acceptable"]]
    if acceptable:
        best = max(acceptable, key=lambda x: x["E_RF_ratio"])
        print(f"\n--- RECOMMENDED OPTIMIZATION ---")
        print(f"Strategy: {best['strategy']}")
        for k, v in best.items():
            if k != "strategy":
                print(f"  {k}: {v}")
        optimization_results["recommended"] = best
    else:
        print(f"\n⚠ No acceptable strategy found that achieves 2× E_RF while keeping T < 1500K")
        optimization_results["recommended"] = None
    
    return optimization_results


def create_plots(results: List[Dict], V_onset: float, bias_curves: Dict) -> None:
    """Create visualization plots."""
    
    # Extract data for heatmaps
    V_fractions = sorted(set(r["V_fraction"] for r in results))
    gamma_values = sorted(set(r["gamma_RF"] for r in results))
    
    n_V = len(V_fractions)
    n_gamma = len(gamma_values)
    
    # Create 2D arrays for powder region metrics
    powder_chi = np.zeros((n_V, n_gamma))
    powder_rf_assist = np.zeros((n_V, n_gamma))
    powder_E_RF = np.zeros((n_V, n_gamma))
    powder_E_eff_lambda = np.zeros((n_V, n_gamma))
    
    for r in results:
        i = V_fractions.index(r["V_fraction"])
        j = gamma_values.index(r["gamma_RF"])
        powder_chi[i, j] = r["powder_chi_fraction_gt_0p5"]
        powder_rf_assist[i, j] = r["powder_RF_assisted_activation_fraction"]
        powder_E_RF[i, j] = r["powder_E_RF_max"]
        powder_E_eff_lambda[i, j] = r["powder_E_eff_over_lambda_fraction"]
    
    # =========================================================================
    # FIGURE 1: Powder Region Overview
    # =========================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: powder_chi vs V for each gamma
    ax1 = axes[0, 0]
    for j, gamma in enumerate(gamma_values):
        ax1.plot([V * V_onset for V in V_fractions], powder_chi[:, j] * 100, 
                 'o-', label=f'γ_RF={gamma}', linewidth=2, markersize=8)
    ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='30% target')
    ax1.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='40% target')
    ax1.set_xlabel('Bias Voltage (V)')
    ax1.set_ylabel('Powder χ > 0.5 Fraction (%)')
    ax1.set_title('Powder Region Flash Activation vs Voltage')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 80])
    
    # Panel 2: RF_assisted heatmap for powder region
    ax2 = axes[0, 1]
    im2 = ax2.imshow(powder_rf_assist.T * 100, aspect='auto', origin='lower',
                     extent=[V_fractions[0]*100-5, V_fractions[-1]*100+5,
                             gamma_values[0]-0.025, gamma_values[-1]+0.025],
                     cmap='YlOrRd')
    ax2.set_xlabel('V / V_onset (%)')
    ax2.set_ylabel('γ_RF')
    ax2.set_title('Powder RF-Assisted Activation (%)')
    plt.colorbar(im2, ax=ax2, label='%')
    
    # Add value annotations
    for i, V_frac in enumerate(V_fractions):
        for j, gamma in enumerate(gamma_values):
            val = powder_rf_assist[i, j] * 100
            if val > 0.1:
                ax2.text(V_frac*100, gamma, f'{val:.1f}', ha='center', va='center', fontsize=8)
    
    # Panel 3: powder_chi heatmap
    ax3 = axes[1, 0]
    im3 = ax3.imshow(powder_chi.T * 100, aspect='auto', origin='lower',
                     extent=[V_fractions[0]*100-5, V_fractions[-1]*100+5,
                             gamma_values[0]-0.025, gamma_values[-1]+0.025],
                     cmap='viridis')
    ax3.set_xlabel('V / V_onset (%)')
    ax3.set_ylabel('γ_RF')
    ax3.set_title('Powder χ > 0.5 Fraction (%)')
    plt.colorbar(im3, ax=ax3, label='%')
    
    # Add contours for targets
    X, Y = np.meshgrid(np.array(V_fractions)*100, gamma_values)
    CS = ax3.contour(X, Y, powder_chi.T * 100, levels=[20, 30, 40], colors=['yellow', 'orange', 'red'], linewidths=2)
    ax3.clabel(CS, fmt='%d%%', fontsize=10)
    
    # Panel 4: E_eff_over_lambda in powder region
    ax4 = axes[1, 1]
    im4 = ax4.imshow(powder_E_eff_lambda.T * 100, aspect='auto', origin='lower',
                     extent=[V_fractions[0]*100-5, V_fractions[-1]*100+5,
                             gamma_values[0]-0.025, gamma_values[-1]+0.025],
                     cmap='plasma')
    ax4.set_xlabel('V / V_onset (%)')
    ax4.set_ylabel('γ_RF')
    ax4.set_title('Powder E_eff > λ Fraction (%)')
    plt.colorbar(im4, ax=ax4, label='%')
    
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / 'powder_rf_sweep_overview.png', dpi=150)
    plt.close(fig1)
    print(f"Saved: {RESULTS_DIR / 'powder_rf_sweep_overview.png'}")
    
    # =========================================================================
    # FIGURE 2: Required bias curves
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Required V for each target
    ax5 = axes2[0]
    targets = [0.2, 0.3, 0.4]
    colors = ['blue', 'orange', 'red']
    
    gamma_list = list(bias_curves.keys())
    for i, target in enumerate(targets):
        V_values = []
        for gamma in gamma_list:
            V_req = bias_curves[gamma].get(target, float('inf'))
            V_values.append(V_req if V_req < float('inf') else np.nan)
        ax5.plot(gamma_list, V_values, 'o-', color=colors[i], 
                label=f'χ>{target*100:.0f}%', linewidth=2, markersize=8)
    
    ax5.set_xlabel('γ_RF')
    ax5.set_ylabel('Required Voltage (V)')
    ax5.set_title('Voltage for Target Powder Activation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 2: chi vs V curves for each gamma
    ax6 = axes2[1]
    for gamma in gamma_list:
        V_vals = bias_curves[gamma].get("V_values", [])
        chi_vals = bias_curves[gamma].get("chi_values", [])
        if V_vals and chi_vals:
            ax6.plot(V_vals, np.array(chi_vals)*100, '-', label=f'γ_RF={gamma}', linewidth=2)
    
    ax6.axhline(y=20, color='blue', linestyle='--', alpha=0.5)
    ax6.axhline(y=30, color='orange', linestyle='--', alpha=0.5)
    ax6.axhline(y=40, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Bias Voltage (V)')
    ax6.set_ylabel('Powder χ > 0.5 Fraction (%)')
    ax6.set_title('Powder Activation vs Bias for Different γ_RF')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 70])
    
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'required_bias_curves.png', dpi=150)
    plt.close(fig2)
    print(f"Saved: {RESULTS_DIR / 'required_bias_curves.png'}")


def main():
    print("\n" + "#"*70)
    print("# RF-ASSISTED ACTIVATION SWEEP - POWDER REGION FOCUS")
    print("#"*70)
    print(f"\nPowder Region Definition:")
    print(f"  r: [{POWDER_REGION['r_min']*1000:.0f}, {POWDER_REGION['r_max']*1000:.0f}] mm")
    print(f"  z: [{POWDER_REGION['z_start']*1000:.0f}, {POWDER_REGION['z_end']*1000:.0f}] mm")
    
    backend = ComsolBackend()
    
    # Step 1: Find V_onset for powder region
    V_onset = find_onset_voltage(backend, target_chi_mean=0.5, use_powder_region=True)
    
    # Step 2: Run the sweep
    results = run_rf_sweep(
        backend,
        V_onset,
        V_fractions=[0.6, 0.7, 0.8, 0.9, 1.0],
        gamma_values=[0, 0.05, 0.1, 0.2, 0.4],
    )
    
    # Step 3: Compute required bias curves
    bias_curves = analyze_required_bias_curves(
        backend,
        gamma_values=[0, 0.1, 0.2, 0.4],
        targets=[0.2, 0.3, 0.4],
    )
    
    # Step 4: Coil optimization analysis
    coil_opt = analyze_coil_optimization(backend, V_bias=int(V_onset*0.85), gamma_RF=0.2)
    
    # Step 5: Create plots
    create_plots(results, V_onset, bias_curves)
    
    # Step 6: Save all results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "V_onset": V_onset,
        "lambda_onset": LAMBDA_ONSET,
        "powder_region": POWDER_REGION,
        "sweep_results": results,
        "bias_curves": {str(k): v for k, v in bias_curves.items()},
        "coil_optimization": coil_opt,
    }
    
    with open(RESULTS_DIR / 'powder_rf_sweep_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # Print summary tables
    print("\n" + "="*70)
    print("POWDER REGION RF SWEEP SUMMARY")
    print("="*70)
    
    print(f"\nV_onset (powder chi_mean ≈ 0.5 at γ_RF=0): {V_onset:.0f}V")
    
    print(f"\n--- Powder χ > 0.5 Fraction Matrix ---")
    print(f"{'V/V_onset':<12} {'γ=0':<10} {'γ=0.05':<10} {'γ=0.1':<10} {'γ=0.2':<10} {'γ=0.4':<10}")
    print("-" * 62)
    
    V_fractions = sorted(set(r["V_fraction"] for r in results))
    gamma_values = sorted(set(r["gamma_RF"] for r in results))
    
    for V_frac in V_fractions:
        row = f"{V_frac*100:.0f}%{'':<8}"
        for gamma in gamma_values:
            match = [r for r in results if r["V_fraction"] == V_frac and r["gamma_RF"] == gamma][0]
            row += f"{match['powder_chi_fraction_gt_0p5']*100:<10.1f}"
        print(row)
    
    print(f"\n--- Powder RF-Assisted Activation Matrix ---")
    print(f"{'V/V_onset':<12} {'γ=0':<10} {'γ=0.05':<10} {'γ=0.1':<10} {'γ=0.2':<10} {'γ=0.4':<10}")
    print("-" * 62)
    
    for V_frac in V_fractions:
        row = f"{V_frac*100:.0f}%{'':<8}"
        for gamma in gamma_values:
            match = [r for r in results if r["V_fraction"] == V_frac and r["gamma_RF"] == gamma][0]
            row += f"{match['powder_RF_assisted_activation_fraction']*100:<10.1f}"
        print(row)
    
    print(f"\n--- Required Voltage for Target Powder Activation ---")
    print(f"{'Target':<15} {'γ=0':<12} {'γ=0.1':<12} {'γ=0.2':<12} {'γ=0.4':<12}")
    print("-" * 63)
    for target in [0.2, 0.3, 0.4]:
        row = f"{target*100:.0f}%{'':<12}"
        for gamma in [0, 0.1, 0.2, 0.4]:
            V_req = bias_curves[gamma].get(target, float('inf'))
            if V_req < float('inf'):
                row += f"{V_req:<12.0f}"
            else:
                row += f"{'N/A':<12}"
        print(row)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return output


if __name__ == "__main__":
    main()
