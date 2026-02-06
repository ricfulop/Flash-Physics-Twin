#!/usr/bin/env python3
"""
rf_assisted_sweep.py

RF-assisted activation sweep in the regime where RF transfer matters.

Steps:
1. Determine V_onset where chi_mean ≈ 0.5 with gamma_RF = 0
2. Sweep V over [0.6, 0.7, 0.8, 0.9, 1.0] × V_onset
3. For each V, sweep gamma_RF over [0, 0.05, 0.1, 0.2, 0.4]
4. Record and plot KPIs
5. Produce heatmap of RF_assisted_activation_fraction
6. Identify region where RF provides maximum voltage margin
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp" / "comsol_mcp"))
from comsol_api import ComsolBackend

# Constants
LAMBDA_ONSET = 60443.0  # V/m (from PhysicsNeMo inversion)
RESULTS_DIR = Path(__file__).parent.parent / "results" / "rf_sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default geometry parameters
DEFAULT_GAP = 0.01  # 10mm gap
DEFAULT_T_PREHEAT = 1100.0  # K
DEFAULT_I_COIL = 50.0  # A (strong RF field)
DEFAULT_SIGMA_EFF = 0.1  # S/m


def run_single_case(
    backend: ComsolBackend,
    run_id: str,
    bias_voltage: float,
    gamma_RF: float,
    gap_distance: float = DEFAULT_GAP,
    T_preheat: float = DEFAULT_T_PREHEAT,
    I_coil: float = DEFAULT_I_COIL,
) -> Dict:
    """Run a single RF-assisted case and return KPIs."""
    
    run_path = RESULTS_DIR / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Create mock model entry
    models_dir = run_path / "models"
    models_dir.mkdir(exist_ok=True)
    model_file = models_dir / "rf_sweep.mph"
    model_file.write_text("# Placeholder")
    backend._models[run_id] = type('MockModel', (), {'java': None, 'name': lambda: 'rf_sweep'})()
    
    # Configure parameters
    em_params = {
        "I_coil": I_coil,
        "sigma_eff": DEFAULT_SIGMA_EFF,
        "n_turns": 8,
        "coil_radius": 0.055,
        "z_coil_start": 0.04,
        "z_coil_end": 0.16,
    }
    
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
    
    # Export KPIs
    backend.export_coupled_kpis(
        run_path,
        em_mode="surrogate",
        thermal_params=thermal_params,
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
        "E_bias": bias_voltage / gap_distance,
        "chi_fraction_gt_0p5": kpis["flash"]["chi_volume_fraction_gt_0p5"],
        "chi_mean": kpis["flash"]["chi_volume_avg"],
        "chi_dc_only_fraction_gt_0p5": kpis["flash"].get("chi_dc_only_fraction_gt_0p5", 0),
        "RF_assisted_activation_fraction": kpis["flash"].get("RF_assisted_activation_fraction", 0),
        "E_eff_over_lambda_fraction": kpis["flash"].get("E_eff_over_lambda_fraction", 0),
        "E_RF_induced_max": kpis["em"].get("E_RF_induced_max", 0),
        "E_eff_max": kpis["em"].get("E_eff_max", 0),
        "T_max": kpis["thermal"]["T_max"],
        "unphysical": kpis["thermal"].get("unphysical_temperature_flag", False),
    }
    
    return result


def find_onset_voltage(
    backend: ComsolBackend,
    gap_distance: float = DEFAULT_GAP,
    target_chi_mean: float = 0.5,
    V_min: float = 300,
    V_max: float = 800,
    tolerance: float = 0.02,
) -> float:
    """
    Find the bias voltage where chi_mean ≈ target_chi_mean with gamma_RF = 0.
    Uses binary search.
    """
    print(f"\n{'='*60}")
    print(f"Finding V_onset where chi_mean ≈ {target_chi_mean} (gamma_RF=0)")
    print(f"{'='*60}")
    
    # Binary search for onset voltage
    iterations = 0
    max_iterations = 15
    
    while (V_max - V_min) > 5 and iterations < max_iterations:
        V_mid = (V_min + V_max) / 2
        
        run_id = f"onset_search_{iterations}"
        result = run_single_case(
            backend, run_id, V_mid, gamma_RF=0.0, gap_distance=gap_distance
        )
        chi_mean = result["chi_mean"]
        
        print(f"  V={V_mid:.0f}V: chi_mean={chi_mean:.4f}")
        
        if abs(chi_mean - target_chi_mean) < tolerance:
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
    gap_distance: float = DEFAULT_GAP,
) -> List[Dict]:
    """
    Run the full RF-assisted sweep.
    """
    print(f"\n{'='*60}")
    print(f"Running RF-assisted sweep")
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
            
            result = run_single_case(
                backend, run_id, V, gamma, gap_distance=gap_distance
            )
            result["V_fraction"] = V_frac
            
            print(f"  chi>0.5: {result['chi_fraction_gt_0p5']:.1%}, "
                  f"RF_assist: {result['RF_assisted_activation_fraction']:.1%}, "
                  f"E_eff/λ: {result['E_eff_over_lambda_fraction']:.1%}")
            
            all_results.append(result)
    
    return all_results


def analyze_and_plot(results: List[Dict], V_onset: float) -> Dict:
    """
    Analyze results and create plots.
    """
    print(f"\n{'='*60}")
    print("Analyzing results and creating plots")
    print(f"{'='*60}")
    
    # Convert to numpy arrays for analysis
    V_fractions = sorted(set(r["V_fraction"] for r in results))
    gamma_values = sorted(set(r["gamma_RF"] for r in results))
    
    n_V = len(V_fractions)
    n_gamma = len(gamma_values)
    
    # Create 2D arrays for heatmaps
    chi_fraction = np.zeros((n_V, n_gamma))
    rf_assisted = np.zeros((n_V, n_gamma))
    e_eff_lambda = np.zeros((n_V, n_gamma))
    e_rf_max = np.zeros((n_V, n_gamma))
    
    for r in results:
        i = V_fractions.index(r["V_fraction"])
        j = gamma_values.index(r["gamma_RF"])
        chi_fraction[i, j] = r["chi_fraction_gt_0p5"]
        rf_assisted[i, j] = r["RF_assisted_activation_fraction"]
        e_eff_lambda[i, j] = r["E_eff_over_lambda_fraction"]
        e_rf_max[i, j] = r["E_RF_induced_max"]
    
    # =========================================================================
    # FIGURE 1: Multi-panel overview
    # =========================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: chi_fraction vs V/V_onset for each gamma
    ax1 = axes[0, 0]
    for j, gamma in enumerate(gamma_values):
        ax1.plot([V * V_onset for V in V_fractions], chi_fraction[:, j] * 100, 
                 'o-', label=f'γ_RF={gamma}', linewidth=2, markersize=8)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% target')
    ax1.axvline(x=V_onset, color='red', linestyle=':', alpha=0.5, label='V_onset')
    ax1.set_xlabel('Bias Voltage (V)')
    ax1.set_ylabel('χ > 0.5 Fraction (%)')
    ax1.set_title('Flash Activation vs Voltage\n(with varying RF coupling)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Panel 2: RF_assisted_activation_fraction heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(rf_assisted.T * 100, aspect='auto', origin='lower',
                     extent=[V_fractions[0]*100-5, V_fractions[-1]*100+5, 
                             gamma_values[0]-0.025, gamma_values[-1]+0.025],
                     cmap='YlOrRd')
    ax2.set_xlabel('V / V_onset (%)')
    ax2.set_ylabel('γ_RF')
    ax2.set_title('RF-Assisted Activation Fraction (%)')
    plt.colorbar(im2, ax=ax2, label='%')
    
    # Add value annotations
    for i, V_frac in enumerate(V_fractions):
        for j, gamma in enumerate(gamma_values):
            val = rf_assisted[i, j] * 100
            if val > 0.1:
                ax2.text(V_frac*100, gamma, f'{val:.1f}', ha='center', va='center', fontsize=8)
    
    # Panel 3: chi_fraction heatmap
    ax3 = axes[1, 0]
    im3 = ax3.imshow(chi_fraction.T * 100, aspect='auto', origin='lower',
                     extent=[V_fractions[0]*100-5, V_fractions[-1]*100+5,
                             gamma_values[0]-0.025, gamma_values[-1]+0.025],
                     cmap='viridis')
    ax3.set_xlabel('V / V_onset (%)')
    ax3.set_ylabel('γ_RF')
    ax3.set_title('χ > 0.5 Fraction (%)')
    plt.colorbar(im3, ax=ax3, label='%')
    
    # Add contour for 50% threshold
    X, Y = np.meshgrid(np.array(V_fractions)*100, gamma_values)
    CS = ax3.contour(X, Y, chi_fraction.T * 100, levels=[50], colors='red', linewidths=2)
    ax3.clabel(CS, fmt='%d%%', fontsize=10)
    
    # Panel 4: E_RF_induced_max
    ax4 = axes[1, 1]
    # E_RF should be constant across gamma (only depends on I_coil)
    e_rf_mean = np.mean(e_rf_max[:, :])
    ax4.bar(range(n_V), e_rf_max[:, 0] / 1000, tick_label=[f'{V*100:.0f}%' for V in V_fractions])
    ax4.axhline(y=e_rf_mean/1000, color='red', linestyle='--', label=f'Mean: {e_rf_mean/1000:.1f} kV/m')
    ax4.set_xlabel('V / V_onset')
    ax4.set_ylabel('E_RF_induced_max (kV/m)')
    ax4.set_title('Maximum RF-Induced Electric Field')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / 'rf_sweep_overview.png', dpi=150)
    plt.close(fig1)
    print(f"Saved: {RESULTS_DIR / 'rf_sweep_overview.png'}")
    
    # =========================================================================
    # FIGURE 2: Voltage margin analysis
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Voltage to achieve target chi fraction
    ax5 = axes2[0]
    chi_targets = [0.3, 0.4, 0.5, 0.6]
    
    for target in chi_targets:
        V_required = []
        for j, gamma in enumerate(gamma_values):
            # Find V where chi >= target (interpolate)
            for i in range(n_V - 1):
                if chi_fraction[i, j] < target <= chi_fraction[i+1, j]:
                    # Linear interpolation
                    slope = (chi_fraction[i+1, j] - chi_fraction[i, j]) / (V_fractions[i+1] - V_fractions[i])
                    if slope > 0:
                        V_req = V_fractions[i] + (target - chi_fraction[i, j]) / slope
                        V_required.append(V_req * V_onset)
                        break
            else:
                if chi_fraction[-1, j] >= target:
                    V_required.append(V_fractions[-1] * V_onset)
                elif chi_fraction[0, j] >= target:
                    V_required.append(V_fractions[0] * V_onset)
                else:
                    V_required.append(np.nan)
        
        ax5.plot(gamma_values, V_required, 'o-', label=f'χ>{target:.0%}', 
                linewidth=2, markersize=8)
    
    ax5.set_xlabel('γ_RF')
    ax5.set_ylabel('Required Voltage (V)')
    ax5.set_title('Voltage Required to Achieve Target χ Fraction')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 2: Voltage savings from RF
    ax6 = axes2[1]
    
    # For each gamma, compute voltage savings compared to gamma=0
    V_savings = []
    for target in [0.4, 0.5]:
        savings_for_target = []
        
        # Find V at gamma=0
        V_at_gamma0 = None
        for i in range(n_V - 1):
            if chi_fraction[i, 0] < target <= chi_fraction[i+1, 0]:
                slope = (chi_fraction[i+1, 0] - chi_fraction[i, 0]) / (V_fractions[i+1] - V_fractions[i])
                if slope > 0:
                    V_at_gamma0 = V_fractions[i] + (target - chi_fraction[i, 0]) / slope
                break
        
        if V_at_gamma0 is None:
            continue
        
        for j, gamma in enumerate(gamma_values):
            if gamma == 0:
                savings_for_target.append(0)
                continue
            
            # Find V at this gamma
            V_at_gamma = None
            for i in range(n_V - 1):
                if chi_fraction[i, j] < target <= chi_fraction[i+1, j]:
                    slope = (chi_fraction[i+1, j] - chi_fraction[i, j]) / (V_fractions[i+1] - V_fractions[i])
                    if slope > 0:
                        V_at_gamma = V_fractions[i] + (target - chi_fraction[i, j]) / slope
                    break
            
            if V_at_gamma is not None:
                savings = (V_at_gamma0 - V_at_gamma) * V_onset
                savings_for_target.append(savings)
            else:
                savings_for_target.append(0)
        
        if len(savings_for_target) == len(gamma_values):
            ax6.plot(gamma_values, savings_for_target, 'o-', 
                    label=f'χ>{target:.0%}', linewidth=2, markersize=8)
    
    ax6.set_xlabel('γ_RF')
    ax6.set_ylabel('Voltage Savings (V)')
    ax6.set_title('Voltage Savings from RF Coupling\n(compared to γ_RF=0)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'rf_voltage_margin.png', dpi=150)
    plt.close(fig2)
    print(f"Saved: {RESULTS_DIR / 'rf_voltage_margin.png'}")
    
    # =========================================================================
    # Identify maximum RF benefit region
    # =========================================================================
    max_rf_assist_idx = np.unravel_index(np.argmax(rf_assisted), rf_assisted.shape)
    max_rf_V_frac = V_fractions[max_rf_assist_idx[0]]
    max_rf_gamma = gamma_values[max_rf_assist_idx[1]]
    max_rf_value = rf_assisted[max_rf_assist_idx[0], max_rf_assist_idx[1]]
    
    print(f"\n--- Maximum RF Benefit Region ---")
    print(f"Peak RF_assisted_activation_fraction: {max_rf_value:.1%}")
    print(f"At V/V_onset = {max_rf_V_frac:.0%}, γ_RF = {max_rf_gamma}")
    print(f"Voltage: {max_rf_V_frac * V_onset:.0f}V")
    
    # Find voltage margin at different targets
    analysis = {
        "V_onset": V_onset,
        "max_rf_assist": {
            "V_fraction": max_rf_V_frac,
            "gamma_RF": max_rf_gamma,
            "value": max_rf_value,
            "voltage": max_rf_V_frac * V_onset,
        },
        "chi_at_V_onset": {
            "gamma_0": chi_fraction[V_fractions.index(1.0), 0] if 1.0 in V_fractions else None,
            "gamma_0.2": chi_fraction[V_fractions.index(1.0), gamma_values.index(0.2)] if 1.0 in V_fractions and 0.2 in gamma_values else None,
            "gamma_0.4": chi_fraction[V_fractions.index(1.0), gamma_values.index(0.4)] if 1.0 in V_fractions and 0.4 in gamma_values else None,
        },
        "V_fractions": V_fractions,
        "gamma_values": gamma_values,
        "chi_fraction_matrix": chi_fraction.tolist(),
        "rf_assisted_matrix": rf_assisted.tolist(),
    }
    
    return analysis


def main():
    print("\n" + "#"*70)
    print("# RF-ASSISTED ACTIVATION SWEEP")
    print("# Finding regime where RF coupling provides maximum voltage margin")
    print("#"*70)
    
    backend = ComsolBackend()
    
    # Step 1: Find V_onset
    V_onset = find_onset_voltage(backend, target_chi_mean=0.5)
    
    # Step 2: Run the sweep
    results = run_rf_sweep(
        backend,
        V_onset,
        V_fractions=[0.6, 0.7, 0.8, 0.9, 1.0],
        gamma_values=[0, 0.05, 0.1, 0.2, 0.4],
    )
    
    # Step 3: Analyze and plot
    analysis = analyze_and_plot(results, V_onset)
    
    # Step 4: Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "V_onset": V_onset,
        "lambda_onset": LAMBDA_ONSET,
        "I_coil": DEFAULT_I_COIL,
        "gap_distance": DEFAULT_GAP,
        "T_preheat": DEFAULT_T_PREHEAT,
        "results": results,
        "analysis": analysis,
    }
    
    with open(RESULTS_DIR / 'rf_sweep_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nV_onset (chi_mean ≈ 0.5 at γ_RF=0): {V_onset:.0f}V")
    print(f"E_RF_induced_max: {results[0]['E_RF_induced_max']/1000:.1f} kV/m")
    print(f"λ_onset: {LAMBDA_ONSET/1000:.1f} kV/m")
    print(f"E_RF / λ: {results[0]['E_RF_induced_max'] / LAMBDA_ONSET:.1%}")
    
    print(f"\n--- χ > 0.5 Fraction at Different Operating Points ---")
    print(f"{'V/V_onset':<12} {'γ_RF=0':<12} {'γ_RF=0.1':<12} {'γ_RF=0.2':<12} {'γ_RF=0.4':<12}")
    print("-" * 60)
    
    for i, V_frac in enumerate(analysis["V_fractions"]):
        chi_0 = analysis["chi_fraction_matrix"][i][0]
        chi_01 = analysis["chi_fraction_matrix"][i][2] if len(analysis["gamma_values"]) > 2 else 0
        chi_02 = analysis["chi_fraction_matrix"][i][3] if len(analysis["gamma_values"]) > 3 else 0
        chi_04 = analysis["chi_fraction_matrix"][i][4] if len(analysis["gamma_values"]) > 4 else 0
        print(f"{V_frac*100:.0f}%{'':<8} {chi_0*100:<11.1f}% {chi_01*100:<11.1f}% {chi_02*100:<11.1f}% {chi_04*100:<11.1f}%")
    
    print(f"\n--- Maximum RF Benefit ---")
    max_info = analysis["max_rf_assist"]
    print(f"Peak RF_assisted_activation: {max_info['value']:.1%}")
    print(f"Occurs at: V = {max_info['voltage']:.0f}V ({max_info['V_fraction']*100:.0f}% of V_onset), γ_RF = {max_info['gamma_RF']}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return output


if __name__ == "__main__":
    main()
