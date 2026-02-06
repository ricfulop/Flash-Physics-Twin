#!/usr/bin/env python3
"""
powder_trade_curves.py

Build bias–preheat–geometry trade curves for the powder region.

For the top 3 candidate geometries:
- Conservative: a=8mm, b=26mm, gap=18mm
- Aggressive: a=12mm, b=20mm, gap=8mm  
- Balanced: a=4mm, b=20mm, gap=16mm

Generate curves of required bias V to reach powder_chi>0.5 targets
[10%, 20%, 40%, 60%] at preheat temperatures [1000, 1100, 1200] K,
with RF off (gamma_RF=0) and RF on (gamma_RF=0.2 and 0.4).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp" / "comsol_mcp"))
from comsol_api import ComsolBackend

# Constants
LAMBDA_ONSET = 60443.0  # V/m
RESULTS_DIR = Path(__file__).parent.parent / "results" / "powder_trade_curves"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate geometries from engineering design
GEOMETRIES = {
    "Conservative": {
        "a_mm": 8.0,      # pin radius
        "b_mm": 26.0,     # mesh radius
        "gap_mm": 18.0,   # gap = b - a
        "description": "Wide gap, moderate E-field, safer thermal margin"
    },
    "Aggressive": {
        "a_mm": 12.0,
        "b_mm": 20.0,
        "gap_mm": 8.0,
        "description": "Narrow gap, high E-field, optimal activation"
    },
    "Balanced": {
        "a_mm": 4.0,
        "b_mm": 20.0,
        "gap_mm": 16.0,
        "description": "Small pin, moderate gap, good coverage"
    }
}

# Powder region (matching ops.yaml)
POWDER_REGION = {
    'r_min': 0.005,
    'r_max': 0.035,
    'z_start': 0.06,
    'z_end': 0.14,
}

# Sweep parameters
PREHEAT_TEMPS = [1000, 1100, 1200]  # K
GAMMA_VALUES = [0, 0.2, 0.4]
CHI_TARGETS = [0.10, 0.20, 0.40, 0.60]
V_RANGE = (200, 1200)
V_STEP = 25

# Default EM parameters
DEFAULT_I_COIL = 50.0  # A
DEFAULT_SIGMA_EFF = 0.1  # S/m


def run_single_case(
    backend: ComsolBackend,
    run_id: str,
    geometry: Dict,
    bias_voltage: float,
    gamma_RF: float,
    T_preheat: float,
) -> Dict:
    """Run a single case and return powder-region KPIs."""
    
    run_path = RESULTS_DIR / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Create mock model entry
    models_dir = run_path / "models"
    models_dir.mkdir(exist_ok=True)
    model_file = models_dir / "trade_curve.mph"
    model_file.write_text("# Placeholder")
    backend._models[run_id] = type('MockModel', (), {'java': None, 'name': lambda: 'trade_curve'})()
    
    # Gap distance in meters
    gap_distance = geometry["gap_mm"] / 1000.0
    
    # Configure parameters
    em_params = {
        "I_coil": DEFAULT_I_COIL,
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
    
    return {
        "run_id": run_id,
        "bias_voltage": bias_voltage,
        "gamma_RF": gamma_RF,
        "T_preheat": T_preheat,
        "powder_chi_fraction": kpis["flash"].get("powder_chi_fraction_gt_0p5", 0),
        "powder_chi_avg": kpis["flash"].get("powder_chi_avg", 0),
        "powder_RF_assist": kpis["flash"].get("powder_RF_assisted_activation_fraction", 0),
        "powder_E_RF_max": kpis["em"].get("powder_E_RF_max", 0),
        "powder_T_mean": kpis["thermal"].get("powder_T_mean", 0),
        "T_max": kpis["thermal"].get("T_max", 0),
        "unphysical": kpis["thermal"].get("unphysical_temperature_flag", False),
    }


def find_required_voltage(
    backend: ComsolBackend,
    geometry: Dict,
    geometry_name: str,
    T_preheat: float,
    gamma_RF: float,
    targets: List[float],
    V_range: Tuple[float, float],
    V_step: float,
) -> Dict[float, Optional[float]]:
    """
    Find required voltage to achieve each target powder_chi_fraction.
    Returns dict mapping target -> required voltage (or None if not achieved).
    """
    V_values = np.arange(V_range[0], V_range[1] + V_step, V_step)
    chi_values = []
    
    for V in V_values:
        run_id = f"{geometry_name}_T{int(T_preheat)}_g{int(gamma_RF*100)}_V{int(V)}"
        result = run_single_case(backend, run_id, geometry, V, gamma_RF, T_preheat)
        chi_values.append(result["powder_chi_fraction"])
    
    chi_values = np.array(chi_values)
    
    # Find required V for each target
    required_V = {}
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
                idx = np.argmax(chi_values >= target)
                V_req = float(V_values[idx])
            # else: V_req stays None (target not achievable)
        
        required_V[target] = V_req
    
    return required_V, V_values.tolist(), chi_values.tolist()


def build_trade_curves(backend: ComsolBackend) -> Dict:
    """Build complete trade curves for all geometries."""
    
    all_results = {}
    
    for geom_name, geometry in GEOMETRIES.items():
        print(f"\n{'='*70}")
        print(f"GEOMETRY: {geom_name}")
        print(f"  a = {geometry['a_mm']:.0f}mm, b = {geometry['b_mm']:.0f}mm, gap = {geometry['gap_mm']:.0f}mm")
        print(f"  {geometry['description']}")
        print(f"{'='*70}")
        
        geom_results = {
            "geometry": geometry,
            "curves": {},
            "required_V": {},
            "V_values": {},
            "chi_values": {},
        }
        
        for T_preheat in PREHEAT_TEMPS:
            geom_results["required_V"][T_preheat] = {}
            geom_results["V_values"][T_preheat] = {}
            geom_results["chi_values"][T_preheat] = {}
            
            for gamma_RF in GAMMA_VALUES:
                key = f"T{T_preheat}_g{int(gamma_RF*100)}"
                print(f"\n  --- T_preheat={T_preheat}K, γ_RF={gamma_RF} ---")
                
                required_V, V_vals, chi_vals = find_required_voltage(
                    backend, geometry, geom_name, T_preheat, gamma_RF,
                    CHI_TARGETS, V_RANGE, V_STEP
                )
                
                geom_results["required_V"][T_preheat][gamma_RF] = required_V
                geom_results["V_values"][T_preheat][gamma_RF] = V_vals
                geom_results["chi_values"][T_preheat][gamma_RF] = chi_vals
                
                for target, V_req in required_V.items():
                    if V_req is not None:
                        print(f"    {target*100:.0f}% activation: V = {V_req:.0f}V")
                    else:
                        print(f"    {target*100:.0f}% activation: NOT ACHIEVED in range")
        
        all_results[geom_name] = geom_results
    
    return all_results


def create_plots(results: Dict) -> None:
    """Create comprehensive visualization plots."""
    
    # =========================================================================
    # FIGURE 1: Multi-panel trade curves for each geometry
    # =========================================================================
    fig1, axes = plt.subplots(3, 3, figsize=(16, 14))
    
    colors_gamma = {0: 'blue', 0.2: 'orange', 0.4: 'red'}
    styles_T = {1000: '-', 1100: '--', 1200: ':'}
    
    for i, (geom_name, geom_data) in enumerate(results.items()):
        ax = axes[i, 0]
        
        # Panel 1: chi vs V curves for each (T, gamma) combination
        for T_preheat in PREHEAT_TEMPS:
            for gamma_RF in GAMMA_VALUES:
                V_vals = geom_data["V_values"][T_preheat][gamma_RF]
                chi_vals = geom_data["chi_values"][T_preheat][gamma_RF]
                label = f'T={T_preheat}K, γ={gamma_RF}'
                ax.plot(V_vals, np.array(chi_vals)*100, 
                       color=colors_gamma[gamma_RF], 
                       linestyle=styles_T[T_preheat],
                       linewidth=1.5, label=label)
        
        # Add target lines
        for target in CHI_TARGETS:
            ax.axhline(y=target*100, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Bias Voltage (V)')
        ax.set_ylabel('Powder χ > 0.5 (%)')
        ax.set_title(f'{geom_name}\n(a={geom_data["geometry"]["a_mm"]:.0f}mm, gap={geom_data["geometry"]["gap_mm"]:.0f}mm)')
        ax.set_xlim([200, 1200])
        ax.set_ylim([0, 80])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='lower right', ncol=2)
    
    # Panels 2-3: Required V heatmaps
    for i, (geom_name, geom_data) in enumerate(results.items()):
        # Panel 2: Required V for 20% target
        ax2 = axes[i, 1]
        data_20 = np.zeros((len(PREHEAT_TEMPS), len(GAMMA_VALUES)))
        for j, T in enumerate(PREHEAT_TEMPS):
            for k, g in enumerate(GAMMA_VALUES):
                V_req = geom_data["required_V"][T][g].get(0.20)
                data_20[j, k] = V_req if V_req is not None else np.nan
        
        im2 = ax2.imshow(data_20, aspect='auto', cmap='viridis_r', 
                        vmin=300, vmax=800)
        ax2.set_xticks(range(len(GAMMA_VALUES)))
        ax2.set_xticklabels([f'{g}' for g in GAMMA_VALUES])
        ax2.set_yticks(range(len(PREHEAT_TEMPS)))
        ax2.set_yticklabels([f'{T}K' for T in PREHEAT_TEMPS])
        ax2.set_xlabel('γ_RF')
        ax2.set_ylabel('T_preheat')
        ax2.set_title(f'{geom_name}: V for 20% target')
        
        # Add value annotations
        for j in range(len(PREHEAT_TEMPS)):
            for k in range(len(GAMMA_VALUES)):
                val = data_20[j, k]
                if not np.isnan(val):
                    ax2.text(k, j, f'{val:.0f}', ha='center', va='center', fontsize=9, fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, label='V')
        
        # Panel 3: Required V for 40% target
        ax3 = axes[i, 2]
        data_40 = np.zeros((len(PREHEAT_TEMPS), len(GAMMA_VALUES)))
        for j, T in enumerate(PREHEAT_TEMPS):
            for k, g in enumerate(GAMMA_VALUES):
                V_req = geom_data["required_V"][T][g].get(0.40)
                data_40[j, k] = V_req if V_req is not None else np.nan
        
        im3 = ax3.imshow(data_40, aspect='auto', cmap='viridis_r',
                        vmin=400, vmax=1000)
        ax3.set_xticks(range(len(GAMMA_VALUES)))
        ax3.set_xticklabels([f'{g}' for g in GAMMA_VALUES])
        ax3.set_yticks(range(len(PREHEAT_TEMPS)))
        ax3.set_yticklabels([f'{T}K' for T in PREHEAT_TEMPS])
        ax3.set_xlabel('γ_RF')
        ax3.set_ylabel('T_preheat')
        ax3.set_title(f'{geom_name}: V for 40% target')
        
        for j in range(len(PREHEAT_TEMPS)):
            for k in range(len(GAMMA_VALUES)):
                val = data_40[j, k]
                if not np.isnan(val):
                    ax3.text(k, j, f'{val:.0f}', ha='center', va='center', fontsize=9, fontweight='bold')
        
        plt.colorbar(im3, ax=ax3, label='V')
    
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / 'geometry_trade_curves.png', dpi=150)
    plt.close(fig1)
    print(f"\nSaved: {RESULTS_DIR / 'geometry_trade_curves.png'}")
    
    # =========================================================================
    # FIGURE 2: Comparison of geometries at fixed conditions
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Compare geometries at T=1100K, gamma=0
    ax = axes2[0, 0]
    for geom_name in results.keys():
        V_vals = results[geom_name]["V_values"][1100][0]
        chi_vals = results[geom_name]["chi_values"][1100][0]
        ax.plot(V_vals, np.array(chi_vals)*100, 'o-', label=geom_name, linewidth=2, markersize=4)
    
    for target in CHI_TARGETS:
        ax.axhline(y=target*100, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Bias Voltage (V)')
    ax.set_ylabel('Powder χ > 0.5 (%)')
    ax.set_title('Geometry Comparison (T=1100K, γ_RF=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([200, 1000])
    ax.set_ylim([0, 70])
    
    # Panel 2: Compare geometries at T=1100K, gamma=0.4
    ax = axes2[0, 1]
    for geom_name in results.keys():
        V_vals = results[geom_name]["V_values"][1100][0.4]
        chi_vals = results[geom_name]["chi_values"][1100][0.4]
        ax.plot(V_vals, np.array(chi_vals)*100, 'o-', label=geom_name, linewidth=2, markersize=4)
    
    for target in CHI_TARGETS:
        ax.axhline(y=target*100, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Bias Voltage (V)')
    ax.set_ylabel('Powder χ > 0.5 (%)')
    ax.set_title('Geometry Comparison (T=1100K, γ_RF=0.4)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([200, 1000])
    ax.set_ylim([0, 70])
    
    # Panel 3: Required V comparison bar chart (40% target)
    ax = axes2[1, 0]
    x = np.arange(len(PREHEAT_TEMPS))
    width = 0.25
    
    for i, geom_name in enumerate(results.keys()):
        V_values = []
        for T in PREHEAT_TEMPS:
            V_req = results[geom_name]["required_V"][T][0].get(0.40)
            V_values.append(V_req if V_req else np.nan)
        ax.bar(x + i*width, V_values, width, label=geom_name)
    
    ax.set_xlabel('Preheat Temperature')
    ax.set_ylabel('Required Voltage (V)')
    ax.set_title('Required V for 40% Powder Activation (γ_RF=0)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{T}K' for T in PREHEAT_TEMPS])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: RF benefit (voltage savings at γ=0.4 vs γ=0)
    ax = axes2[1, 1]
    
    for i, geom_name in enumerate(results.keys()):
        savings = []
        for T in PREHEAT_TEMPS:
            V_g0 = results[geom_name]["required_V"][T][0].get(0.40)
            V_g04 = results[geom_name]["required_V"][T][0.4].get(0.40)
            if V_g0 and V_g04:
                savings.append(V_g0 - V_g04)
            else:
                savings.append(0)
        ax.bar(x + i*width, savings, width, label=geom_name)
    
    ax.set_xlabel('Preheat Temperature')
    ax.set_ylabel('Voltage Savings (V)')
    ax.set_title('RF Benefit: Voltage Savings (γ_RF=0.4 vs 0) for 40% target')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{T}K' for T in PREHEAT_TEMPS])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'geometry_comparison.png', dpi=150)
    plt.close(fig2)
    print(f"Saved: {RESULTS_DIR / 'geometry_comparison.png'}")
    
    # =========================================================================
    # FIGURE 3: Power electronics sizing chart
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    # Create comprehensive sizing chart
    markers = {'Conservative': 'o', 'Aggressive': 's', 'Balanced': '^'}
    colors_T = {1000: 'blue', 1100: 'green', 1200: 'red'}
    
    for geom_name in results.keys():
        for T in PREHEAT_TEMPS:
            for target in [0.20, 0.40, 0.60]:
                V_req = results[geom_name]["required_V"][T][0].get(target)
                if V_req and V_req < 1200:
                    # Calculate approximate power (P = V * I_est)
                    # Rough estimate: I ~ 0.1A at low V, scales with gap conductivity
                    gap_mm = results[geom_name]["geometry"]["gap_mm"]
                    I_est = 0.1 * (10 / gap_mm)  # rough scaling
                    P_est = V_req * I_est
                    
                    ax3.scatter(V_req, target*100, 
                              marker=markers[geom_name],
                              c=[colors_T[T]], s=100, alpha=0.7,
                              edgecolors='black', linewidths=0.5)
    
    # Add legend elements
    legend_elements = []
    for geom_name, marker in markers.items():
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                         markerfacecolor='gray', markersize=10,
                                         label=geom_name))
    for T, color in colors_T.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=10,
                                         label=f'{T}K'))
    
    ax3.legend(handles=legend_elements, loc='lower right', ncol=2)
    ax3.set_xlabel('Required Bias Voltage (V)', fontsize=12)
    ax3.set_ylabel('Target Powder Activation (%)', fontsize=12)
    ax3.set_title('Power Electronics Sizing: Voltage vs Activation Target\n(γ_RF=0, all geometries)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([300, 1000])
    ax3.set_ylim([0, 70])
    
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / 'power_sizing_chart.png', dpi=150)
    plt.close(fig3)
    print(f"Saved: {RESULTS_DIR / 'power_sizing_chart.png'}")


def generate_compact_table(results: Dict) -> str:
    """Generate compact table for power electronics + heater sizing."""
    
    lines = []
    lines.append("=" * 100)
    lines.append("POWDER REGION TRADE CURVES - COMPACT SIZING TABLE")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Required Bias Voltage (V) for Target Powder Activation")
    lines.append("γ_RF=0 (DC only) | γ_RF=0.2 | γ_RF=0.4")
    lines.append("")
    
    for geom_name, geom_data in results.items():
        lines.append("-" * 100)
        lines.append(f"GEOMETRY: {geom_name}")
        lines.append(f"  Pin: a={geom_data['geometry']['a_mm']:.0f}mm | Mesh: b={geom_data['geometry']['b_mm']:.0f}mm | Gap: {geom_data['geometry']['gap_mm']:.0f}mm")
        lines.append("-" * 100)
        
        # Header
        header = f"{'T_pre':<8}"
        for target in CHI_TARGETS:
            header += f"| χ>{target*100:.0f}% (γ=0/0.2/0.4) "
        lines.append(header)
        lines.append("-" * 100)
        
        for T in PREHEAT_TEMPS:
            row = f"{T}K{'':<4}"
            for target in CHI_TARGETS:
                V_g0 = geom_data["required_V"][T][0].get(target)
                V_g02 = geom_data["required_V"][T][0.2].get(target)
                V_g04 = geom_data["required_V"][T][0.4].get(target)
                
                v0 = f"{V_g0:.0f}" if V_g0 else "N/A"
                v02 = f"{V_g02:.0f}" if V_g02 else "N/A"
                v04 = f"{V_g04:.0f}" if V_g04 else "N/A"
                
                row += f"| {v0:>4}/{v02:>4}/{v04:>4}     "
            lines.append(row)
        lines.append("")
    
    # Heater sizing section
    lines.append("=" * 100)
    lines.append("HEATER SIZING GUIDE")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Preheat Power Estimate (for 100mm ID tube, 200mm heated length):")
    lines.append("  P_heater ≈ h * A * ΔT")
    lines.append("  h ≈ 10 W/(m²·K) (free convection in Ar/H2)")
    lines.append("  A = π * D * L ≈ 0.063 m²")
    lines.append("")
    lines.append(f"  {'T_preheat':<12} {'ΔT from 300K':<15} {'P_heater (est)':<15} {'P_heater + 50% margin':<20}")
    lines.append("-" * 65)
    for T in PREHEAT_TEMPS:
        dT = T - 300
        P_est = 10 * 0.063 * dT
        P_margin = P_est * 1.5
        lines.append(f"  {T}K{'':<8} {dT}K{'':<11} {P_est:.0f}W{'':<11} {P_margin:.0f}W")
    
    lines.append("")
    lines.append("=" * 100)
    lines.append("POWER ELECTRONICS SIZING GUIDE")
    lines.append("=" * 100)
    lines.append("")
    lines.append("DC Bias Supply Requirements (per geometry at T=1100K, γ_RF=0):")
    lines.append("")
    lines.append(f"  {'Geometry':<15} {'V for 20%':<12} {'V for 40%':<12} {'V for 60%':<12} {'I_est (mA)':<12} {'P_est (W)':<12}")
    lines.append("-" * 75)
    
    for geom_name, geom_data in results.items():
        V_20 = geom_data["required_V"][1100][0].get(0.20)
        V_40 = geom_data["required_V"][1100][0].get(0.40)
        V_60 = geom_data["required_V"][1100][0].get(0.60)
        
        gap_mm = geom_data['geometry']['gap_mm']
        # Rough current estimate (plasma conduction at ~1e-4 S/m, A ~ 0.01 m²)
        sigma_eff = 1e-4  # S/m (weak plasma)
        A_cross = 0.01  # m² (approximate cross-section)
        gap_m = gap_mm / 1000
        
        V_max = V_60 if V_60 else (V_40 if V_40 else V_20)
        if V_max:
            I_est = sigma_eff * A_cross * V_max / gap_m * 1000  # mA
            P_est = V_max * I_est / 1000  # W
        else:
            I_est = 0
            P_est = 0
        
        v20 = f"{V_20:.0f}V" if V_20 else "N/A"
        v40 = f"{V_40:.0f}V" if V_40 else "N/A"
        v60 = f"{V_60:.0f}V" if V_60 else "N/A"
        
        lines.append(f"  {geom_name:<15} {v20:<12} {v40:<12} {v60:<12} {I_est:<12.1f} {P_est:<12.1f}")
    
    lines.append("")
    lines.append("Note: Current estimates assume weak plasma conductivity (σ ~ 1e-4 S/m)")
    lines.append("      Actual current depends strongly on plasma density and temperature")
    lines.append("      Recommend HV supply rated for: 0-1500V, 0-100mA minimum")
    lines.append("")
    
    # RF system sizing
    lines.append("=" * 100)
    lines.append("RF SYSTEM PARAMETERS (for reference)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  Frequency: 13.56 MHz")
    lines.append(f"  Coil current (baseline): {DEFAULT_I_COIL} A")
    lines.append(f"  Plasma conductivity: {DEFAULT_SIGMA_EFF} S/m")
    lines.append(f"  λ_onset: {LAMBDA_ONSET/1000:.1f} kV/m")
    lines.append("")
    lines.append("RF provides modest voltage savings (~5-15V) at γ_RF=0.4")
    lines.append("Primary activation mechanism is DC bias; RF is supplementary")
    lines.append("")
    
    return "\n".join(lines)


def main():
    print("\n" + "#"*70)
    print("# POWDER REGION TRADE CURVES")
    print("# Bias–Preheat–Geometry Analysis for Power Electronics Sizing")
    print("#"*70)
    
    backend = ComsolBackend()
    
    # Build all trade curves
    results = build_trade_curves(backend)
    
    # Create plots
    create_plots(results)
    
    # Generate compact table
    table = generate_compact_table(results)
    print("\n" + table)
    
    # Save table
    with open(RESULTS_DIR / 'sizing_table.txt', 'w') as f:
        f.write(table)
    print(f"\nSaved: {RESULTS_DIR / 'sizing_table.txt'}")
    
    # Save full results
    # Convert results for JSON serialization
    def convert_keys(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_key = str(k) if isinstance(k, (int, float)) else k
                new_dict[new_key] = convert_keys(v)
            return new_dict
        elif isinstance(obj, list):
            return [convert_keys(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "geometries": GEOMETRIES,
        "powder_region": POWDER_REGION,
        "preheat_temps": PREHEAT_TEMPS,
        "gamma_values": GAMMA_VALUES,
        "chi_targets": CHI_TARGETS,
        "results": convert_keys(results),
    }
    
    with open(RESULTS_DIR / 'powder_trade_curves.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'powder_trade_curves.json'}")
    
    return results


if __name__ == "__main__":
    main()
