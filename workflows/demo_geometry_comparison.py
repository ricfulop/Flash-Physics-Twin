#!/usr/bin/env python3
"""
PFR Digital Twin Demo: Focused Geometry Comparison
===================================================

Compares 3 candidate reactor configurations to demonstrate:
- Multi-physics flash activation modeling
- Residence time analysis (τ_powder_active, τ_gas_refresh)
- Enhanced dusting/plasma risk with Stokes-number proxy
- Multi-objective optimization (min V, min risk, Pareto)

Author: PFR Digital Twin System
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from datetime import datetime

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
RESULTS_DIR = Path(__file__).parent.parent / "results" / "demo_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CANDIDATE GEOMETRIES
# =============================================================================
GEOMETRIES = {
    "Tight Gap": {
        "a_mm": 10,
        "b_mm": 18,
        "description": "Aggressive E-field, 8mm gap",
        "color": "#e74c3c"  # red
    },
    "Wide Gap": {
        "a_mm": 4,
        "b_mm": 22,
        "description": "Conservative, 18mm gap",
        "color": "#3498db"  # blue
    },
    "Balanced": {
        "a_mm": 6,
        "b_mm": 20,
        "description": "Compromise, 14mm gap",
        "color": "#2ecc71"  # green
    },
}

TUBE_RADIUS = 50  # mm

# =============================================================================
# SWEEP PARAMETERS
# =============================================================================
T_PREHEAT_VALUES = [1000, 1100, 1200]  # K
V_BIAS_VALUES = [200, 400, 600, 800]    # V
GAMMA_RF_VALUES = [0, 0.2, 0.4]
Q_H2_SLPM = 20  # Fixed flow rate

# =============================================================================
# PHYSICS PARAMETERS
# =============================================================================
PHYSICS = {
    "n": 2,
    "F": 96485,
    "r_act": 2e-6,
    "k_soft": 8.0,
    "DeltaG0_ref": 370e3,
    "T_ref": 1000,
    "beta_T": -100,
    "p_bar": 0.15,
    "L_active": 0.3,
}

# Constraints
TAU_MIN, TAU_MAX = 0.3, 3.0
RISK_LIMIT = 1.0
CHI_TARGET = 0.40

# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================
def get_calibration(T: float) -> Tuple[float, float]:
    A_base, B_base = 2.30, 0.029
    T_ref = 1100
    A = A_base + 0.0015 * (T - T_ref)
    B = B_base + 0.00006 * (T - T_ref)
    return A, B

def compute_DeltaG0(T: float) -> float:
    return PHYSICS["DeltaG0_ref"] + PHYSICS["beta_T"] * (T - PHYSICS["T_ref"])

def compute_lambda_onset(T: float) -> float:
    DeltaG0 = compute_DeltaG0(T)
    n, F, r_act, k_soft = PHYSICS["n"], PHYSICS["F"], PHYSICS["r_act"], PHYSICS["k_soft"]
    return n * F * r_act * DeltaG0 / k_soft

def compute_gas_properties(T_gas: float, p_bar: float) -> Tuple[float, float]:
    M_H2 = 2.016e-3
    R = 8.314
    rho_g = (p_bar * 1e5) * M_H2 / (R * T_gas)
    mu_0, T_0, S = 8.9e-6, 300.0, 72.0
    mu_g = mu_0 * (T_gas / T_0)**1.5 * (T_0 + S) / (T_gas + S)
    return rho_g, mu_g

def compute_stokes_number(v_gas: float, L_char: float, T: float, p_bar: float,
                          d_p: float = 7.5e-6, rho_p: float = 5000.0) -> float:
    _, mu_g = compute_gas_properties(T, p_bar)
    if L_char <= 0:
        return 0.0
    return (rho_p * d_p**2 * v_gas) / (18 * mu_g * L_char)

# =============================================================================
# EVALUATION
# =============================================================================
@dataclass
class EvalResult:
    geometry_name: str
    a_mm: float
    b_mm: float
    gap_mm: float
    standoff_mm: float
    T_preheat: float
    V_bias: float
    gamma_RF: float
    chi_fraction: float
    E_max: float
    lambda_onset: float
    tau_powder_active: float
    tau_gas_refresh: float
    tau_gas_flag: bool
    v_crossflow: float
    stokes_number: float
    risk_index: float
    risk_stokes: float
    risk_velocity: float
    risk_gap: float
    risk_standoff: float
    feasible: bool
    meets_target: bool


def evaluate_point(geom_name: str, geom: Dict, T: float, V: float, gamma: float) -> EvalResult:
    """Evaluate a single design point."""
    a = geom["a_mm"] / 1000
    b = geom["b_mm"] / 1000
    R_tube = TUBE_RADIUS / 1000
    gap = b - a
    standoff = R_tube - b
    L = PHYSICS["L_active"]
    p_bar = PHYSICS["p_bar"]
    
    # Areas and volumes
    A_active = np.pi * (b**2 - a**2)
    A_annulus = np.pi * (R_tube**2 - b**2)
    V_annulus = A_annulus * L
    
    # Flow rate
    T_STP, p_STP = 273.15, 1.0
    Q_m3s = (Q_H2_SLPM / 60000) * (T / T_STP) * (p_STP / p_bar)
    
    # Flash physics
    lambda_onset = compute_lambda_onset(T)
    E_max = V / (a * np.log(b / a)) if a > 0 and b > a else 0
    E_RF = 5000
    E_eff = E_max + gamma * E_RF
    
    # Activation fraction
    if E_max > 0:
        r_onset = V / (lambda_onset * np.log(b / a))
        r_onset = np.clip(r_onset, a, b)
    else:
        r_onset = a
    
    A_cal, B_cal = get_calibration(T)
    if b > a:
        f_raw = (r_onset**2 - a**2) / (b**2 - a**2)
        chi_fraction = np.clip(A_cal * np.clip(f_raw, 0, 1) + B_cal, 0, 1)
    else:
        chi_fraction = 0
    
    # Residence times (conservative placeholder: 1.0 m/s median fall)
    v_fall = 1.0
    tau_powder_active = L / v_fall
    tau_gas_refresh = V_annulus / Q_m3s if Q_m3s > 0 else float('inf')
    tau_gas_flag = tau_gas_refresh > 2.0
    
    # Flow
    v_crossflow = Q_m3s / A_active if A_active > 0 else 0
    stokes_number = compute_stokes_number(v_crossflow, gap, T, p_bar)
    
    # Risk index
    v_max, g_min, gap_min, Stk_crit = 5.0, 0.015, 0.010, 1.0
    
    risk_stokes = max(0, (Stk_crit / stokes_number) - 1) if stokes_number > 0.01 else 5.0
    risk_velocity = max(0, (v_crossflow / v_max) - 1) * 0.5
    risk_gap = max(0, (gap_min / gap) - 1) if gap > 0.003 else 5.0
    risk_standoff = max(0, (g_min / standoff) - 1) if standoff > 0.005 else 5.0
    
    risk_index = 0.30 * risk_stokes + 0.25 * risk_velocity + 0.20 * risk_gap + 0.25 * risk_standoff
    
    # Feasibility
    feasible = (TAU_MIN <= tau_powder_active <= TAU_MAX) and (risk_index < RISK_LIMIT)
    meets_target = chi_fraction >= CHI_TARGET
    
    return EvalResult(
        geometry_name=geom_name,
        a_mm=geom["a_mm"], b_mm=geom["b_mm"],
        gap_mm=gap*1000, standoff_mm=standoff*1000,
        T_preheat=T, V_bias=V, gamma_RF=gamma,
        chi_fraction=chi_fraction, E_max=E_max, lambda_onset=lambda_onset,
        tau_powder_active=tau_powder_active, tau_gas_refresh=tau_gas_refresh,
        tau_gas_flag=tau_gas_flag, v_crossflow=v_crossflow,
        stokes_number=stokes_number, risk_index=risk_index,
        risk_stokes=risk_stokes, risk_velocity=risk_velocity,
        risk_gap=risk_gap, risk_standoff=risk_standoff,
        feasible=feasible, meets_target=meets_target
    )


def run_sweep() -> Dict[str, List[EvalResult]]:
    """Run sweep for all geometries."""
    results = {name: [] for name in GEOMETRIES}
    
    for geom_name, geom in GEOMETRIES.items():
        for T in T_PREHEAT_VALUES:
            for V in V_BIAS_VALUES:
                for gamma in GAMMA_RF_VALUES:
                    result = evaluate_point(geom_name, geom, T, V, gamma)
                    results[geom_name].append(result)
    
    return results


def find_required_voltage(results: List[EvalResult], T: float, gamma: float) -> Optional[float]:
    """Find minimum V to achieve target chi."""
    matching = [r for r in results if r.T_preheat == T and r.gamma_RF == gamma and r.feasible]
    for r in sorted(matching, key=lambda x: x.V_bias):
        if r.chi_fraction >= CHI_TARGET:
            return r.V_bias
    return None


def find_pareto_optimal(results: List[EvalResult]) -> EvalResult:
    """Find Pareto-optimal point (min V subject to feasibility and target)."""
    candidates = [r for r in results if r.feasible and r.meets_target]
    if not candidates:
        return min(results, key=lambda x: x.risk_index)  # fallback
    
    # Score: balance V and risk
    for r in candidates:
        r._score = 0.5 * (r.V_bias / 800) + 0.5 * r.risk_index
    
    return min(candidates, key=lambda x: x._score)


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_demo_plots(all_results: Dict[str, List[EvalResult]]):
    """Create comprehensive demo visualization."""
    
    print("\n📊 Generating visualizations...")
    
    # =========================================================================
    # FIGURE 1: Chi vs V curves for each geometry
    # =========================================================================
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (geom_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        geom = GEOMETRIES[geom_name]
        
        for T in T_PREHEAT_VALUES:
            for gamma in [0, 0.4]:
                subset = [r for r in results if r.T_preheat == T and r.gamma_RF == gamma]
                V_vals = [r.V_bias for r in subset]
                chi_vals = [r.chi_fraction * 100 for r in subset]
                
                ls = '-' if gamma == 0 else '--'
                label = f'T={T}K' if gamma == 0 else f'T={T}K, γ=0.4'
                ax.plot(V_vals, chi_vals, ls, linewidth=2, label=label, 
                       color=plt.cm.plasma((T-1000)/200))
        
        ax.axhline(y=40, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.fill_between([200, 800], [40, 40], [100, 100], alpha=0.1, color='green')
        ax.set_xlabel('Bias Voltage (V)', fontsize=11)
        ax.set_ylabel('Powder Activation χ (%)', fontsize=11)
        ax.set_title(f'{geom_name}\n(a={geom["a_mm"]}mm, gap={geom["a_mm"]+int(geom["b_mm"]-geom["a_mm"])-geom["a_mm"]}mm)', 
                    fontsize=12, fontweight='bold', color=geom['color'])
        ax.set_xlim([200, 800])
        ax.set_ylim([0, 105])
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Flash Activation vs Bias Voltage', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / 'activation_curves.png', dpi=150)
    plt.close(fig1)
    print(f"  ✓ Saved: activation_curves.png")
    
    # =========================================================================
    # FIGURE 2: Required Voltage Comparison
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(T_PREHEAT_VALUES))
    width = 0.25
    
    for idx, (geom_name, results) in enumerate(all_results.items()):
        geom = GEOMETRIES[geom_name]
        V_required = []
        for T in T_PREHEAT_VALUES:
            V_req = find_required_voltage(results, T, gamma=0)
            V_required.append(V_req if V_req else 900)
        
        bars = ax2.bar(x + idx*width, V_required, width, label=geom_name, 
                      color=geom['color'], edgecolor='black', linewidth=1.5)
        
        for bar, v in zip(bars, V_required):
            if v < 900:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                        f'{v:.0f}V', ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Preheat Temperature (K)', fontsize=12)
    ax2.set_ylabel('Required Voltage for 40% Activation (V)', fontsize=12)
    ax2.set_title('Voltage Requirements by Geometry\n(DC-only, γ=0)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'{T}K' for T in T_PREHEAT_VALUES])
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 900])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'voltage_comparison.png', dpi=150)
    plt.close(fig2)
    print(f"  ✓ Saved: voltage_comparison.png")
    
    # =========================================================================
    # FIGURE 3: Risk Breakdown
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (geom_name, results) in enumerate(all_results.items()):
        ax = axes3[idx]
        geom = GEOMETRIES[geom_name]
        
        # Get Pareto point
        pareto = find_pareto_optimal(results)
        
        risk_components = {
            'Stokes\n(entrainment)': pareto.risk_stokes * 0.30,
            'Velocity\n(lofting)': pareto.risk_velocity * 0.25,
            'Gap\n(E-field)': pareto.risk_gap * 0.20,
            'Standoff\n(wall)': pareto.risk_standoff * 0.25,
        }
        
        colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
        bars = ax.bar(risk_components.keys(), risk_components.values(), 
                     color=colors, edgecolor='black', linewidth=1.5)
        
        ax.axhline(y=RISK_LIMIT/4, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Weighted Risk Contribution', fontsize=11)
        ax.set_title(f'{geom_name}\nTotal Risk: {pareto.risk_index:.3f}', 
                    fontsize=12, fontweight='bold', color=geom['color'])
        ax.set_ylim([0, max(0.5, max(risk_components.values())*1.2)])
        
        for bar, val in zip(bars, risk_components.values()):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', fontsize=9)
    
    plt.suptitle('Dusting/Plasma Risk Breakdown\n(Stokes-number enhanced)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / 'risk_breakdown.png', dpi=150)
    plt.close(fig3)
    print(f"  ✓ Saved: risk_breakdown.png")
    
    # =========================================================================
    # FIGURE 4: Residence Time Analysis
    # =========================================================================
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: tau_powder_active (same for all at 0.3s)
    ax = axes4[0]
    geom_names = list(GEOMETRIES.keys())
    tau_powder = [0.3] * 3  # All same with v_fall = 1.0 m/s
    colors = [GEOMETRIES[g]['color'] for g in geom_names]
    
    bars = ax.barh(geom_names, tau_powder, color=colors, edgecolor='black', height=0.5)
    ax.axvline(x=TAU_MIN, color='red', linestyle='--', linewidth=2, label=f'Min ({TAU_MIN}s)')
    ax.axvline(x=TAU_MAX, color='red', linestyle='--', linewidth=2, label=f'Max ({TAU_MAX}s)')
    ax.fill_betweenx([-0.5, 2.5], TAU_MIN, TAU_MAX, alpha=0.1, color='green')
    ax.set_xlabel('τ_powder_active (s)', fontsize=12)
    ax.set_title('Powder Residence Time\n(L=300mm, v_fall=1.0 m/s)', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.legend(loc='upper right')
    
    # Right: tau_gas_refresh
    ax = axes4[1]
    tau_gas = []
    for geom_name in geom_names:
        results = all_results[geom_name]
        # Get at T=1100K, V=400V, gamma=0
        r = [r for r in results if r.T_preheat == 1100 and r.V_bias == 400 and r.gamma_RF == 0][0]
        tau_gas.append(r.tau_gas_refresh)
    
    bars = ax.barh(geom_names, tau_gas, color=colors, edgecolor='black', height=0.5)
    ax.axvline(x=2.0, color='orange', linestyle='--', linewidth=2, label='Warning (2s)')
    
    for bar, val in zip(bars, tau_gas):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}s', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('τ_gas_refresh (s)', fontsize=12)
    ax.set_title('Gas Crossflow Refresh Time\n(Q=20 SLPM, T=1100K)', fontsize=12, fontweight='bold')
    ax.set_xlim([0, max(tau_gas)*1.3])
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig4.savefig(RESULTS_DIR / 'residence_times.png', dpi=150)
    plt.close(fig4)
    print(f"  ✓ Saved: residence_times.png")
    
    # =========================================================================
    # FIGURE 5: Pareto Front (V vs Risk)
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    
    for geom_name, results in all_results.items():
        geom = GEOMETRIES[geom_name]
        feasible = [r for r in results if r.feasible and r.meets_target]
        
        if feasible:
            V_vals = [r.V_bias for r in feasible]
            risk_vals = [r.risk_index for r in feasible]
            ax5.scatter(V_vals, risk_vals, c=geom['color'], s=100, alpha=0.6, 
                       label=geom_name, edgecolors='black', linewidths=1)
            
            # Mark Pareto optimal
            pareto = find_pareto_optimal(results)
            ax5.scatter([pareto.V_bias], [pareto.risk_index], c=geom['color'], 
                       s=300, marker='*', edgecolors='black', linewidths=2, zorder=10)
    
    ax5.axhline(y=RISK_LIMIT, color='red', linestyle='--', linewidth=2, label='Risk Limit')
    ax5.set_xlabel('Bias Voltage (V)', fontsize=12)
    ax5.set_ylabel('Risk Index', fontsize=12)
    ax5.set_title('Pareto Front: Voltage vs Risk Trade-off\n(★ = Pareto-optimal point)', 
                 fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([150, 850])
    ax5.set_ylim([0, 1.2])
    
    # Add annotation
    ax5.annotate('Lower-left\nis better', xy=(250, 0.2), fontsize=12, 
                fontstyle='italic', alpha=0.7)
    
    plt.tight_layout()
    fig5.savefig(RESULTS_DIR / 'pareto_front.png', dpi=150)
    plt.close(fig5)
    print(f"  ✓ Saved: pareto_front.png")
    
    # =========================================================================
    # FIGURE 6: Summary Dashboard
    # =========================================================================
    fig6 = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig6, hspace=0.3, wspace=0.3)
    
    # Panel 1: Geometry schematic
    ax1 = fig6.add_subplot(gs[0, 0])
    ax1.set_xlim([-55, 55])
    ax1.set_ylim([-55, 55])
    ax1.set_aspect('equal')
    
    # Draw tube
    circle_tube = plt.Circle((0, 0), 50, fill=False, color='black', linewidth=3)
    ax1.add_patch(circle_tube)
    
    for geom_name, geom in GEOMETRIES.items():
        a, b = geom['a_mm'], geom['b_mm']
        circle_mesh = plt.Circle((0, 0), b, fill=False, color=geom['color'], 
                                 linewidth=2, linestyle='--')
        circle_pin = plt.Circle((0, 0), a, fill=True, color=geom['color'], alpha=0.3)
        ax1.add_patch(circle_mesh)
        ax1.add_patch(circle_pin)
    
    ax1.set_title('Geometry Configurations\n(cross-section view)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('r (mm)')
    ax1.set_ylabel('r (mm)')
    
    # Legend
    for geom_name, geom in GEOMETRIES.items():
        ax1.plot([], [], color=geom['color'], linewidth=3, 
                label=f"{geom_name}: a={geom['a_mm']}mm, b={geom['b_mm']}mm")
    ax1.legend(loc='upper right', fontsize=8)
    
    # Panel 2: Required voltage bar chart
    ax2 = fig6.add_subplot(gs[0, 1])
    x = np.arange(3)
    width = 0.6
    V_req_1100 = []
    for geom_name, results in all_results.items():
        V = find_required_voltage(results, T=1100, gamma=0)
        V_req_1100.append(V if V else 900)
    
    colors = [GEOMETRIES[g]['color'] for g in GEOMETRIES.keys()]
    bars = ax2.bar(list(GEOMETRIES.keys()), V_req_1100, color=colors, 
                  edgecolor='black', linewidth=2)
    
    for bar, v in zip(bars, V_req_1100):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{v:.0f}V', ha='center', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Required V for 40% (V)', fontsize=11)
    ax2.set_title('Voltage Requirement\n(T=1100K, DC-only)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 700])
    
    # Panel 3: Risk comparison
    ax3 = fig6.add_subplot(gs[0, 2])
    risk_vals = []
    for geom_name, results in all_results.items():
        pareto = find_pareto_optimal(results)
        risk_vals.append(pareto.risk_index)
    
    bars = ax3.bar(list(GEOMETRIES.keys()), risk_vals, color=colors,
                  edgecolor='black', linewidth=2)
    ax3.axhline(y=RISK_LIMIT, color='red', linestyle='--', linewidth=2)
    
    for bar, v in zip(bars, risk_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Risk Index', fontsize=11)
    ax3.set_title('Dusting/Plasma Risk\n(at Pareto-optimal point)', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.2])
    
    # Panel 4-6: Recommendation table
    ax4 = fig6.add_subplot(gs[1, :])
    ax4.axis('off')
    
    # Build table data
    table_data = [['Geometry', 'Gap', 'Standoff', 'V@40%', 'Risk', 'τ_powder', 'τ_gas', 'Stokes', 'Recommendation']]
    
    recommendations = []
    for geom_name, results in all_results.items():
        geom = GEOMETRIES[geom_name]
        pareto = find_pareto_optimal(results)
        V_req = find_required_voltage(results, T=1100, gamma=0)
        
        rec = "⚠ High risk" if pareto.risk_index > 0.7 else ("✓ Best balance" if pareto.risk_index < 0.4 else "○ Moderate")
        recommendations.append((geom_name, pareto, V_req, rec))
        
        table_data.append([
            geom_name,
            f'{pareto.gap_mm:.0f} mm',
            f'{pareto.standoff_mm:.0f} mm',
            f'{V_req:.0f} V' if V_req else 'N/A',
            f'{pareto.risk_index:.3f}',
            f'{pareto.tau_powder_active:.2f} s',
            f'{pareto.tau_gas_refresh:.2f} s',
            f'{pareto.stokes_number:.2f}',
            rec
        ])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.12, 0.08, 0.09, 0.08, 0.08, 0.09, 0.09, 0.08, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.5)
    
    # Style header
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color code rows by geometry
    for i, (geom_name, _, _, _) in enumerate(recommendations, 1):
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(GEOMETRIES[geom_name]['color'] + '30')
    
    ax4.set_title('Multi-Objective Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('PFR Digital Twin: Geometry Comparison Demo', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig6.savefig(RESULTS_DIR / 'summary_dashboard.png', dpi=150)
    plt.close(fig6)
    print(f"  ✓ Saved: summary_dashboard.png")


# =============================================================================
# REPORT
# =============================================================================
def generate_report(all_results: Dict[str, List[EvalResult]]) -> str:
    """Generate demo report."""
    
    lines = []
    lines.append("=" * 90)
    lines.append("PFR DIGITAL TWIN: GEOMETRY COMPARISON DEMO")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)
    lines.append("")
    
    lines.append("CANDIDATE GEOMETRIES")
    lines.append("-" * 90)
    for geom_name, geom in GEOMETRIES.items():
        gap = geom['b_mm'] - geom['a_mm']
        standoff = TUBE_RADIUS - geom['b_mm']
        lines.append(f"  {geom_name:<12}: a={geom['a_mm']:>2}mm, b={geom['b_mm']:>2}mm | gap={gap:>2}mm | standoff={standoff:>2}mm | {geom['description']}")
    lines.append("")
    
    lines.append("REQUIRED VOLTAGE FOR 40% ACTIVATION")
    lines.append("-" * 90)
    lines.append(f"  {'Geometry':<12} | {'T=1000K':<15} | {'T=1100K':<15} | {'T=1200K':<15}")
    lines.append("  " + "-" * 60)
    
    for geom_name, results in all_results.items():
        row = f"  {geom_name:<12} |"
        for T in T_PREHEAT_VALUES:
            V_dc = find_required_voltage(results, T, gamma=0)
            V_rf = find_required_voltage(results, T, gamma=0.4)
            dc_str = f"{V_dc:.0f}V" if V_dc else "N/A"
            rf_str = f"{V_rf:.0f}V" if V_rf else "N/A"
            row += f" DC:{dc_str:>4}, RF:{rf_str:>4} |"
        lines.append(row)
    lines.append("")
    
    lines.append("DUSTING/PLASMA RISK BREAKDOWN (at Pareto-optimal point)")
    lines.append("-" * 90)
    lines.append(f"  {'Geometry':<12} | {'Total':<8} | {'Stokes':<10} | {'Velocity':<10} | {'Gap':<10} | {'Standoff':<10}")
    lines.append("  " + "-" * 75)
    
    for geom_name, results in all_results.items():
        pareto = find_pareto_optimal(results)
        lines.append(f"  {geom_name:<12} | {pareto.risk_index:<8.3f} | "
                    f"{pareto.risk_stokes*0.30:<10.3f} | {pareto.risk_velocity*0.25:<10.3f} | "
                    f"{pareto.risk_gap*0.20:<10.3f} | {pareto.risk_standoff*0.25:<10.3f}")
    lines.append("")
    
    lines.append("RESIDENCE TIMES (at T=1100K, Q=20 SLPM)")
    lines.append("-" * 90)
    lines.append(f"  {'Geometry':<12} | {'τ_powder':<12} | {'τ_gas_refresh':<15} | {'Stokes #':<10} | {'Status':<15}")
    lines.append("  " + "-" * 70)
    
    for geom_name, results in all_results.items():
        r = [r for r in results if r.T_preheat == 1100 and r.V_bias == 400 and r.gamma_RF == 0][0]
        status = "⚠ τ_gas > 2s" if r.tau_gas_flag else "✓ OK"
        lines.append(f"  {geom_name:<12} | {r.tau_powder_active:<12.2f} | {r.tau_gas_refresh:<15.2f} | "
                    f"{r.stokes_number:<10.3f} | {status:<15}")
    lines.append("")
    
    # Final recommendation
    lines.append("=" * 90)
    lines.append("RECOMMENDATION")
    lines.append("=" * 90)
    
    # Find best by each objective
    best_V = min(all_results.items(), 
                 key=lambda x: find_required_voltage(x[1], 1100, 0) or 999)
    best_risk = min(all_results.items(),
                    key=lambda x: find_pareto_optimal(x[1]).risk_index)
    
    lines.append("")
    lines.append(f"  MINIMIZE VOLTAGE:  {best_V[0]:<12} → {find_required_voltage(best_V[1], 1100, 0):.0f}V for 40% activation")
    lines.append(f"  MINIMIZE RISK:     {best_risk[0]:<12} → risk index = {find_pareto_optimal(best_risk[1]).risk_index:.3f}")
    lines.append("")
    
    # Pareto recommendation
    all_pareto = [(name, find_pareto_optimal(results)) for name, results in all_results.items()]
    best_pareto = min(all_pareto, key=lambda x: 0.5*(find_required_voltage(all_results[x[0]], 1100, 0) or 999)/800 + 0.5*x[1].risk_index)
    
    lines.append(f"  PARETO OPTIMAL:    {best_pareto[0]}")
    lines.append(f"    → V = {find_required_voltage(all_results[best_pareto[0]], 1100, 0):.0f}V, Risk = {best_pareto[1].risk_index:.3f}")
    lines.append(f"    → τ_powder = {best_pareto[1].tau_powder_active:.2f}s, τ_gas = {best_pareto[1].tau_gas_refresh:.2f}s")
    lines.append(f"    → Stokes = {best_pareto[1].stokes_number:.3f}")
    lines.append("")
    
    lines.append("=" * 90)
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "🔬 " + "=" * 60)
    print("   PFR DIGITAL TWIN: GEOMETRY COMPARISON DEMO")
    print("   Comparing 3 candidate reactor configurations")
    print("🔬 " + "=" * 60)
    
    # Run sweep
    print("\n📈 Running parameter sweep...")
    print(f"   Geometries: {list(GEOMETRIES.keys())}")
    print(f"   T_preheat: {T_PREHEAT_VALUES} K")
    print(f"   V_bias: {V_BIAS_VALUES} V")
    print(f"   γ_RF: {GAMMA_RF_VALUES}")
    
    all_results = run_sweep()
    
    total_points = sum(len(r) for r in all_results.values())
    print(f"   ✓ Evaluated {total_points} design points")
    
    # Generate visualizations
    create_demo_plots(all_results)
    
    # Generate report
    report = generate_report(all_results)
    print("\n" + report)
    
    # Save outputs
    with open(RESULTS_DIR / 'demo_report.txt', 'w') as f:
        f.write(report)
    print(f"\n📄 Saved: {RESULTS_DIR / 'demo_report.txt'}")
    
    # Save JSON
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            return {k: convert(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    json_data = {name: [convert(r) for r in results] for name, results in all_results.items()}
    with open(RESULTS_DIR / 'demo_results.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"📄 Saved: {RESULTS_DIR / 'demo_results.json'}")
    
    print(f"\n🎯 All outputs saved to: {RESULTS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
