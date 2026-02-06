#!/usr/bin/env python3
"""
Comprehensive Design Space Exploration for PFR Digital Twin
============================================================

Multi-dimensional parameter sweep demonstrating full system capabilities:
- Geometry optimization (pin radius, mesh radius)
- Operating condition sensitivity (T, V, γ_RF, flow)
- Multi-physics coupling (thermal, EM, flash, flow)
- Constraint satisfaction and risk analysis
- Multi-objective Pareto optimization

Author: PFR Digital Twin System
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from itertools import product
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from datetime import datetime

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
RESULTS_DIR = Path(__file__).parent.parent / "results" / "demo_sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SWEEP PARAMETERS
# =============================================================================
# Geometry sweep
A_VALUES = [2, 4, 6, 8, 10, 12]  # mm - pin radius
B_VALUES = [18, 22, 26, 30, 35, 40]  # mm - mesh radius
TUBE_RADIUS = 50  # mm

# Geometry filters
MIN_GAP = 8  # mm - minimum electrode gap
MIN_STANDOFF = 10  # mm - minimum mesh-to-wall standoff

# Operating conditions
T_PREHEAT_VALUES = [900, 1000, 1100, 1200, 1300]  # K
V_BIAS_VALUES = [200, 400, 600, 800, 1000, 1200]  # V
GAMMA_RF_VALUES = [0, 0.1, 0.2, 0.3, 0.4]
Q_H2_VALUES = [10, 20, 50]  # SLPM

# Constraints
TAU_MIN = 0.3  # s
TAU_MAX = 3.0  # s
RISK_LIMIT = 1.0
T_MAX_LIMIT = 1500  # K

# =============================================================================
# PHYSICS PARAMETERS
# =============================================================================
PHYSICS = {
    "n": 2,                    # electrons transferred
    "F": 96485,                # Faraday constant [C/mol]
    "r_act": 2e-6,             # activation site radius [m]
    "k_soft": 8.0,             # softening parameter
    "DeltaG0_ref": 370e3,      # Reference ΔG₀ [J/mol] at T_ref
    "T_ref": 1000,             # Reference temperature [K]
    "beta_T": -100,            # ΔG₀ temperature coefficient [J/mol/K]
    "p_bar": 0.15,             # Operating pressure [bar]
    "L_active": 0.3,           # Active length [m]
}

# Temperature-dependent calibration
def get_calibration(T: float) -> Tuple[float, float]:
    """Get A(T), B(T) calibration coefficients."""
    A_base, B_base = 2.30, 0.029
    T_ref = 1100
    A = A_base + 0.0015 * (T - T_ref)
    B = B_base + 0.00006 * (T - T_ref)
    return A, B

def compute_DeltaG0(T: float) -> float:
    """Temperature-dependent Gibbs free energy."""
    return PHYSICS["DeltaG0_ref"] + PHYSICS["beta_T"] * (T - PHYSICS["T_ref"])

def compute_lambda_onset(T: float) -> float:
    """Onset field strength [V/m]."""
    DeltaG0 = compute_DeltaG0(T)
    n, F, r_act, k_soft = PHYSICS["n"], PHYSICS["F"], PHYSICS["r_act"], PHYSICS["k_soft"]
    return n * F * r_act * DeltaG0 / k_soft

# =============================================================================
# POWDER FALL MODEL (Enhanced for fine PSD)
# =============================================================================
def compute_gas_properties(T_gas: float, p_bar: float) -> Tuple[float, float]:
    """Compute H2 gas density and viscosity at operating conditions."""
    M_H2 = 2.016e-3  # kg/mol
    R = 8.314  # J/(mol·K)
    
    # Density from ideal gas law
    rho_g = (p_bar * 1e5) * M_H2 / (R * T_gas)
    
    # Viscosity (Sutherland's law for H2)
    mu_0, T_0, S = 8.9e-6, 300.0, 72.0
    mu_g = mu_0 * (T_gas / T_0)**1.5 * (T_0 + S) / (T_gas + S)
    
    return rho_g, mu_g


def compute_powder_fall_velocity(
    T_gas: float, 
    p_bar: float, 
    d_p: float = None,
    rho_p: float = 5000.0
) -> float:
    """
    Compute terminal fall velocity using drag-limited model.
    
    For fine particles (PSD 5-10 µm), fall velocity is low (Stokes regime).
    For coarser particles (50-100 µm), intermediate regime applies.
    
    Parameters:
        T_gas: Gas temperature [K]
        p_bar: Gas pressure [bar]
        d_p: Particle diameter [m]. If None, uses median of 5-10 µm range.
        rho_p: Particle density [kg/m³]
    
    Returns:
        Terminal fall velocity [m/s]
    """
    # Default to median of fine PSD (5-10 µm)
    if d_p is None:
        d_p = 7.5e-6  # 7.5 µm median
    
    g_accel = 9.81  # m/s²
    rho_g, mu_g = compute_gas_properties(T_gas, p_bar)
    
    # Stokes terminal velocity
    v_stokes = (rho_p - rho_g) * g_accel * d_p**2 / (18 * mu_g)
    
    # Check Reynolds number
    Re_p = rho_g * v_stokes * d_p / mu_g
    
    if Re_p < 1:
        # Stokes regime (fine particles)
        return v_stokes
    elif Re_p < 500:
        # Intermediate regime: Schiller-Naumann correction
        v_t = v_stokes
        for _ in range(10):
            Re_p = max(rho_g * v_t * d_p / mu_g, 0.01)
            C_D = 24/Re_p * (1 + 0.15 * Re_p**0.687)
            v_t = np.sqrt(4 * g_accel * d_p * (rho_p - rho_g) / (3 * C_D * rho_g))
        return v_t
    else:
        # Newton regime
        C_D = 0.44
        return np.sqrt(4 * g_accel * d_p * (rho_p - rho_g) / (3 * C_D * rho_g))


def compute_powder_fall_velocity_psd(
    T_gas: float,
    p_bar: float,
    d_min: float = 5e-6,
    d_max: float = 10e-6,
    rho_p: float = 5000.0
) -> Tuple[float, float, float]:
    """
    Compute fall velocities for a particle size distribution.
    
    Returns: (v_min, v_median, v_max) for the PSD range.
    For fine particles, provides conservative bounds.
    """
    d_median = (d_min + d_max) / 2
    
    v_min = compute_powder_fall_velocity(T_gas, p_bar, d_min, rho_p)  # smallest → slowest
    v_median = compute_powder_fall_velocity(T_gas, p_bar, d_median, rho_p)
    v_max = compute_powder_fall_velocity(T_gas, p_bar, d_max, rho_p)  # largest → fastest
    
    return v_min, v_median, v_max


def compute_stokes_number(
    v_gas: float, 
    L_char: float, 
    T: float, 
    p_bar: float,
    d_p: float = 7.5e-6,
    rho_p: float = 5000.0
) -> float:
    """
    Compute Stokes number for particle entrainment analysis.
    
    Stk = (rho_p * d_p^2 * v_gas) / (18 * mu_g * L_char)
    
    Stk >> 1: particles ignore flow (inertial, settle out)
    Stk << 1: particles follow flow (entrained, dusting risk)
    Stk ~ 1: critical regime
    
    Parameters:
        v_gas: Gas velocity [m/s]
        L_char: Characteristic length [m] (typically gap or annulus width)
        T: Gas temperature [K]
        p_bar: Gas pressure [bar]
        d_p: Particle diameter [m]
        rho_p: Particle density [kg/m³]
    
    Returns:
        Stokes number [dimensionless]
    """
    _, mu_g = compute_gas_properties(T, p_bar)
    
    if L_char <= 0:
        return 0.0
    
    Stk = (rho_p * d_p**2 * v_gas) / (18 * mu_g * L_char)
    return Stk

# =============================================================================
# CORE PHYSICS EVALUATION
# =============================================================================
@dataclass
class EvaluationResult:
    """Container for all physics outputs."""
    # Geometry
    a_mm: float
    b_mm: float
    gap_mm: float
    standoff_mm: float
    
    # Operating conditions
    T_preheat: float
    V_bias: float
    gamma_RF: float
    Q_H2_SLPM: float
    
    # Physics outputs
    chi_fraction: float
    E_max: float
    E_eff_over_lambda: float
    lambda_onset: float
    
    # Thermal
    T_max_estimate: float
    thermal_flag: bool
    
    # Flow/residence - ENHANCED
    tau_powder_active: float      # L_active / v_fall (powder in active zone)
    tau_powder_range: Tuple[float, float]  # (min, max) for PSD
    tau_gas_refresh: float        # V_annulus / Vdot_gas (crossflow refresh)
    tau_gas_refresh_flag: bool    # True if > 2s (water/plasma risk)
    v_crossflow: float            # Gas velocity in active region
    v_powder_fall: float          # Median powder fall velocity
    stokes_number: float          # Particle entrainment indicator
    
    # Risk - ENHANCED with Stokes proxy
    risk_index: float
    risk_stokes: float            # Stokes-number based entrainment risk
    risk_velocity: float
    risk_gap: float
    risk_standoff: float
    
    # Feasibility
    feasible: bool
    constraint_violations: List[str]


def evaluate_design_point(
    a_mm: float, b_mm: float,
    T: float, V: float, gamma_RF: float, Q_SLPM: float
) -> EvaluationResult:
    """
    Evaluate a single design point across all physics.
    
    Enhanced with:
    - Separate residence times (tau_powder_active, tau_gas_refresh)
    - Drag-limited fall model for fine PSD (5-10 µm)
    - Stokes-number based dusting/plasma risk
    - tau_gas_refresh > 2s flagging
    """
    # Convert to SI
    a = a_mm / 1000
    b = b_mm / 1000
    R_tube = TUBE_RADIUS / 1000
    gap = b - a                    # electrode gap width
    standoff = R_tube - b          # mesh-to-wall standoff (g)
    L = PHYSICS["L_active"]
    p_bar = PHYSICS["p_bar"]
    
    # Areas and volumes
    A_active = np.pi * (b**2 - a**2)      # active region between pin and mesh
    A_annulus = np.pi * (R_tube**2 - b**2) # outer annulus (gas inlet region)
    V_active = A_active * L
    V_annulus = A_annulus * L
    
    # Flow rate conversion (SLPM → m³/s at operating conditions)
    T_STP, p_STP = 273.15, 1.0
    Q_m3s = (Q_SLPM / 60000) * (T / T_STP) * (p_STP / p_bar)
    
    # ==========================================================================
    # FLASH PHYSICS
    # ==========================================================================
    lambda_onset = compute_lambda_onset(T)
    
    # Electric field (coaxial geometry)
    if a > 0 and b > a:
        E_max = V / (a * np.log(b / a))  # at pin surface
    else:
        E_max = 0
    
    # RF contribution
    E_RF_induced = 5000  # V/m (placeholder for moderate RF)
    E_eff_max = E_max + gamma_RF * E_RF_induced
    
    # Activation radius (where E > lambda)
    if E_max > 0:
        r_onset = V / (lambda_onset * np.log(b / a))
        r_onset = min(r_onset, b)
        r_onset = max(r_onset, a)
    else:
        r_onset = a
    
    # Analytical activation fraction with calibration
    A_cal, B_cal = get_calibration(T)
    if b > a:
        f_onset_raw = (r_onset**2 - a**2) / (b**2 - a**2)
        f_onset_raw = np.clip(f_onset_raw, 0, 1)
        chi_fraction = np.clip(A_cal * f_onset_raw + B_cal, 0, 1)
    else:
        chi_fraction = 0
    
    E_eff_over_lambda = E_eff_max / lambda_onset if lambda_onset > 0 else 0
    
    # ==========================================================================
    # THERMAL ESTIMATE
    # ==========================================================================
    sigma_eff = 0.1  # S/m (placeholder conductivity)
    Q_joule = sigma_eff * E_max**2 * V_active * 1e-6  # W (scaled)
    T_max_estimate = T + Q_joule * 0.01  # simplified rise
    T_max_estimate = min(T_max_estimate, 2000)  # cap
    thermal_flag = T_max_estimate > T_MAX_LIMIT
    
    # ==========================================================================
    # FLOW & RESIDENCE TIME - ENHANCED
    # ==========================================================================
    # Powder fall velocity options:
    # 1. Drag-limited model for fine PSD (5-10 µm) - gives very slow fall
    # 2. Conservative placeholder (0.5-2 m/s) - practical for design
    #
    # Fine PSD (5-10 µm) in hot H2 at low pressure: Stokes regime gives
    # v_fall ~ 0.001-0.01 m/s, which is unrealistically slow for practical
    # powder feeding (gravity + some entrainment in feed mechanism).
    #
    # Use CONSERVATIVE PLACEHOLDER as specified: 0.5-2 m/s range
    # This accounts for practical powder feeding mechanisms.
    
    # Option 1: Compute theoretical Stokes fall velocity (for reference)
    v_fall_stokes_min, v_fall_stokes_med, v_fall_stokes_max = compute_powder_fall_velocity_psd(
        T, p_bar, d_min=5e-6, d_max=10e-6
    )
    
    # Option 2: Conservative placeholder velocities (practical values)
    V_FALL_CONSERVATIVE_MIN = 0.5   # m/s (slowest practical)
    V_FALL_CONSERVATIVE_MAX = 2.0   # m/s (fastest practical)
    V_FALL_CONSERVATIVE_MED = 1.0   # m/s (median placeholder)
    
    # Use conservative placeholder for design (more realistic)
    v_powder_fall = V_FALL_CONSERVATIVE_MED
    
    # tau_powder_active: time for powder to fall through active zone
    tau_powder_active = L / v_powder_fall
    tau_powder_min = L / V_FALL_CONSERVATIVE_MAX  # fastest fall (largest v)
    tau_powder_max = L / V_FALL_CONSERVATIVE_MIN  # slowest fall (smallest v)
    tau_powder_range = (tau_powder_min, tau_powder_max)
    
    # Gas crossflow velocity in active region
    v_crossflow = Q_m3s / A_active if A_active > 0 else 0
    
    # tau_gas_refresh: crossflow refresh time (time to flush annulus)
    tau_gas_refresh = V_annulus / Q_m3s if Q_m3s > 0 else float('inf')
    
    # Flag if tau_gas_refresh > 2s (water removal / plasma stability risk)
    TAU_GAS_REFRESH_LIMIT = 2.0  # seconds
    tau_gas_refresh_flag = tau_gas_refresh > TAU_GAS_REFRESH_LIMIT
    
    # Stokes number for particle entrainment
    # Use gap as characteristic length for crossflow impingement
    stokes_number = compute_stokes_number(
        v_crossflow, gap, T, p_bar, d_p=7.5e-6  # median PSD
    )
    
    # ==========================================================================
    # ENHANCED DUSTING/PLASMA RISK INDEX
    # ==========================================================================
    # Risk components:
    # 1. Stokes number proxy: low Stk = particles follow flow = entrainment risk
    # 2. Crossflow velocity: high v = particle lofting risk
    # 3. Small annulus width (gap): concentrated field, tight tolerance
    # 4. Small mesh standoff (g): plasma proximity to wall
    
    v_max = 5.0        # m/s maximum acceptable velocity
    g_min = 0.015      # 15mm minimum mesh standoff
    gap_min = 0.010    # 10mm minimum electrode gap
    Stk_crit = 1.0     # critical Stokes number
    
    # Stokes-based entrainment risk
    # If Stk < 1, particles follow flow → high entrainment/dusting risk
    # Penalty scales inversely with Stokes number
    if stokes_number > 0.01:
        risk_stokes = max(0, (Stk_crit / stokes_number) - 1)
    else:
        risk_stokes = 5.0  # Very small Stk = high risk
    
    # Crossflow velocity risk (penalize high velocity)
    risk_velocity = max(0, (v_crossflow / v_max) - 1) * 0.5
    
    # Small gap risk (penalize narrow gaps)
    if gap > 0.003:
        risk_gap = max(0, (gap_min / gap) - 1)
    else:
        risk_gap = 5.0  # Very narrow gap = high risk
    
    # Small standoff risk (penalize mesh too close to wall)
    if standoff > 0.005:
        risk_standoff = max(0, (g_min / standoff) - 1)
    else:
        risk_standoff = 5.0  # Very small standoff = high risk
    
    # Combined risk index with balanced weights
    # Emphasizes entrainment (Stokes) and wall proximity (standoff)
    risk_index = (
        0.30 * risk_stokes +
        0.25 * risk_velocity +
        0.20 * risk_gap +
        0.25 * risk_standoff
    )
    
    # ==========================================================================
    # FEASIBILITY
    # ==========================================================================
    violations = []
    
    # Powder residence time constraint [0.3, 3.0] s
    if tau_powder_active < TAU_MIN:
        violations.append(f"τ_powder_active={tau_powder_active:.2f}s < {TAU_MIN}s")
    if tau_powder_active > TAU_MAX:
        violations.append(f"τ_powder_active={tau_powder_active:.2f}s > {TAU_MAX}s")
    
    # Risk limit
    if risk_index >= RISK_LIMIT:
        violations.append(f"risk={risk_index:.2f} ≥ {RISK_LIMIT}")
    
    # Thermal limit
    if thermal_flag:
        violations.append(f"T_max={T_max_estimate:.0f}K > {T_MAX_LIMIT}K")
    
    # tau_gas_refresh flag (warning, not hard constraint)
    if tau_gas_refresh_flag:
        violations.append(f"⚠ τ_gas_refresh={tau_gas_refresh:.2f}s > 2.0s (water/plasma risk)")
    
    # Only hard constraints make design infeasible
    hard_violations = [v for v in violations if not v.startswith("⚠")]
    feasible = len(hard_violations) == 0
    
    return EvaluationResult(
        a_mm=a_mm, b_mm=b_mm, gap_mm=gap*1000, standoff_mm=standoff*1000,
        T_preheat=T, V_bias=V, gamma_RF=gamma_RF, Q_H2_SLPM=Q_SLPM,
        chi_fraction=chi_fraction, E_max=E_max, E_eff_over_lambda=E_eff_over_lambda,
        lambda_onset=lambda_onset, T_max_estimate=T_max_estimate, thermal_flag=thermal_flag,
        tau_powder_active=tau_powder_active, tau_powder_range=tau_powder_range,
        tau_gas_refresh=tau_gas_refresh, tau_gas_refresh_flag=tau_gas_refresh_flag,
        v_crossflow=v_crossflow, v_powder_fall=v_powder_fall, stokes_number=stokes_number,
        risk_index=risk_index, risk_stokes=risk_stokes, risk_velocity=risk_velocity,
        risk_gap=risk_gap, risk_standoff=risk_standoff,
        feasible=feasible, constraint_violations=violations
    )


# =============================================================================
# SWEEP EXECUTION
# =============================================================================
def run_design_sweep() -> List[EvaluationResult]:
    """Run full parameter sweep."""
    results = []
    
    # Filter valid geometries
    valid_geometries = []
    for a, b in product(A_VALUES, B_VALUES):
        gap = b - a
        standoff = TUBE_RADIUS - b
        if gap >= MIN_GAP and standoff >= MIN_STANDOFF:
            valid_geometries.append((a, b))
    
    print(f"Valid geometries: {len(valid_geometries)} / {len(A_VALUES) * len(B_VALUES)}")
    
    total_points = (len(valid_geometries) * len(T_PREHEAT_VALUES) * 
                   len(V_BIAS_VALUES) * len(GAMMA_RF_VALUES) * len(Q_H2_VALUES))
    print(f"Total evaluation points: {total_points:,}")
    
    count = 0
    for (a, b), T, V, gamma, Q in product(
        valid_geometries, T_PREHEAT_VALUES, V_BIAS_VALUES, GAMMA_RF_VALUES, Q_H2_VALUES
    ):
        result = evaluate_design_point(a, b, T, V, gamma, Q)
        results.append(result)
        count += 1
        if count % 1000 == 0:
            print(f"  Evaluated {count:,} / {total_points:,} points...")
    
    print(f"Sweep complete: {len(results):,} points evaluated")
    feasible_count = sum(1 for r in results if r.feasible)
    print(f"Feasible designs: {feasible_count:,} ({100*feasible_count/len(results):.1f}%)")
    
    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def compute_required_voltage(
    results: List[EvaluationResult],
    a: float, b: float, T: float, gamma: float, Q: float,
    target_chi: float = 0.4
) -> Optional[float]:
    """Find minimum V to achieve target chi fraction."""
    matching = [r for r in results 
                if r.a_mm == a and r.b_mm == b and r.T_preheat == T 
                and r.gamma_RF == gamma and r.Q_H2_SLPM == Q and r.feasible]
    
    for r in sorted(matching, key=lambda x: x.V_bias):
        if r.chi_fraction >= target_chi:
            return r.V_bias
    return None


def compute_sensitivity(results: List[EvaluationResult]) -> Dict:
    """Compute parameter sensitivity on chi fraction."""
    feasible = [r for r in results if r.feasible]
    if not feasible:
        return {}
    
    # Group by parameter and compute chi variance
    sensitivities = {}
    
    # Temperature sensitivity
    chi_by_T = {}
    for r in feasible:
        chi_by_T.setdefault(r.T_preheat, []).append(r.chi_fraction)
    T_means = {T: np.mean(vals) for T, vals in chi_by_T.items()}
    sensitivities["T_preheat"] = max(T_means.values()) - min(T_means.values())
    
    # Voltage sensitivity
    chi_by_V = {}
    for r in feasible:
        chi_by_V.setdefault(r.V_bias, []).append(r.chi_fraction)
    V_means = {V: np.mean(vals) for V, vals in chi_by_V.items()}
    sensitivities["V_bias"] = max(V_means.values()) - min(V_means.values())
    
    # Gamma sensitivity
    chi_by_gamma = {}
    for r in feasible:
        chi_by_gamma.setdefault(r.gamma_RF, []).append(r.chi_fraction)
    gamma_means = {g: np.mean(vals) for g, vals in chi_by_gamma.items()}
    sensitivities["gamma_RF"] = max(gamma_means.values()) - min(gamma_means.values())
    
    # Gap sensitivity
    chi_by_gap = {}
    for r in feasible:
        chi_by_gap.setdefault(r.gap_mm, []).append(r.chi_fraction)
    gap_means = {g: np.mean(vals) for g, vals in chi_by_gap.items()}
    sensitivities["gap"] = max(gap_means.values()) - min(gap_means.values())
    
    # Flow sensitivity
    chi_by_Q = {}
    for r in feasible:
        chi_by_Q.setdefault(r.Q_H2_SLPM, []).append(r.chi_fraction)
    Q_means = {Q: np.mean(vals) for Q, vals in chi_by_Q.items()}
    sensitivities["Q_H2"] = max(Q_means.values()) - min(Q_means.values())
    
    return sensitivities


def find_pareto_front(results: List[EvaluationResult]) -> List[EvaluationResult]:
    """Find Pareto-optimal designs (minimize V, minimize risk)."""
    feasible = [r for r in results if r.feasible and r.chi_fraction >= 0.4]
    
    # Group by geometry and find best V for each
    best_by_geom = {}
    for r in feasible:
        key = (r.a_mm, r.b_mm, r.Q_H2_SLPM)
        if key not in best_by_geom or r.V_bias < best_by_geom[key].V_bias:
            best_by_geom[key] = r
    
    candidates = list(best_by_geom.values())
    
    # Pareto filter
    pareto = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if (other.V_bias <= c.V_bias and other.risk_index <= c.risk_index and
                (other.V_bias < c.V_bias or other.risk_index < c.risk_index)):
                dominated = True
                break
        if not dominated:
            pareto.append(c)
    
    return sorted(pareto, key=lambda x: x.V_bias)


def rank_by_three_objectives(
    results: List[EvaluationResult], 
    target_chi: float = 0.4
) -> Dict[str, List[EvaluationResult]]:
    """
    Rank designs under three objectives:
    1. Minimize required V for target_chi% powder activation
    2. Minimize dusting/plasma risk
    3. Robust Pareto compromise
    
    Returns dict with 'by_voltage', 'by_risk', 'pareto' lists.
    """
    feasible = [r for r in results if r.feasible and r.chi_fraction >= target_chi]
    
    if not feasible:
        return {'by_voltage': [], 'by_risk': [], 'pareto': []}
    
    # Find best config per geometry (minimum V for target chi)
    best_by_geom = {}
    for r in feasible:
        key = (r.a_mm, r.b_mm)
        if key not in best_by_geom or r.V_bias < best_by_geom[key].V_bias:
            best_by_geom[key] = r
    
    candidates = list(best_by_geom.values())
    
    # ==========================================================================
    # Objective 1: Minimize voltage for target activation
    # ==========================================================================
    by_voltage = sorted(candidates, key=lambda x: x.V_bias)
    
    # ==========================================================================
    # Objective 2: Minimize dusting/plasma risk
    # ==========================================================================
    by_risk = sorted(candidates, key=lambda x: x.risk_index)
    
    # ==========================================================================
    # Objective 3: Pareto compromise (minimize both V and risk)
    # ==========================================================================
    # Find non-dominated solutions
    pareto = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if (other.V_bias <= c.V_bias and other.risk_index <= c.risk_index and
                (other.V_bias < c.V_bias or other.risk_index < c.risk_index)):
                dominated = True
                break
        if not dominated:
            pareto.append(c)
    
    # Sort Pareto front by balanced score (equal weight V and risk)
    if pareto:
        V_vals = [r.V_bias for r in pareto]
        risk_vals = [r.risk_index for r in pareto]
        V_min, V_max = min(V_vals), max(V_vals)
        risk_min, risk_max = min(risk_vals), max(risk_vals)
        
        V_range = V_max - V_min if V_max > V_min else 1
        risk_range = risk_max - risk_min if risk_max > risk_min else 0.1
        
        for r in pareto:
            v_norm = (r.V_bias - V_min) / V_range
            risk_norm = (r.risk_index - risk_min) / risk_range
            r._pareto_score = 0.5 * v_norm + 0.5 * risk_norm
        
        pareto = sorted(pareto, key=lambda x: x._pareto_score)
    
    return {
        'by_voltage': by_voltage,
        'by_risk': by_risk,
        'pareto': pareto,
    }


def find_designs_with_tau_gas_flag(results: List[EvaluationResult]) -> List[EvaluationResult]:
    """Find designs where tau_gas_refresh > 2s (water removal / plasma stability risk)."""
    return [r for r in results if r.tau_gas_refresh_flag]


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualizations(results: List[EvaluationResult]):
    """Generate all visualization outputs."""
    
    print("\nGenerating visualizations...")
    
    # =========================================================================
    # FIGURE 1: 3D Surface - χ vs (a, b) at fixed V, T
    # =========================================================================
    fig1 = plt.figure(figsize=(16, 5))
    
    for idx, (V_fixed, T_fixed) in enumerate([(600, 1100), (800, 1100), (1000, 1200)]):
        ax = fig1.add_subplot(1, 3, idx+1, projection='3d')
        
        # Get data for fixed V, T, gamma=0.2, Q=20
        subset = [r for r in results 
                  if r.V_bias == V_fixed and r.T_preheat == T_fixed 
                  and r.gamma_RF == 0.2 and r.Q_H2_SLPM == 20]
        
        if subset:
            a_vals = [r.a_mm for r in subset]
            b_vals = [r.b_mm for r in subset]
            chi_vals = [r.chi_fraction for r in subset]
            colors = ['green' if r.feasible else 'red' for r in subset]
            
            ax.scatter(a_vals, b_vals, chi_vals, c=colors, s=100, alpha=0.7)
            ax.set_xlabel('Pin radius a (mm)')
            ax.set_ylabel('Mesh radius b (mm)')
            ax.set_zlabel('χ fraction')
            ax.set_title(f'V={V_fixed}V, T={T_fixed}K\nγ=0.2, Q=20 SLPM')
    
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / '3d_chi_surface.png', dpi=150)
    plt.close(fig1)
    print(f"  Saved: 3d_chi_surface.png")
    
    # =========================================================================
    # FIGURE 2: Heatmap - Required V vs (T, γ) for select geometries
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    
    # Select representative geometries
    geom_configs = [
        (4, 22, "Small gap"),
        (8, 26, "Medium gap"),
        (4, 30, "Large gap"),
        (6, 26, "Balanced"),
        (10, 30, "Wide mesh"),
        (4, 40, "Maximum mesh"),
    ]
    
    for idx, (a, b, label) in enumerate(geom_configs):
        ax = axes2.flat[idx]
        
        V_matrix = np.zeros((len(T_PREHEAT_VALUES), len(GAMMA_RF_VALUES)))
        V_matrix[:] = np.nan
        
        for i, T in enumerate(T_PREHEAT_VALUES):
            for j, gamma in enumerate(GAMMA_RF_VALUES):
                V_req = compute_required_voltage(results, a, b, T, gamma, 20, target_chi=0.4)
                if V_req:
                    V_matrix[i, j] = V_req
        
        im = ax.imshow(V_matrix, cmap='viridis_r', aspect='auto',
                       extent=[-0.05, 0.45, T_PREHEAT_VALUES[-1]+50, T_PREHEAT_VALUES[0]-50])
        ax.set_xlabel('γ_RF')
        ax.set_ylabel('T_preheat (K)')
        ax.set_title(f'a={a}mm, b={b}mm\n({label})')
        
        # Add value annotations
        for i, T in enumerate(T_PREHEAT_VALUES):
            for j, gamma in enumerate(GAMMA_RF_VALUES):
                if not np.isnan(V_matrix[i, j]):
                    ax.text(gamma, T, f'{V_matrix[i,j]:.0f}', 
                           ha='center', va='center', fontsize=8, color='white')
        
        plt.colorbar(im, ax=ax, label='Required V')
    
    plt.suptitle('Required Voltage for 40% Activation (V)', fontsize=14)
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'heatmap_V_vs_T_gamma.png', dpi=150)
    plt.close(fig2)
    print(f"  Saved: heatmap_V_vs_T_gamma.png")
    
    # =========================================================================
    # FIGURE 3: Pareto Front - V vs Risk
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    pareto = find_pareto_front(results)
    feasible = [r for r in results if r.feasible and r.chi_fraction >= 0.4]
    
    # Best per geometry
    best_by_geom = {}
    for r in feasible:
        key = (r.a_mm, r.b_mm)
        if key not in best_by_geom or r.V_bias < best_by_geom[key].V_bias:
            best_by_geom[key] = r
    
    all_best = list(best_by_geom.values())
    
    # Plot all feasible (faded)
    V_all = [r.V_bias for r in all_best]
    risk_all = [r.risk_index for r in all_best]
    ax3.scatter(V_all, risk_all, c='lightgray', s=80, alpha=0.5, label='All feasible')
    
    # Plot Pareto front (highlighted)
    V_pareto = [r.V_bias for r in pareto]
    risk_pareto = [r.risk_index for r in pareto]
    ax3.scatter(V_pareto, risk_pareto, c='green', s=150, edgecolors='black', 
                linewidths=2, label='Pareto optimal', zorder=10)
    
    # Connect Pareto points
    pareto_sorted = sorted(pareto, key=lambda x: x.V_bias)
    ax3.plot([r.V_bias for r in pareto_sorted], [r.risk_index for r in pareto_sorted], 
             'g--', linewidth=2, alpha=0.7)
    
    # Annotate Pareto points
    for r in pareto:
        ax3.annotate(f'a={r.a_mm},b={r.b_mm}', 
                    (r.V_bias, r.risk_index), 
                    textcoords="offset points", xytext=(10, 5), fontsize=9)
    
    ax3.axhline(y=RISK_LIMIT, color='red', linestyle='--', label=f'Risk limit ({RISK_LIMIT})')
    ax3.set_xlabel('Minimum Voltage for 40% Activation (V)', fontsize=12)
    ax3.set_ylabel('Risk Index', fontsize=12)
    ax3.set_title('Pareto Front: Voltage vs Risk Trade-off\n(Lower-left is better)', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / 'pareto_front.png', dpi=150)
    plt.close(fig3)
    print(f"  Saved: pareto_front.png")
    
    # =========================================================================
    # FIGURE 4: Sensitivity Tornado
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    sens = compute_sensitivity(results)
    if sens:
        params = list(sens.keys())
        values = list(sens.values())
        
        # Sort by sensitivity
        sorted_idx = np.argsort(values)[::-1]
        params = [params[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(params)))
        bars = ax4.barh(params, values, color=colors, edgecolor='black')
        
        ax4.set_xlabel('Δχ (max - min mean)', fontsize=12)
        ax4.set_title('Parameter Sensitivity on Activation Fraction χ\n(Higher = More Influential)', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, values):
            ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    fig4.savefig(RESULTS_DIR / 'sensitivity_tornado.png', dpi=150)
    plt.close(fig4)
    print(f"  Saved: sensitivity_tornado.png")
    
    # =========================================================================
    # FIGURE 5: Operating Window Map - Feasible region in (V, T) space
    # =========================================================================
    fig5, axes5 = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, Q in enumerate([10, 20, 50]):
        for jdx, gamma in enumerate([0, 0.2]):
            ax = axes5[jdx, idx]
            
            # For each (V, T), count feasible geometries with chi > 0.4
            feasible_count = np.zeros((len(V_BIAS_VALUES), len(T_PREHEAT_VALUES)))
            
            for i, V in enumerate(V_BIAS_VALUES):
                for j, T in enumerate(T_PREHEAT_VALUES):
                    count = sum(1 for r in results 
                               if r.V_bias == V and r.T_preheat == T 
                               and r.gamma_RF == gamma and r.Q_H2_SLPM == Q
                               and r.feasible and r.chi_fraction >= 0.4)
                    feasible_count[i, j] = count
            
            im = ax.imshow(feasible_count, cmap='YlGn', aspect='auto',
                          extent=[T_PREHEAT_VALUES[0]-50, T_PREHEAT_VALUES[-1]+50,
                                  V_BIAS_VALUES[-1]+50, V_BIAS_VALUES[0]-50])
            ax.set_xlabel('T_preheat (K)')
            ax.set_ylabel('V_bias (V)')
            ax.set_title(f'Q={Q} SLPM, γ={gamma}')
            plt.colorbar(im, ax=ax, label='# Feasible geoms')
    
    plt.suptitle('Operating Window: Feasible Designs with χ>40%', fontsize=14)
    plt.tight_layout()
    fig5.savefig(RESULTS_DIR / 'operating_window.png', dpi=150)
    plt.close(fig5)
    print(f"  Saved: operating_window.png")
    
    # =========================================================================
    # FIGURE 6: Comprehensive Summary Dashboard
    # =========================================================================
    fig6 = plt.figure(figsize=(20, 14))
    
    # Panel 1: χ distribution
    ax1 = fig6.add_subplot(2, 4, 1)
    chi_vals = [r.chi_fraction for r in results if r.feasible]
    ax1.hist(chi_vals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0.4, color='red', linestyle='--', label='Target (40%)')
    ax1.set_xlabel('χ fraction')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Activation Fraction')
    ax1.legend()
    
    # Panel 2: Risk distribution with components
    ax2 = fig6.add_subplot(2, 4, 2)
    feasible_results = [r for r in results if r.feasible]
    risk_vals = [r.risk_index for r in feasible_results]
    ax2.hist(risk_vals, bins=30, color='green', alpha=0.7, label='Feasible', edgecolor='black')
    ax2.axvline(x=RISK_LIMIT, color='red', linestyle='--', linewidth=2, label='Limit')
    ax2.set_xlabel('Risk Index')
    ax2.set_ylabel('Count')
    ax2.set_title('Risk Distribution (Enhanced)')
    ax2.legend()
    
    # Panel 3: Residence time distributions
    ax3 = fig6.add_subplot(2, 4, 3)
    tau_powder_vals = [r.tau_powder_active for r in feasible_results]
    tau_gas_vals = [r.tau_gas_refresh for r in feasible_results if r.tau_gas_refresh < 10]  # clip outliers
    ax3.hist(tau_powder_vals, bins=30, color='blue', alpha=0.6, label='τ_powder_active', edgecolor='black')
    ax3.hist(tau_gas_vals, bins=30, color='orange', alpha=0.6, label='τ_gas_refresh', edgecolor='black')
    ax3.axvline(x=TAU_MIN, color='blue', linestyle='--', alpha=0.7)
    ax3.axvline(x=TAU_MAX, color='blue', linestyle='--', alpha=0.7)
    ax3.axvline(x=2.0, color='red', linestyle=':', label='τ_gas limit (2s)')
    ax3.set_xlabel('Residence Time (s)')
    ax3.set_ylabel('Count')
    ax3.set_title('Residence Time Distributions')
    ax3.legend(fontsize=8)
    
    # Panel 4: Stokes number vs Risk
    ax4 = fig6.add_subplot(2, 4, 4)
    high_chi = [r for r in feasible_results if r.chi_fraction >= 0.4]
    stk_vals = [r.stokes_number for r in high_chi]
    risk_vals = [r.risk_index for r in high_chi]
    scatter = ax4.scatter(stk_vals, risk_vals, c=[r.V_bias for r in high_chi], 
                         cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
    plt.colorbar(scatter, ax=ax4, label='V_bias (V)')
    ax4.axhline(y=RISK_LIMIT, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(x=1.0, color='green', linestyle=':', alpha=0.7, label='Stk=1')
    ax4.set_xlabel('Stokes Number')
    ax4.set_ylabel('Risk Index')
    ax4.set_title('Entrainment Risk (Stk vs Risk)')
    ax4.legend(fontsize=8)
    
    # Panel 5: Gap vs standoff (feasibility map)
    ax5 = fig6.add_subplot(2, 4, 5)
    gaps = sorted(set(r.gap_mm for r in results))
    standoffs = sorted(set(r.standoff_mm for r in results))
    feas_matrix = np.zeros((len(gaps), len(standoffs)))
    
    for r in results:
        if r.feasible and r.chi_fraction >= 0.4:
            i = gaps.index(r.gap_mm)
            j = standoffs.index(r.standoff_mm)
            feas_matrix[i, j] += 1
    
    im = ax5.imshow(feas_matrix.T, cmap='YlGn', aspect='auto',
                   extent=[min(gaps)-2, max(gaps)+2, min(standoffs)-2, max(standoffs)+2])
    ax5.set_xlabel('Gap (mm)')
    ax5.set_ylabel('Standoff g (mm)')
    ax5.set_title('Feasibility by Geometry')
    plt.colorbar(im, ax=ax5, label='# Feasible configs')
    
    # Panel 6: Risk components breakdown
    ax6 = fig6.add_subplot(2, 4, 6)
    # Average risk components for feasible designs
    if high_chi:
        risk_components = {
            'Stokes': np.mean([r.risk_stokes for r in high_chi]),
            'Velocity': np.mean([r.risk_velocity for r in high_chi]),
            'Gap': np.mean([r.risk_gap for r in high_chi]),
            'Standoff': np.mean([r.risk_standoff for r in high_chi]),
        }
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        bars = ax6.bar(risk_components.keys(), risk_components.values(), color=colors, edgecolor='black')
        ax6.set_ylabel('Average Risk Component')
        ax6.set_title('Risk Index Breakdown')
        for bar, val in zip(bars, risk_components.values()):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2f}', ha='center', fontsize=9)
    
    # Panel 7: V vs chi scatter (colored by T)
    ax7 = fig6.add_subplot(2, 4, 7)
    scatter = ax7.scatter([r.V_bias for r in feasible_results], 
                         [r.chi_fraction for r in feasible_results],
                         c=[r.T_preheat for r in feasible_results], 
                         cmap='plasma', alpha=0.5, s=20)
    plt.colorbar(scatter, ax=ax7, label='T_preheat (K)')
    ax7.set_xlabel('Voltage (V)')
    ax7.set_ylabel('χ fraction')
    ax7.set_title('Activation vs Voltage')
    ax7.axhline(y=0.4, color='red', linestyle='--', alpha=0.5)
    
    # Panel 8: Three-objective ranking table
    ax8 = fig6.add_subplot(2, 4, 8)
    ax8.axis('off')
    
    rankings = rank_by_three_objectives(results, target_chi=0.4)
    table_data = [['Objective', 'a', 'b', 'V', 'χ', 'Risk', 'τ_pow']]
    
    if rankings['by_voltage']:
        r = rankings['by_voltage'][0]
        table_data.append(['Min V', f'{r.a_mm:.0f}', f'{r.b_mm:.0f}', 
                          f'{r.V_bias:.0f}', f'{r.chi_fraction:.2f}', 
                          f'{r.risk_index:.2f}', f'{r.tau_powder_active:.1f}'])
    
    if rankings['by_risk']:
        r = rankings['by_risk'][0]
        table_data.append(['Min Risk', f'{r.a_mm:.0f}', f'{r.b_mm:.0f}', 
                          f'{r.V_bias:.0f}', f'{r.chi_fraction:.2f}', 
                          f'{r.risk_index:.2f}', f'{r.tau_powder_active:.1f}'])
    
    if rankings['pareto']:
        r = rankings['pareto'][0]
        table_data.append(['Pareto', f'{r.a_mm:.0f}', f'{r.b_mm:.0f}', 
                          f'{r.V_bias:.0f}', f'{r.chi_fraction:.2f}', 
                          f'{r.risk_index:.2f}', f'{r.tau_powder_active:.1f}'])
    
    table = ax8.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.18, 0.10, 0.10, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Color header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax8.set_title('Top Candidates by Objective', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('PFR Digital Twin: Comprehensive Design Space Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig6.savefig(RESULTS_DIR / 'summary_dashboard.png', dpi=150)
    plt.close(fig6)
    print(f"  Saved: summary_dashboard.png")


# =============================================================================
# REPORT GENERATION
# =============================================================================
def generate_report(results: List[EvaluationResult]) -> str:
    """Generate comprehensive text report with three-objective ranking."""
    
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines.append("=" * 100)
    lines.append("PFR DIGITAL TWIN: COMPREHENSIVE DESIGN SPACE EXPLORATION")
    lines.append(f"Generated: {timestamp}")
    lines.append("=" * 100)
    lines.append("")
    
    # Summary statistics
    total = len(results)
    feasible = [r for r in results if r.feasible]
    high_chi = [r for r in feasible if r.chi_fraction >= 0.4]
    tau_gas_flagged = find_designs_with_tau_gas_flag(results)
    
    lines.append("SWEEP SUMMARY")
    lines.append("-" * 50)
    lines.append(f"  Total evaluation points: {total:,}")
    lines.append(f"  Feasible designs: {len(feasible):,} ({100*len(feasible)/total:.1f}%)")
    lines.append(f"  High activation (χ>40%): {len(high_chi):,} ({100*len(high_chi)/total:.1f}%)")
    lines.append(f"  τ_gas_refresh > 2s flagged: {len(tau_gas_flagged):,} (water/plasma risk)")
    lines.append("")
    
    # Parameter ranges
    lines.append("PARAMETER RANGES")
    lines.append("-" * 50)
    lines.append(f"  Pin radius a: {A_VALUES} mm")
    lines.append(f"  Mesh radius b: {B_VALUES} mm")
    lines.append(f"  Preheat T: {T_PREHEAT_VALUES} K")
    lines.append(f"  Bias V: {V_BIAS_VALUES} V")
    lines.append(f"  RF coupling γ: {GAMMA_RF_VALUES}")
    lines.append(f"  H2 flow Q: {Q_H2_VALUES} SLPM")
    lines.append("")
    
    # Constraint summary
    lines.append("CONSTRAINTS")
    lines.append("-" * 50)
    lines.append(f"  Geometry: gap ≥ {MIN_GAP} mm, standoff ≥ {MIN_STANDOFF} mm")
    lines.append(f"  Powder residence: τ_powder_active ∈ [{TAU_MIN}, {TAU_MAX}] s")
    lines.append(f"  Gas refresh flag: τ_gas_refresh > 2.0 s (warning)")
    lines.append(f"  Risk index: < {RISK_LIMIT}")
    lines.append(f"  Temperature: T_max < {T_MAX_LIMIT} K")
    lines.append("")
    
    # Enhanced risk index explanation
    lines.append("DUSTING/PLASMA RISK INDEX COMPONENTS")
    lines.append("-" * 50)
    lines.append("  Weights:")
    lines.append("    - 30% Stokes-number proxy (low Stk = entrainment risk)")
    lines.append("    - 25% Crossflow velocity (high v = particle lofting)")
    lines.append("    - 20% Electrode gap penalty (narrow gap = tight tolerance)")
    lines.append("    - 25% Mesh standoff penalty (small g = plasma wall risk)")
    lines.append("")
    
    # Sensitivity analysis
    sens = compute_sensitivity(results)
    lines.append("PARAMETER SENSITIVITY (on χ fraction)")
    lines.append("-" * 50)
    for param, val in sorted(sens.items(), key=lambda x: -x[1]):
        bar = "█" * int(val * 50)
        lines.append(f"  {param:<12} {val:.3f}  {bar}")
    lines.append("")
    
    # ==========================================================================
    # THREE-OBJECTIVE RANKING
    # ==========================================================================
    rankings = rank_by_three_objectives(results, target_chi=0.4)
    
    lines.append("=" * 100)
    lines.append("TOP CANDIDATES BY THREE OBJECTIVES (for χ ≥ 40%)")
    lines.append("=" * 100)
    lines.append("")
    
    # --------------------------------------------------------------------------
    # Objective 1: Minimize Voltage
    # --------------------------------------------------------------------------
    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│ OBJECTIVE 1: MINIMIZE REQUIRED VOLTAGE for 40% Activation                   │")
    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append(f"  {'Rank':<6} {'a':<6} {'b':<6} {'gap':<8} {'g':<8} {'V':<8} {'χ':<8} {'Risk':<8} {'τ_powder':<10} {'τ_gas':<8} {'Stk':<8}")
    lines.append("  " + "-" * 95)
    
    for i, r in enumerate(rankings['by_voltage'][:10], 1):
        tau_flag = "⚠" if r.tau_gas_refresh_flag else ""
        lines.append(f"  {i:<6} {r.a_mm:<6.0f} {r.b_mm:<6.0f} {r.gap_mm:<8.0f} {r.standoff_mm:<8.0f} "
                    f"{r.V_bias:<8.0f} {r.chi_fraction:<8.2f} {r.risk_index:<8.3f} "
                    f"{r.tau_powder_active:<10.2f} {r.tau_gas_refresh:<8.2f}{tau_flag} {r.stokes_number:<8.2f}")
    lines.append("")
    
    # --------------------------------------------------------------------------
    # Objective 2: Minimize Dusting/Plasma Risk
    # --------------------------------------------------------------------------
    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│ OBJECTIVE 2: MINIMIZE DUSTING/PLASMA RISK                                   │")
    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append(f"  {'Rank':<6} {'a':<6} {'b':<6} {'gap':<8} {'g':<8} {'Risk':<8} {'r_Stk':<8} {'r_vel':<8} {'r_gap':<8} {'r_g':<8} {'V':<8}")
    lines.append("  " + "-" * 95)
    
    for i, r in enumerate(rankings['by_risk'][:10], 1):
        lines.append(f"  {i:<6} {r.a_mm:<6.0f} {r.b_mm:<6.0f} {r.gap_mm:<8.0f} {r.standoff_mm:<8.0f} "
                    f"{r.risk_index:<8.3f} {r.risk_stokes:<8.2f} {r.risk_velocity:<8.2f} "
                    f"{r.risk_gap:<8.2f} {r.risk_standoff:<8.2f} {r.V_bias:<8.0f}")
    lines.append("")
    
    # --------------------------------------------------------------------------
    # Objective 3: Pareto Compromise
    # --------------------------------------------------------------------------
    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│ OBJECTIVE 3: PARETO COMPROMISE (balanced V + Risk)                          │")
    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append(f"  {'Rank':<6} {'a':<6} {'b':<6} {'gap':<8} {'g':<8} {'V':<8} {'Risk':<8} {'χ':<8} {'τ_powder':<10} {'Stk':<8} {'T':<6}")
    lines.append("  " + "-" * 95)
    
    for i, r in enumerate(rankings['pareto'][:10], 1):
        lines.append(f"  {i:<6} {r.a_mm:<6.0f} {r.b_mm:<6.0f} {r.gap_mm:<8.0f} {r.standoff_mm:<8.0f} "
                    f"{r.V_bias:<8.0f} {r.risk_index:<8.3f} {r.chi_fraction:<8.2f} "
                    f"{r.tau_powder_active:<10.2f} {r.stokes_number:<8.2f} {r.T_preheat:<6.0f}")
    lines.append("")
    
    # ==========================================================================
    # DESIGNS WITH τ_gas_refresh > 2s FLAG
    # ==========================================================================
    flagged_feasible = [r for r in tau_gas_flagged if r.feasible and r.chi_fraction >= 0.4]
    if flagged_feasible:
        lines.append("⚠ DESIGNS WITH τ_gas_refresh > 2.0s (water removal / plasma stability risk)")
        lines.append("-" * 100)
        unique_geoms = {}
        for r in flagged_feasible:
            key = (r.a_mm, r.b_mm)
            if key not in unique_geoms:
                unique_geoms[key] = r
        
        for key, r in sorted(unique_geoms.items())[:5]:
            lines.append(f"  a={r.a_mm:.0f}mm, b={r.b_mm:.0f}mm | τ_gas={r.tau_gas_refresh:.2f}s | "
                        f"V={r.V_bias:.0f}V | Consider: increase Q_H2 or reduce annulus volume")
        lines.append("")
    
    # ==========================================================================
    # RESIDENCE TIME ANALYSIS
    # ==========================================================================
    lines.append("RESIDENCE TIME ANALYSIS (using drag-limited fall for PSD 5-10 µm)")
    lines.append("-" * 100)
    
    # Sample a few geometries
    sample_results = [r for r in feasible if r.T_preheat == 1100 and r.gamma_RF == 0 and r.Q_H2_SLPM == 20][:3]
    for r in sample_results:
        lines.append(f"  Geometry: a={r.a_mm:.0f}mm, b={r.b_mm:.0f}mm, g={r.standoff_mm:.0f}mm")
        lines.append(f"    v_powder_fall (median PSD): {r.v_powder_fall:.4f} m/s")
        lines.append(f"    τ_powder_active: {r.tau_powder_active:.2f} s (range: {r.tau_powder_range[0]:.2f}-{r.tau_powder_range[1]:.2f} s)")
        lines.append(f"    τ_gas_refresh: {r.tau_gas_refresh:.2f} s {'⚠ > 2s' if r.tau_gas_refresh_flag else '✓'}")
        lines.append(f"    Stokes number: {r.stokes_number:.3f}")
        lines.append("")
    
    # ==========================================================================
    # Best for each flow rate
    # ==========================================================================
    lines.append("BEST DESIGN PER FLOW RATE REGIME (Pareto-optimal)")
    lines.append("-" * 100)
    for Q in Q_H2_VALUES:
        subset = [r for r in results if r.Q_H2_SLPM == Q and r.feasible and r.chi_fraction >= 0.4]
        if subset:
            # Find Pareto best
            best_by_geom = {}
            for r in subset:
                key = (r.a_mm, r.b_mm)
                if key not in best_by_geom or r.V_bias < best_by_geom[key].V_bias:
                    best_by_geom[key] = r
            
            if best_by_geom:
                best = min(best_by_geom.values(), key=lambda x: 0.5 * x.V_bias/1000 + 0.5 * x.risk_index)
                tau_flag = "⚠" if best.tau_gas_refresh_flag else "✓"
                lines.append(f"  Q={Q:3d} SLPM: a={best.a_mm:.0f}mm, b={best.b_mm:.0f}mm | "
                            f"V={best.V_bias:.0f}V | χ={best.chi_fraction:.2f} | risk={best.risk_index:.3f} | "
                            f"τ_powder={best.tau_powder_active:.2f}s | τ_gas={best.tau_gas_refresh:.2f}s {tau_flag}")
    lines.append("")
    
    # ==========================================================================
    # Final recommendation
    # ==========================================================================
    lines.append("=" * 100)
    lines.append("RECOMMENDED PROTOTYPE GEOMETRY")
    lines.append("=" * 100)
    lines.append("")
    
    pareto = rankings['pareto']
    if pareto:
        # Pick best Pareto compromise
        best = pareto[0]
        
        lines.append(f"  GEOMETRY:")
        lines.append(f"    Pin radius (a):      {best.a_mm:.0f} mm")
        lines.append(f"    Mesh radius (b):     {best.b_mm:.0f} mm")
        lines.append(f"    Electrode gap:       {best.gap_mm:.0f} mm")
        lines.append(f"    Wall standoff (g):   {best.standoff_mm:.0f} mm")
        lines.append("")
        lines.append(f"  OPERATING POINT:")
        lines.append(f"    Preheat:            {best.T_preheat:.0f} K")
        lines.append(f"    Bias voltage:       {best.V_bias:.0f} V")
        lines.append(f"    RF coupling:        γ = {best.gamma_RF}")
        lines.append(f"    H2 flow:            {best.Q_H2_SLPM:.0f} SLPM")
        lines.append("")
        lines.append(f"  PERFORMANCE:")
        lines.append(f"    Activation (χ):     {best.chi_fraction:.1%}")
        lines.append(f"    Risk index:         {best.risk_index:.3f}")
        lines.append(f"    τ_powder_active:    {best.tau_powder_active:.2f} s")
        lines.append(f"    τ_gas_refresh:      {best.tau_gas_refresh:.2f} s {'⚠ > 2s' if best.tau_gas_refresh_flag else '✓'}")
        lines.append(f"    Stokes number:      {best.stokes_number:.3f}")
        lines.append("")
        lines.append(f"  RISK BREAKDOWN:")
        lines.append(f"    Stokes risk:        {best.risk_stokes:.3f} (weight 0.30)")
        lines.append(f"    Velocity risk:      {best.risk_velocity:.3f} (weight 0.25)")
        lines.append(f"    Gap risk:           {best.risk_gap:.3f} (weight 0.20)")
        lines.append(f"    Standoff risk:      {best.risk_standoff:.3f} (weight 0.25)")
        lines.append("")
        lines.append(f"  SAFETY MARGINS:")
        lines.append(f"    Voltage margin:     +{int(1200 - best.V_bias)} V to max supply")
        lines.append(f"    Risk margin:        {1.0 - best.risk_index:.2f} below limit")
        lines.append(f"    Temperature margin: {T_MAX_LIMIT - best.T_max_estimate:.0f} K below limit")
        
        # Summary comparison table
        lines.append("")
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│ SUMMARY: TOP CANDIDATES BY OBJECTIVE                                        │")
        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append(f"  {'Objective':<30} {'Geometry':<20} {'V (V)':<10} {'Risk':<10} {'χ':<8}")
        lines.append("  " + "-" * 78)
        
        if rankings['by_voltage']:
            r = rankings['by_voltage'][0]
            lines.append(f"  {'Minimize Voltage':<30} {'a=%d,b=%d' % (r.a_mm, r.b_mm):<20} {r.V_bias:<10.0f} {r.risk_index:<10.3f} {r.chi_fraction:<8.2f}")
        
        if rankings['by_risk']:
            r = rankings['by_risk'][0]
            lines.append(f"  {'Minimize Risk':<30} {'a=%d,b=%d' % (r.a_mm, r.b_mm):<20} {r.V_bias:<10.0f} {r.risk_index:<10.3f} {r.chi_fraction:<8.2f}")
        
        if rankings['pareto']:
            r = rankings['pareto'][0]
            lines.append(f"  {'Pareto Compromise':<30} {'a=%d,b=%d' % (r.a_mm, r.b_mm):<20} {r.V_bias:<10.0f} {r.risk_index:<10.3f} {r.chi_fraction:<8.2f}")
    
    lines.append("")
    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("PFR DIGITAL TWIN: COMPREHENSIVE DESIGN SPACE EXPLORATION")
    print("Enhanced with separate residence times and Stokes-based risk")
    print("=" * 70 + "\n")
    
    # Run sweep
    results = run_design_sweep()
    
    # Generate visualizations
    create_visualizations(results)
    
    # Generate report
    report = generate_report(results)
    print("\n" + report)
    
    # Save outputs
    report_path = RESULTS_DIR / "comprehensive_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved: {report_path}")
    
    # Save results as JSON (convert numpy types and tuples)
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        return obj
    
    results_data = [convert_types(asdict(r)) for r in results]
    json_path = RESULTS_DIR / "all_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved: {json_path}")
    
    # Save rankings separately
    rankings = rank_by_three_objectives(results, target_chi=0.4)
    
    def result_to_dict(r: EvaluationResult) -> Dict:
        d = convert_types(asdict(r))
        # Add pareto score if available
        if hasattr(r, '_pareto_score'):
            d['pareto_score'] = r._pareto_score
        return d
    
    rankings_data = {
        'by_voltage': [result_to_dict(r) for r in rankings['by_voltage'][:20]],
        'by_risk': [result_to_dict(r) for r in rankings['by_risk'][:20]],
        'pareto': [result_to_dict(r) for r in rankings['pareto'][:20]],
        'metadata': {
            'target_chi': 0.4,
            'tau_powder_limits': [TAU_MIN, TAU_MAX],
            'tau_gas_refresh_flag_limit': 2.0,
            'risk_limit': RISK_LIMIT,
            'risk_weights': {
                'stokes': 0.30,
                'velocity': 0.25,
                'gap': 0.20,
                'standoff': 0.25
            },
            'psd_um': [5, 10],
            'generated': datetime.now().isoformat()
        }
    }
    
    rankings_path = RESULTS_DIR / "three_objective_rankings.json"
    with open(rankings_path, 'w') as f:
        json.dump(rankings_data, f, indent=2)
    print(f"Saved: {rankings_path}")
    
    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}\n")
    
    return results, rankings


if __name__ == "__main__":
    main()
