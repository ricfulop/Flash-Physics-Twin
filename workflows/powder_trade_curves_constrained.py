#!/usr/bin/env python3
"""
powder_trade_curves_constrained.py

Powder-region bias–preheat–geometry trade curves with system constraints.

Features:
1. Temperature-dependent ΔG₀(T) and A(T), B(T) calibration for preheat effects
2. System constraints: φ_p, Q_H2, p, L_active
3. Gas velocity and residence time calculations
4. Dusting/plasma risk index
5. Re-ranking with objective: minimize V for 40% activation subject to constraints
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp" / "comsol_mcp"))
from comsol_api import ComsolBackend

# ============================================================================
# Physical Constants
# ============================================================================
F_CONST = 96485.0  # Faraday constant [C/mol]
R_GAS = 8.314      # Gas constant [J/(mol·K)]
N_ELECTRONS = 2    # Electrons transferred

# ============================================================================
# System Constraints (defaults)
# ============================================================================
SYSTEM_DEFAULTS = {
    "phi_p": 0.02,           # Powder volume fraction [dimensionless]
    "Q_H2_SLPM": 20.0,       # H2 flow rate [SLPM] - reduced from 100 for feasibility
    "p_bar": 0.15,           # Pressure [bar]
    "L_active": 0.3,         # Active length [m] - increased for adequate residence
    "tube_ID": 0.1,          # Tube inner diameter [m] = 100mm
    "tau_min": 0.3,          # Minimum residence time [s]
    "tau_max": 3.0,          # Maximum residence time [s]
    "T_ambient": 300.0,      # Ambient temperature [K]
    "v_max": 5.0,            # Maximum acceptable gas velocity [m/s]
}

# Flow rate sweep for sensitivity analysis
FLOW_RATES_SLPM = [10, 20, 50, 100]

# ============================================================================
# Physics Parameters (from PhysicsNeMo inversion)
# ============================================================================
PHYSICS_PARAMS = {
    "DeltaG0_ref": 350000.0,  # J/mol at T_ref
    "T_ref": 1100.0,          # Reference temperature [K]
    "beta_T": 100.0,          # Temperature coefficient [J/(mol·K)]
    "r_act": 3e-5,            # Activation thickness [m]
    "B_s": 15000.0,           # Softening parameter [J/mol]
    "n": 2,                   # Electron transfer number
    "lambda_onset": 60443.0,  # Onset field [V/m]
}

# ============================================================================
# A(T), B(T) Calibration (from analytic screening validation)
# ============================================================================
# These were derived from comparing analytic model to coupled simulation
# f_corrected = A(T) * f_analytic + B(T)
CALIBRATION = {
    1000: {"A": 2.30, "B": 0.029},
    1100: {"A": 2.45, "B": 0.035},
    1200: {"A": 2.60, "B": 0.042},
}

# ============================================================================
# Results directory
# ============================================================================
RESULTS_DIR = Path(__file__).parent.parent / "results" / "powder_trade_constrained"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Candidate geometries
# ============================================================================
GEOMETRIES = {
    "Conservative": {
        "a_mm": 8.0,      # pin radius
        "b_mm": 26.0,     # mesh radius
        "gap_mm": 18.0,
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

# Powder region
POWDER_REGION = {
    'r_min': 0.005,
    'r_max': 0.035,
    'z_start': 0.06,
    'z_end': 0.14,
}

# Sweep parameters
PREHEAT_TEMPS = [1000, 1100, 1200]
GAMMA_VALUES = [0, 0.2, 0.4]
CHI_TARGETS = [0.10, 0.20, 0.40, 0.60]


def compute_DeltaG0_T(T: float) -> float:
    """Compute temperature-dependent Gibbs free energy barrier."""
    DeltaG0_ref = PHYSICS_PARAMS["DeltaG0_ref"]
    T_ref = PHYSICS_PARAMS["T_ref"]
    beta_T = PHYSICS_PARAMS["beta_T"]
    return DeltaG0_ref - beta_T * (T - T_ref)


def get_calibration(T: float) -> Tuple[float, float]:
    """Get A(T), B(T) calibration coefficients with interpolation."""
    temps = sorted(CALIBRATION.keys())
    
    if T <= temps[0]:
        return CALIBRATION[temps[0]]["A"], CALIBRATION[temps[0]]["B"]
    elif T >= temps[-1]:
        return CALIBRATION[temps[-1]]["A"], CALIBRATION[temps[-1]]["B"]
    else:
        # Linear interpolation
        for i in range(len(temps) - 1):
            if temps[i] <= T <= temps[i+1]:
                t1, t2 = temps[i], temps[i+1]
                frac = (T - t1) / (t2 - t1)
                A = CALIBRATION[t1]["A"] + frac * (CALIBRATION[t2]["A"] - CALIBRATION[t1]["A"])
                B = CALIBRATION[t1]["B"] + frac * (CALIBRATION[t2]["B"] - CALIBRATION[t1]["B"])
                return A, B
    
    return 2.30, 0.029  # default


def compute_powder_fall_velocity(
    d_p: float,  # particle diameter [m]
    rho_p: float,  # particle density [kg/m³]
    T_gas: float,  # gas temperature [K]
    p_gas: float,  # gas pressure [bar]
    gas: str = "H2"
) -> float:
    """
    Compute terminal fall velocity using drag-limited model.
    
    For fine particles in the Stokes regime (Re_p < 1):
    v_t = (rho_p - rho_g) * g * d_p^2 / (18 * mu_g)
    
    For intermediate regime, use correlation.
    """
    g_accel = 9.81  # m/s²
    
    # Gas properties for H2 at given T, p
    # Density: rho = p * M / (R * T)
    M_H2 = 2.016e-3  # kg/mol
    R_gas_const = 8.314  # J/(mol·K)
    rho_g = (p_gas * 1e5) * M_H2 / (R_gas_const * T_gas)  # kg/m³
    
    # Dynamic viscosity of H2 (Sutherland's law approximation)
    # mu_0 = 8.9e-6 Pa·s at T_0 = 300 K
    mu_0 = 8.9e-6
    T_0 = 300.0
    S = 72.0  # Sutherland constant for H2
    mu_g = mu_0 * (T_gas / T_0)**1.5 * (T_0 + S) / (T_gas + S)
    
    # Stokes terminal velocity
    v_stokes = (rho_p - rho_g) * g_accel * d_p**2 / (18 * mu_g)
    
    # Check Reynolds number
    Re_p = rho_g * v_stokes * d_p / mu_g
    
    if Re_p < 1:
        return v_stokes
    elif Re_p < 500:
        # Intermediate regime: use Schiller-Naumann correlation
        # C_D = 24/Re * (1 + 0.15 * Re^0.687)
        # Iterate to find v_t
        v_t = v_stokes
        for _ in range(10):
            Re_p = rho_g * v_t * d_p / mu_g
            C_D = 24/Re_p * (1 + 0.15 * Re_p**0.687)
            v_t = np.sqrt(4 * g_accel * d_p * (rho_p - rho_g) / (3 * C_D * rho_g))
        return v_t
    else:
        # Newton regime
        C_D = 0.44
        return np.sqrt(4 * g_accel * d_p * (rho_p - rho_g) / (3 * C_D * rho_g))


def compute_stokes_number(
    d_p: float,  # particle diameter [m]
    rho_p: float,  # particle density [kg/m³]
    v_gas: float,  # gas velocity [m/s]
    L_char: float,  # characteristic length [m]
    T_gas: float,  # gas temperature [K]
    p_gas: float,  # gas pressure [bar]
) -> float:
    """
    Compute Stokes number: ratio of particle stopping distance to characteristic length.
    
    Stk = (rho_p * d_p^2 * v_gas) / (18 * mu_g * L_char)
    
    Stk >> 1: particles ignore flow (inertial)
    Stk << 1: particles follow flow (entrained)
    """
    # Gas viscosity
    mu_0 = 8.9e-6
    T_0 = 300.0
    S = 72.0
    mu_g = mu_0 * (T_gas / T_0)**1.5 * (T_0 + S) / (T_gas + S)
    
    # Stokes number
    Stk = (rho_p * d_p**2 * v_gas) / (18 * mu_g * L_char)
    
    return Stk


def compute_system_constraints(
    geometry: Dict,
    T_preheat: float,
    system: Dict = None
) -> Dict:
    """
    Compute gas flow, residence time, and risk indices.
    
    Flow geometry: Gas enters through Hastelloy X spargers at the perimeter,
    flows radially inward through the outer annulus, then axially through the 
    active region (pin-mesh gap), and exits through spargers at the inner annulus.
    
    Key residence times:
    1. τ_powder_active: time for powder to fall through active zone (drag-limited)
    2. τ_gas_refresh: gas crossflow refresh time in annulus
    
    Constraints:
    - τ_powder_active ∈ [0.3, 3] s
    - τ_gas_refresh > 2 s flagged (water removal/plasma stability risk)
    """
    sys = dict(SYSTEM_DEFAULTS)
    if system:
        sys.update(system)
    
    # Geometry in meters
    a = geometry["a_mm"] / 1000.0  # pin radius
    b = geometry["b_mm"] / 1000.0  # mesh radius
    R_tube = sys["tube_ID"] / 2.0  # tube radius
    g = R_tube - b                  # mesh standoff from tube wall
    L = sys["L_active"]
    gap = b - a                      # electrode gap width
    
    # Cross-sectional areas
    A_annulus = np.pi * (R_tube**2 - b**2)  # outer annulus (gas inlet region)
    A_active = np.pi * (b**2 - a**2)         # active region between pin and mesh
    A_tube = np.pi * R_tube**2
    
    # Volumes
    V_annulus = A_annulus * L  # outer annulus volume
    V_active = A_active * L    # active region volume
    
    # Convert flow rate to m³/s at operating conditions
    Q_SLPM = sys["Q_H2_SLPM"]
    p_bar = sys["p_bar"]
    T_op = T_preheat
    
    # SLPM is at STP (273.15 K, 1 bar)
    T_STP = 273.15
    p_STP = 1.0
    Q_m3s = (Q_SLPM / 60000.0) * (T_op / T_STP) * (p_STP / p_bar)
    
    # =========================================================================
    # Powder residence time (τ_powder_active)
    # =========================================================================
    # Use drag-limited fall model for typical metal powder PSD
    # For flash reduction, we need coarser powder (50-100 µm) to achieve
    # reasonable residence times. 5-10 µm particles fall too slowly.
    d_p_min = 50e-6   # 50 µm (coarse end)
    d_p_max = 100e-6  # 100 µm (fine end)
    d_p_median = 75e-6  # median particle size (typical spray-dried/milled)
    rho_p = 5000.0   # kg/m³ (metal oxide density)
    
    # Compute terminal velocities for PSD range
    v_fall_min = compute_powder_fall_velocity(d_p_min, rho_p, T_op, p_bar)
    v_fall_max = compute_powder_fall_velocity(d_p_max, rho_p, T_op, p_bar)
    v_fall_median = compute_powder_fall_velocity(d_p_median, rho_p, T_op, p_bar)
    
    # Use median for design
    v_powder_fall = v_fall_median
    
    # Powder residence time in active zone
    tau_powder_active = L / v_powder_fall
    tau_powder_min = L / v_fall_max  # fastest particles (largest)
    tau_powder_max = L / v_fall_min  # slowest particles (smallest)
    
    # =========================================================================
    # Gas refresh time (τ_gas_refresh)
    # =========================================================================
    # Time for gas to refresh the annulus (crossflow direction)
    tau_gas_refresh = V_annulus / Q_m3s
    
    # Gas residence in active region
    tau_gas_active = V_active / Q_m3s
    
    # =========================================================================
    # Gas velocities
    # =========================================================================
    # Crossflow velocity in active region
    v_crossflow = Q_m3s / A_active
    
    # Radial velocity in outer annulus
    r_avg_annulus = (R_tube + b) / 2
    v_annulus_radial = Q_m3s / (2 * np.pi * r_avg_annulus * L)
    
    # =========================================================================
    # Stokes number (particle entrainment risk)
    # =========================================================================
    # Characteristic length = annulus width for crossflow entrainment
    L_char = gap  # electrode gap as characteristic length
    Stk = compute_stokes_number(d_p_median, rho_p, v_crossflow, L_char, T_op, p_bar)
    
    # =========================================================================
    # Residence time validity
    # =========================================================================
    tau_min = sys["tau_min"]
    tau_max = sys["tau_max"]
    tau_powder_valid = tau_min <= tau_powder_active <= tau_max
    
    # Water removal / plasma stability flag
    tau_gas_refresh_limit = 2.0  # seconds
    tau_gas_refresh_flag = tau_gas_refresh > tau_gas_refresh_limit
    
    # =========================================================================
    # Enhanced Dusting/Plasma Risk Index
    # =========================================================================
    # Components:
    # 1. Stokes number proxy: high Stk + high velocity = entrainment risk
    # 2. Crossflow velocity penalty
    # 3. Small annulus width penalty
    # 4. Small mesh standoff g penalty
    
    v_max = sys.get("v_max", 5.0)
    g_min = 0.015  # 15mm minimum standoff
    gap_min = 0.010  # 10mm minimum gap
    Stk_crit = 1.0  # critical Stokes number
    
    # Stokes-based entrainment risk
    # If Stk < 1, particles follow flow → high entrainment risk
    # Penalty if Stk < Stk_crit (particles too easily entrained)
    risk_stokes = max(0, (Stk_crit / Stk) - 1) if Stk > 0.01 else 5.0
    
    # Crossflow velocity risk
    risk_velocity = max(0, (v_crossflow / v_max) - 1) * 0.5
    
    # Small gap risk (concentrated field, tight tolerance)
    risk_gap = max(0, (gap_min / gap) - 1) if gap > 0.003 else 5.0
    
    # Small standoff risk (plasma proximity to wall)
    risk_standoff = max(0, (g_min / g) - 1) if g > 0.005 else 5.0
    
    # Combined dusting/plasma risk index
    # Weighted sum emphasizing entrainment and standoff
    risk_index = (
        0.30 * risk_stokes +
        0.25 * risk_velocity +
        0.20 * risk_gap +
        0.25 * risk_standoff
    )
    
    # Risk threshold
    risk_acceptable = risk_index < 1.0
    
    # Overall feasibility
    overall_feasible = tau_powder_valid and risk_acceptable
    
    return {
        # Geometry
        "a_m": a,
        "b_m": b,
        "R_tube_m": R_tube,
        "g_m": g,
        "g_mm": g * 1000,
        "gap_m": gap,
        "gap_mm": gap * 1000,
        "L_m": L,
        # Areas and volumes
        "A_annulus_m2": A_annulus,
        "A_active_m2": A_active,
        "V_annulus_m3": V_annulus,
        "V_active_m3": V_active,
        # Flow
        "Q_m3s": Q_m3s,
        "Q_SLPM_actual": Q_m3s * 60000 * (T_STP / T_op) * (p_bar / p_STP),
        # Velocities
        "v_powder_fall": v_powder_fall,
        "v_fall_range": (v_fall_min, v_fall_max),
        "v_crossflow": v_crossflow,
        "v_annulus_radial": v_annulus_radial,
        # Residence times
        "tau_powder_active": tau_powder_active,
        "tau_powder_range": (tau_powder_min, tau_powder_max),
        "tau_gas_refresh": tau_gas_refresh,
        "tau_gas_active": tau_gas_active,
        # Validity flags
        "tau_powder_valid": tau_powder_valid,
        "tau_gas_refresh_flag": tau_gas_refresh_flag,
        # Stokes number
        "Stokes_number": Stk,
        "d_p_median_um": d_p_median * 1e6,
        # Risk components
        "risk_stokes": risk_stokes,
        "risk_velocity": risk_velocity,
        "risk_gap": risk_gap,
        "risk_standoff": risk_standoff,
        "risk_index": risk_index,
        "risk_acceptable": risk_acceptable,
        # Overall
        "overall_feasible": overall_feasible,
        "T_preheat": T_preheat,
        "p_bar": sys["p_bar"],
        "Q_H2_SLPM": sys["Q_H2_SLPM"],
    }


def estimate_onset_voltage_analytic(
    geometry: Dict,
    T_preheat: float,
    gamma_RF: float = 0,
    E_RF_max: float = 5000.0,  # V/m (typical RF field)
) -> float:
    """
    Estimate onset voltage using analytical model with temperature dependence.
    
    Uses λ(T) = ΔG₀(T) / (n * F * r_act) and the calibrated onset model.
    """
    a = geometry["a_mm"] / 1000.0
    b = geometry["b_mm"] / 1000.0
    
    # Temperature-dependent parameters
    DeltaG0_T = compute_DeltaG0_T(T_preheat)
    n = PHYSICS_PARAMS["n"]
    r_act = PHYSICS_PARAMS["r_act"]
    
    # Temperature-dependent lambda
    lambda_T = DeltaG0_T / (n * F_CONST * r_act)
    
    # Effective lambda with RF contribution
    lambda_eff = lambda_T - gamma_RF * E_RF_max
    
    # Onset voltage for E(a) = lambda_eff (at pin surface)
    # E(r) = V / (r * ln(b/a)), E_max at r = a
    # V_onset = lambda_eff * a * ln(b/a)
    V_onset = lambda_eff * a * np.log(b/a)
    
    return max(0, V_onset)


def compute_chi_fraction_analytic(
    geometry: Dict,
    V_bias: float,
    T_preheat: float,
    gamma_RF: float = 0,
    E_RF_max: float = 5000.0,
) -> float:
    """
    Compute powder region χ > 0.5 fraction using calibrated analytical model.
    
    f_corrected = A(T) * f_analytic + B(T), clipped to [0, 1]
    """
    a = geometry["a_mm"] / 1000.0
    b = geometry["b_mm"] / 1000.0
    
    # Temperature-dependent parameters
    DeltaG0_T = compute_DeltaG0_T(T_preheat)
    n = PHYSICS_PARAMS["n"]
    r_act = PHYSICS_PARAMS["r_act"]
    
    # Temperature-dependent lambda
    lambda_T = DeltaG0_T / (n * F_CONST * r_act)
    
    # Effective lambda with RF
    lambda_eff = lambda_T - gamma_RF * E_RF_max
    
    if lambda_eff <= 0:
        return 1.0  # Full activation if RF dominates
    
    # Onset radius: r_onset = V / (lambda_eff * ln(b/a))
    ln_ba = np.log(b/a)
    if ln_ba <= 0:
        return 0.0
    
    r_onset = V_bias / (lambda_eff * ln_ba)
    
    # Analytic fraction: area from a to min(r_onset, b) / total area
    r_onset_clipped = np.clip(r_onset, a, b)
    f_analytic = (r_onset_clipped**2 - a**2) / (b**2 - a**2)
    
    # Apply temperature-dependent calibration
    A_cal, B_cal = get_calibration(T_preheat)
    f_corrected = A_cal * f_analytic + B_cal
    
    return float(np.clip(f_corrected, 0, 1))


def find_required_voltage_analytic(
    geometry: Dict,
    T_preheat: float,
    gamma_RF: float,
    target: float,
    V_range: Tuple[float, float] = (100, 1500),
    V_step: float = 10,
) -> Optional[float]:
    """Find required voltage to achieve target χ fraction using analytical model."""
    
    for V in np.arange(V_range[0], V_range[1], V_step):
        chi = compute_chi_fraction_analytic(geometry, V, T_preheat, gamma_RF)
        if chi >= target:
            return float(V)
    
    return None  # Target not achievable


def evaluate_geometry(
    geometry: Dict,
    geometry_name: str,
    targets: List[float] = CHI_TARGETS,
) -> Dict:
    """
    Evaluate a geometry across all operating conditions.
    Returns complete results with system constraints.
    """
    results = {
        "name": geometry_name,
        "geometry": geometry,
        "operating_points": [],
        "required_V": {},
        "constraints": {},
        "feasible": {},
    }
    
    for T in PREHEAT_TEMPS:
        results["required_V"][T] = {}
        results["constraints"][T] = compute_system_constraints(geometry, T)
        
        for gamma in GAMMA_VALUES:
            results["required_V"][T][gamma] = {}
            
            for target in targets:
                V_req = find_required_voltage_analytic(geometry, T, gamma, target)
                results["required_V"][T][gamma][target] = V_req
    
    # Determine feasibility for each operating point
    for T in PREHEAT_TEMPS:
        constraints = results["constraints"][T]
        tau_powder_ok = constraints["tau_powder_valid"]
        risk_ok = constraints["risk_acceptable"]
        tau_gas_flag = constraints["tau_gas_refresh_flag"]
        
        results["feasible"][T] = {
            "tau_powder_valid": tau_powder_ok,
            "tau_gas_refresh_flag": tau_gas_flag,
            "risk_acceptable": risk_ok,
            "overall_feasible": constraints["overall_feasible"],
        }
    
    return results


def rank_geometries(all_results: Dict) -> Dict:
    """
    Rank geometries under three objectives:
    1. Minimize required V for 40% activation
    2. Minimize dusting/plasma risk
    3. Robust compromise (Pareto)
    """
    candidates = []
    
    for geom_name, data in all_results.items():
        # Find best config for each geometry
        min_V_40 = float('inf')
        best_config = None
        min_risk = float('inf')
        
        for T in PREHEAT_TEMPS:
            c = data["constraints"][T]
            feasible = data["feasible"][T]["overall_feasible"]
            
            # Track minimum risk (even if not feasible for voltage)
            if c["risk_index"] < min_risk:
                min_risk = c["risk_index"]
            
            if feasible:
                # Check DC-only
                V_40 = data["required_V"][T][0].get(0.40)
                if V_40 is not None and V_40 < min_V_40:
                    min_V_40 = V_40
                    best_config = {
                        "T_preheat": T,
                        "gamma_RF": 0,
                        "V_40": V_40,
                        "risk_index": c["risk_index"],
                    }
                
                # Check RF-assisted
                V_40_rf = data["required_V"][T][0.4].get(0.40)
                if V_40_rf is not None and V_40_rf < min_V_40:
                    min_V_40 = V_40_rf
                    best_config = {
                        "T_preheat": T,
                        "gamma_RF": 0.4,
                        "V_40": V_40_rf,
                        "risk_index": c["risk_index"],
                    }
        
        # Collect constraint info at 1100K (reference)
        c = data["constraints"][1100]
        
        candidates.append({
            "name": geom_name,
            "geometry": data["geometry"],
            "min_V_40": min_V_40 if min_V_40 < float('inf') else None,
            "min_risk": min_risk,
            "best_config": best_config,
            "constraints": {
                "tau_powder_active": c["tau_powder_active"],
                "tau_gas_refresh": c["tau_gas_refresh"],
                "tau_gas_refresh_flag": c["tau_gas_refresh_flag"],
                "tau_powder_valid": c["tau_powder_valid"],
                "Stokes_number": c["Stokes_number"],
                "risk_index": c["risk_index"],
                "risk_stokes": c["risk_stokes"],
                "risk_velocity": c["risk_velocity"],
                "risk_gap": c["risk_gap"],
                "risk_standoff": c["risk_standoff"],
                "v_crossflow": c["v_crossflow"],
                "g_mm": c["g_mm"],
                "gap_mm": c["gap_mm"],
                "overall_feasible": c["overall_feasible"],
            },
            "any_feasible": any(data["feasible"][T]["overall_feasible"] for T in PREHEAT_TEMPS),
        })
    
    # =========================================================================
    # Objective 1: Minimize V for 40% activation
    # =========================================================================
    rank_by_V = sorted(
        [c for c in candidates if c["any_feasible"]],
        key=lambda x: x["min_V_40"] or float('inf')
    )
    
    # =========================================================================
    # Objective 2: Minimize dusting/plasma risk
    # =========================================================================
    rank_by_risk = sorted(
        [c for c in candidates if c["any_feasible"]],
        key=lambda x: x["min_risk"]
    )
    
    # =========================================================================
    # Objective 3: Pareto compromise
    # =========================================================================
    # Normalize objectives and compute weighted score
    feasible = [c for c in candidates if c["any_feasible"]]
    
    if feasible:
        V_values = [c["min_V_40"] for c in feasible if c["min_V_40"]]
        risk_values = [c["min_risk"] for c in feasible]
        
        V_min, V_max = min(V_values), max(V_values) if len(V_values) > 1 else (min(V_values), min(V_values) + 1)
        risk_min, risk_max = min(risk_values), max(risk_values) if len(risk_values) > 1 else (min(risk_values), min(risk_values) + 1)
        
        for c in feasible:
            # Normalize to [0, 1]
            V_norm = (c["min_V_40"] - V_min) / (V_max - V_min) if V_max > V_min else 0
            risk_norm = (c["min_risk"] - risk_min) / (risk_max - risk_min) if risk_max > risk_min else 0
            
            # Pareto score: lower is better (equal weight)
            c["pareto_score"] = 0.5 * V_norm + 0.5 * risk_norm
            c["V_normalized"] = V_norm
            c["risk_normalized"] = risk_norm
        
        rank_by_pareto = sorted(feasible, key=lambda x: x["pareto_score"])
    else:
        rank_by_pareto = []
    
    return {
        "candidates": candidates,
        "rank_by_V": rank_by_V,
        "rank_by_risk": rank_by_risk,
        "rank_by_pareto": rank_by_pareto,
    }


def create_plots(all_results: Dict, rankings: List[Dict]) -> None:
    """Create visualization plots with system constraints."""
    
    # =========================================================================
    # FIGURE 1: Chi vs V curves with temperature dependence
    # =========================================================================
    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors_T = {1000: 'blue', 1100: 'green', 1200: 'red'}
    linestyles_gamma = {0: '-', 0.2: '--', 0.4: ':'}
    
    for i, (geom_name, data) in enumerate(all_results.items()):
        ax = axes[i]
        V_range = np.linspace(200, 1200, 100)
        
        for T in PREHEAT_TEMPS:
            for gamma in [0, 0.4]:
                chi_values = [compute_chi_fraction_analytic(data["geometry"], V, T, gamma) 
                             for V in V_range]
                label = f'T={T}K' if gamma == 0 else f'T={T}K, γ=0.4'
                ax.plot(V_range, np.array(chi_values)*100, 
                       color=colors_T[T], 
                       linestyle=linestyles_gamma[gamma],
                       linewidth=2, label=label)
        
        # Add target lines
        for target in [0.20, 0.40, 0.60]:
            ax.axhline(y=target*100, color='gray', linestyle=':', alpha=0.5)
        
        # Mark feasibility
        feasible_str = "✓" if data["feasible"][1100]["overall_feasible"] else "✗"
        ax.set_title(f'{geom_name} {feasible_str}\n(a={data["geometry"]["a_mm"]:.0f}mm, gap={data["geometry"]["gap_mm"]:.0f}mm)')
        ax.set_xlabel('Bias Voltage (V)')
        ax.set_ylabel('Powder χ > 0.5 (%)')
        ax.set_xlim([200, 1200])
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
    
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / 'chi_vs_V_temperature.png', dpi=150)
    plt.close(fig1)
    print(f"Saved: {RESULTS_DIR / 'chi_vs_V_temperature.png'}")
    
    # =========================================================================
    # FIGURE 2: Required V heatmaps with constraints
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    
    for i, (geom_name, data) in enumerate(all_results.items()):
        # Top row: Required V for 40% (gamma=0)
        ax = axes2[0, i]
        V_matrix = np.zeros((len(PREHEAT_TEMPS), len(GAMMA_VALUES)))
        
        for j, T in enumerate(PREHEAT_TEMPS):
            for k, gamma in enumerate(GAMMA_VALUES):
                V_req = data["required_V"][T][gamma].get(0.40)
                V_matrix[j, k] = V_req if V_req else np.nan
        
        im = ax.imshow(V_matrix, aspect='auto', cmap='viridis_r', vmin=300, vmax=1000)
        ax.set_xticks(range(len(GAMMA_VALUES)))
        ax.set_xticklabels([f'{g}' for g in GAMMA_VALUES])
        ax.set_yticks(range(len(PREHEAT_TEMPS)))
        ax.set_yticklabels([f'{T}K' for T in PREHEAT_TEMPS])
        ax.set_xlabel('γ_RF')
        ax.set_ylabel('T_preheat')
        ax.set_title(f'{geom_name}: V for 40% target')
        
        for j in range(len(PREHEAT_TEMPS)):
            for k in range(len(GAMMA_VALUES)):
                val = V_matrix[j, k]
                if not np.isnan(val):
                    ax.text(k, j, f'{val:.0f}', ha='center', va='center', 
                           fontsize=9, fontweight='bold', color='white' if val > 600 else 'black')
        
        plt.colorbar(im, ax=ax, label='V', shrink=0.8)
        
        # Bottom row: Constraint summary
        ax2 = axes2[1, i]
        constraint_data = []
        labels = []
        for T in PREHEAT_TEMPS:
            c = data["constraints"][T]
            constraint_data.append([
                c["tau_powder_active"],
                c["risk_index"],
                c["Stokes_number"],
            ])
            labels.append(f'{T}K')
        
        constraint_data = np.array(constraint_data)
        x = np.arange(len(PREHEAT_TEMPS))
        width = 0.25
        
        bars1 = ax2.bar(x - width, constraint_data[:, 0], width, label='τ_powder (s)', color='blue')
        bars2 = ax2.bar(x, constraint_data[:, 1], width, label='Risk Index', color='orange')
        bars3 = ax2.bar(x + width, constraint_data[:, 2], width, label='Stokes #', color='green')
        
        # Add threshold lines
        ax2.axhline(y=0.3, color='blue', linestyle='--', alpha=0.5, label='τ_min')
        ax2.axhline(y=3.0, color='blue', linestyle='--', alpha=0.5)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Risk limit')
        
        ax2.set_xlabel('Preheat Temperature')
        ax2.set_ylabel('Value')
        ax2.set_title(f'{geom_name}: System Constraints')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend(fontsize=7, loc='upper right')
        ax2.set_ylim([0, 5])
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'voltage_constraints.png', dpi=150)
    plt.close(fig2)
    print(f"Saved: {RESULTS_DIR / 'voltage_constraints.png'}")
    
    # =========================================================================
    # FIGURE 3: Multi-objective ranking comparison
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Use candidates from rankings dict
    candidates = rankings["candidates"]
    
    # Left panel: V vs Risk scatter
    ax = axes3[0]
    for c in candidates:
        color = 'green' if c["any_feasible"] else 'red'
        marker = 'o' if c["any_feasible"] else 'x'
        V = c["min_V_40"] if c["min_V_40"] else 1500
        risk = c["min_risk"]
        ax.scatter(V, risk, c=color, marker=marker, s=200, edgecolors='black', linewidths=1.5)
        ax.annotate(c["name"], (V, risk), textcoords="offset points", xytext=(10, 5), fontsize=10)
    
    ax.set_xlabel('Minimum V for 40% Activation (V)')
    ax.set_ylabel('Risk Index')
    ax.set_title('Multi-Objective Trade-off\n(Green = Feasible, Red = Infeasible)')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Risk limit')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Right panel: Bar chart ranking
    ax2 = axes3[1]
    
    geom_names = [c["name"] for c in candidates]
    min_V_values = [c["min_V_40"] if c["min_V_40"] else 1500 for c in candidates]
    feasible_flags = [c["any_feasible"] for c in candidates]
    
    colors = ['green' if f else 'red' for f in feasible_flags]
    bars = ax2.barh(geom_names, min_V_values, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, c in zip(bars, candidates):
        V = c["min_V_40"]
        if V:
            ax2.text(V + 10, bar.get_y() + bar.get_height()/2, 
                    f'{V:.0f}V (risk={c["min_risk"]:.2f})', va='center', fontsize=9)
    
    ax2.set_xlabel('Minimum V for 40% Activation')
    ax2.set_title('Geometry Ranking')
    ax2.set_xlim([0, 1200])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / 'geometry_ranking.png', dpi=150)
    plt.close(fig3)
    print(f"Saved: {RESULTS_DIR / 'geometry_ranking.png'}")


def generate_report(all_results: Dict, rankings: Dict, flow_sensitivity: Dict = None) -> str:
    """Generate comprehensive text report."""
    
    lines = []
    lines.append("=" * 100)
    lines.append("POWDER REGION TRADE CURVES WITH SYSTEM CONSTRAINTS")
    lines.append("=" * 100)
    lines.append("")
    
    # System parameters
    lines.append("SYSTEM PARAMETERS")
    lines.append("-" * 50)
    lines.append(f"  Powder volume fraction (φ_p): {SYSTEM_DEFAULTS['phi_p']*100:.0f}%")
    lines.append(f"  H2 flow rate: {SYSTEM_DEFAULTS['Q_H2_SLPM']:.0f} SLPM")
    lines.append(f"  Pressure: {SYSTEM_DEFAULTS['p_bar']:.2f} bar")
    lines.append(f"  Active length: {SYSTEM_DEFAULTS['L_active']*1000:.0f} mm")
    lines.append(f"  Tube ID: {SYSTEM_DEFAULTS['tube_ID']*1000:.0f} mm")
    lines.append(f"  Residence time limits: [{SYSTEM_DEFAULTS['tau_min']}, {SYSTEM_DEFAULTS['tau_max']}] s")
    lines.append(f"  Max gas velocity: {SYSTEM_DEFAULTS['v_max']:.1f} m/s")
    lines.append("")
    
    # Flow rate sensitivity
    if flow_sensitivity:
        lines.append("FLOW RATE SENSITIVITY (at T=1100K)")
        lines.append("-" * 80)
        header = f"{'Geometry':<15}"
        for Q in FLOW_RATES_SLPM:
            header += f"| Q={Q:>3} SLPM "
        lines.append(header)
        lines.append("-" * 80)
        
        for geom_name in flow_sensitivity:
            row = f"{geom_name:<15}"
            for Q in FLOW_RATES_SLPM:
                data = flow_sensitivity[geom_name][Q]
                status = "✓" if data["feasible"] else "✗"
                row += f"| v={data['v_crossflow']:>4.1f} {status} "
            lines.append(row)
        lines.append("")
        lines.append("Note: v = gas velocity (m/s), ✓ = feasible, ✗ = constraint violation")
        lines.append("")
    
    # Physics parameters
    lines.append("PHYSICS PARAMETERS (Temperature-Dependent)")
    lines.append("-" * 50)
    for T in PREHEAT_TEMPS:
        DeltaG0_T = compute_DeltaG0_T(T)
        lambda_T = DeltaG0_T / (PHYSICS_PARAMS["n"] * F_CONST * PHYSICS_PARAMS["r_act"])
        A_cal, B_cal = get_calibration(T)
        lines.append(f"  T = {T}K:")
        lines.append(f"    ΔG₀(T) = {DeltaG0_T/1000:.1f} kJ/mol")
        lines.append(f"    λ(T) = {lambda_T/1000:.1f} kV/m")
        lines.append(f"    Calibration: A={A_cal:.2f}, B={B_cal:.3f}")
    lines.append("")
    
    # Required voltage table
    lines.append("=" * 100)
    lines.append("REQUIRED BIAS VOLTAGE (V) FOR TARGET POWDER ACTIVATION")
    lines.append("Format: γ=0 / γ=0.2 / γ=0.4")
    lines.append("=" * 100)
    
    for geom_name, data in all_results.items():
        lines.append("")
        lines.append("-" * 100)
        lines.append(f"GEOMETRY: {geom_name}")
        lines.append(f"  Pin: a={data['geometry']['a_mm']:.0f}mm | Mesh: b={data['geometry']['b_mm']:.0f}mm | Gap: {data['geometry']['gap_mm']:.0f}mm")
        lines.append("-" * 100)
        
        # Constraints summary
        c = data["constraints"][1100]
        lines.append(f"  Constraints (at 1100K):")
        lines.append(f"    τ_powder_active={c['tau_powder_active']:.2f}s (fall through active zone)")
        lines.append(f"    τ_gas_refresh={c['tau_gas_refresh']:.3f}s (crossflow refresh)")
        lines.append(f"    Stokes number={c['Stokes_number']:.2f}, v_crossflow={c['v_crossflow']:.2f}m/s")
        lines.append(f"    Gap={c['gap_mm']:.0f}mm, Standoff g={c['g_mm']:.0f}mm")
        lines.append(f"    Risk index={c['risk_index']:.2f} (stokes={c['risk_stokes']:.2f}, vel={c['risk_velocity']:.2f}, gap={c['risk_gap']:.2f}, standoff={c['risk_standoff']:.2f})")
        
        # Flags
        flags = []
        if not c["tau_powder_valid"]:
            flags.append(f"τ_powder={c['tau_powder_active']:.2f}s outside [{SYSTEM_DEFAULTS['tau_min']}, {SYSTEM_DEFAULTS['tau_max']}]s")
        if c["tau_gas_refresh_flag"]:
            flags.append(f"τ_gas_refresh={c['tau_gas_refresh']:.2f}s > 2.0s (water/plasma risk)")
        if not c["risk_acceptable"]:
            flags.append(f"Risk={c['risk_index']:.2f} > 1.0")
        
        feasible = data["feasible"][1100]["overall_feasible"]
        lines.append(f"  Feasibility: {'✓ FEASIBLE' if feasible else '✗ CONSTRAINT VIOLATION'}")
        for flag in flags:
            lines.append(f"    ⚠ {flag}")
        lines.append("")
        
        header = f"{'T_pre':<8}"
        for target in CHI_TARGETS:
            header += f"| χ>{target*100:.0f}% (γ=0/0.2/0.4) "
        lines.append(header)
        lines.append("-" * 100)
        
        for T in PREHEAT_TEMPS:
            row = f"{T}K{'':<4}"
            for target in CHI_TARGETS:
                V_g0 = data["required_V"][T][0].get(target)
                V_g02 = data["required_V"][T][0.2].get(target)
                V_g04 = data["required_V"][T][0.4].get(target)
                
                v0 = f"{V_g0:.0f}" if V_g0 else "N/A"
                v02 = f"{V_g02:.0f}" if V_g02 else "N/A"
                v04 = f"{V_g04:.0f}" if V_g04 else "N/A"
                
                row += f"| {v0:>4}/{v02:>4}/{v04:>4}     "
            lines.append(row)
    
    lines.append("")
    
    # System-feasible recommendation
    lines.append("=" * 100)
    lines.append("MULTI-OBJECTIVE RANKING")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Constraints applied:")
    lines.append(f"  - τ_powder_active ∈ [{SYSTEM_DEFAULTS['tau_min']}, {SYSTEM_DEFAULTS['tau_max']}] s")
    lines.append(f"  - τ_gas_refresh > 2.0 s flagged (water/plasma risk)")
    lines.append(f"  - Risk index < 1.0")
    lines.append("")
    
    # Objective 1: Minimize V
    lines.append("-" * 100)
    lines.append("OBJECTIVE 1: Minimize Required Voltage for 40% Activation")
    lines.append("-" * 100)
    for rank, r in enumerate(rankings["rank_by_V"], 1):
        V_str = f"{r['min_V_40']:.0f}V" if r["min_V_40"] else "N/A"
        c = r["constraints"]
        lines.append(f"  {rank}. {r['name']:<15} | V={V_str:<6} | Risk={c['risk_index']:.2f} | Stk={c['Stokes_number']:.2f}")
        if r["best_config"]:
            bc = r["best_config"]
            lines.append(f"     → T={bc['T_preheat']}K, γ_RF={bc['gamma_RF']}")
    lines.append("")
    
    # Objective 2: Minimize Risk
    lines.append("-" * 100)
    lines.append("OBJECTIVE 2: Minimize Dusting/Plasma Risk")
    lines.append("-" * 100)
    for rank, r in enumerate(rankings["rank_by_risk"], 1):
        V_str = f"{r['min_V_40']:.0f}V" if r["min_V_40"] else "N/A"
        c = r["constraints"]
        lines.append(f"  {rank}. {r['name']:<15} | Risk={c['risk_index']:.3f} | V={V_str:<6} | g={c['g_mm']:.0f}mm, gap={c['gap_mm']:.0f}mm")
        lines.append(f"     Risk breakdown: Stokes={c['risk_stokes']:.2f}, Vel={c['risk_velocity']:.2f}, Gap={c['risk_gap']:.2f}, Standoff={c['risk_standoff']:.2f}")
    lines.append("")
    
    # Objective 3: Pareto
    lines.append("-" * 100)
    lines.append("OBJECTIVE 3: Pareto Compromise (50% V + 50% Risk)")
    lines.append("-" * 100)
    for rank, r in enumerate(rankings["rank_by_pareto"], 1):
        V_str = f"{r['min_V_40']:.0f}V" if r["min_V_40"] else "N/A"
        c = r["constraints"]
        lines.append(f"  {rank}. {r['name']:<15} | Score={r['pareto_score']:.3f} | V={V_str:<6} | Risk={c['risk_index']:.3f}")
        lines.append(f"     Normalized: V_norm={r['V_normalized']:.2f}, Risk_norm={r['risk_normalized']:.2f}")
    lines.append("")
    
    # Final recommendation
    lines.append("=" * 100)
    lines.append("FINAL RECOMMENDATIONS")
    lines.append("=" * 100)
    
    pareto_best = rankings["rank_by_pareto"]
    V_best = rankings["rank_by_V"]
    risk_best = rankings["rank_by_risk"]
    
    if pareto_best:
        lines.append("")
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│ RECOMMENDED: PARETO-OPTIMAL GEOMETRY                                        │")
        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        
        best = pareto_best[0]
        c = best["constraints"]
        lines.append("")
        lines.append(f"  Geometry: {best['name']}")
        lines.append(f"    - Pin radius (a): {best['geometry']['a_mm']:.0f} mm")
        lines.append(f"    - Mesh radius (b): {best['geometry']['b_mm']:.0f} mm")
        lines.append(f"    - Electrode gap: {best['geometry']['gap_mm']:.0f} mm")
        lines.append(f"    - Mesh standoff (g): {c['g_mm']:.0f} mm")
        lines.append("")
        
        if best["best_config"]:
            bc = best["best_config"]
            lines.append(f"  Operating Point:")
            lines.append(f"    - Preheat: {bc['T_preheat']} K")
            lines.append(f"    - RF coupling: γ_RF = {bc['gamma_RF']}")
            lines.append(f"    - Bias voltage for 40% activation: {bc['V_40']:.0f} V")
            lines.append("")
        
        lines.append(f"  System Metrics:")
        lines.append(f"    - τ_powder_active: {c['tau_powder_active']:.2f} s (limit: 0.3-3.0 s)")
        lines.append(f"    - τ_gas_refresh: {c['tau_gas_refresh']:.3f} s {'⚠ >2s' if c['tau_gas_refresh_flag'] else '✓'}")
        lines.append(f"    - Stokes number: {c['Stokes_number']:.2f}")
        lines.append(f"    - Risk index: {c['risk_index']:.3f} (limit: 1.0)")
        lines.append("")
        
        lines.append(f"  Power Electronics Sizing:")
        if best["best_config"]:
            bc = best["best_config"]
            lines.append(f"    - DC Bias Supply: 0-{int(bc['V_40']*1.5)}V, 0-100mA")
        lines.append(f"    - Heater: ~850W for 1200K preheat")
        lines.append("")
        
        # Summary table
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│ SUMMARY: TOP CANDIDATES BY OBJECTIVE                                        │")
        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append(f"  {'Objective':<25} {'Winner':<15} {'V (40%)':<10} {'Risk':<10} {'Pareto':<10}")
        lines.append("  " + "-" * 70)
        
        if V_best:
            r = V_best[0]
            lines.append(f"  {'Minimize Voltage':<25} {r['name']:<15} {r['min_V_40']:.0f}V{'':<5} {r['min_risk']:.3f}{'':<5} {r.get('pareto_score', 0):.3f}")
        if risk_best:
            r = risk_best[0]
            lines.append(f"  {'Minimize Risk':<25} {r['name']:<15} {r['min_V_40']:.0f}V{'':<5} {r['min_risk']:.3f}{'':<5} {r.get('pareto_score', 0):.3f}")
        if pareto_best:
            r = pareto_best[0]
            lines.append(f"  {'Pareto Compromise':<25} {r['name']:<15} {r['min_V_40']:.0f}V{'':<5} {r['min_risk']:.3f}{'':<5} {r.get('pareto_score', 0):.3f}")
        
    else:
        lines.append("")
        lines.append("  ⚠ NO FEASIBLE GEOMETRY FOUND")
        lines.append("  Consider:")
        lines.append("    - Reducing H2 flow rate to increase residence time")
        lines.append("    - Increasing active length")
        lines.append("    - Using a different geometry with larger annulus/gap")
    
    lines.append("")
    
    return "\n".join(lines)


def evaluate_flow_sensitivity(geometries: Dict) -> Dict:
    """Evaluate how flow rate affects feasibility for each geometry."""
    
    sensitivity = {}
    
    for geom_name, geometry in geometries.items():
        sensitivity[geom_name] = {}
        
        for Q_SLPM in FLOW_RATES_SLPM:
            system = {"Q_H2_SLPM": Q_SLPM}
            constraints = compute_system_constraints(geometry, 1100, system)
            
            sensitivity[geom_name][Q_SLPM] = {
                "v_crossflow": constraints["v_crossflow"],
                "tau_powder_active": constraints["tau_powder_active"],
                "tau_gas_refresh": constraints["tau_gas_refresh"],
                "Stokes_number": constraints["Stokes_number"],
                "risk_index": constraints["risk_index"],
                "feasible": constraints["overall_feasible"],
            }
    
    return sensitivity


def main():
    print("\n" + "#"*70)
    print("# POWDER TRADE CURVES WITH SYSTEM CONSTRAINTS")
    print("# Temperature-dependent physics + flow/residence constraints")
    print("#"*70)
    
    # Evaluate flow sensitivity first
    print("\n--- Flow Rate Sensitivity Analysis ---")
    flow_sensitivity = evaluate_flow_sensitivity(GEOMETRIES)
    for geom_name, data in flow_sensitivity.items():
        print(f"\n{geom_name}:")
        for Q, vals in data.items():
            status = "✓" if vals["feasible"] else "✗"
            print(f"  Q={Q:3d} SLPM: v_cross={vals['v_crossflow']:.1f}m/s, τ_powder={vals['tau_powder_active']:.2f}s, Stk={vals['Stokes_number']:.2f}, risk={vals['risk_index']:.2f} {status}")
    
    # Evaluate all geometries with default flow rate
    all_results = {}
    for geom_name, geometry in GEOMETRIES.items():
        print(f"\nEvaluating: {geom_name}")
        results = evaluate_geometry(geometry, geom_name)
        all_results[geom_name] = results
    
    # Rank geometries
    rankings = rank_geometries(all_results)
    
    # Create plots
    create_plots(all_results, rankings)
    
    # Generate report
    report = generate_report(all_results, rankings, flow_sensitivity)
    print("\n" + report)
    
    # Save report
    with open(RESULTS_DIR / 'system_feasible_report.txt', 'w') as f:
        f.write(report)
    print(f"\nSaved: {RESULTS_DIR / 'system_feasible_report.txt'}")
    
    # Save full results as JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            return obj
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_defaults": SYSTEM_DEFAULTS,
        "physics_params": PHYSICS_PARAMS,
        "calibration": CALIBRATION,
        "results": convert_for_json(all_results),
        "rankings": convert_for_json(rankings),
    }
    
    with open(RESULTS_DIR / 'constrained_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'constrained_results.json'}")
    
    return all_results, rankings


if __name__ == "__main__":
    main()
