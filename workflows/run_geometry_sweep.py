#!/usr/bin/env python3
"""
Coarse geometry sweep for 100mm ID quartz tube with coaxial pin + mesh.

Sweep parameters:
- Pin radius a: [2, 5, 8] mm
- Mesh standoff g: [2, 5, 10, 15] mm (b = R_t - g)
- Preheat temperature: [1000, 1100, 1200] K
- sigma_eff: [0.01, 0.1, 1.0] S/m
- Bias voltage: [300, 600, 900, 1200] V

E_bias field for coaxial geometry:
    E_bias(r) = V_bias / (r * ln(b/a))  for a <= r <= b
"""

import argparse
import json
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import h5py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
FARADAY = 96485.0  # C/mol


def compute_coaxial_E_bias(R: np.ndarray, V_bias: float, a: float, b: float) -> np.ndarray:
    """
    Compute E_bias field for coaxial pin-mesh geometry.
    
    E(r) = V / (r * ln(b/a)) for a <= r <= b
    
    Args:
        R: Radial coordinate array (m)
        V_bias: Bias voltage (V)
        a: Pin radius (m)
        b: Mesh radius (m)
    
    Returns:
        E_bias field array (V/m)
    """
    E = np.zeros_like(R)
    log_ratio = np.log(b / a)
    
    # Inside the pin (r < a): E = 0
    # Between pin and mesh (a <= r <= b): E = V / (r * ln(b/a))
    # Outside mesh (r > b): E = 0
    
    mask = (R >= a) & (R <= b)
    E[mask] = np.abs(V_bias) / (R[mask] * log_ratio)
    
    # Handle r = 0 case (set to field at inner boundary)
    E[R < a] = 0  # Inside pin, no field
    E[R > b] = 0  # Outside mesh, no field
    
    return E


def run_single_case(
    tube_id: float,
    pin_radius: float,
    mesh_standoff: float,
    T_preheat: float,
    sigma_eff: float,
    bias_voltage: float,
    results_dir: Path,
    flash_params: dict,
) -> dict:
    """
    Run a single geometry/operating point case.
    
    Returns:
        Dictionary of KPIs
    """
    # Geometry
    R_t = tube_id / 2  # Tube radius (m)
    a = pin_radius  # Pin radius (m)
    b = R_t - mesh_standoff  # Mesh radius (m)
    
    if b <= a:
        return {'error': f'Invalid geometry: mesh radius ({b*1000:.1f}mm) <= pin radius ({a*1000:.1f}mm)'}
    
    # Grid setup
    nr, nz = 100, 200
    L = 0.2  # Reactor length (m)
    r = np.linspace(0, R_t, nr)
    z = np.linspace(0, L, nz)
    R, Z = np.meshgrid(r, z, indexing='ij')
    shape = R.shape
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    # ========================================
    # EM Fields
    # ========================================
    
    # RF coil parameters (fixed for now)
    I_coil = 10.0
    n_turns = 5
    f_RF = 13.56e6
    omega = 2 * np.pi * f_RF
    mu_0 = 4 * np.pi * 1e-7
    coil_radius = R_t + 0.01  # Coil outside quartz tube
    z_coil_start = 0.05
    z_coil_end = 0.15
    L_coil = z_coil_end - z_coil_start
    n_density = n_turns / L_coil
    
    # B_z using finite solenoid approximation
    B_z = np.zeros(shape)
    for i in range(nr):
        for j in range(nz):
            r_pt = R[i, j]
            z_pt = Z[i, j]
            
            d1 = np.sqrt(coil_radius**2 + (z_pt - z_coil_start)**2)
            d2 = np.sqrt(coil_radius**2 + (z_pt - z_coil_end)**2)
            
            cos_theta1 = (z_pt - z_coil_start) / d1 if d1 > 0 else 0
            cos_theta2 = (z_pt - z_coil_end) / d2 if d2 > 0 else 0
            
            B_z_axis = (mu_0 * n_density * I_coil / 2) * (cos_theta1 - cos_theta2)
            radial_factor = np.exp(-(r_pt / coil_radius)**2 * 0.5)
            
            if z_coil_start <= z_pt <= z_coil_end and r_pt < coil_radius:
                B_z[i, j] = mu_0 * n_density * I_coil * radial_factor
            else:
                B_z[i, j] = B_z_axis * radial_factor
    
    B_mag = np.abs(B_z)
    
    # E_phi (induced azimuthal field)
    E_phi = np.zeros(shape)
    for i in range(nr):
        r_pt = R[i, 0]
        if r_pt > 1e-10:
            E_phi[i, :] = omega * r_pt * B_z[i, :] / 2
    
    E_mag = np.abs(E_phi)
    
    # Q_RF heating
    Q_RF = 0.5 * sigma_eff * E_mag**2
    Q_RF_total = np.sum(Q_RF * 2 * np.pi * R * dr * dz)
    
    # E_bias from coaxial geometry
    E_bias = compute_coaxial_E_bias(R, bias_voltage, a, b)
    E_bias_max = np.max(E_bias)
    E_bias_mean = np.mean(E_bias[E_bias > 0]) if np.any(E_bias > 0) else 0
    
    # E_bias over threshold (e.g., 1 MV/m could cause breakdown)
    E_threshold = 1e6  # V/m
    E_over_threshold_fraction = np.sum(E_bias > E_threshold) / E_bias.size
    
    # ========================================
    # Thermal Fields
    # ========================================
    
    rho_gas = 0.35  # kg/m³
    cp_gas = 1000  # J/(kg·K)
    tau = 0.1  # residence time (s)
    T_wall = T_preheat  # Wall temperature = preheat temperature
    T_inlet = T_preheat
    
    # Temperature rise from Q_RF heating
    dT_from_QRF = Q_RF * tau / (rho_gas * cp_gas)
    radial_factor = 1.0 - (R / R_t)**2
    T_gas = T_wall + dT_from_QRF * (0.5 + 0.5 * radial_factor)
    T_gas = np.clip(T_gas, T_inlet, 3000.0)
    
    T_max = np.max(T_gas)
    T_mean = np.mean(T_gas)
    T_wall_max = T_wall  # Fixed wall temperature
    
    unphysical = T_max > 1500  # Flag for unphysical temperatures
    
    # ========================================
    # Flash Physics
    # ========================================
    
    # Temperature-dependent DeltaG0
    DeltaG0_ref = flash_params['DeltaG0']
    beta_T = flash_params.get('beta_T', 100.0)
    T_ref = flash_params.get('T_ref', 400.0)
    k_soft = flash_params.get('k_soft', 1.0)
    n_electrons = int(flash_params.get('n', 2))
    r_act = flash_params['r_act']
    B_s = flash_params['B_s']
    W_ph = flash_params.get('W_ph', 0.0)
    DeltaMu_chem = flash_params.get('DeltaMu_chem', 0.0)
    
    # DeltaG0(T)
    DeltaG0_T = DeltaG0_ref - beta_T * (T_gas - T_ref)
    
    # DeltaB
    reduction = n_electrons * FARADAY * E_bias * r_act + W_ph + DeltaMu_chem
    DeltaB = k_soft * DeltaG0_T - reduction
    
    # chi
    with np.errstate(over='ignore'):
        exp_term = np.exp(DeltaB / B_s)
    chi = 1.0 / (1.0 + exp_term)
    chi = np.clip(chi, 0.0, 1.0)
    
    chi_mean = np.mean(chi)
    chi_fraction_gt_0p5 = np.mean(chi > 0.5)
    
    # ========================================
    # Compile KPIs
    # ========================================
    
    return {
        'chi_volume_avg': float(chi_mean),
        'chi_fraction_gt_0p5': float(chi_fraction_gt_0p5),
        'T_max': float(T_max),
        'T_mean': float(T_mean),
        'T_wall_max': float(T_wall_max),
        'unphysical_temperature_flag': bool(unphysical),
        'E_bias_max': float(E_bias_max),
        'E_bias_mean': float(E_bias_mean),
        'E_over_threshold_fraction': float(E_over_threshold_fraction),
        'Q_RF_total': float(Q_RF_total),
        'DeltaB_mean': float(np.mean(DeltaB)),
        'geometry': {
            'tube_id_mm': float(tube_id * 1000),
            'pin_radius_mm': float(pin_radius * 1000),
            'mesh_standoff_mm': float(mesh_standoff * 1000),
            'mesh_radius_mm': float(b * 1000),
            'gap_mm': float((b - a) * 1000),
        },
        'operating': {
            'T_preheat_K': float(T_preheat),
            'sigma_eff': float(sigma_eff),
            'bias_V': float(bias_voltage),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Run coarse geometry sweep')
    parser.add_argument('--output', type=str, default='results/sweeps/geometry_sweep.json',
                        help='Output JSON file')
    args = parser.parse_args()
    
    # Fixed parameters
    tube_id = 0.100  # 100 mm ID
    
    # Sweep parameters
    pin_radii = [0.002, 0.005, 0.008]  # [2, 5, 8] mm
    mesh_standoffs = [0.002, 0.005, 0.010, 0.015]  # [2, 5, 10, 15] mm
    T_preheats = [1000, 1100, 1200]  # K
    sigma_effs = [0.01, 0.1, 1.0]  # S/m
    bias_voltages = [300, 600, 900, 1200]  # V
    
    # Flash physics parameters (science mode with temperature dependence)
    flash_params = {
        'DeltaG0': 350000.0,  # J/mol
        'r_act': 3e-5,  # m
        'B_s': 15000.0,  # J/mol
        'k_soft': 1.0,
        'n': 2,
        'beta_T': 100.0,  # J/(mol·K)
        'T_ref': 400.0,  # K
    }
    
    # Create output directory
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate total combinations
    all_combinations = list(product(
        pin_radii, mesh_standoffs, T_preheats, sigma_effs, bias_voltages
    ))
    total = len(all_combinations)
    
    logger.info(f"Starting geometry sweep: {total} combinations")
    logger.info(f"  Pin radii: {[r*1000 for r in pin_radii]} mm")
    logger.info(f"  Mesh standoffs: {[g*1000 for g in mesh_standoffs]} mm")
    logger.info(f"  Preheat temps: {T_preheats} K")
    logger.info(f"  sigma_eff: {sigma_effs} S/m")
    logger.info(f"  Bias voltages: {bias_voltages} V")
    
    results = []
    skipped = 0
    
    for idx, (a, g, T, sigma, V) in enumerate(all_combinations):
        # Check for invalid geometry
        b = tube_id / 2 - g
        if b <= a:
            skipped += 1
            continue
        
        result = run_single_case(
            tube_id=tube_id,
            pin_radius=a,
            mesh_standoff=g,
            T_preheat=T,
            sigma_eff=sigma,
            bias_voltage=V,
            results_dir=output_file.parent,
            flash_params=flash_params,
        )
        
        if 'error' not in result:
            results.append(result)
        else:
            skipped += 1
        
        if (idx + 1) % 50 == 0:
            logger.info(f"  Progress: {idx + 1}/{total} ({len(results)} valid, {skipped} skipped)")
    
    logger.info(f"Completed: {len(results)} valid cases, {skipped} skipped")
    
    # Save all results
    output_data = {
        'sweep_parameters': {
            'tube_id_mm': 100,
            'pin_radii_mm': [r*1000 for r in pin_radii],
            'mesh_standoffs_mm': [g*1000 for g in mesh_standoffs],
            'T_preheats_K': T_preheats,
            'sigma_effs': sigma_effs,
            'bias_voltages_V': bias_voltages,
        },
        'flash_params': flash_params,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'total_valid': len(results),
        'total_skipped': skipped,
        'results': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Quick feasibility summary
    feasible = [r for r in results if not r['unphysical_temperature_flag'] and r['T_max'] < 1500]
    logger.info(f"\nFeasibility summary:")
    logger.info(f"  Total valid: {len(results)}")
    logger.info(f"  Feasible (T_max < 1500K): {len(feasible)}")
    
    if feasible:
        # Group by geometry and find best chi at lowest bias
        from collections import defaultdict
        by_geometry = defaultdict(list)
        for r in feasible:
            key = (r['geometry']['pin_radius_mm'], r['geometry']['mesh_standoff_mm'])
            by_geometry[key].append(r)
        
        logger.info(f"\nTop feasible geometries by chi_fraction at lowest bias:")
        geom_scores = []
        for (pin, standoff), runs in by_geometry.items():
            # Get runs at lowest bias
            min_bias = min(r['operating']['bias_V'] for r in runs)
            low_bias_runs = [r for r in runs if r['operating']['bias_V'] == min_bias]
            max_chi = max(r['chi_fraction_gt_0p5'] for r in low_bias_runs)
            geom_scores.append((pin, standoff, min_bias, max_chi))
        
        geom_scores.sort(key=lambda x: -x[3])
        for pin, standoff, bias, chi in geom_scores[:5]:
            logger.info(f"  a={pin:.0f}mm, g={standoff:.0f}mm: χ>0.5 = {chi*100:.1f}% at {bias}V")


if __name__ == '__main__':
    main()
