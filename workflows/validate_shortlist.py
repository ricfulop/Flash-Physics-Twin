#!/usr/bin/env python3
"""
validate_shortlist.py
Validate shortlisted geometries with full coupled simulations.

Runs the complete EM-coupled reactor model on the top geometries from fast screening,
computes error between estimated and simulated chi>0.5 fraction, and optionally
refits temperature-dependent calibration coefficients.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
FARADAY = 96485.0


def run_single_validation(
    a: float,
    b: float,
    V: float,
    T_preheat: float,
    sigma_eff: float,
    flash_params: Dict,
) -> Dict:
    """
    Run a single validation case with full coupled model.
    
    Args:
        a: Pin radius (m)
        b: Mesh radius (m)
        V: Bias voltage (V)
        T_preheat: Preheat temperature (K)
        sigma_eff: Effective conductivity (S/m)
        flash_params: Flash physics parameters
    
    Returns:
        Dict with simulation results
    """
    # Import here to avoid circular dependency
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Simplified inline simulation (same as run_geometry_sweep.py)
    R_t = b + 0.002  # Tube radius slightly larger than mesh
    
    # Grid setup
    nr, nz = 100, 200
    L = 0.2
    r = np.linspace(0, R_t, nr)
    z = np.linspace(0, L, nz)
    R, Z = np.meshgrid(r, z, indexing='ij')
    shape = R.shape
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    # EM fields (simplified)
    I_coil = 10.0
    n_turns = 5
    f_RF = 13.56e6
    omega = 2 * np.pi * f_RF
    mu_0 = 4 * np.pi * 1e-7
    coil_radius = R_t + 0.01
    z_coil_start = 0.05
    z_coil_end = 0.15
    L_coil = z_coil_end - z_coil_start
    n_density = n_turns / L_coil
    
    # B_z (finite solenoid)
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
    
    # E_phi
    E_phi = np.zeros(shape)
    for i in range(nr):
        r_pt = R[i, 0]
        if r_pt > 1e-10:
            E_phi[i, :] = omega * r_pt * B_z[i, :] / 2
    
    E_mag = np.abs(E_phi)
    Q_RF = 0.5 * sigma_eff * E_mag**2
    
    # E_bias (coaxial)
    E_bias = np.zeros(shape)
    log_ratio = np.log(b / a)
    mask = (R >= a) & (R <= b)
    E_bias[mask] = np.abs(V) / (R[mask] * log_ratio)
    
    # Thermal
    rho_gas = 0.35
    cp_gas = 1000
    tau = 0.1
    T_wall = T_preheat
    
    dT_from_QRF = Q_RF * tau / (rho_gas * cp_gas)
    radial_factor = 1.0 - (R / R_t)**2
    T_gas = T_wall + dT_from_QRF * (0.5 + 0.5 * radial_factor)
    T_gas = np.clip(T_gas, T_preheat, 3000.0)
    
    # Flash physics
    DeltaG0_ref = flash_params['DeltaG0']
    beta_T = flash_params.get('beta_T', 100.0)
    T_ref = flash_params.get('T_ref', 400.0)
    k_soft = flash_params.get('k_soft', 1.0)
    n_electrons = int(flash_params.get('n', 2))
    r_act = flash_params['r_act']
    B_s = flash_params['B_s']
    
    DeltaG0_T = DeltaG0_ref - beta_T * (T_gas - T_ref)
    reduction = n_electrons * FARADAY * E_bias * r_act
    DeltaB = k_soft * DeltaG0_T - reduction
    
    with np.errstate(over='ignore'):
        exp_term = np.exp(DeltaB / B_s)
    chi = 1.0 / (1.0 + exp_term)
    chi = np.clip(chi, 0.0, 1.0)
    
    return {
        'chi_mean': float(np.mean(chi)),
        'chi_fraction_gt_0p5': float(np.mean(chi > 0.5)),
        'T_max': float(np.max(T_gas)),
        'T_mean': float(np.mean(T_gas)),
        'E_bias_max': float(np.max(E_bias)),
    }


def main():
    parser = argparse.ArgumentParser(description='Validate shortlisted geometries')
    parser.add_argument('--shortlist', type=str, 
                        default='results/sweeps/fast_screen_shortlist_only.json',
                        help='Path to shortlist JSON')
    parser.add_argument('--T_values', type=str, default='1000,1100,1200',
                        help='Comma-separated preheat temperatures (K)')
    parser.add_argument('--sigma_eff', type=float, default=0.01,
                        help='Effective conductivity (S/m)')
    parser.add_argument('--output', type=str, 
                        default='results/sweeps/fast_screen_validation.json',
                        help='Output JSON file')
    parser.add_argument('--fit_temperature', action='store_true',
                        help='Fit temperature-dependent calibration')
    
    args = parser.parse_args()
    
    # Parse temperature values
    T_values = [float(T) for T in args.T_values.split(',')]
    
    # Load shortlist
    shortlist_path = Path(args.shortlist)
    if not shortlist_path.exists():
        logger.error(f'Shortlist file not found: {shortlist_path}')
        return
    
    with open(shortlist_path) as f:
        shortlist_data = json.load(f)
    
    shortlist = shortlist_data['shortlist']
    metadata = shortlist_data['metadata']
    
    logger.info('=' * 60)
    logger.info('VALIDATING SHORTLISTED GEOMETRIES')
    logger.info('=' * 60)
    logger.info(f'Shortlist: {len(shortlist)} geometries')
    logger.info(f'Temperatures: {T_values} K')
    logger.info(f'sigma_eff: {args.sigma_eff} S/m')
    logger.info('')
    
    # Flash parameters
    flash_params = {
        'DeltaG0': metadata['physics']['DeltaG0'],
        'r_act': metadata['physics']['r_act'],
        'n': metadata['physics']['n'],
        'B_s': 15000.0,
        'beta_T': 100.0,
        'T_ref': 400.0,
        'k_soft': 1.0,
    }
    
    # Run validations
    validation_results = []
    total_runs = len(shortlist) * len(T_values)
    run_count = 0
    
    for item in shortlist:
        a = item['a_mm'] / 1000  # mm -> m
        b = item['b_mm'] / 1000
        V = item['V']
        f_estimated = item['f_corrected']
        f_analytic = item['f_analytic']
        
        for T in T_values:
            run_count += 1
            if run_count % 10 == 0:
                logger.info(f'Progress: {run_count}/{total_runs}')
            
            result = run_single_validation(a, b, V, T, args.sigma_eff, flash_params)
            
            f_simulated = result['chi_fraction_gt_0p5']
            error = f_simulated - f_estimated
            error_analytic = f_simulated - f_analytic
            
            validation_results.append({
                'a_mm': item['a_mm'],
                'b_mm': item['b_mm'],
                'gap_mm': item['gap_mm'],
                'V': V,
                'T_preheat_K': T,
                'f_analytic': f_analytic,
                'f_estimated': f_estimated,
                'f_simulated': f_simulated,
                'error': error,
                'error_analytic': error_analytic,
                'T_max': result['T_max'],
                'chi_mean': result['chi_mean'],
            })
    
    logger.info(f'Completed {len(validation_results)} validation runs')
    logger.info('')
    
    # Compute statistics
    errors = np.array([r['error'] for r in validation_results])
    errors_analytic = np.array([r['error_analytic'] for r in validation_results])
    
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    
    logger.info('VALIDATION RESULTS')
    logger.info('-' * 60)
    logger.info(f'Mean Absolute Error (corrected): {mae*100:.2f}%')
    logger.info(f'RMSE (corrected): {rmse*100:.2f}%')
    logger.info(f'Bias (corrected): {bias*100:+.2f}%')
    logger.info('')
    
    mae_analytic = np.mean(np.abs(errors_analytic))
    logger.info(f'Mean Absolute Error (analytic): {mae_analytic*100:.2f}%')
    logger.info('')
    
    # Check for systematic error vs temperature
    logger.info('ERROR BY TEMPERATURE')
    logger.info('-' * 60)
    for T in T_values:
        T_results = [r for r in validation_results if r['T_preheat_K'] == T]
        T_errors = [r['error'] for r in T_results]
        T_bias = np.mean(T_errors)
        T_std = np.std(T_errors)
        logger.info(f'  T = {T:.0f} K: bias = {T_bias*100:+.2f}%, std = {T_std*100:.2f}%')
    logger.info('')
    
    # Fit temperature-dependent calibration if requested
    temp_calibration = None
    if args.fit_temperature:
        logger.info('FITTING TEMPERATURE-DEPENDENT CALIBRATION')
        logger.info('-' * 60)
        
        # Prepare data for fitting
        f_analytic_arr = np.array([r['f_analytic'] for r in validation_results])
        f_simulated_arr = np.array([r['f_simulated'] for r in validation_results])
        T_arr = np.array([r['T_preheat_K'] for r in validation_results])
        T_ref = 1100.0
        
        from scipy.optimize import least_squares
        
        def model(params, f_a, T):
            A0, A1, B0, B1 = params
            A = A0 + A1 * (T - T_ref)
            B = B0 + B1 * (T - T_ref)
            return np.clip(A * f_a + B, 0, 1)
        
        def residuals(params, f_a, f_s, T):
            return f_s - model(params, f_a, T)
        
        x0 = [2.30, 0.0, 0.029, 0.0]
        result = least_squares(residuals, x0, args=(f_analytic_arr, f_simulated_arr, T_arr))
        A0, A1, B0, B1 = result.x
        
        logger.info(f'Fitted model: f = A(T) * f_analytic + B(T)')
        logger.info(f'  A(T) = {A0:.4f} + {A1:.6f} * (T - {T_ref})')
        logger.info(f'  B(T) = {B0:.4f} + {B1:.6f} * (T - {T_ref})')
        logger.info('')
        
        # Compute improved error
        f_fitted = model([A0, A1, B0, B1], f_analytic_arr, T_arr)
        improved_errors = f_simulated_arr - f_fitted
        improved_mae = np.mean(np.abs(improved_errors))
        improved_rmse = np.sqrt(np.mean(improved_errors**2))
        
        logger.info(f'Improved MAE: {improved_mae*100:.2f}% (was {mae*100:.2f}%)')
        logger.info(f'Improved RMSE: {improved_rmse*100:.2f}% (was {rmse*100:.2f}%)')
        logger.info('')
        
        temp_calibration = {
            'T_ref': T_ref,
            'A0': float(A0),
            'A1': float(A1),
            'B0': float(B0),
            'B1': float(B1),
            'improved_mae': float(improved_mae),
            'improved_rmse': float(improved_rmse),
        }
        
        # Show calibration at each temperature
        logger.info('Calibration at each temperature:')
        for T in T_values:
            A_T = A0 + A1 * (T - T_ref)
            B_T = B0 + B1 * (T - T_ref)
            logger.info(f'  T = {T:.0f} K: A = {A_T:.4f}, B = {B_T:.4f}')
    
    # Sample validation table
    logger.info('')
    logger.info('SAMPLE VALIDATION RESULTS (first 15)')
    logger.info('-' * 80)
    logger.info('| a(mm) | b(mm) | V(V) | T(K) | f_est(%) | f_sim(%) | error(%) |')
    logger.info('|-------|-------|------|------|----------|----------|----------|')
    for r in validation_results[:15]:
        logger.info(f"| {r['a_mm']:5.1f} | {r['b_mm']:5.1f} | {r['V']:4.0f} | {r['T_preheat_K']:.0f} | "
                   f"{r['f_estimated']*100:8.2f} | {r['f_simulated']*100:8.2f} | {r['error']*100:+8.2f} |")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'shortlist_source': str(shortlist_path),
            'T_values': T_values,
            'sigma_eff': args.sigma_eff,
        },
        'statistics': {
            'mae_corrected': float(mae),
            'rmse_corrected': float(rmse),
            'bias_corrected': float(bias),
            'mae_analytic': float(mae_analytic),
        },
        'temperature_calibration': temp_calibration,
        'validation_results': validation_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info('')
    logger.info(f'Results saved to {output_path}')


if __name__ == '__main__':
    main()
