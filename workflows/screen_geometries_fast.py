#!/usr/bin/env python3
"""
screen_geometries_fast.py
Fast analytical screening of coaxial pin-mesh geometries.

Uses the calibrated analytical formula to rapidly evaluate many geometry/voltage
combinations without running full coupled simulations.

Output: Ranked shortlist of top 20 geometry/voltage pairs for validation.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from analytic_screening import (
    compute_lambda_onset,
    estimate_active_fraction,
    screen_geometry_grid,
    ScreeningCalibration,
    DEFAULT_CALIBRATION,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Fast analytical geometry screening')
    
    # Geometry ranges
    parser.add_argument('--a_min', type=float, default=1.0, help='Min pin radius (mm)')
    parser.add_argument('--a_max', type=float, default=10.0, help='Max pin radius (mm)')
    parser.add_argument('--a_step', type=float, default=1.0, help='Pin radius step (mm)')
    
    parser.add_argument('--b_min', type=float, default=30.0, help='Min mesh radius (mm)')
    parser.add_argument('--b_max', type=float, default=48.0, help='Max mesh radius (mm)')
    parser.add_argument('--b_step', type=float, default=2.0, help='Mesh radius step (mm)')
    
    # Voltage range
    parser.add_argument('--V_min', type=float, default=200.0, help='Min bias voltage (V)')
    parser.add_argument('--V_max', type=float, default=1500.0, help='Max bias voltage (V)')
    parser.add_argument('--V_step', type=float, default=100.0, help='Voltage step (V)')
    
    # Physics parameters
    parser.add_argument('--DeltaG0', type=float, default=350000.0, help='Gibbs barrier (J/mol)')
    parser.add_argument('--n', type=int, default=2, help='Number of electrons')
    parser.add_argument('--r_act', type=float, default=3e-5, help='Activation length (m)')
    
    # Temperature for correction
    parser.add_argument('--T', type=float, default=1100.0, help='Temperature for correction (K)')
    
    # Output
    parser.add_argument('--output', type=str, default='results/sweeps/fast_screen_shortlist.json',
                        help='Output JSON file')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top geometries to output')
    
    args = parser.parse_args()
    
    # Generate ranges
    a_values = np.arange(args.a_min, args.a_max + args.a_step/2, args.a_step) / 1000  # mm -> m
    b_values = np.arange(args.b_min, args.b_max + args.b_step/2, args.b_step) / 1000  # mm -> m
    V_values = np.arange(args.V_min, args.V_max + args.V_step/2, args.V_step)
    
    total_combinations = len(a_values) * len(b_values) * len(V_values)
    
    logger.info('=' * 60)
    logger.info('FAST GEOMETRY SCREENING')
    logger.info('=' * 60)
    logger.info(f'Pin radii: {len(a_values)} values from {args.a_min} to {args.a_max} mm')
    logger.info(f'Mesh radii: {len(b_values)} values from {args.b_min} to {args.b_max} mm')
    logger.info(f'Voltages: {len(V_values)} values from {args.V_min} to {args.V_max} V')
    logger.info(f'Total combinations: {total_combinations}')
    logger.info('')
    
    # Compute lambda_onset
    lambda_onset = compute_lambda_onset(args.DeltaG0, args.n, args.r_act)
    logger.info(f'λ_onset = {lambda_onset:.0f} V/m ({lambda_onset/1000:.2f} kV/m)')
    logger.info('')
    
    # Run screening
    logger.info('Running analytical screening...')
    results = screen_geometry_grid(
        a_values=list(a_values),
        b_values=list(b_values),
        V_values=list(V_values),
        lambda_onset=lambda_onset,
        calibration=DEFAULT_CALIBRATION,
        T=args.T,
    )
    
    valid_count = len(results)
    logger.info(f'Valid configurations: {valid_count}')
    logger.info('')
    
    # Get top N
    top_n = min(args.top_n, len(results))
    shortlist = results[:top_n]
    
    logger.info(f'TOP {top_n} GEOMETRIES BY CORRECTED ONSET FRACTION')
    logger.info('-' * 60)
    logger.info('| Rank | a(mm) | b(mm) | gap(mm) | V(V) | f_analytic | f_corrected |')
    logger.info('|------|-------|-------|---------|------|------------|-------------|')
    for i, r in enumerate(shortlist, 1):
        logger.info(f"| {i:4d} | {r['a_mm']:5.1f} | {r['b_mm']:5.1f} | {r['gap_mm']:7.1f} | {r['V']:4.0f} | "
                   f"{r['f_analytic']*100:10.2f}% | {r['f_corrected']*100:11.2f}% |")
    
    logger.info('')
    
    # Summary statistics
    f_corrected = [r['f_corrected'] for r in results]
    logger.info('DISTRIBUTION SUMMARY')
    logger.info('-' * 60)
    logger.info(f'  f_corrected max: {max(f_corrected)*100:.2f}%')
    logger.info(f'  f_corrected mean: {np.mean(f_corrected)*100:.2f}%')
    logger.info(f'  Configurations with f > 10%: {sum(1 for f in f_corrected if f > 0.1)}')
    logger.info(f'  Configurations with f > 20%: {sum(1 for f in f_corrected if f > 0.2)}')
    logger.info(f'  Configurations with f > 50%: {sum(1 for f in f_corrected if f > 0.5)}')
    logger.info('')
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'screening_type': 'fast_analytical',
            'lambda_onset_V_m': lambda_onset,
            'calibration': {
                'A': DEFAULT_CALIBRATION.A_global,
                'B': DEFAULT_CALIBRATION.B_global,
            },
            'temperature_K': args.T,
            'physics': {
                'DeltaG0': args.DeltaG0,
                'n': args.n,
                'r_act': args.r_act,
            },
        },
        'sweep_ranges': {
            'a_mm': list(a_values * 1000),
            'b_mm': list(b_values * 1000),
            'V': list(V_values),
        },
        'statistics': {
            'total_combinations': total_combinations,
            'valid_configurations': valid_count,
            'max_f_corrected': max(f_corrected),
            'mean_f_corrected': float(np.mean(f_corrected)),
        },
        'shortlist': shortlist,
        'all_results': results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f'Results saved to {output_path}')
    
    # Also save shortlist separately for validation workflow
    shortlist_path = output_path.parent / 'fast_screen_shortlist_only.json'
    with open(shortlist_path, 'w') as f:
        json.dump({
            'metadata': output_data['metadata'],
            'shortlist': shortlist,
        }, f, indent=2)
    
    logger.info(f'Shortlist saved to {shortlist_path}')


if __name__ == '__main__':
    main()
