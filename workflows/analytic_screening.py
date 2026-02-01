"""
analytic_screening.py
Analytical screening functions for rapid geometry evaluation.

Provides fast estimates of Flash onset fraction using the derived formula:
    r_onset = V / (lambda_onset * ln(b/a))
    f_onset_analytic = (min(r_onset, b)^2 - a^2) / (b^2 - a^2)
    f_onset_corrected = A * f_onset_analytic + B  (calibrated from full simulations)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

# Physical constants
FARADAY = 96485.0  # C/mol


@dataclass
class ScreeningCalibration:
    """Calibration parameters for corrected onset estimate."""
    # Global calibration (default)
    A_global: float = 2.30
    B_global: float = 0.029
    
    # Temperature-dependent calibration (optional)
    # f_corrected = A(T) * f_analytic + B(T)
    # A(T) = A0 + A1 * (T - T_ref)
    # B(T) = B0 + B1 * (T - T_ref)
    A0: Optional[float] = None
    A1: Optional[float] = None
    B0: Optional[float] = None
    B1: Optional[float] = None
    T_ref: float = 1100.0  # Reference temperature (K)
    
    use_temperature_dependent: bool = False
    
    def get_coefficients(self, T: Optional[float] = None) -> Tuple[float, float]:
        """Get A, B coefficients, optionally temperature-dependent."""
        if self.use_temperature_dependent and T is not None and self.A0 is not None:
            A = self.A0 + self.A1 * (T - self.T_ref)
            B = self.B0 + self.B1 * (T - self.T_ref)
            return A, B
        return self.A_global, self.B_global


# Default calibration from geometry sweep analysis (100mm tube, g=2-15mm standoff)
DEFAULT_CALIBRATION = ScreeningCalibration()

# Temperature-dependent calibration fitted from validation
# For high-temperature preheat (1000-1200 K) with coaxial pin-mesh geometry
TEMP_DEPENDENT_CALIBRATION = ScreeningCalibration(
    A_global=0.7001,
    B_global=0.2168,
    A0=0.7001,
    A1=0.000283,
    B0=0.2168,
    B1=0.000172,
    T_ref=1100.0,
    use_temperature_dependent=True,
)


def compute_lambda_onset(
    DeltaG0: float = 350000.0,
    n: int = 2,
    r_act: float = 3e-5,
    F: float = FARADAY,
) -> float:
    """
    Compute the characteristic onset electric field.
    
    lambda_onset = DeltaG0 / (n * F * r_act)
    
    Args:
        DeltaG0: Gibbs free energy barrier (J/mol)
        n: Number of electrons
        r_act: Activation length (m)
        F: Faraday constant (C/mol)
    
    Returns:
        lambda_onset: Onset electric field (V/m)
    """
    return DeltaG0 / (n * F * r_act)


def estimate_active_fraction(
    a: float,
    b: float,
    V: float,
    lambda_onset: float,
    calibration: Optional[ScreeningCalibration] = None,
    T: Optional[float] = None,
) -> Dict[str, float]:
    """
    Estimate the Flash active fraction for coaxial pin-mesh geometry.
    
    For coaxial geometry, E(r) = V / (r * ln(b/a))
    At onset, E(r_onset) = lambda_onset
    So: r_onset = V / (lambda_onset * ln(b/a))
    
    The area fraction where E > lambda_onset (i.e., r < r_onset):
    f_onset_analytic = (min(r_onset, b)^2 - a^2) / (b^2 - a^2)
    
    Args:
        a: Pin radius (m)
        b: Mesh radius (m)
        V: Bias voltage (V)
        lambda_onset: Onset electric field (V/m)
        calibration: Calibration parameters for correction
        T: Temperature (K) for temperature-dependent correction
    
    Returns:
        Dict with 'f_analytic', 'f_corrected', 'r_onset'
    """
    if b <= a:
        raise ValueError(f"Invalid geometry: b ({b}) must be > a ({a})")
    
    if calibration is None:
        calibration = DEFAULT_CALIBRATION
    
    # Compute onset radius
    log_ratio = np.log(b / a)
    r_onset = V / (lambda_onset * log_ratio)
    
    # Compute analytic fraction
    r_eff = min(r_onset, b)
    if r_eff <= a:
        f_analytic = 0.0
    else:
        f_analytic = (r_eff**2 - a**2) / (b**2 - a**2)
    f_analytic = np.clip(f_analytic, 0.0, 1.0)
    
    # Apply correction
    A, B = calibration.get_coefficients(T)
    f_corrected = A * f_analytic + B
    f_corrected = np.clip(f_corrected, 0.0, 1.0)
    
    return {
        'f_analytic': float(f_analytic),
        'f_corrected': float(f_corrected),
        'r_onset': float(r_onset),
        'r_onset_mm': float(r_onset * 1000),
        'A': float(A),
        'B': float(B),
    }


def screen_geometry_grid(
    a_values: List[float],
    b_values: List[float],
    V_values: List[float],
    lambda_onset: float,
    calibration: Optional[ScreeningCalibration] = None,
    T: Optional[float] = None,
    filter_invalid: bool = True,
) -> List[Dict]:
    """
    Screen a grid of geometries and voltages.
    
    Args:
        a_values: List of pin radii (m)
        b_values: List of mesh radii (m)
        V_values: List of bias voltages (V)
        lambda_onset: Onset electric field (V/m)
        calibration: Calibration parameters
        T: Temperature for correction (K)
        filter_invalid: Skip invalid geometries (b <= a)
    
    Returns:
        List of result dicts, sorted by f_corrected descending
    """
    results = []
    
    for a in a_values:
        for b in b_values:
            if filter_invalid and b <= a:
                continue
            
            for V in V_values:
                try:
                    est = estimate_active_fraction(a, b, V, lambda_onset, calibration, T)
                    est['a_mm'] = a * 1000
                    est['b_mm'] = b * 1000
                    est['V'] = V
                    est['gap_mm'] = (b - a) * 1000
                    results.append(est)
                except ValueError:
                    continue
    
    # Sort by corrected fraction descending
    results.sort(key=lambda x: -x['f_corrected'])
    
    return results


def fit_temperature_calibration(
    sweep_data: List[Dict],
    T_ref: float = 1100.0,
) -> ScreeningCalibration:
    """
    Fit temperature-dependent calibration from sweep data.
    
    Model: f_sim = A(T) * f_analytic + B(T)
    where: A(T) = A0 + A1 * (T - T_ref)
           B(T) = B0 + B1 * (T - T_ref)
    
    Args:
        sweep_data: List of dicts with 'f_onset_analytic', 'f_onset_simulated', 'T_preheat_K'
        T_ref: Reference temperature
    
    Returns:
        ScreeningCalibration with fitted parameters
    """
    from scipy.optimize import least_squares
    
    # Extract data
    f_analytic = np.array([d['f_onset_analytic'] for d in sweep_data])
    f_simulated = np.array([d['f_onset_simulated'] for d in sweep_data])
    T = np.array([d.get('operating', {}).get('T_preheat_K', d.get('T_preheat_K', 1100)) 
                  for d in sweep_data])
    
    def model(params, f_a, T):
        A0, A1, B0, B1 = params
        A = A0 + A1 * (T - T_ref)
        B = B0 + B1 * (T - T_ref)
        return np.clip(A * f_a + B, 0, 1)
    
    def residuals(params, f_a, f_s, T):
        return f_s - model(params, f_a, T)
    
    # Initial guess
    x0 = [2.30, 0.0, 0.029, 0.0]
    
    # Fit
    result = least_squares(residuals, x0, args=(f_analytic, f_simulated, T))
    A0, A1, B0, B1 = result.x
    
    return ScreeningCalibration(
        A_global=A0,
        B_global=B0,
        A0=A0,
        A1=A1,
        B0=B0,
        B1=B1,
        T_ref=T_ref,
        use_temperature_dependent=True,
    )
