"""
trainer_inverse.py
Parameter inversion for Flash physics using analytical relations.

This module provides tools to infer Flash physics parameters (r_act, B_s, etc.)
from observed chi vs E_bias data using the documented relation:

    chi = 1 / (1 + exp(DeltaB / B_s))
    DeltaB = k_soft * DeltaG0 - (n * F * E_bias * r_act + W_ph + DeltaMu_chem)

For the simplified case (W_ph = 0, DeltaMu_chem = 0):
    chi = 1 / (1 + exp((k_soft * DeltaG0 - n * F * E_bias * r_act) / B_s))
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import bootstrap
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Physical constants
FARADAY_CONSTANT = 96485.0  # C/mol


@dataclass
class InversionResult:
    """Result of a parameter inversion."""
    problem_id: str
    status: str  # "success", "failed", "running"
    
    # Fitted parameters
    fitted_params: Dict[str, float] = field(default_factory=dict)
    
    # Uncertainties (from bootstrap)
    param_uncertainties: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g., {"r_act": {"mean": 3e-5, "std": 1e-6, "ci_low": 2.8e-5, "ci_high": 3.2e-5}}
    
    # Known/fixed parameters
    known_params: Dict[str, float] = field(default_factory=dict)
    
    # Fit quality metrics
    mse: float = 0.0
    rmse: float = 0.0
    r_squared: float = 0.0
    
    # Data used
    n_points: int = 0
    run_ids: List[str] = field(default_factory=list)
    
    # Paths to outputs
    output_dir: str = ""
    plots: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chi_model_with_temp(
    E_bias: np.ndarray, 
    T_gas: np.ndarray,
    r_act: float, 
    B_s: float,
    DeltaG0_ref: float,
    beta_T: float,
    T_ref: float,
    k_soft: float, 
    n: int, 
    F: float = FARADAY_CONSTANT,
    W_ph: float = 0.0, 
    DeltaMu_chem: float = 0.0
) -> np.ndarray:
    """
    Compute chi from E_bias and T_gas using temperature-dependent DeltaG0.
    
    DeltaG0(T) = DeltaG0_ref - beta_T * (T_gas - T_ref)
    DeltaB = k_soft * DeltaG0(T) - (n * F * E_bias * r_act + W_ph + DeltaMu_chem)
    chi = 1 / (1 + exp(DeltaB / B_s))
    
    Args:
        E_bias: Electric field array (V/m)
        T_gas: Gas temperature array (K)
        r_act: Activation length (m)
        B_s: Smoothing scale (J/mol)
        DeltaG0_ref: Reference Gibbs barrier at T_ref (J/mol)
        beta_T: Temperature coefficient (J/mol/K)
        T_ref: Reference temperature (K)
        k_soft: Softening factor
        n: Number of electrons
        F: Faraday constant (C/mol)
        W_ph: Photon work (J/mol)
        DeltaMu_chem: Chemical potential contribution (J/mol)
    
    Returns:
        chi: Flash order parameter array (0-1)
    """
    # Temperature-dependent Gibbs barrier
    DeltaG0_T = DeltaG0_ref - beta_T * (T_gas - T_ref)
    
    # Barrier reduction from electrochemistry
    reduction = n * F * E_bias * r_act + W_ph + DeltaMu_chem
    
    # Remaining barrier
    DeltaB = k_soft * DeltaG0_T - reduction
    
    # Sigmoid mapping
    chi = 1.0 / (1.0 + np.exp(DeltaB / B_s))
    
    return chi


def chi_model(E_bias: np.ndarray, r_act: float, B_s: float, 
              DeltaG0: float, k_soft: float, n: int, F: float = FARADAY_CONSTANT,
              W_ph: float = 0.0, DeltaMu_chem: float = 0.0) -> np.ndarray:
    """
    Compute chi from E_bias using the analytical Flash physics relation.
    
    chi = 1 / (1 + exp(DeltaB / B_s))
    DeltaB = k_soft * DeltaG0 - (n * F * E_bias * r_act + W_ph + DeltaMu_chem)
    
    Args:
        E_bias: Electric field array (V/m)
        r_act: Activation length (m)
        B_s: Smoothing scale (J/mol)
        DeltaG0: Intrinsic Gibbs barrier (J/mol)
        k_soft: Softening factor
        n: Number of electrons
        F: Faraday constant (C/mol)
        W_ph: Photon work (J/mol)
        DeltaMu_chem: Chemical potential contribution (J/mol)
    
    Returns:
        chi: Flash order parameter array (0-1)
    """
    # Barrier reduction term
    reduction = n * F * E_bias * r_act + W_ph + DeltaMu_chem
    
    # Remaining barrier
    DeltaB = k_soft * DeltaG0 - reduction
    
    # Sigmoid mapping
    chi = 1.0 / (1.0 + np.exp(DeltaB / B_s))
    
    return chi


def mse_loss_with_temp(
    params: np.ndarray, 
    E_bias: np.ndarray, 
    T_gas: np.ndarray,
    chi_obs: np.ndarray,
    param_names: List[str], 
    known: Dict[str, float]
) -> float:
    """
    Compute MSE loss for temperature-dependent DeltaG0(T) model.
    
    Args:
        params: Array of parameter values to optimize
        E_bias: Observed E_bias values
        T_gas: Observed T_gas values
        chi_obs: Observed chi values
        param_names: Names of parameters being optimized
        known: Dictionary of known/fixed parameters
    
    Returns:
        Mean squared error
    """
    # Build full parameter dict
    all_params = dict(known)
    for name, val in zip(param_names, params):
        all_params[name] = val
    
    # Extract parameters
    r_act = all_params.get('r_act', 1e-9)
    B_s = all_params.get('B_s', 50000.0)
    DeltaG0_ref = all_params.get('DeltaG0_ref', all_params.get('DeltaG0', 350000.0))
    beta_T = all_params.get('beta_T', 100.0)
    T_ref = all_params.get('T_ref', 400.0)
    k_soft = all_params.get('k_soft', 1.0)
    n = int(all_params.get('n', 2))
    W_ph = all_params.get('W_ph', 0.0)
    DeltaMu_chem = all_params.get('DeltaMu_chem', 0.0)
    
    # Compute predicted chi using temperature-dependent model
    chi_pred = chi_model_with_temp(
        E_bias, T_gas, r_act, B_s, DeltaG0_ref, beta_T, T_ref,
        k_soft, n, W_ph=W_ph, DeltaMu_chem=DeltaMu_chem
    )
    
    # MSE
    mse = np.mean((chi_pred - chi_obs) ** 2)
    
    return mse


def fit_inverse_with_temp(
    E_bias: np.ndarray, 
    T_gas: np.ndarray,
    chi_obs: np.ndarray,
    infer: List[str], 
    known: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[Dict[str, float], float]:
    """
    Fit temperature-dependent Flash physics parameters by minimizing MSE.
    
    Args:
        E_bias: Observed E_bias values (V/m)
        T_gas: Observed T_gas values (K)
        chi_obs: Observed chi values (0-1)
        infer: List of parameter names to infer
        known: Dictionary of known/fixed parameters
        bounds: Optional bounds for each parameter
    
    Returns:
        Tuple of (fitted_params dict, final_mse)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for parameter inversion")
    
    # Default bounds for common parameters
    default_bounds = {
        'r_act': (1e-9, 1e-3),
        'B_s': (1000.0, 500000.0),
        'DeltaG0_ref': (10000.0, 1e6),
        'DeltaG0': (10000.0, 1e6),
        'beta_T': (1.0, 500.0),  # J/mol/K
        'k_soft': (0.1, 10.0),
        'W_ph': (0.0, 100000.0),
        'DeltaMu_chem': (-100000.0, 100000.0),
    }
    
    if bounds is None:
        bounds = {}
    
    # Build bounds array for optimizer
    param_bounds = []
    for name in infer:
        if name in bounds:
            param_bounds.append(bounds[name])
        elif name in default_bounds:
            param_bounds.append(default_bounds[name])
        else:
            raise ValueError(f"No bounds defined for parameter: {name}")
    
    # Use differential evolution for global optimization
    result = differential_evolution(
        mse_loss_with_temp,
        bounds=param_bounds,
        args=(E_bias, T_gas, chi_obs, infer, known),
        seed=42,
        maxiter=1000,
        tol=1e-10,
        polish=True,
    )
    
    # Build result dict
    fitted = {}
    for name, val in zip(infer, result.x):
        fitted[name] = val
    
    return fitted, result.fun


def mse_loss(params: np.ndarray, E_bias: np.ndarray, chi_obs: np.ndarray,
             param_names: List[str], known: Dict[str, float]) -> float:
    """
    Compute MSE loss for parameter optimization.
    
    Args:
        params: Array of parameter values to optimize
        E_bias: Observed E_bias values
        chi_obs: Observed chi values
        param_names: Names of parameters being optimized
        known: Dictionary of known/fixed parameters
    
    Returns:
        Mean squared error
    """
    # Build full parameter dict
    all_params = dict(known)
    for name, val in zip(param_names, params):
        all_params[name] = val
    
    # Extract parameters
    r_act = all_params.get('r_act', 1e-9)
    B_s = all_params.get('B_s', 50000.0)
    DeltaG0 = all_params.get('DeltaG0', 43500.0)
    k_soft = all_params.get('k_soft', 1.0)
    n = int(all_params.get('n', 2))
    W_ph = all_params.get('W_ph', 0.0)
    DeltaMu_chem = all_params.get('DeltaMu_chem', 0.0)
    
    # Compute predicted chi
    chi_pred = chi_model(E_bias, r_act, B_s, DeltaG0, k_soft, n, 
                         W_ph=W_ph, DeltaMu_chem=DeltaMu_chem)
    
    # MSE
    mse = np.mean((chi_pred - chi_obs) ** 2)
    
    return mse


def fit_inverse(E_bias: np.ndarray, chi_obs: np.ndarray,
                infer: List[str], known: Dict[str, float],
                bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[Dict[str, float], float]:
    """
    Fit Flash physics parameters by minimizing MSE.
    
    Args:
        E_bias: Observed E_bias values (V/m)
        chi_obs: Observed chi values (0-1)
        infer: List of parameter names to infer
        known: Dictionary of known/fixed parameters
        bounds: Optional bounds for each parameter
    
    Returns:
        Tuple of (fitted_params dict, final_mse)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for parameter inversion")
    
    # Default bounds for common parameters
    default_bounds = {
        'r_act': (1e-9, 1e-3),      # 1 nm to 1 mm
        'B_s': (1000.0, 500000.0),  # 1 kJ/mol to 500 kJ/mol
        'DeltaG0': (10000.0, 1e6), # 10 kJ/mol to 1 MJ/mol
        'k_soft': (0.1, 10.0),
        'W_ph': (0.0, 100000.0),
        'DeltaMu_chem': (-100000.0, 100000.0),
    }
    
    if bounds is None:
        bounds = {}
    
    # Build bounds array for optimizer
    param_bounds = []
    for name in infer:
        if name in bounds:
            param_bounds.append(bounds[name])
        elif name in default_bounds:
            param_bounds.append(default_bounds[name])
        else:
            raise ValueError(f"No bounds defined for parameter: {name}")
    
    # Initial guess (middle of bounds, log-scale for r_act and B_s)
    x0 = []
    for name, (lo, hi) in zip(infer, param_bounds):
        if name in ['r_act', 'B_s', 'DeltaG0']:
            # Log-scale midpoint
            x0.append(np.sqrt(lo * hi))
        else:
            x0.append((lo + hi) / 2)
    x0 = np.array(x0)
    
    # Use differential evolution for global optimization (more robust)
    result = differential_evolution(
        mse_loss,
        bounds=param_bounds,
        args=(E_bias, chi_obs, infer, known),
        seed=42,
        maxiter=1000,
        tol=1e-10,
        polish=True,  # Use L-BFGS-B to polish the result
    )
    
    # Build result dict
    fitted = {}
    for name, val in zip(infer, result.x):
        fitted[name] = val
    
    return fitted, result.fun


@dataclass
class PriorSpec:
    """Specification for a parameter prior distribution."""
    distribution: str  # "normal", "uniform", "fixed"
    mean: float = 0.0
    std: float = 1.0
    low: float = 0.0
    high: float = 1.0
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from the prior distribution."""
        if self.distribution == "normal":
            return np.random.normal(self.mean, self.std, n)
        elif self.distribution == "uniform":
            return np.random.uniform(self.low, self.high, n)
        elif self.distribution == "fixed":
            return np.full(n, self.mean)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass 
class BayesianInversionResult:
    """Result of Bayesian parameter inversion with uncertain priors."""
    problem_id: str
    status: str
    
    # Posterior samples for all parameters (including those with priors)
    posterior_samples: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Posterior statistics
    posterior_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Prior specifications
    priors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Correlation matrix
    correlation_matrix: Optional[np.ndarray] = None
    param_names: List[str] = field(default_factory=list)
    
    # Fit quality (mean over samples)
    mean_mse: float = 0.0
    mean_r_squared: float = 0.0
    
    # Data info
    n_samples: int = 0
    n_points: int = 0
    run_ids: List[str] = field(default_factory=list)
    output_dir: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "problem_id": self.problem_id,
            "status": self.status,
            "posterior_stats": self.posterior_stats,
            "priors": self.priors,
            "correlation_matrix": self.correlation_matrix.tolist() if self.correlation_matrix is not None else None,
            "param_names": self.param_names,
            "mean_mse": self.mean_mse,
            "mean_r_squared": self.mean_r_squared,
            "n_samples": self.n_samples,
            "n_points": self.n_points,
            "run_ids": self.run_ids,
            "output_dir": self.output_dir,
        }
        # Don't include raw samples in JSON (too large)
        return d


def bayesian_inversion_with_prior(
    E_bias: np.ndarray,
    chi_obs: np.ndarray,
    infer: List[str],
    fixed: Dict[str, float],
    priors: Dict[str, PriorSpec],
    n_samples: int = 500,
    confidence: float = 0.95
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], np.ndarray]:
    """
    Bayesian-style inversion with uncertain prior parameters.
    
    For each sample:
    1. Draw prior parameters (e.g., DeltaG0) from their distributions
    2. Fit the infer parameters (e.g., r_act, B_s) conditioned on the drawn priors
    3. Collect posterior samples
    
    Args:
        E_bias: Observed E_bias values (V/m)
        chi_obs: Observed chi values (0-1)
        infer: Parameters to infer by optimization (e.g., ['r_act', 'B_s'])
        fixed: Parameters with fixed values (e.g., {'k_soft': 1.0, 'n': 2})
        priors: Parameters with prior distributions (e.g., {'DeltaG0': PriorSpec(...)})
        n_samples: Number of samples to draw
        confidence: Confidence level for intervals
    
    Returns:
        Tuple of (posterior_samples dict, posterior_stats dict, correlation_matrix)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Bayesian inversion")
    
    # Initialize storage for all parameters
    all_param_names = list(infer) + list(priors.keys())
    posterior_samples = {name: [] for name in all_param_names}
    mse_samples = []
    r2_samples = []
    
    n_data = len(E_bias)
    
    for i in range(n_samples):
        # 1. Draw prior parameters
        prior_values = {}
        for name, prior in priors.items():
            prior_values[name] = prior.sample(1)[0]
        
        # 2. Bootstrap resample data
        idx = np.random.choice(n_data, size=n_data, replace=True)
        E_boot = E_bias[idx]
        chi_boot = chi_obs[idx]
        
        # 3. Build known dict for this sample
        known_for_sample = dict(fixed)
        known_for_sample.update(prior_values)
        
        # 4. Fit infer parameters
        try:
            fitted, mse = fit_inverse(E_boot, chi_boot, infer, known_for_sample)
            
            # Store samples
            for name in infer:
                posterior_samples[name].append(fitted[name])
            for name, val in prior_values.items():
                posterior_samples[name].append(val)
            
            # Compute R² on original data
            all_params = dict(known_for_sample)
            all_params.update(fitted)
            chi_pred = chi_model(
                E_bias,
                r_act=all_params.get('r_act', 1e-9),
                B_s=all_params.get('B_s', 50000),
                DeltaG0=all_params.get('DeltaG0', 300000),
                k_soft=all_params.get('k_soft', 1.0),
                n=int(all_params.get('n', 2)),
            )
            ss_res = np.sum((chi_obs - chi_pred) ** 2)
            ss_tot = np.sum((chi_obs - np.mean(chi_obs)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            mse_samples.append(mse)
            r2_samples.append(r2)
            
        except Exception as e:
            logger.debug(f"Sample {i} failed: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Completed {i+1}/{n_samples} samples")
    
    # Convert to arrays
    for name in all_param_names:
        posterior_samples[name] = np.array(posterior_samples[name])
    
    n_valid = len(mse_samples)
    logger.info(f"Completed {n_valid}/{n_samples} valid samples")
    
    # Compute posterior statistics
    alpha = 1 - confidence
    posterior_stats = {}
    for name in all_param_names:
        samples = posterior_samples[name]
        if len(samples) > 0:
            posterior_stats[name] = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "median": float(np.median(samples)),
                "ci_low": float(np.percentile(samples, 100 * alpha / 2)),
                "ci_high": float(np.percentile(samples, 100 * (1 - alpha / 2))),
                "n_samples": len(samples),
            }
    
    # Compute correlation matrix
    param_arrays = [posterior_samples[name] for name in all_param_names]
    if all(len(a) > 0 for a in param_arrays):
        correlation_matrix = np.corrcoef(param_arrays)
    else:
        correlation_matrix = np.eye(len(all_param_names))
    
    # Mean fit quality
    mean_mse = np.mean(mse_samples) if mse_samples else 0.0
    mean_r2 = np.mean(r2_samples) if r2_samples else 0.0
    
    return posterior_samples, posterior_stats, correlation_matrix, mean_mse, mean_r2


def bootstrap_uncertainty(E_bias: np.ndarray, chi_obs: np.ndarray,
                          infer: List[str], known: Dict[str, float],
                          n_bootstrap: int = 100,
                          confidence: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Estimate parameter uncertainties using bootstrap resampling.
    
    Args:
        E_bias: Observed E_bias values
        chi_obs: Observed chi values
        infer: Parameters to infer
        known: Known parameters
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level for intervals
    
    Returns:
        Dict mapping parameter names to uncertainty stats
    """
    n_points = len(E_bias)
    bootstrap_results = {name: [] for name in infer}
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_points, size=n_points, replace=True)
        E_boot = E_bias[idx]
        chi_boot = chi_obs[idx]
        
        try:
            fitted, _ = fit_inverse(E_boot, chi_boot, infer, known)
            for name in infer:
                bootstrap_results[name].append(fitted[name])
        except Exception as e:
            logger.warning(f"Bootstrap sample {i} failed: {e}")
            continue
    
    # Compute statistics
    uncertainties = {}
    alpha = 1 - confidence
    
    for name in infer:
        samples = np.array(bootstrap_results[name])
        if len(samples) < 10:
            logger.warning(f"Too few successful bootstrap samples for {name}")
            uncertainties[name] = {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            continue
        
        uncertainties[name] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "ci_low": float(np.percentile(samples, 100 * alpha / 2)),
            "ci_high": float(np.percentile(samples, 100 * (1 - alpha / 2))),
            "n_samples": len(samples),
        }
    
    return uncertainties


class InverseTrainer:
    """
    Trainer for Flash physics parameter inversion.
    """
    
    def __init__(self, results_dir: Path = Path("results/inversions")):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._problems: Dict[str, InversionResult] = {}
    
    def load_dataset(self, run_ids: List[str], runs_dir: Path = Path("results/runs"),
                     target: str = "flash/chi", feature: str = "em/E_bias",
                     aggregation: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from registered runs.
        
        Args:
            run_ids: List of run IDs to load
            runs_dir: Directory containing runs
            target: Target field path in HDF5
            feature: Feature field path in HDF5
            aggregation: How to aggregate field data
        
        Returns:
            Tuple of (feature_values, target_values) arrays
        """
        import h5py
        
        features = []
        targets = []
        
        for run_id in run_ids:
            run_path = runs_dir / run_id
            fields_file = run_path / "outputs" / "fields.h5"
            
            if not fields_file.exists():
                logger.warning(f"Fields file not found for run {run_id}")
                continue
            
            with h5py.File(fields_file, 'r') as f:
                # Extract feature
                if feature in f:
                    feat_data = f[feature][:]
                elif feature.replace('/', '') in str(f.keys()):
                    # Try nested path
                    parts = feature.split('/')
                    feat_data = f[parts[0]][parts[1]][:]
                else:
                    logger.warning(f"Feature {feature} not found in {run_id}")
                    continue
                
                # Extract target
                if target in f:
                    tgt_data = f[target][:]
                elif '/' in target:
                    parts = target.split('/')
                    tgt_data = f[parts[0]][parts[1]][:]
                else:
                    logger.warning(f"Target {target} not found in {run_id}")
                    continue
                
                # Aggregate
                if aggregation == "mean":
                    features.append(np.mean(feat_data))
                    targets.append(np.mean(tgt_data))
                elif aggregation == "median":
                    features.append(np.median(feat_data))
                    targets.append(np.median(tgt_data))
                elif aggregation == "min":
                    features.append(np.min(feat_data))
                    targets.append(np.min(tgt_data))
                elif aggregation == "max":
                    features.append(np.max(feat_data))
                    targets.append(np.max(tgt_data))
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return np.array(features), np.array(targets)
    
    def train(self, problem_id: str, run_ids: List[str],
              infer: List[str], known: Dict[str, Any],
              output_dir: Optional[str] = None,
              n_bootstrap: int = 100) -> InversionResult:
        """
        Train an inverse model to infer Flash parameters.
        
        Args:
            problem_id: Unique problem identifier
            run_ids: Run IDs to use as training data
            infer: Parameters to infer (e.g., ['r_act', 'B_s'])
            known: Known/fixed parameters
            output_dir: Output directory (default: results/inversions/<problem_id>)
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            InversionResult with fitted parameters and uncertainties
        """
        # Setup output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.results_dir / problem_id
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize result
        result = InversionResult(
            problem_id=problem_id,
            status="running",
            known_params=dict(known),
            run_ids=list(run_ids),
            output_dir=str(out_path),
        )
        
        try:
            # Load dataset
            logger.info(f"Loading dataset from {len(run_ids)} runs...")
            E_bias, chi_obs = self.load_dataset(run_ids)
            
            if len(E_bias) < 2:
                raise ValueError(f"Need at least 2 data points, got {len(E_bias)}")
            
            result.n_points = len(E_bias)
            logger.info(f"Dataset: {len(E_bias)} points, E_bias range [{E_bias.min():.0f}, {E_bias.max():.0f}] V/m")
            
            # Convert known params to correct types
            known_typed = {}
            for k, v in known.items():
                if k == 'n':
                    known_typed[k] = int(v)
                else:
                    known_typed[k] = float(v)
            
            # Fit parameters
            logger.info(f"Fitting parameters: {infer}")
            fitted, mse = fit_inverse(E_bias, chi_obs, infer, known_typed)
            
            result.fitted_params = fitted
            result.mse = float(mse)
            result.rmse = float(np.sqrt(mse))
            
            # Compute R²
            chi_pred = chi_model(
                E_bias,
                r_act=fitted.get('r_act', known_typed.get('r_act', 1e-9)),
                B_s=fitted.get('B_s', known_typed.get('B_s', 50000)),
                DeltaG0=known_typed.get('DeltaG0', 350000),
                k_soft=known_typed.get('k_soft', 1.0),
                n=int(known_typed.get('n', 2)),
            )
            ss_res = np.sum((chi_obs - chi_pred) ** 2)
            ss_tot = np.sum((chi_obs - np.mean(chi_obs)) ** 2)
            result.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            logger.info(f"Fit complete: MSE={mse:.2e}, R²={result.r_squared:.4f}")
            for name, val in fitted.items():
                if name == 'r_act':
                    logger.info(f"  {name} = {val:.3e} m ({val*1e6:.1f} μm)")
                elif name == 'B_s':
                    logger.info(f"  {name} = {val:.0f} J/mol ({val/1000:.1f} kJ/mol)")
                else:
                    logger.info(f"  {name} = {val}")
            
            # Bootstrap uncertainty estimation
            if n_bootstrap > 0:
                logger.info(f"Running {n_bootstrap} bootstrap samples for uncertainty estimation...")
                uncertainties = bootstrap_uncertainty(E_bias, chi_obs, infer, known_typed, n_bootstrap)
                result.param_uncertainties = uncertainties
                
                for name, stats in uncertainties.items():
                    if name == 'r_act':
                        logger.info(f"  {name}: {stats['mean']:.3e} ± {stats['std']:.3e} m "
                                   f"(95% CI: [{stats['ci_low']:.3e}, {stats['ci_high']:.3e}])")
                    elif name == 'B_s':
                        logger.info(f"  {name}: {stats['mean']:.0f} ± {stats['std']:.0f} J/mol "
                                   f"(95% CI: [{stats['ci_low']:.0f}, {stats['ci_high']:.0f}])")
            
            result.status = "success"
            
        except Exception as e:
            logger.error(f"Inversion failed: {e}")
            result.status = "failed"
            raise
        
        finally:
            # Save result
            self._problems[problem_id] = result
            result_file = out_path / "inversion_result.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        return result
    
    def generate_report(self, problem_id: str, include_plots: bool = True) -> Dict[str, Any]:
        """
        Generate a report for a completed inversion.
        
        Args:
            problem_id: Problem ID to report on
            include_plots: Whether to generate plots
        
        Returns:
            Report dictionary
        """
        # Load result
        result_file = self.results_dir / problem_id / "inversion_result.json"
        if result_file.exists():
            with open(result_file) as f:
                result_dict = json.load(f)
            result = InversionResult(**result_dict)
        elif problem_id in self._problems:
            result = self._problems[problem_id]
        else:
            raise ValueError(f"Unknown problem_id: {problem_id}")
        
        report = {
            "problem_id": result.problem_id,
            "status": result.status,
            "fitted_parameters": result.fitted_params,
            "uncertainties": result.param_uncertainties,
            "known_parameters": result.known_params,
            "fit_quality": {
                "mse": result.mse,
                "rmse": result.rmse,
                "r_squared": result.r_squared,
            },
            "data": {
                "n_points": result.n_points,
                "run_ids": result.run_ids,
            },
            "output_dir": result.output_dir,
        }
        
        if include_plots and result.status == "success":
            try:
                plot_paths = self._generate_plots(result)
                report["plots"] = plot_paths
            except Exception as e:
                logger.warning(f"Failed to generate plots: {e}")
                report["plots"] = {}
        
        return report
    
    def train_bayesian(
        self,
        problem_id: str,
        run_ids: List[str],
        infer: List[str],
        fixed: Dict[str, Any],
        priors: Dict[str, Dict[str, float]],
        output_dir: Optional[str] = None,
        n_samples: int = 500
    ) -> BayesianInversionResult:
        """
        Train an inverse model with uncertain prior parameters.
        
        This method samples prior parameters from their distributions and
        jointly infers the specified parameters, providing full posterior
        distributions and correlations.
        
        Args:
            problem_id: Unique problem identifier
            run_ids: Run IDs to use as training data
            infer: Parameters to infer by optimization (e.g., ['r_act', 'B_s'])
            fixed: Parameters with fixed values (e.g., {'k_soft': 1.0, 'n': 2})
            priors: Parameters with prior distributions, specified as:
                    {'DeltaG0': {'distribution': 'normal', 'mean': 300000, 'std': 30000}}
            output_dir: Output directory
            n_samples: Number of samples for posterior estimation
        
        Returns:
            BayesianInversionResult with posterior distributions and correlations
        """
        # Setup output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.results_dir / problem_id
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize result
        result = BayesianInversionResult(
            problem_id=problem_id,
            status="running",
            run_ids=list(run_ids),
            output_dir=str(out_path),
        )
        
        try:
            # Load dataset
            logger.info(f"Loading dataset from {len(run_ids)} runs...")
            E_bias, chi_obs = self.load_dataset(run_ids)
            
            if len(E_bias) < 2:
                raise ValueError(f"Need at least 2 data points, got {len(E_bias)}")
            
            result.n_points = len(E_bias)
            logger.info(f"Dataset: {len(E_bias)} points")
            
            # Convert fixed params
            fixed_typed = {}
            for k, v in fixed.items():
                if k == 'n':
                    fixed_typed[k] = int(v)
                else:
                    fixed_typed[k] = float(v)
            
            # Convert prior specs
            prior_specs = {}
            for name, spec in priors.items():
                prior_specs[name] = PriorSpec(
                    distribution=spec.get('distribution', 'normal'),
                    mean=spec.get('mean', 0),
                    std=spec.get('std', 1),
                    low=spec.get('low', 0),
                    high=spec.get('high', 1),
                )
            result.priors = priors
            
            # Run Bayesian inversion
            logger.info(f"Running Bayesian inversion with {n_samples} samples...")
            logger.info(f"  Inferring: {infer}")
            logger.info(f"  Fixed: {list(fixed_typed.keys())}")
            logger.info(f"  Priors: {list(prior_specs.keys())}")
            
            posterior_samples, posterior_stats, corr_matrix, mean_mse, mean_r2 = \
                bayesian_inversion_with_prior(
                    E_bias, chi_obs, infer, fixed_typed, prior_specs, n_samples
                )
            
            result.posterior_samples = posterior_samples
            result.posterior_stats = posterior_stats
            result.correlation_matrix = corr_matrix
            result.param_names = list(infer) + list(prior_specs.keys())
            result.mean_mse = mean_mse
            result.mean_r_squared = mean_r2
            result.n_samples = len(list(posterior_samples.values())[0]) if posterior_samples else 0
            
            # Log results
            logger.info(f"\nBayesian inversion complete:")
            logger.info(f"  Mean R²: {mean_r2:.4f}")
            for name, stats in posterior_stats.items():
                if name == 'r_act':
                    logger.info(f"  {name}: {stats['mean']*1e6:.2f} ± {stats['std']*1e6:.2f} μm "
                               f"(95% CI: [{stats['ci_low']*1e6:.2f}, {stats['ci_high']*1e6:.2f}])")
                elif name in ['B_s', 'DeltaG0']:
                    logger.info(f"  {name}: {stats['mean']/1000:.2f} ± {stats['std']/1000:.2f} kJ/mol "
                               f"(95% CI: [{stats['ci_low']/1000:.2f}, {stats['ci_high']/1000:.2f}])")
                else:
                    logger.info(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            result.status = "success"
            
        except Exception as e:
            logger.error(f"Bayesian inversion failed: {e}")
            result.status = "failed"
            raise
        
        finally:
            # Save result
            result_file = out_path / "bayesian_result.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        return result
    
    def generate_bayesian_report(
        self,
        result: BayesianInversionResult,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """Generate a comprehensive report for Bayesian inversion."""
        out_path = Path(result.output_dir)
        
        report = {
            "problem_id": result.problem_id,
            "status": result.status,
            "posterior_stats": result.posterior_stats,
            "priors": result.priors,
            "correlation_matrix": result.correlation_matrix.tolist() if result.correlation_matrix is not None else None,
            "param_names": result.param_names,
            "mean_r_squared": result.mean_r_squared,
            "mean_mse": result.mean_mse,
            "n_samples": result.n_samples,
            "n_points": result.n_points,
        }
        
        if include_plots and result.status == "success":
            try:
                plots = self._generate_bayesian_plots(result)
                report["plots"] = plots
            except Exception as e:
                logger.warning(f"Failed to generate Bayesian plots: {e}")
                report["plots"] = {}
        
        return report
    
    def _generate_bayesian_plots(self, result: BayesianInversionResult) -> Dict[str, str]:
        """Generate plots for Bayesian inversion results."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        
        out_path = Path(result.output_dir)
        plots = {}
        
        # Number of parameters
        n_params = len(result.param_names)
        
        # Create corner plot
        fig, axes = plt.subplots(n_params, n_params, figsize=(4*n_params, 4*n_params))
        
        for i, name_i in enumerate(result.param_names):
            for j, name_j in enumerate(result.param_names):
                ax = axes[i, j]
                
                samples_i = result.posterior_samples.get(name_i, np.array([]))
                samples_j = result.posterior_samples.get(name_j, np.array([]))
                
                # Scale for display
                scale_i = 1e6 if name_i == 'r_act' else (1e-3 if name_i in ['B_s', 'DeltaG0'] else 1)
                scale_j = 1e6 if name_j == 'r_act' else (1e-3 if name_j in ['B_s', 'DeltaG0'] else 1)
                unit_i = 'μm' if name_i == 'r_act' else ('kJ/mol' if name_i in ['B_s', 'DeltaG0'] else '')
                unit_j = 'μm' if name_j == 'r_act' else ('kJ/mol' if name_j in ['B_s', 'DeltaG0'] else '')
                
                if i == j:
                    # Diagonal: histogram
                    if len(samples_i) > 0:
                        ax.hist(samples_i * scale_i, bins=30, density=True, alpha=0.7, color='steelblue')
                        ax.axvline(np.mean(samples_i) * scale_i, color='red', lw=2)
                        
                        # Show prior if available
                        if name_i in result.priors:
                            prior = result.priors[name_i]
                            if prior.get('distribution') == 'normal':
                                x = np.linspace(
                                    (prior['mean'] - 3*prior['std']) * scale_i,
                                    (prior['mean'] + 3*prior['std']) * scale_i,
                                    100
                                )
                                from scipy.stats import norm
                                ax.plot(x, norm.pdf(x, prior['mean']*scale_i, prior['std']*scale_i),
                                       'g--', lw=2, label='Prior')
                    
                    ax.set_xlabel(f'{name_i} ({unit_i})' if unit_i else name_i)
                    if i == 0:
                        ax.set_title(f'{name_i}')
                        
                elif i > j:
                    # Lower triangle: scatter plot with correlation
                    if len(samples_i) > 0 and len(samples_j) > 0:
                        ax.scatter(samples_j * scale_j, samples_i * scale_i, alpha=0.1, s=5)
                        
                        # Add correlation coefficient
                        corr = result.correlation_matrix[i, j] if result.correlation_matrix is not None else 0
                        ax.text(0.05, 0.95, f'ρ={corr:.3f}', transform=ax.transAxes,
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    ax.set_xlabel(f'{name_j} ({unit_j})' if unit_j else name_j)
                    ax.set_ylabel(f'{name_i} ({unit_i})' if unit_i else name_i)
                else:
                    # Upper triangle: hide
                    ax.axis('off')
        
        plt.tight_layout()
        corner_plot = out_path / "posterior_corner.png"
        plt.savefig(corner_plot, dpi=150)
        plt.close()
        plots["posterior_corner"] = str(corner_plot)
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if result.correlation_matrix is not None:
            im = ax.imshow(result.correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # Add labels
            ax.set_xticks(range(n_params))
            ax.set_yticks(range(n_params))
            ax.set_xticklabels(result.param_names, rotation=45, ha='right')
            ax.set_yticklabels(result.param_names)
            
            # Add correlation values as text
            for i in range(n_params):
                for j in range(n_params):
                    text = ax.text(j, i, f'{result.correlation_matrix[i, j]:.2f}',
                                  ha='center', va='center', fontsize=12,
                                  color='white' if abs(result.correlation_matrix[i, j]) > 0.5 else 'black')
            
            plt.colorbar(im, label='Correlation')
            ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        corr_plot = out_path / "correlation_heatmap.png"
        plt.savefig(corr_plot, dpi=150)
        plt.close()
        plots["correlation_heatmap"] = str(corr_plot)
        
        return plots
    
    def _generate_plots(self, result: InversionResult) -> Dict[str, str]:
        """Generate fit quality plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        out_path = Path(result.output_dir)
        plots = {}
        
        # Load data
        E_bias, chi_obs = self.load_dataset(result.run_ids)
        
        # Build full params
        all_params = dict(result.known_params)
        all_params.update(result.fitted_params)
        
        # Compute predictions
        E_dense = np.linspace(0, E_bias.max() * 1.1, 200)
        chi_pred_dense = chi_model(
            E_dense,
            r_act=all_params.get('r_act', 1e-9),
            B_s=all_params.get('B_s', 50000),
            DeltaG0=all_params.get('DeltaG0', 350000),
            k_soft=all_params.get('k_soft', 1.0),
            n=int(all_params.get('n', 2)),
        )
        
        chi_pred = chi_model(
            E_bias,
            r_act=all_params.get('r_act', 1e-9),
            B_s=all_params.get('B_s', 50000),
            DeltaG0=all_params.get('DeltaG0', 350000),
            k_soft=all_params.get('k_soft', 1.0),
            n=int(all_params.get('n', 2)),
        )
        
        # Plot 1: Fit quality
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        ax1.scatter(E_bias / 1000, chi_obs, s=100, c='blue', marker='o', label='Observed', zorder=3)
        ax1.plot(E_dense / 1000, chi_pred_dense, 'r-', lw=2, label='Fitted model')
        ax1.set_xlabel('E_bias (kV/m)', fontsize=12)
        ax1.set_ylabel('χ', fontsize=12)
        ax1.set_title(f'Fit Quality (R² = {result.r_squared:.4f})', fontsize=14, fontweight='bold')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add parameter annotation
        r_act = all_params.get('r_act', 0)
        B_s = all_params.get('B_s', 0)
        ax1.text(0.05, 0.95, f"r_act = {r_act*1e6:.2f} μm\nB_s = {B_s/1000:.2f} kJ/mol",
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2 = axes[1]
        residuals = chi_obs - chi_pred
        ax2.scatter(E_bias / 1000, residuals, s=100, c='green', marker='s')
        ax2.axhline(0, color='gray', ls='--')
        ax2.set_xlabel('E_bias (kV/m)', fontsize=12)
        ax2.set_ylabel('Residual (obs - pred)', fontsize=12)
        ax2.set_title('Residuals', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fit_plot = out_path / "fit_quality.png"
        plt.savefig(fit_plot, dpi=150)
        plt.close()
        plots["fit_quality"] = str(fit_plot)
        
        # Plot 2: Bootstrap distributions (if available)
        if result.param_uncertainties:
            fig, axes = plt.subplots(1, len(result.param_uncertainties), figsize=(5*len(result.param_uncertainties), 4))
            if len(result.param_uncertainties) == 1:
                axes = [axes]
            
            for ax, (name, stats) in zip(axes, result.param_uncertainties.items()):
                # Generate samples for histogram (approximation from stats)
                if stats.get('std', 0) > 0:
                    samples = np.random.normal(stats['mean'], stats['std'], 1000)
                    
                    if name == 'r_act':
                        ax.hist(samples * 1e6, bins=30, density=True, alpha=0.7, color='steelblue')
                        ax.axvline(stats['mean'] * 1e6, color='red', lw=2, label=f"Mean: {stats['mean']*1e6:.2f} μm")
                        ax.axvline(stats['ci_low'] * 1e6, color='orange', ls='--', label='95% CI')
                        ax.axvline(stats['ci_high'] * 1e6, color='orange', ls='--')
                        ax.set_xlabel('r_act (μm)')
                    elif name == 'B_s':
                        ax.hist(samples / 1000, bins=30, density=True, alpha=0.7, color='steelblue')
                        ax.axvline(stats['mean'] / 1000, color='red', lw=2, label=f"Mean: {stats['mean']/1000:.2f} kJ/mol")
                        ax.axvline(stats['ci_low'] / 1000, color='orange', ls='--', label='95% CI')
                        ax.axvline(stats['ci_high'] / 1000, color='orange', ls='--')
                        ax.set_xlabel('B_s (kJ/mol)')
                    else:
                        ax.hist(samples, bins=30, density=True, alpha=0.7, color='steelblue')
                        ax.axvline(stats['mean'], color='red', lw=2, label=f"Mean: {stats['mean']:.2e}")
                        ax.axvline(stats['ci_low'], color='orange', ls='--', label='95% CI')
                        ax.axvline(stats['ci_high'], color='orange', ls='--')
                        ax.set_xlabel(name)
                    
                    ax.set_ylabel('Density')
                    ax.set_title(f'{name} Distribution')
                    ax.legend(fontsize=8)
            
            plt.tight_layout()
            dist_plot = out_path / "parameter_distributions.png"
            plt.savefig(dist_plot, dpi=150)
            plt.close()
            plots["parameter_distributions"] = str(dist_plot)
        
        return plots
    
    def load_dataset_with_temp(
        self, 
        run_ids: List[str], 
        runs_dir: Path = Path("results/runs"),
        aggregation: str = "mean"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load dataset from registered runs including temperature.
        
        Args:
            run_ids: List of run IDs to load
            runs_dir: Directory containing runs
            aggregation: How to aggregate field data
        
        Returns:
            Tuple of (E_bias, T_gas, chi) arrays
        """
        import h5py
        
        E_bias_list = []
        T_gas_list = []
        chi_list = []
        
        for run_id in run_ids:
            run_path = runs_dir / run_id
            fields_file = run_path / "outputs" / "fields.h5"
            
            if not fields_file.exists():
                logger.warning(f"Fields file not found for run {run_id}")
                continue
            
            with h5py.File(fields_file, 'r') as f:
                E_bias_data = f['em/E_bias'][:]
                T_gas_data = f['thermal/T_gas'][:]
                chi_data = f['flash/chi'][:]
                
                if aggregation == "mean":
                    E_bias_list.append(np.mean(E_bias_data))
                    T_gas_list.append(np.mean(T_gas_data))
                    chi_list.append(np.mean(chi_data))
        
        return np.array(E_bias_list), np.array(T_gas_list), np.array(chi_list)
    
    def train_with_temp(
        self, 
        problem_id: str, 
        run_ids: List[str],
        infer: List[str], 
        known: Dict[str, Any],
        output_dir: Optional[str] = None,
        n_bootstrap: int = 100
    ) -> InversionResult:
        """
        Train inverse model with temperature-dependent DeltaG0(T).
        """
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.results_dir / problem_id
        out_path.mkdir(parents=True, exist_ok=True)
        
        result = InversionResult(
            problem_id=problem_id,
            status="running",
            known_params=dict(known),
            run_ids=list(run_ids),
            output_dir=str(out_path),
        )
        
        try:
            logger.info(f"Loading dataset from {len(run_ids)} runs...")
            E_bias, T_gas, chi_obs = self.load_dataset_with_temp(run_ids)
            
            result.n_points = len(E_bias)
            logger.info(f"Dataset: {len(E_bias)} points, T range [{T_gas.min():.1f}, {T_gas.max():.1f}] K")
            
            known_typed = {k: int(v) if k == 'n' else float(v) for k, v in known.items()}
            
            logger.info(f"Fitting parameters: {infer}")
            fitted, mse = fit_inverse_with_temp(E_bias, T_gas, chi_obs, infer, known_typed)
            
            result.fitted_params = fitted
            result.mse = float(mse)
            result.rmse = float(np.sqrt(mse))
            
            all_params = dict(known_typed)
            all_params.update(fitted)
            
            chi_pred = chi_model_with_temp(
                E_bias, T_gas,
                r_act=all_params.get('r_act', 1e-9),
                B_s=all_params.get('B_s', 50000),
                DeltaG0_ref=all_params.get('DeltaG0_ref', all_params.get('DeltaG0', 350000)),
                beta_T=all_params.get('beta_T', 100),
                T_ref=all_params.get('T_ref', 400),
                k_soft=all_params.get('k_soft', 1.0),
                n=int(all_params.get('n', 2)),
            )
            ss_res = np.sum((chi_obs - chi_pred) ** 2)
            ss_tot = np.sum((chi_obs - np.mean(chi_obs)) ** 2)
            result.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            logger.info(f"Fit: MSE={mse:.2e}, R²={result.r_squared:.4f}")
            for name, val in fitted.items():
                if name == 'r_act':
                    logger.info(f"  {name} = {val*1e6:.2f} μm")
                elif name in ['B_s', 'DeltaG0_ref']:
                    logger.info(f"  {name} = {val/1000:.2f} kJ/mol")
                elif name == 'beta_T':
                    logger.info(f"  {name} = {val:.2f} J/(mol·K)")
            
            if n_bootstrap > 0:
                logger.info(f"Bootstrap uncertainty ({n_bootstrap} samples)...")
                uncertainties = self._bootstrap_with_temp(E_bias, T_gas, chi_obs, infer, known_typed, n_bootstrap)
                result.param_uncertainties = uncertainties
            
            result.status = "success"
            
        except Exception as e:
            logger.error(f"Inversion failed: {e}")
            result.status = "failed"
            raise
        finally:
            self._problems[problem_id] = result
            with open(out_path / "inversion_result.json", 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        return result
    
    def _bootstrap_with_temp(self, E_bias, T_gas, chi_obs, infer, known, n_bootstrap=100, confidence=0.95):
        """Bootstrap for temperature-dependent model."""
        n_points = len(E_bias)
        results = {name: [] for name in infer}
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_points, size=n_points, replace=True)
            try:
                fitted, _ = fit_inverse_with_temp(E_bias[idx], T_gas[idx], chi_obs[idx], infer, known)
                for name in infer:
                    results[name].append(fitted[name])
            except:
                continue
        
        alpha = 1 - confidence
        uncertainties = {}
        for name in infer:
            samples = np.array(results[name])
            if len(samples) >= 10:
                uncertainties[name] = {
                    "mean": float(np.mean(samples)),
                    "std": float(np.std(samples)),
                    "ci_low": float(np.percentile(samples, 100 * alpha / 2)),
                    "ci_high": float(np.percentile(samples, 100 * (1 - alpha / 2))),
                    "n_samples": len(samples),
                }
            else:
                uncertainties[name] = {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        return uncertainties
    
    def train_bayesian_with_temp(
        self, problem_id: str, run_ids: List[str], infer: List[str],
        fixed: Dict[str, Any], priors: Dict[str, Dict[str, float]],
        output_dir: Optional[str] = None, n_samples: int = 500
    ) -> BayesianInversionResult:
        """Bayesian inversion with temperature-dependent DeltaG0(T) and priors."""
        out_path = Path(output_dir) if output_dir else self.results_dir / problem_id
        out_path.mkdir(parents=True, exist_ok=True)
        
        result = BayesianInversionResult(
            problem_id=problem_id, status="running",
            run_ids=list(run_ids), output_dir=str(out_path)
        )
        
        try:
            logger.info(f"Loading dataset from {len(run_ids)} runs...")
            E_bias, T_gas, chi_obs = self.load_dataset_with_temp(run_ids)
            result.n_points = len(E_bias)
            
            fixed_typed = {k: int(v) if k == 'n' else float(v) for k, v in fixed.items()}
            prior_specs = {name: PriorSpec(
                distribution=spec.get('distribution', 'normal'),
                mean=spec.get('mean', 0), std=spec.get('std', 1),
                low=spec.get('low', 0), high=spec.get('high', 1),
            ) for name, spec in priors.items()}
            result.priors = priors
            
            logger.info(f"Bayesian inversion ({n_samples} samples), infer: {infer}, priors: {list(priors.keys())}")
            
            all_names = list(infer) + list(prior_specs.keys())
            posterior_samples = {name: [] for name in all_names}
            mse_list, r2_list = [], []
            
            for i in range(n_samples):
                prior_vals = {}
                for name, prior in prior_specs.items():
                    val = prior.sample(1)[0]
                    if name == 'beta_T' and val < 1.0:
                        val = abs(val) + 1.0
                    prior_vals[name] = val
                
                idx = np.random.choice(len(E_bias), size=len(E_bias), replace=True)
                known_sample = dict(fixed_typed)
                known_sample.update(prior_vals)
                
                try:
                    fitted, mse = fit_inverse_with_temp(E_bias[idx], T_gas[idx], chi_obs[idx], infer, known_sample)
                    for name in infer:
                        posterior_samples[name].append(fitted[name])
                    for name, val in prior_vals.items():
                        posterior_samples[name].append(val)
                    
                    all_p = dict(known_sample)
                    all_p.update(fitted)
                    chi_pred = chi_model_with_temp(
                        E_bias, T_gas, all_p.get('r_act', 1e-9), all_p.get('B_s', 50000),
                        all_p.get('DeltaG0_ref', 350000), all_p.get('beta_T', 100),
                        all_p.get('T_ref', 400), all_p.get('k_soft', 1.0), int(all_p.get('n', 2))
                    )
                    ss_res = np.sum((chi_obs - chi_pred) ** 2)
                    ss_tot = np.sum((chi_obs - np.mean(chi_obs)) ** 2)
                    mse_list.append(mse)
                    r2_list.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
                except:
                    continue
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  {i+1}/{n_samples} samples")
            
            for name in all_names:
                posterior_samples[name] = np.array(posterior_samples[name])
            
            posterior_stats = {}
            for name in all_names:
                samples = posterior_samples[name]
                if len(samples) > 0:
                    posterior_stats[name] = {
                        "mean": float(np.mean(samples)), "std": float(np.std(samples)),
                        "median": float(np.median(samples)),
                        "ci_low": float(np.percentile(samples, 2.5)),
                        "ci_high": float(np.percentile(samples, 97.5)),
                        "n_samples": len(samples),
                    }
            
            param_arrays = [posterior_samples[n] for n in all_names]
            corr_mat = np.corrcoef(param_arrays) if all(len(a) > 0 for a in param_arrays) else np.eye(len(all_names))
            
            result.posterior_samples = posterior_samples
            result.posterior_stats = posterior_stats
            result.correlation_matrix = corr_mat
            result.param_names = all_names
            result.mean_mse = float(np.mean(mse_list)) if mse_list else 0.0
            result.mean_r_squared = float(np.mean(r2_list)) if r2_list else 0.0
            result.n_samples = len(mse_list)
            
            logger.info(f"Complete (R²={result.mean_r_squared:.4f}):")
            for name, stats in posterior_stats.items():
                if name == 'r_act':
                    logger.info(f"  {name}: {stats['mean']*1e6:.2f} ± {stats['std']*1e6:.2f} μm")
                elif name in ['B_s', 'DeltaG0_ref']:
                    logger.info(f"  {name}: {stats['mean']/1000:.2f} ± {stats['std']/1000:.2f} kJ/mol")
                elif name == 'beta_T':
                    logger.info(f"  {name}: {stats['mean']:.1f} ± {stats['std']:.1f} J/(mol·K)")
            
            result.status = "success"
        except Exception as e:
            logger.error(f"Failed: {e}")
            result.status = "failed"
            raise
        finally:
            with open(out_path / "bayesian_result.json", 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        return result
