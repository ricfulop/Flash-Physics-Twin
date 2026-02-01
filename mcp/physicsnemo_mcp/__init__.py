"""
PhysicsNeMo MCP Server
Parameter inversion and surrogate training for Flash physics.
"""

from .trainer_inverse import (
    InverseTrainer, 
    chi_model, 
    InversionResult,
    BayesianInversionResult,
    PriorSpec,
    bayesian_inversion_with_prior,
)

__all__ = [
    "InverseTrainer", 
    "chi_model", 
    "InversionResult",
    "BayesianInversionResult",
    "PriorSpec",
    "bayesian_inversion_with_prior",
]
