"""
schema_checks.py
Validation helpers for PFR_Data_Schema.md
"""

import h5py
import numpy as np
from pathlib import Path

REQUIRED_GROUPS = [
    "/em", "/plasma", "/thermal", "/species", "/flash"
]

REQUIRED_DATASETS = [
    "/flash/DeltaB",
    "/flash/chi"
]

def validate_fields_h5(path: Path):
    issues = []
    with h5py.File(path, "r") as f:
        for grp in REQUIRED_GROUPS:
            if grp not in f:
                issues.append(f"Missing group {grp}")
        for ds in REQUIRED_DATASETS:
            if ds not in f:
                issues.append(f"Missing dataset {ds}")
        if "/flash/chi" in f:
            chi = f["/flash/chi"][:]
            if np.any(chi < 0) or np.any(chi > 1):
                issues.append("chi out of bounds [0,1]")
            if np.allclose(chi, 0):
                issues.append("chi identically zero everywhere")
    return issues
