# PFR_Data_Schema.md
**Plasma Flash Reduction (PFR) — Canonical Data & Artifact Schema**

**Status:** required  
**Audience:** simulation, experiment, MCP, ML, and analysis workflows  
**Purpose:** Define a single, strict schema for all inputs, outputs, and derived quantities so that COMSOL, Flexcompute, Modulus, and experiments interoperate cleanly.

---

## 1. Design principles
1. Everything is a run  
2. Inputs ≠ outputs ≠ derived KPIs ≠ plots  
3. Machine-readable first (HDF5 + JSON)  
4. Spatial fields are never flattened  
5. Flash physics (ΔB, χ) is first-class

---

## 2. Run directory layout
```
results/runs/<run_id>/
  run_card.json
  inputs/
    geometry.yaml
    ops.yaml
    materials.yaml
    chemistry.yaml
  outputs/
    fields.h5
    kpis.json
  plots/
    summary.png
    fields/
  logs/
    solver.log
    mcp.log
```

---

## 3. run_card.json
```json
{
  "run_id": "2026-01-30_001_baseline",
  "timestamp_utc": "2026-01-30T15:42:12Z",
  "toolchain": {
    "em": "flexcompute-tidy3d",
    "reactor": "comsol",
    "ml": "modulus"
  },
  "git": {
    "repo": "ricfulop/Flash-Physics-Twin",
    "commit": "abc1234",
    "dirty": false
  },
  "notes": "Baseline PFR run"
}
```

---

## 4. Inputs
All inputs are copied verbatim into `inputs/`.  
Solvers must **never** read from `params/` directly.

---

## 5. fields.h5 (canonical)
**Units:** SI  
**Coordinates:** axisymmetric (r,z) or Cartesian (x,y,z)

### 5.1 Attributes
- coordinate_system
- geometry_hash
- units = "SI"

### 5.2 Grid
- `/grid/r`
- `/grid/z`

### 5.3 Electromagnetics
- `/em/E_mag`
- `/em/E_bias`
- `/em/B_mag`
- `/em/Q_RF`

### 5.4 Plasma
- `/plasma/ne`
- `/plasma/Te`
- `/plasma/ni`

### 5.5 Thermal & Flow
- `/thermal/T_gas`
- `/thermal/T_wall`
- `/flow/u_r`
- `/flow/u_z`
- `/flow/p`

### 5.6 Species
- `/species/H2`
- `/species/H`
- `/species/H2O`

### 5.7 Flash
- `/flash/DeltaB`
- `/flash/chi`

---

## 6. kpis.json
```json
{
  "flash": {
    "chi_volume_avg": 0.42,
    "chi_volume_fraction_gt_0p5": 0.31
  },
  "reduction": {
    "rate_integral": 1.7e-3,
    "extent_proxy": 0.65
  },
  "power": {
    "rf_absorbed": 8200.0,
    "wall_losses": 1900.0
  },
  "thermal": {
    "T_wall_max": 1220.0,
    "T_gas_avg": 1080.0
  },
  "plasma": {
    "ne_avg": 2.1e18,
    "Te_avg": 9800.0
  }
}
```

---

## 7. Validation rules
A run is invalid if:
- required datasets are missing
- ΔB or χ is absent
- units are inconsistent
- KPIs cannot be recomputed from fields

---

## 8. ML compatibility
- χ ∈ [0,1]
- ΔB continuous
- identical grids per dataset batch

---

## 9. Philosophy
Flash is new physics.  
A strict data contract is the guardrail that makes inversion, comparison, and discovery possible.

---

**End of schema.**
