# PFR_Data_Schema.md
**Plasma Flash Reduction (PFR) — Canonical Data & Artifact Schema**

**Status:** required  
**Audience:** simulation, experiment, MCP, NVIDIA PhysicsNeMo / ML, and analysis workflows  
**Purpose:** Define a single, strict schema for all inputs, outputs, and derived quantities so that **COMSOL**, **Flexcompute**, **NVIDIA PhysicsNeMo**, and experiments interoperate cleanly.

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
    "ml": "nvidia-physicsnemo"
  },
  "git": {
    "repo": "ricfulop/Flash-Physics-Twin",
    "commit": "abc1234",
    "dirty": false
  },
  "notes": "Baseline PFR run"
}
```

**Notes**
- `ml` records the *discovery / inversion engine*, not the physics solver.
- COMSOL remains the authoritative source of continuum physics fields.

---

## 4. Inputs
All inputs are copied verbatim into `inputs/`.  

**Rules**
- Solvers must **never** read directly from `params/`.
- Every run is self-contained and reproducible from its `inputs/` snapshot.

---

## 5. fields.h5 (canonical)
**Format:** HDF5  
**Units:** SI  
**Coordinates:** axisymmetric `(r,z)` or Cartesian `(x,y,z)`

---

### 5.1 Global attributes (required)
- `coordinate_system` : `"axisymmetric"` or `"cartesian"`
- `geometry_hash` : SHA-256 of geometry definition
- `units` : `"SI"`

---

### 5.2 Grid
- `/grid/r` (m)  
- `/grid/z` (m)  

(or `/grid/x`, `/grid/y`, `/grid/z` for 3D)

---

### 5.3 Electromagnetics
- `/em/E_mag`  (V/m)  
- `/em/E_bias` (V/m)  
- `/em/B_mag` (T)  
- `/em/Q_RF`  (W/m³)  

---

### 5.4 Plasma (fluid)
- `/plasma/ne` (m⁻³)  
- `/plasma/Te` (K)  
- `/plasma/ni` (m⁻³)  

---

### 5.5 Thermal & flow
- `/thermal/T_gas` (K)  
- `/thermal/T_wall` (K)  
- `/flow/u_r` (m/s)  
- `/flow/u_z` (m/s)  
- `/flow/p` (Pa)  

---

### 5.6 Species
- `/species/H2` (mol/m³)  
- `/species/H` (mol/m³)  
- `/species/H2O` (mol/m³)  

---

### 5.7 Flash physics (mandatory)
- `/flash/DeltaB` (J/mol)  
- `/flash/chi`  (dimensionless, 0–1)  

**Rule:**  
Any run missing `DeltaB` or `chi` is **invalid**.

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
A run is **invalid** if:
- required datasets are missing
- `DeltaB` or `chi` is absent
- units are inconsistent
- KPIs cannot be recomputed from `fields.h5`

These checks must be enforced by `pfr_data_mcp`.

---

## 8. NVIDIA PhysicsNeMo compatibility
This schema is intentionally designed for:
- parameter inversion
- uncertainty quantification
- surrogate and operator learning

**Invariants**
- χ ∈ [0,1]
- ΔB is continuous
- identical grids across training batches

PhysicsNeMo **consumes** this schema; it does not redefine it.

---

## 9. Philosophy
Flash is new physics.  
A strict data contract is the guardrail that makes **inversion, comparison, and discovery** possible.

If the schema drifts, the science collapses.

---

**End of schema.**
