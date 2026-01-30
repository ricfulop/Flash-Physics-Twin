# PFR Digital Twin Stack (MIT) — Zero‑Shot IDE Spec
**Goal:** Build a Cursor‑driven (MCP‑orchestrated) simulation + inference stack for the PFR reactor: **EM (Flexcompute)** + **core multiphysics (COMSOL)** + **Flash closure** + **data/inversion (NVIDIA PhysicsNeMo)** with clean interfaces and reproducible outputs.

This spec assumes:
- You will use **FlexAgent MCP** for Flexcompute products (Tidy3D workflows integrated with Cursor).  
- Cursor will talk to MCP servers via **stdio** transport (local subprocess JSON‑RPC) by default.  
- **NVIDIA PhysicsNeMo** is used for **physics‑informed / data‑driven ML**, parameter inversion, and surrogate/operator learning.  

---

## 0) What runs in Cursor
Cursor is the **single cockpit** for:
1) Generating/validating **simulation inputs** (geometry, parameters, chemistry sets)  
2) Launching runs through MCP servers  
3) Pulling field maps & KPIs into consistent artifacts (HDF5/VTK/PNG)  
4) Running **inversion & surrogate training** (NVIDIA PhysicsNeMo)  
5) Maintaining a living “canonical context” for PFR (docs + datasets + decisions)

---

## 1) Recommended tool stack (roles + boundaries)

### 1.1 Flexcompute (via FlexAgent MCP)
- **Role:** EM/coil design and (optionally) providing RF field/coupling curves used by the reactor engine.
- **What we use it for:**  
  - coil geometry sweeps, impedance/coupling metrics  
  - baseline |E|, |B| distribution maps in geometry  
  - exporting a compact “EM surrogate” for the reactor engine: `Q_RF(r,z)` or `Pabs(σ_eff)` look‑up.
- **Why:** FlexAgent MCP exists specifically as an AI+Cursor integration for Tidy3D workflows.

### 1.2 COMSOL (custom MCP server)
- **Role:** “truth” multiphysics reactor engine: **plasma fluid + flow + heat + species + bias** + **Flash closure**.
- **What we use it for:**  
  - stable coupled PDE solves (axisymmetric first)  
  - compute: `n_e, T_e, Q_RF, T_gas, T_wall, c_H2, c_H, c_H2O, E_bias, ΔB, χ`  
  - param sweeps and design point evaluation

### 1.3 NVIDIA PhysicsNeMo (PINNs / neural operators)
- **Role:** inference + surrogates + parameter discovery:
  - infer `r_act`, `k_soft`, `B_smooth`, effective `Δμ_chem` proxies from experimental data
  - build fast predictors of maps/KPIs for optimization loops
- **Basis:** PhysicsNeMo solves physics‑constrained learning problems by minimizing PDE residuals, BC losses, and data mismatch.  
  It **does not replace COMSOL**; it learns from COMSOL outputs and experiments.

### 1.4 Optional: kinetic/PIC validation
- **Role:** validate sheath/SEE corner cases; fit boundary parameters that feed COMSOL.
- **Not required for v1** if you adopt Bohm + energy‑flux sheath BCs (reduced model).

---

## 2) Repository: YES, start a GitHub repo
You should start a dedicated repo. Benefits:
- reproducible simulation provenance (inputs → outputs)
- team collaboration (issues/PRs)
- version control for the Flash closure and parameter tables (this is the IP)

### 2.1 Suggested repo name
`Flash-Physics-Twin`

### 2.2 Monorepo layout (works well in Cursor)
```
Flash-Physics-Twin/
  README.md
  docs/
    PFR_COMSOL_CONTEXT.md
    PFR_Physics_Equations.md
    PFR_BC_Sheath_Model.md
    PFR_Chemistry_Minimal.md
    PFR_Validation_Plan.md
    PFR_Data_Schema.md
    PFR_Modeling_Assumptions.md
    PFR_Runbook.md
  params/
    geometry.yaml
    ops.yaml
    materials.yaml
    chemistry.yaml
  mcp/
    comsol_mcp/
      README.md
      server.py
      tools.json
      templates/
        pfr_axisym_template.mph
    physicsnemo_mcp/
      README.md
      server.py
      tools.json
      models/
        flash_inverse/
        flash_surrogate/
    pfr_data_mcp/
      server.py
      tools.json
  workflows/
    run_design_point.py
    sweep_rf_power.py
    sweep_bias.py
    calibrate_flash_params.py
  data/
    raw/
    processed/
  results/
    runs/
  notebooks/
  ci/
```

---

## 3) Physics packages you must implement (minimal complete PFR v1)

### 3.1 Domains (what’s solved)
- RF EM → absorbed power (`Q_RF`) (from Flexcompute or internal COMSOL RF interface)
- Plasma fluid (ICP): `n_e, T_e`, ions (minimal)
- Flow: `u, p` (laminar)
- Heat: `T_gas, T_wall`
- Species: `c_H2, c_H, c_H2O` (minimal set)
- Bias electrostatics: `φ`, `E_bias`
- Flash closure: `ΔB`, `χ`, blended properties and rate multipliers

### 3.2 Flash closure (canonical)
Use the barrier balance and χ smoothing:
- `ΔB = k_soft*ΔG0 - (n*F*E_bias*r_act + W_ph + Δμ_chem)`
- `χ = 1/(1+exp(ΔB/B_smooth))`
and apply χ to blend:
- conductivity `σ(χ)`
- diffusion `D_eff(χ,E,T)`
- reaction rate `k_eff(χ,E,T,chem)`

(Defined in `PFR_COMSOL_CONTEXT.md`.)

### 3.3 Sheath (v1 reduced BCs)
Implement sheath as boundary conditions (not a resolved region):
- ion Bohm flux (sets ion wall flux)
- electron suppression via exp(−eVs/kT_e)
- energy flux terms for thermal balance  
Deliverable doc: `docs/PFR_BC_Sheath_Model.md`.

---

## 4) MCP servers you should have (beyond FlexAgent)

### 4.1 `comsol_mcp`
**Purpose:** Create/modify/run a COMSOL model headlessly, return maps + KPIs.

### 4.2 `physicsnemo_mcp`
**Purpose:** Train and query NVIDIA PhysicsNeMo models for inversion and surrogates.

**Core tools**
- `physicsnemo.train_inverse(problem_id, data_ref, priors)`
- `physicsnemo.train_surrogate(problem_id, dataset_ref)`
- `physicsnemo.predict(problem_id, inputs)`
- `physicsnemo.evaluate(problem_id, testset_ref)`

### 4.3 `pfr_data_mcp`
**Purpose:** One “source of truth” for runs, artifacts, schemas.

---

## 5) End‑to‑end workflow (zero‑shot goal)
A new team member should be able to:
1. Run a baseline case
2. Sweep bias or RF power
3. Validate against experiment
4. Invert Flash parameters with PhysicsNeMo
5. Train a surrogate for rapid exploration

All steps are encoded in `PFR_Runbook.md`.

---

## 6) What this spec enforces

- Clear separation between **physics enforcement** (COMSOL) and **physics discovery** (NVIDIA PhysicsNeMo)
- MCP‑first automation compatible with Cursor
- Reproducible, schema‑validated outputs
- Controlled upgrade path when assumptions fail

---

**End of Zero‑Shot IDE Spec.**
