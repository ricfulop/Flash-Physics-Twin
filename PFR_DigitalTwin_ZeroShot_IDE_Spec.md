# PFR Digital Twin Stack (MIT) — Zero‑Shot IDE Spec
**Goal:** Build a Cursor‑driven (MCP‑orchestrated) simulation + inference stack for the PFR reactor: **EM (Flexcompute)** + **core multiphysics (COMSOL)** + **Flash closure** + **data/inversion (Modulus / PINNs)** with clean interfaces and reproducible outputs.

This spec assumes:
- You will use **FlexAgent MCP** for Flexcompute products (Tidy3D workflows integrated with Cursor). citeturn0search0turn0search3  
- Cursor will talk to MCP servers via **stdio** transport (local subprocess JSON‑RPC) by default. citeturn0search4turn0search2  
- Modulus Sym is used for **physics‑informed / data‑driven ML** where you express the physics as an optimization problem on geometry/data. citeturn0search10turn0search1

---

## 0) What runs in Cursor
Cursor should be the **single cockpit** for:
1) Generating/validating **simulation inputs** (geometry, parameters, chemistry sets)  
2) Launching runs through MCP servers  
3) Pulling field maps & KPIs into consistent artifacts (HDF5/VTK/PNG)  
4) Running **inversion & surrogate training** (Modulus)  
5) Maintaining a living “canonical context” for PFR (docs + datasets + decisions)

---

## 1) Recommended tool stack (roles + boundaries)

### 1.1 Flexcompute (via FlexAgent MCP)
- **Role:** EM/coil design and (optionally) providing RF field/coupling curves used by the reactor engine.
- **What we use it for:**  
  - coil geometry sweeps, impedance/coupling metrics  
  - baseline |E|, |B| distribution maps in geometry  
  - exporting a compact “EM surrogate” for the reactor engine: `Q_RF(r,z)` or `Pabs(σ_eff)` look‑up.
- **Why:** FlexAgent MCP exists specifically as an AI+Cursor integration for Tidy3D workflows. citeturn0search0turn0search3

### 1.2 COMSOL (custom MCP server)
- **Role:** “truth” multiphysics reactor engine: **plasma fluid + flow + heat + species + bias** + **Flash closure**.
- **What we use it for:**  
  - stable coupled PDE solves (axisymmetric first)  
  - compute: `n_e, T_e, Q_RF, T_gas, T_wall, c_H2, c_H, c_H2O, E_bias, ΔB, χ`  
  - param sweeps and design point evaluation

### 1.3 Modulus (PINNs / neural operators)
- **Role:** inference + surrogates + parameter discovery:
  - infer `r_act`, `k_soft`, `B_smooth`, effective `Δμ_chem` proxies from experimental data
  - build fast predictors of maps/KPIs for optimization loops
- **Basis:** Modulus Sym solves PDEs via NN by minimizing physics residual + BC loss; supports building geometry and optimization‑posed physics. citeturn0search1turn0search10

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
`pfr-digital-twin` (or `generalflash-pfr-sim` if internal)

### 2.2 Monorepo layout (works well in Cursor)
```
pfr-digital-twin/
  README.md
  docs/
    PFR_COMSOL_CONTEXT.md
    PFR_Physics_Equations.md
    PFR_BC_Sheath_Model.md
    PFR_Chemistry_Minimal.md
    PFR_Validation_Plan.md
    PFR_Data_Schema.md
    PFR_Modeling_Assumptions.md
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
    modulus_mcp/
      README.md
      server.py
      tools.json
      models/
        flash_inverse/
        flash_surrogate/
  workflows/
    run_design_point.py
    sweep_rf_power.py
    sweep_bias.py
    calibrate_r_act.py
  data/
    raw/
    processed/
  results/
    runs/
      2026-01-30_001_baseline/
        inputs/
        outputs/
        logs/
  notebooks/
  ci/
```

---

## 3) Physics packages you must implement (minimal complete PFR v1)
This section is “what success means” for the model.

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

(These are already in `PFR_COMSOL_CONTEXT.md`.)

### 3.3 Sheath (v1 reduced BCs)
Implement sheath as boundary conditions (not a resolved region):
- ion Bohm flux (sets ion wall flux)
- electron suppression via exp(−eVs/kT_e)
- energy flux terms for thermal balance
Deliverable doc: `docs/PFR_BC_Sheath_Model.md`.

---

## 4) MCP servers you should have (beyond FlexAgent)
Cursor MCP is meant to connect to external systems. citeturn0search2turn0search4  
Here are the MCP servers that will “make this go right”:

### 4.1 `comsol_mcp` (you likely need to write this)
**Purpose:** Create/modify/run a COMSOL model headlessly, return maps + KPIs.

**Core tools (MCP “functions”)**
- `comsol.open_model(path_or_template)`
- `comsol.set_params(dict)` (supports units)
- `comsol.build_geometry()`
- `comsol.mesh(config)`
- `comsol.run_study(study_id)` (A/B/C/D pipeline)
- `comsol.export_fields(fields: [...], format: h5/vtk/csv)`
- `comsol.export_kpis()` (JSON)
- `comsol.render_png(plotspec)` (quick visual sanity checks)

**Notes**
- Use **stdio transport** for Cursor to launch server locally. citeturn0search4
- Keep COMSOL model templates in `mcp/comsol_mcp/templates/`.

### 4.2 `modulus_mcp`
**Purpose:** Train and query Modulus models for inversion/surrogates.

**Core tools**
- `modulus.train_inverse(problem_id, data_ref, priors)`
- `modulus.train_surrogate(problem_id, dataset_ref)`
- `modulus.predict(problem_id, inputs)` → maps/KPIs
- `modulus.export_model(problem_id)` (for deployment)
- `modulus.evaluate(problem_id, testset_ref)` (metrics)

**Notes**
- Modulus frames physics solutions as optimization problems on geometry/data. citeturn0search10turn0search1
- Start with inverse problems on **1D/2D reduced geometry** before trying 3D.

### 4.3 `pfr_data_mcp`
**Purpose:** One “source of truth” for runs, artifacts, schemas.
- manages run directories, checksums, metadata, and produces a “run card”
- validates outputs conform to `docs/PFR_Data_Schema.md`
- provides query tools: “show runs that match ops/geometry/material constraints”

### 4.4 `literature_mcp` (optional but helpful)
**Purpose:** Keep PFR citations/plots traceable.
- stores PDFs, extracted tables (ΔG°, kinetics), and figure digitizations
- links each parameter in `materials.yaml` to a source entry

### 4.5 `instrument_mcp` (phase 2)
**Purpose:** Pull experiment logs into the same schema.
- power supply logs (I,V,t)
- flow/pressure logs
- spectroscopy logs (if any)
- oxygen/water monitoring (TDLAS)
Produces standardized `data/raw/...` entries with metadata.

---

## 5) “Documents we need” checklist (what must exist in `docs/`)
You asked for all physics/docs needed for success. Minimum set:

1. **`PFR_COMSOL_CONTEXT.md`** (canonical equations + parameters) ✅
2. **`PFR_Physics_Equations.md`**  
   - same content as context, but expanded derivations + units checks
3. **`PFR_Data_Schema.md`**  
   - defines HDF5 groups/datasets and KPI JSON schema
4. **`PFR_Modeling_Assumptions.md`**  
   - what’s neglected (kinetic EEDF, sheath resolution, turbulence, etc.) + why
5. **`PFR_BC_Sheath_Model.md`**  
   - reduced sheath BCs, knobs, validity regime
6. **`PFR_Chemistry_Minimal.md`**  
   - minimal reactions + rate coefficient sources
7. **`PFR_Validation_Plan.md`**  
   - what experiments validate which submodels; acceptance criteria; plots to compare
8. **`PFR_Runbook.md`**  
   - “how to run a design point”, “how to add a new material”, “how to train inversion”

---

## 6) How to set up Modulus (team‑friendly)
Modulus is Python‑based; best practice is to standardize on a single environment.

### 6.1 Environment
- Use **uv** or **conda**; keep a lockfile.
- Prefer GPU nodes if available; CPU works for small 1D/2D.

### 6.2 Project conventions
- Put Modulus training configs in `mcp/modulus_mcp/models/<project>/config.yaml`
- Save:
  - training data snapshot hash
  - model weights
  - metrics
  - plots

### 6.3 First Modulus deliverables (do these before anything big)
1) **Inverse fit of `r_act` and `B_smooth`** from a small dataset (one material)  
2) **Surrogate of χ fraction / reduction KPI** over `(P_RF, V_bias, p, Q_H2)` in 2D axisym

This is aligned with Modulus’ strength: solving/inverting physics as optimization. citeturn0search1turn0search10

---

## 7) End‑to‑end workflow (what “zero‑shot” means)
A new team member should be able to:

### 7.1 Run a baseline
1. `cursor` → run `workflows/run_design_point.py`
2. MCP calls:
   - FlexAgent (optional): generate EM coupling artifact
   - COMSOL MCP: run A→B→C→D studies
3. Results written to `results/runs/<run_id>/`

### 7.2 Sweep & optimize
- run `workflows/sweep_bias.py` and `workflows/sweep_rf_power.py`
- produce plots: χ volume fraction, reduction KPI, wall heat load

### 7.3 Calibrate constants
- run `workflows/calibrate_r_act.py` (Modulus MCP)
- updates `params/materials.yaml` via a PR

---

## 8) Milestones (sequence that avoids dead ends)

### Milestone 1 — “Coupled reactor solve” (COMSOL only)
- stable solution at baseline conditions (0.15 bar, 100 SLPM, 10 kW, 1 kV) with χ enabled
- exports all maps + KPIs

### Milestone 2 — “EM‑informed heating” (Flexcompute + COMSOL)
- use FlexAgent/Tidy3D to generate a coil coupling artifact
- feed as `Q_RF(r,z)` or `Pabs(σ)` into COMSOL
- demonstrate sensitivity to coil parameters

### Milestone 3 — “Inversion loop”
- infer `r_act` for one oxide using experiment KPI curves

### Milestone 4 — “Fast surrogate”
- Modulus surrogate predicts χ fraction within acceptable error vs COMSOL

---

## 9) Appendix — COMSOL MCP implementation notes
Cursor provides documentation and cookbook guidance for building MCP servers. citeturn0search20turn0search2  
MCP stdio servers are subprocesses reading/writing JSON‑RPC line‑delimited messages. citeturn0search4

**Design choice:** implement `comsol_mcp` as:
- Python server (stdio)
- uses COMSOL Java API / LiveLink (depending on available licensing)
- exports results with deterministic filenames and unit metadata

---

## 10) What we are *not* solving in v1 (explicit)
- RF‑period sheath oscillations (use reduced BCs)
- Non‑Maxwellian EEDF kinetics (use parameterized rate coefficients)
- Fully resolved powder particles (effective medium first; particle tracing later)

---

## 11) Immediate next actions (no meetings needed)
1) Create the repo and commit:
   - `docs/PFR_COMSOL_CONTEXT.md` (already generated)
   - skeleton directories and YAML param files
2) Implement MCP servers in this order:
   1. `pfr_data_mcp` (small, fast win)
   2. `comsol_mcp` (core capability)
   3. `modulus_mcp` (inversion/surrogate)
3) Add first “baseline” COMSOL template model and one workflow script.

---

### Links (for the team)
- FlexAgent MCP overview / Cursor integration: citeturn0search0turn0search3  
- Cursor MCP docs + cookbook: citeturn0search2turn0search20  
- MCP stdio transport spec: citeturn0search4  
- Modulus Sym overview + PINN theory: citeturn0search10turn0search1
