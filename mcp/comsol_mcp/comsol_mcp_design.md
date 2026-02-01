# comsol_mcp — Design Specification
**COMSOL execution MCP for Flash-Physics-Twin**

**Role:** Provide a Cursor/MCP-friendly control plane for running COMSOL models headlessly, with strict integration to `pfr_data_mcp` and `PFR_Data_Schema.md`.

This MCP:
- enforces a standard study pipeline (A→B→C→D) for PFR
- sets parameters from run inputs
- exports fields + KPIs in canonical formats
- never edits physics definitions (only parameters & study execution)

---

## 1) Boundaries and responsibilities

### MUST do
1. Open a COMSOL model template (`.mph`) or an existing run model
2. Apply parameters from `results/runs/<run_id>/inputs/*.yaml`
3. Build geometry + mesh (optional, configurable)
4. Run one or more studies (default: A→B→C→D)
5. Export:
   - `outputs/fields.h5`
   - `outputs/kpis.json`
   - `plots/summary.png` (sanity)
6. Return structured status and errors to Cursor
7. (Optional) cache compiled models to speed up sweeps

### MUST NOT do
- Define new physics that contradicts `PFR_COMSOL_CONTEXT.md`
- Change constitutive laws (Flash closure, sheath model) without a template update
- Write into `params/` or global state outside a run directory

---

## 2) Execution model

Transport: stdio MCP server  
State: file-backed, per-run  
Primary unit of work: a `run_id` created and snapshotted by `pfr_data_mcp`

Canonical orchestration:
1. `pfr.create_run`
2. `pfr.snapshot_inputs`
3. `comsol.open_or_create_model`
4. `comsol.apply_inputs`
5. `comsol.run_pipeline`
6. `comsol.export_all`
7. `pfr.validate_outputs`
8. `pfr.register_run`

---

## 3) COMSOL connectivity options (choose one)

### Option A (recommended): COMSOL Java API via `mphserver`
- COMSOL provides a Java API; Python can drive it via:
  - a JVM bridge (JPype) OR
  - COMSOL’s own server (`mphserver`) + client bindings
- Pros: robust, closest to native
- Cons: requires COMSOL installation + proper license setup

### Option B: Python wrapper client (if available)
- Pros: faster to implement
- Cons: wrapper availability varies

This MCP skeleton supports both via an adapter class (`comsol_api.py`) where you implement the chosen backend.

---

## 4) Outputs: canonical contract

Writes outputs to:
- `results/runs/<run_id>/outputs/fields.h5`
- `results/runs/<run_id>/outputs/kpis.json`
- `results/runs/<run_id>/plots/summary.png`
- `results/runs/<run_id>/logs/solver.log`

Field content must satisfy `PFR_Data_Schema.md`.

---

## 5) Study pipeline (A→B→C→D)

A. Flow + Thermal baseline (no plasma)  
B. Plasma (fixed flow/thermal)  
C. Fully coupled plasma ↔ thermal  
D. Bias + Flash closure (must export DeltaB, chi)

Pipeline may be one study or multiple studies depending on the template.

---

## 6) Parameter application rules

- Parameters come from `inputs/*.yaml`
- Units must be interpreted consistently (COMSOL parses `50.8[mm]` style)
- Never silently drop unknown parameters: warn and log

---

## 7) MCP Tool API (see `comsol_mcp_tools.json`)

- `comsol.open_or_create_model`
- `comsol.apply_inputs`
- `comsol.build_geometry`
- `comsol.mesh`
- `comsol.run_pipeline`
- `comsol.run_study`
- `comsol.export_fields`
- `comsol.export_kpis`
- `comsol.render_png`
- `comsol.close_model`

---

## 8) Error handling philosophy

- Fail fast, fail loud
- Surface solver errors in `logs/solver.log`
- Always return structured error payload:
  - stage
  - exception
  - log path
  - suggested next action

---

## 9) Minimal “v0” scope

1. `open_or_create_model`
2. `apply_inputs` (global params only)
3. `run_pipeline` (single study id acceptable)
4. `export_fields` (HDF5 or intermediate formats)
5. `export_kpis` (via COMSOL evaluations or computed post-export)

Integrate with `pfr_data_mcp.validate_outputs`.

---

End of comsol_mcp design specification.
