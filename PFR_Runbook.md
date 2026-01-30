# PFR_Runbook.md
**Plasma Flash Reduction (PFR) — Operational Runbook (v1)**

**Status:** canonical  
**Audience:** MIT students, postdocs, collaborators, MCP automation, reviewers  
**Purpose:** Define **exactly how the PFR digital twin is executed**, end-to-end, using Cursor + MCP, with **no tribal knowledge** required.

If you follow this document, you will:
- produce valid, comparable runs
- avoid silent physics drift
- know when a result is trustworthy or invalid
- know when to escalate model complexity

---

## 0) Golden rules (read once, obey always)

1. **Never edit physics inside a solver without updating the docs**
   - Physics is defined in:
     - `PFR_COMSOL_CONTEXT.md`
     - `PFR_BC_Sheath_Model.md`
     - `PFR_Chemistry_Minimal.md`

2. **Never tune Flash parameters by hand**
   - `r_act`, `k_soft`, `B_s` are inferred, not guessed.

3. **Never compare runs across model versions**
   - Check `model_version` in `run_card.json`.

4. **If a run fails schema validation, it does not exist**

5. **NVIDIA PhysicsNeMo does not replace COMSOL**
   - COMSOL enforces continuum physics
   - NVIDIA PhysicsNeMo learns, inverts, and accelerates that physics

---

## 1) Canonical execution stack

| Layer | Tool | Role |
|----|----|----|
| Orchestration | Cursor + MCP | Single cockpit |
| EM (optional) | Flexcompute (Tidy3D via FlexAgent MCP) | Coil & RF coupling |
| Reactor physics | COMSOL (custom MCP server) | Ground-truth multiphysics |
| Discovery & surrogates | **NVIDIA PhysicsNeMo** | Inversion, uncertainty, surrogates |
| Data integrity | `pfr_data_mcp` | Schema enforcement & provenance |

---

## 2) Directory conventions (mandatory)

All work happens inside the repository.

```
Flash-Physics-Twin/
  params/        # Editable inputs only
  docs/          # Canonical physics & procedures
  mcp/           # MCP servers (comsol_mcp, physicsnemo_mcp, pfr_data_mcp)
  workflows/     # Cursor-invoked execution scripts
  results/       # Immutable run artifacts
```

Rules:
- **Never edit files under `results/`**
- **Never read directly from `params/` during a run**
- Every run must copy inputs into its own `inputs/` directory

---

## 3) Workflow overview (strict order)

Every valid run must execute **in this order**:

1. Input validation (`pfr_data_mcp`)
2. EM coupling (FlexAgent MCP, optional or cached)
3. COMSOL multiphysics solve (`comsol_mcp`)
4. Output validation (schema + invariants)
5. Artifact storage (`results/runs/<run_id>/`)
6. Optional NVIDIA PhysicsNeMo step (inversion or surrogate training)

Skipping or re-ordering steps invalidates the run.

---

## 4) Workflow A — Baseline run

### 4.1 Purpose
Establish a reproducible reference operating point.

### 4.2 Inputs
Edit only in `params/`:
- `geometry.yaml`
- `ops.yaml`
- `materials.yaml`
- `chemistry.yaml`

Do **not** edit solver templates or MCP code.

### 4.3 Execution
From Cursor:
```bash
python workflows/run_design_point.py
```

Behind the scenes:
1. Inputs are snapshotted into `results/runs/<run_id>/inputs/`
2. `pfr_data_mcp` validates all inputs
3. `comsol_mcp` runs the A→B→C→D study sequence
4. Fields and KPIs are exported per schema

### 4.4 Required outputs
- `fields.h5`
- `kpis.json`
- `plots/summary.png`

### 4.5 Pass / fail checks
✔ `DeltaB` and `chi` exist everywhere  
✔ χ ∈ [0,1]  
✔ Power balance closes  
❌ Missing χ or ΔB → **run invalid**

---

## 5) Workflow B — Bias sweep (Flash control test)

### Purpose
Test whether **bias controls reduction through χ**, independent of plasma power.

### Execution
```bash
python workflows/sweep_bias.py
```

### Acceptance criteria
✔ χ onset shifts with bias  
✔ Reduction tracks χ, not temperature alone  
❌ Thermal Arrhenius explains onset equally well → **Flash falsified**

---

## 6) Workflow C — RF power sweep

### Purpose
Separate plasma chemistry effects from Flash physics.

### Execution
```bash
python workflows/sweep_rf_power.py
```

### Acceptance criteria
✔ Radical density changes without proportional reduction change once χ is active  
❌ Reduction scales purely with radical density → **upgrade chemistry**

---

## 7) Workflow D — Flow sweep

### Purpose
Test transport limitation vs Flash-controlled behavior.

### Execution
```bash
python workflows/sweep_flow.py
```

### Acceptance criteria
✔ Reduction robust across flow at fixed χ  
❌ Strong flow dependence → **transport-limited regime suspected**

---

## 8) Workflow E — Material swap

### Purpose
Test cross-material generality of Flash physics.

### Procedure
1. Add a new entry to `materials.yaml`
2. Update ΔG⁰, σ₀, and baseline properties
3. Re-run baseline and bias sweep

### Acceptance criteria
✔ Different onset fields but similar χ-collapsed behavior  
❌ Each material requires unrelated tuning → **model incomplete**

---

## 9) Workflow F — Validation vs experiment

### Inputs
- Experimental data mapped into `PFR_Data_Schema.md`
- Stored under `data/raw/experiment_<id>/`

### Execution
```bash
python workflows/compare_experiment.py
```

### Required plots
- Reduction vs bias (model vs experiment)
- χ vs bias
- Temperature vs bias
- Water production vs reduction

Failure here blocks all model upgrades.

---

## 10) Workflow G — Parameter inversion (NVIDIA PhysicsNeMo)

### Purpose
Infer Flash parameters from data:
- `r_act`
- `B_s`
- `k_soft` (if enabled)

### Execution
```bash
python workflows/calibrate_flash_params.py
```

This invokes **`physicsnemo_mcp`**.

### Rules
- NVIDIA PhysicsNeMo **reads** COMSOL outputs
- It **does not overwrite** reactor physics
- Posterior uncertainty must be reported

### Acceptance criteria
✔ Narrow, stable posteriors  
❌ Broad or drifting posteriors → **physics under-specified**

---

## 11) Workflow H — Surrogate training (optional)

### Purpose
Accelerate design-space exploration.

### Execution
```bash
python workflows/train_surrogate.py
```

### Restrictions
- Surrogates may replace parameter sweeps
- Surrogates may **not** define physics
- All surrogates must be benchmarked against COMSOL truth

---

## 12) What you must never do

🚫 Tune Flash parameters manually  
🚫 Skip schema validation  
🚫 Compare runs across model versions  
🚫 Use NVIDIA PhysicsNeMo to “solve the reactor”  
🚫 Add physics without updating documentation  

Violating these rules invalidates results.

---

## 13) Versioning and provenance

Every run records:
- model version
- git commit hash
- toolchain identifiers

Example (`run_card.json`):
```json
{
  "model_version": "v1.0-minimal",
  "physics_engine": "COMSOL",
  "discovery_engine": "NVIDIA-PhysicsNeMo"
}
```

---

## 14) When to escalate complexity

Escalation is allowed **only** when:
- a validation criterion fails
- the failure maps to an upgrade trigger in:
  - `PFR_Modeling_Assumptions.md`
  - `PFR_Validation_Plan.md`

No speculative upgrades.

---

## 15) Final checklist

✔ Schema-valid runs  
✔ χ drives reduction trends  
✔ Thermal-only explanations ruled out  
✔ Cross-material consistency  
✔ PhysicsNeMo posteriors reported  

If all boxes are checked, results are defensible.

---

## 16) Final philosophy

This runbook exists to make PFR modeling:
- reproducible
- falsifiable
- scalable
- transferable

If a new student can run the system in one day, the runbook has succeeded.

---

**End of PFR runbook.**
