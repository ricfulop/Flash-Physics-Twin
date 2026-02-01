# pfr_data_mcp — Design Specification
**Plasma Flash Reduction (PFR) Data & Provenance MCP**

**Role:**  
Enforce the PFR data contract, manage runs, and prevent invalid physics from entering the stack.

This MCP **does not run physics**.  
It **guards physics**.

---

## 1. Why `pfr_data_mcp` exists (non-negotiable)

Without this MCP:
- invalid COMSOL runs get used “temporarily”
- χ or ΔB goes missing and no one notices
- NVIDIA PhysicsNeMo trains on garbage
- results become non-comparable across time

With this MCP:
- **invalid runs literally cannot exist**
- Cursor workflows become safe by default
- every run is reproducible and auditable

---

## 2. Responsibilities

### Must do
1. Create runs
2. Snapshot inputs
3. Validate outputs against `PFR_Data_Schema.md`
4. Refuse invalid runs
5. Register provenance & metadata
6. Answer structured queries about runs

### Must NOT do
- run COMSOL
- run FlexAgent
- run NVIDIA PhysicsNeMo
- modify physics or parameters

---

## 3. Location in repo

```
mcp/
  pfr_data_mcp/
    server.py
    tools.json
    schema/
      PFR_Data_Schema.md
      schema_checks.py
    README.md
```

Pure Python. No heavy dependencies.

---

## 4. Transport & runtime

- MCP transport: `stdio`
- Execution model: long-running process launched by Cursor
- State: filesystem-backed (no database)

---

## 5. Canonical run lifecycle

```
created → inputs_snapshotted → outputs_validated → registered
                      ↘︎ invalid (hard stop)
```

No path to `registered` without validation.

---

## 6. Run directory contract (enforced)

```
results/runs/<run_id>/
  run_card.json
  inputs/
  outputs/
  plots/
  logs/
```

Any deviation → invalid run.

---

## 7. MCP Tool API

These are the **only** tools Cursor and other MCPs may call.

---

### 7.1 `pfr.create_run`

Create a new run directory and run card.

**Input**
```json
{
  "run_id": "optional-string",
  "notes": "string",
  "model_version": "v1.0-minimal",
  "physics_engine": "COMSOL",
  "discovery_engine": "NVIDIA-PhysicsNeMo"
}
```

**Behavior**
- Auto-generate run_id if omitted
- Create `results/runs/<run_id>/`
- Write initial `run_card.json`

**Output**
```json
{
  "run_id": "2026-02-01_003",
  "path": "results/runs/2026-02-01_003/"
}
```

---

### 7.2 `pfr.snapshot_inputs`

Freeze inputs for reproducibility.

**Input**
```json
{
  "run_id": "2026-02-01_003",
  "params_dir": "params/"
}
```

**Behavior**
- Copy geometry.yaml, ops.yaml, materials.yaml, chemistry.yaml
- Write to `inputs/`
- Hash files (SHA256)
- Store hashes in `run_card.json`
- Refuse overwrite

---

### 7.3 `pfr.validate_outputs`

Validate outputs against `PFR_Data_Schema.md`.

**Input**
```json
{
  "run_id": "2026-02-01_003"
}
```

**Checks**
- outputs/fields.h5 exists
- outputs/kpis.json exists
- Required groups and datasets exist
- flash/DeltaB and flash/chi present
- χ ∈ [0,1]
- No NaNs or infs
- Grid shapes consistent
- KPIs finite and non-negative

**Output**
```json
{
  "valid": true,
  "issues": []
}
```

or
```json
{
  "valid": false,
  "issues": ["Missing flash/chi"]
}
```

Invalid runs are marked and blocked.

---

### 7.4 `pfr.register_run`

Register a validated run.

**Input**
```json
{
  "run_id": "2026-02-01_003"
}
```

**Rules**
- Only allowed if validation passed
- Marks run as immutable and queryable

---

### 7.5 `pfr.list_runs`

List runs with filters.

**Input**
```json
{
  "filters": {
    "model_version": "v1.0-minimal",
    "status": "registered"
  }
}
```

---

### 7.6 `pfr.get_run`

Retrieve full metadata for a run.

**Input**
```json
{
  "run_id": "2026-01-30_001"
}
```

---

## 8. Hard rules enforced

- Missing χ or ΔB → invalid
- Inputs cannot change after snapshot
- Invalid runs cannot be used by PhysicsNeMo
- Registered runs are immutable
- No “temporary” runs

---

## 9. MCP interactions

### With `comsol_mcp`
Order is mandatory:
1. create_run
2. snapshot_inputs
3. run COMSOL
4. validate_outputs
5. register_run

### With `physicsnemo_mcp`
- May only consume registered runs
- Must record run_ids used for training

### With Cursor
Cursor never touches run filesystem directly.

---

## 10. Error philosophy

Be strict and loud.
Never auto-fix data.
Errors must explain exactly what is wrong.

---

## 11. Why this MCP is the keystone

If you build only one MCP correctly, build this one.
It prevents silent failure and protects scientific integrity.

---

## 12. Next steps

Recommended next:
1. Implement tools.json
2. Implement server.py skeleton
3. Add HDF5 validation helpers

---

**End of pfr_data_mcp design specification.**
