# PFR_Validation_Plan.md
**Plasma Flash Reduction (PFR) — Validation, Falsification, and Acceptance Criteria**

**Status:** canonical  
**Audience:** modeling, experiments, NVIDIA PhysicsNeMo / ML, reviewers  
**Purpose:** Define **how the PFR model is tested against reality**, what constitutes success or failure, and which measurements validate (or falsify) each physics component.

This document turns the PFR stack from a simulation into a **discovery engine**.

---

## 0) Validation philosophy (non-negotiable)

This plan is designed to:
- **falsify** the Flash hypothesis if it is wrong
- prevent “model rescue” by adding ad-hoc physics
- ensure every model upgrade is driven by **specific experimental failure**

Validation is **hierarchical**:
1. Flash onset & scaling  
2. Spatial structure  
3. Energy consistency  
4. Chemistry consistency  
5. Robustness across materials  

If the model fails at a lower level, higher-level validation is irrelevant.

---

## 1) Validation layers and what each proves

### Layer A — Flash onset & control variables
**Question:** *Is reduction controlled by Flash (χ), not temperature or plasma alone?*

### Layer B — Spatial localization
**Question:** *Does the model predict where reduction happens in space?*

### Layer C — Energy consistency
**Question:** *Can the observed reduction occur without hidden heat?*

### Layer D — Chemistry consistency
**Question:** *Are radical/water trends consistent with minimal chemistry?*

### Layer E — Cross-material collapse
**Question:** *Do different materials collapse onto a common Flash framework?*

---

## 2) Required experimental inputs (baseline)

These measurements are **mandatory** for v1 validation.

### 2.1 Electrical & EM
- RF power (forward/reflected)
- Bias voltage and current
- Coil current / impedance (if available)

### 2.2 Thermal
- Wall temperature (IR camera / pyrometer)
- Gas temperature (if available)
- Heater power (independent loop)

### 2.3 Gas chemistry
- Inlet H₂ flow
- Outlet H₂O (TDLAS preferred)
- Background O₂ / H₂O (ppm level)

### 2.4 Reduction outcome
- Mass loss or oxygen removal
- Phase analysis (XRD / XPS / LECO)
- Conductivity change (if measurable)

All experimental runs must be mapped into the **`PFR_Data_Schema.md`** format.

---

## 3) Layer A — Flash onset validation (most critical)

### 3.1 Primary validation experiment
Perform a sweep where:
- Plasma conditions are held constant  
- Temperature is held approximately constant  
- **Bias field is varied**

### 3.2 Model prediction
The model predicts:
- χ transitions from ≪1 to O(1) at a critical bias
- Reduction rate increases sharply at χ ≳ 0.5

### 3.3 Acceptance criteria
✔ Reduction onset correlates with χ, not T alone  
✔ Shifting bias shifts onset even at fixed plasma power  
❌ Reduction explained equally well by thermal Arrhenius alone → **model fails**

---

## 4) Layer B — Spatial validation

### 4.1 Required comparison
Compare:
- predicted χ(r,z) maps  
- predicted reduction rate density  

with:
- post-mortem spatial reduction patterns  
- optical emission / imaging proxies (if available)

### 4.2 Acceptance criteria
✔ Reduction localized where χ is highest  
✔ Changing geometry or bias reshapes reduction zone as predicted  
❌ Uniform reduction despite highly nonuniform χ → **model fails**

---

## 5) Layer C — Energy balance validation

### 5.1 Power accounting
For each run:
- RF absorbed power
- Bias electrical work
- Heater input
- Wall losses (measured)
- Chemical enthalpy (modeled)

### 5.2 Acceptance criteria
✔ Total modeled power matches measured inputs within uncertainty  
✔ Reduction occurs without exceeding measured wall temperatures  
❌ Reduction requires unaccounted heat → **model fails**

This is the strongest guard against “hidden thermal explanations.”

---

## 6) Layer D — Chemistry consistency

### 6.1 Radical trends (qualitative)
Model predicts:
- H radical availability scales with plasma power
- Reduction does *not* scale linearly with radical density once χ is active

### 6.2 Water production
- Predicted H₂O production zones correlate with reduction zones
- Outlet H₂O scales with integrated R_red

### 6.3 Acceptance criteria
✔ Water trends consistent with reduction extent  
✔ Radical changes alone cannot explain rate without χ  
❌ Reduction scales purely with radical density → **upgrade chemistry**

---

## 7) Layer E — Cross-material validation

### 7.1 Material sweep
Apply the same framework to:
- at least two oxides with different ΔG⁰, conductivity, lattice type

### 7.2 Expected collapse
Using:
- material-specific ΔG⁰, r_act, k_soft  
- same χ definition  

Model should:
- predict different onset fields  
- but similar χ-based reduction behavior  

### 7.3 Acceptance criteria
✔ Data collapses when plotted vs χ or ΔB  
❌ Each material requires unrelated tuning → **model incomplete**

---

## 8) Validation-driven upgrade triggers

| Failure observed | Required upgrade |
|------------------|------------------|
| Radical mismatch | Add plasma chemistry detail |
| Surface sensitivity | Add surface microkinetics |
| Intermediate phase bottleneck | Add suboxide pathways |
| Internal diffusion limits | Add explicit defect transport |
| Energy mismatch | Refine thermochemistry |

No upgrade is allowed without a **specific failed validation test**.

---

## 9) Role of NVIDIA PhysicsNeMo in validation

### 9.1 Inversion and discovery
Use **NVIDIA PhysicsNeMo** to:
- infer `r_act`, `B_s`, `k_soft`
- quantify posterior uncertainty
- test identifiability of Flash parameters

PhysicsNeMo **learns from COMSOL outputs and experiments**; it does not replace the reactor physics model.

### 9.2 Success condition
✔ Narrow posterior distributions  
✔ Parameters stable across datasets  
❌ Wide or drifting posteriors → **physics under-specified**

---

## 10) Reporting & plots (required for every material)

Each validation package must include:
1. Reduction rate vs bias (model vs experiment)
2. Reduction rate vs temperature (model vs experiment)
3. χ vs bias
4. χ-collapsed reduction curves
5. Power balance bar chart

No cherry-picking.

---

## 11) Final falsification statement

The Flash hypothesis is **falsified** if:
- reduction can be fully explained by thermal + plasma chemistry
- χ does not correlate with onset or spatial extent
- no consistent `r_act` can be inferred

The hypothesis is **supported** if:
- χ predicts onset, location, and scaling across materials
- bias provides non-thermal control
- minimal chemistry suffices to explain trends

---

## 12) Philosophy (why this plan is strict)

A model that cannot fail is not science.

This validation plan ensures:
- negative results are valuable
- upgrades are principled
- PFR either stands or falls on its core claim

---

**End of validation plan.**
