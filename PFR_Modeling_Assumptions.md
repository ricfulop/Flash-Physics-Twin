# PFR_Modeling_Assumptions.md
**Plasma Flash Reduction (PFR) — Modeling Assumptions, Scope, and Risk Register**

**Status:** canonical  
**Audience:** modeling, MCP automation, NVIDIA PhysicsNeMo / ML, experimental validation, reviewers  
**Purpose:** Explicitly document **what is assumed, approximated, or neglected**, why those choices are justified, and **what evidence forces upgrades**.  
This file is the guardrail against scope creep, overfitting, and post-hoc explanations.

---

## 0) Philosophy (why this document exists)

PFR modeling targets a **new driven regime**. Early models must be:
- **mechanism-discriminating**, not encyclopedic
- **falsifiable**, not over-parameterized
- **hierarchical**, so complexity is added only when demanded by data

This document makes the assumptions **explicit and auditable** so that:
- disagreement with experiment is informative
- upgrades are controlled
- results remain comparable across versions

---

## 1) Scope of validity (where the model is intended to work)

The v1 PFR model is intended to be valid under the following conditions:

### 1.1 Plasma regime
- Collisional, fluid plasma
- Electron density: ~10¹⁷–10¹⁹ m⁻³
- Electron temperature: few eV (order)
- Pressure ≥ O(10 mbar)
- No sustained RF breakdown or multipactor

### 1.2 Geometry & scale
- Reactor dimensions: cm-scale
- Sheath thickness ≪ reactor dimensions
- 2D axisymmetric geometry captures dominant physics
- Walls are smooth, macroscopically planar

### 1.3 Temporal regime
- Quasi-steady operation
- RF-period dynamics averaged out
- No nanosecond pulsed bias effects (v1)

### 1.4 Materials regime
- Oxides treated as homogeneous continua
- No explicit grain-scale resolution
- No morphology evolution (cracking, melting) in v1

---

## 2) Explicit assumptions (what we assume to be true)

### 2.1 Flash physics assumptions
- Reduction and transport are enabled by **field-driven barrier collapse**
- This collapse is captured by a **single order parameter χ**
- χ represents coarse-grained defect percolation, not a microscopic species
- χ is continuous and smooth in space (no hard phase boundaries)

### 2.2 Plasma assumptions
- Electron population at sheath edge is Maxwellian
- Plasma chemistry affects *availability* of H radicals, not the rate-limiting step
- Plasma conductivity is adequately represented by a Drude-like form

### 2.3 Chemistry assumptions
- Reduction can be represented as a single effective step MO → M
- Reaction rate is Arrhenius-like with χ-modified activation energy
- Wall recombination dominates H radical losses
- Gas-phase chemistry beyond H₂/H/H₂O is second order in v1

### 2.4 Thermal assumptions
- Heat transport is continuum (no ballistic effects)
- Reaction enthalpy can be represented as an effective volumetric source/sink
- Radiative losses are negligible compared to conduction/convection (v1)

### 2.5 Electrical assumptions
- Bias field is quasi-static
- Sheath is unresolved and represented by reduced BCs
- No significant dielectric charging dynamics in v1

---

## 3) Approximations (what we know is approximate)

These are **deliberate approximations**, not oversights.

### 3.1 Effective plasma dissociation
- H₂ dissociation is represented by an effective rate k_diss(T_e)
- Detailed EEDF effects are collapsed into this coefficient

### 3.2 χ smoothing
- The Flash transition is smoothed with a logistic function
- This represents spatial disorder, heterogeneity, and percolation
- Sharp, discontinuous transitions are intentionally avoided

### 3.3 Effective diffusion and conductivity
- Defect-mediated transport is captured via χ-dependent multipliers
- Explicit vacancy concentrations are not solved

### 3.4 Solid state representation
- Solids are implicit; only their reaction extent matters
- No explicit elastic, plastic, or fracture mechanics in v1

---

## 4) What is intentionally neglected (and why)

### 4.1 Kinetic plasma effects
- Non-Maxwellian EEDF
- PIC-resolved sheath dynamics
- Secondary electron cascades

**Why:** not identifiable from v1 observables; would mask Flash mechanism.

### 4.2 Detailed surface microkinetics
- Adsorption/desorption steps
- Site densities and coverage
- Catalytic wall effects

**Why:** Flash hypothesis asserts bulk barrier collapse; surface-limited models would invalidate it.

### 4.3 Multi-step oxide chemistry
- Explicit suboxide intermediates
- Stoichiometric phase tracking

**Why:** intermediates are consequences of oxygen chemical potential collapse, not primary drivers.

### 4.4 Explicit defect chemistry
- Vacancy concentration fields
- Coupled electron–ion defect transport

**Why:** χ is a coarse-grained replacement aligned with measurable observables.

### 4.5 Radiation and optical effects
- Line radiation
- Plasma emission heating

**Why:** expected to be small compared to conductive/convective heat fluxes at v1 conditions.

---

## 5) Upgrade triggers (non-negotiable)

| Area | Trigger |
|----|--------|
| Plasma chemistry | Radical densities cannot be matched by any reasonable k_diss |
| Surface kinetics | Reduction rate scales with surface area, not χ |
| Suboxide pathways | Conversion stalls at known intermediate stoichiometries |
| Defect transport | Internal gradients inconsistent with χ-modified diffusion |
| Thermal model | Wall or gas temperatures disagree systematically |
| Sheath model | Evidence of SEE, arcing, or strong bias instabilities |

---

## 6) Risk register (how this model could be wrong)

### Risk 1 — Reduction is purely thermal
**Mitigation:** explicit comparison of reduction onset vs χ and vs T alone.

### Risk 2 — Plasma chemistry dominates rate
**Mitigation:** vary plasma conditions at fixed χ; check scaling.

### Risk 3 — Surface-limited reduction
**Mitigation:** geometry and surface-area scaling tests.

### Risk 4 — χ is not the correct order parameter
**Mitigation:** inversion via NVIDIA PhysicsNeMo; failure to collapse data indicates revision needed.

### Risk 5 — Hidden heat sources
**Mitigation:** strict power balance and wall temperature constraints.

---

## 7) Versioning rules

- Any change to assumptions → **model version bump**
- Results across versions are **not directly comparable**
- Version ID must be recorded in run_card.json

Example:
```json
{
  "model_version": "v1.0-minimal"
}
```

---

## 8) Final note

This file is intentionally conservative.  
If the model works under these assumptions, the Flash hypothesis is strong.  
If it fails, the failure will point directly to the missing physics.

---

**End of modeling assumptions.**
