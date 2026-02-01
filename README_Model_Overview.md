# README_Model_Overview.md
**Flash Physics Twin — Model Overview (PFR)**

**Audience:** new team members, collaborators, sponsors, reviewers  
**Purpose:** Provide a **high-level, human-readable overview** of what the PFR model is, how it works, and what questions it is designed to answer — without requiring the reader to dive into equations or code.

This document explains the *why* and *what*.  
The detailed *how* lives in the other docs.

---

## 1. What this model is

The **Flash Physics Twin** is a **physics-first digital twin** of a **Plasma Flash Reduction (PFR)** reactor.

It is designed to test a single, central hypothesis:

> **That strong electric fields, coupled to plasma and phonons, collapse activation barriers and enable ultrafast reduction and transport in solids — a driven non-equilibrium state we call “Flash.”**

The model is **not** a generic plasma reactor simulator.  
It is a **mechanism-discriminating model** built to determine whether Flash is *necessary* and *sufficient* to explain experimental observations.

---

## 2. What problem this model solves

Traditional explanations for high-rate reduction rely on:
- high temperature
- plasma radical density
- surface catalysis

These explanations fail to explain:
- sharp onset thresholds
- strong bias dependence at fixed temperature
- spatial localization of reduction
- cross-material scaling collapse

The Flash Physics Twin is built to answer:

- *Is reduction controlled by field-driven barrier collapse, not temperature alone?*
- *Where in the reactor does Flash occur, and why there?*
- *Can plasma chemistry be secondary rather than rate-controlling?*
- *Do different materials collapse onto a common Flash framework?*

---

## 3. Core modeling idea (conceptual)

The model introduces a **single order parameter**, χ (chi), which represents the **fraction of the material in the Flash state**.

- χ = 0 → conventional thermally activated regime  
- χ ≈ 1 → Flash-enabled, barrier-collapsed regime  

χ is determined by a **barrier balance**:

\[
\Delta B = k_{soft}\Delta G^0 - (nFEr + W_{ph} + \Delta\mu_{chem})
\]

and mapped smoothly via:
\[
\chi = \frac{1}{1+\exp(\Delta B/B_s)}
\]

Once χ activates:
- diffusion accelerates
- conductivity increases
- reaction barriers collapse

This is how Flash enters the model — **explicitly, not implicitly**.

---

## 4. What physics is included (at a glance)

The model includes:

- **RF plasma (fluid)**
  - electron density and temperature
  - radical generation
- **Electric fields (bias)**
  - electromigration work
- **Heat transfer**
  - gas and wall temperatures
- **Flow**
  - hydrogen transport
- **Minimal chemistry**
  - H₂, H, H₂O
  - effective reduction reaction
- **Reduced plasma sheath**
  - Bohm ion flux
  - electron suppression
- **Flash closure**
  - χ-dependent transport and kinetics

All of these are coupled self-consistently.

---

## 5. What is intentionally *not* included (v1)

The model **does not** initially include:

- full hydrogen plasma chemistry networks
- detailed surface adsorption microkinetics
- explicit vacancy or defect concentration fields
- multi-step suboxide reaction pathways
- RF-period sheath oscillations
- turbulence or particle-resolved powder dynamics

These omissions are **intentional** and documented — not limitations.

They ensure the model remains:
- falsifiable
- interpretable
- aligned with available data

Upgrades are only added when validation fails.

---

## 6. How the tool stack is organized

The Flash Physics Twin is implemented as a **layered system**, where each tool is used only for what it does best.

### Physics enforcement
- **COMSOL Multiphysics**
  - solves coupled continuum PDEs
  - enforces conservation laws
  - produces spatial fields and KPIs
  - serves as the **ground-truth physics engine**

### Electromagnetics (optional but recommended)
- **Flexcompute (Tidy3D via FlexAgent MCP)**
  - RF coil design
  - coupling efficiency
  - EM field distributions
  - provides EM inputs or surrogates to COMSOL

### Discovery, inversion, and surrogates
- **NVIDIA PhysicsNeMo**
  - parameter inversion (e.g., activation length *r*)
  - uncertainty quantification
  - surrogate model training
  - operator learning for rapid design iteration

> **PhysicsNeMo does not replace COMSOL.**  
> COMSOL enforces physics; PhysicsNeMo learns, inverts, and accelerates it.

### Orchestration
- **Cursor + MCP**
  - single execution cockpit
  - reproducible workflows
  - strict data and schema enforcement

---

## 7. What makes this model different

This project differs from conventional “digital twins” because:

- it does **not** tune parameters to fit outcomes
- it forces Flash to carry explanatory weight
- it fails cleanly if Flash is not real
- it separates:
  - **physics enforcement** (COMSOL)
  - **physics discovery** (NVIDIA PhysicsNeMo)

If the model works, it works for the right reason.

---

## 8. What success looks like

The model is considered successful if it demonstrates:

- reduction onset controlled by χ, not temperature alone
- bias-driven control independent of plasma power
- spatial localization of reduction predicted by χ maps
- consistent behavior across different oxides
- narrow inferred Flash parameters (*r*, *Bₛ*, *k_soft*)

If these conditions are met, Flash is a **real, predictive state**.

---

## 9. How to get started (new team members)

1. Read:
   - `PFR_COMSOL_CONTEXT.md`
   - `PFR_Runbook.md`
2. Run:
   - a baseline case
3. Inspect:
   - χ maps
   - ΔB maps
   - power balance
4. Do **not** change physics without updating docs.

You should be productive in one day.

---

## 10. How this model evolves

This is a **living model**, but evolution is controlled:

- new physics only enters after failed validation
- every change bumps model version
- old results remain reproducible

The goal is **understanding**, not just prediction.

---

## 11. Final takeaway

This repository is not just a simulator.

It is:
- a hypothesis test
- a discovery engine
- a guardrail against self-deception

If Flash exists, this model will show it.  
If it doesn’t, this model will tell us why.

---

**End of model overview.**
