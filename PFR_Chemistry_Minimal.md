# PFR_Chemistry_Minimal.md
**Plasma Flash Reduction (PFR) — Minimal Chemistry Specification (v1)**

**Status:** canonical (v1)  
**Audience:** COMSOL modeling, MCP automation, NVIDIA PhysicsNeMo / ML, experimental validation  
**Purpose:** Define a **minimal, falsifiable chemistry model** that is consistent across solvers and sufficient to test the **Flash hypothesis** without introducing non-identifiable degrees of freedom.

This document intentionally specifies **what is included** and **what is excluded**, and defines **explicit upgrade triggers**.

---

## 0) Modeling philosophy (why “minimal” is intentional)

This chemistry model is designed to answer a *single, hard question*:

> **Given conservative plasma chemistry and no surface “magic,” does field-driven barrier collapse (Flash) enable reduction?**

To preserve falsifiability:
- chemistry is **collapsed to effective source/sink terms**
- plasma provides radicals but does **not control the rate**
- Flash (ΔB → χ) is the *only* mechanism that can unlock fast reduction

If the model succeeds under these constraints → PFR is real.  
If it fails → adding complexity should *not* rescue it.

---

## 1) Species set (v1)

The minimal species set is:

| Symbol | Description | Phase |
|------|------------|-------|
| H₂ | molecular hydrogen | gas |
| H | atomic hydrogen (radical) | gas |
| H₂O | water vapor (product) | gas |
| MO | generic metal oxide | solid (implicit) |
| M | reduced metal | solid (implicit) |

**Notes**
- Solids (MO, M) are **not transported as species** in v1.
- Oxygen removal is represented implicitly via reaction rates and χ.
- All gas species are tracked as **mol/m³** (not number density).

---

## 2) Plasma source term (radical generation)

### 2.1 Effective dissociation reaction
\[
\mathrm{H_2} \xrightarrow{\text{plasma}} 2\,\mathrm{H}
\]

### 2.2 Rate law (effective, v1)
\[
\boxed{
R_{diss}
=
k_{diss}(T_e)\; n_e\; c_{H_2}
}
\]

Where:
- \(n_e\) = electron density (m⁻³)
- \(T_e\) = electron temperature (K)
- \(c_{H_2}\) = H₂ concentration (mol/m³)
- \(k_{diss}(T_e)\) = **effective dissociation coefficient**

**Important**
- \(k_{diss}\) is **not explanatory** in v1.
- It may be tabulated, parameterized, or calibrated from experiment.
- It is assumed monotonic in \(T_e\).

---

## 3) Bulk gas-phase reactions (optional, weak)

### 3.1 H recombination in gas (optional stabilization)
\[
\mathrm{H} + \mathrm{H} (+\mathrm{M}) \rightarrow \mathrm{H_2} (+\mathrm{M})
\]

Rate law (if enabled):
\[
R_{rec,gas} = k_{rec}\; c_H^2
\]

**Default (v1):** disabled  
**Rationale:** wall recombination dominates radical loss in typical PFR regimes.

---

## 4) Wall recombination (mandatory)

Wall recombination of H radicals is defined **exclusively** by:

➡ **`PFR_BC_Sheath_Model.md`**

Chemistry models must **not redefine** wall losses.

The effective wall sink is:
\[
\Gamma_H^{wall}
=
\gamma_H \frac{1}{4} c_H \bar{v}_H
\]

with material-dependent \(\gamma_H\).

---

## 5) Reduction reaction (core chemistry)

### 5.1 Effective reduction step
\[
\mathrm{MO} + \mathrm{H_2} \;\;(\text{or } \mathrm{H}) \;\rightarrow\; \mathrm{M} + \mathrm{H_2O}
\]

This represents net oxygen removal from the solid.

### 5.2 Baseline Arrhenius form
\[
k_{th} = k_0 \exp\!\left(-\frac{E_a}{k_B T}\right)
\]

### 5.3 Flash-modified kinetics (mandatory)
\[
\boxed{
E_a^{eff}
=
E_a
-
\chi\; n F E_{bias} r_{act}
}
\]

\[
\boxed{
k_{eff}
=
k_0
\exp\!\left(-\frac{E_a^{eff}}{k_B T}\right)
}
\]

### 5.4 Reduction rate
\[
\boxed{
R_{red}
=
k_{eff}\; c_{H}\; c_{MO}
}
\]

**Notes**
- \(c_{MO}\) is treated as a **normalized solid fraction** (often unity in v1).
- χ (from `PFR_COMSOL_CONTEXT.md`) is the *only* mechanism that collapses the barrier.
- If χ = 0, reduction must be slow.

---

## 6) Water production and transport

### 6.1 Source term
\[
R_{H_2O} = R_{red}
\]

### 6.2 Transport
H₂O is transported via:
- convection
- diffusion

No condensation or adsorption is modeled in v1.

---

## 7) Heat of reaction (effective)

### 7.1 Volumetric heat source/sink
Reduction enthalpy is included as an **effective volumetric term**:
\[
Q_{chem} = R_{red}\; \Delta H_{red}
\]

Where:
- \(\Delta H_{red}\) is an **effective enthalpy** per mole of MO reduced.

**Important**
- \(\Delta H_{red}\) is included for **energy consistency**, not micro-thermochemical accuracy.
- Exact partitioning (solid vs gas) is not resolved in v1.

---

## 8) Parameter classification

### 8.1 Calibratable parameters (v1)
- \(k_{diss}(T_e)\)
- \(k_0\)
- \(E_a\)
- \(\Delta H_{red}\)

### 8.2 Flash parameters (from Flash physics)
- \(r_{act}\)
- \(k_{soft}\)
- \(B_s\)

These should be inferred via **NVIDIA PhysicsNeMo / inversion**, **not tuned ad hoc**.

---

## 9) Known omissions and when to upgrade (critical)

The following physics is **intentionally omitted** in v1.  
Each item has a **clear upgrade trigger**.

---

### 9.1 Full hydrogen plasma chemistry network
**Omitted**
- H⁺, H₂⁺, H₃⁺
- vibrationally excited H₂
- EEDF-resolved rate coefficients

**Upgrade if**
- measured radical densities cannot be matched by any reasonable \(k_{diss}(T_e)\)
- diagnostics show strong vibrational excitation effects
- plasma chemistry, not Flash, appears rate-controlling

---

### 9.2 Surface reaction microphysics
**Omitted**
- adsorption kinetics
- sticking coefficients
- Langmuir–Hinshelwood / Eley–Rideal mechanisms
- catalytic wall effects

**Upgrade if**
- reduction rate scales with surface area, not χ
- different wall materials change outcomes dramatically
- reduction remains slow even when χ → 1

---

### 9.3 Multi-oxide / suboxide pathways
**Omitted**
- explicit TiO₂ → Ti₂O₃ → TiO → Ti steps (etc.)

**Upgrade if**
- intermediate phases dominate measurable kinetics
- conductivity or χ evolution stalls at known suboxide stoichiometries

---

### 9.4 Oxygen transport / defect chemistry inside the solid
**Omitted**
- explicit vacancy concentration fields
- oxygen chemical potential gradients
- coupled electron–ion defect transport

**Upgrade if**
- reduction is clearly bulk-diffusion-limited *after* χ activation
- experiments show internal gradients inconsistent with χ-based diffusion

---

### 9.5 Gas-phase radicals beyond H
**Omitted**
- OH, O, HO₂, impurity species

**Upgrade if**
- spectroscopy shows OH/O comparable to H
- back-oxidation or poisoning effects dominate
- water management cannot be captured with H₂O alone

---

### 9.6 Full thermochemistry and phase energetics
**Omitted**
- detailed enthalpy partitioning
- phase transitions, latent heats
- solid morphology evolution

**Upgrade if**
- predicted temperatures disagree systematically with measurements
- reduction appears purely thermal with no χ dependence

---

## 10) What success looks like (v1)

The minimal chemistry model is **successful** if:
- reduction onset correlates with χ, not just T or radical density
- increasing bias shifts reduction even at fixed plasma conditions
- water production follows reduction zones predicted by χ
- results cannot be explained by thermal chemistry alone

If these conditions are met, **PFR works**.

---

## 11) Philosophy (why this file exists)

This file prevents the most dangerous failure mode:
> *“The model works, but we don’t know why.”*

By keeping chemistry minimal:
- Flash must carry explanatory weight
- upgrades are additive, not rescuing
- disagreement with experiment is informative

---

**End of minimal chemistry specification.**
