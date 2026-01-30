# PFR_COMSOL_CONTEXT.md
**Plasma Flash Reduction (PFR) — Canonical Physics & Multiphysics Context**

**Status:** canonical  
**Audience:** COMSOL modeling, MCP automation, NVIDIA PhysicsNeMo / ML (discovery layer), experiments  
**Purpose:** Single source of truth for the physics, equations, variables, and couplings that define a PFR reactor model.  
This document defines *what the physics is*. **COMSOL enforces it; NVIDIA PhysicsNeMo learns from it.**

---

## 0. One-paragraph system definition

A Plasma Flash Reduction (PFR) reactor combines **inductively coupled hydrogen plasma**, **imposed electric fields (bias)**, and **non-equilibrium defect physics (Flash)** to drive ultrafast reduction and transport in solids.  
RF fields create a plasma with controlled electron density and radical chemistry; a DC or low-frequency bias supplies electromigration work; phonon softening and defect percolation collapse activation barriers, producing the **Flash state**, quantified by a continuous order parameter χ.

---

## 1. Geometry (canonical)

Baseline assumption: **2D axisymmetric (r–z)**.

Domains:
- Quartz tube (solid)
- Plasma / gas volume
- RF coil (external conductor)
- Optional inner pin / electrode
- Inlet and outlet boundaries

All geometry parameters are defined in `geometry.yaml`.

---

## 2. Physics stack (COMSOL mapping)

| Physics | COMSOL Interface |
|------|------------------|
| RF electromagnetics | Magnetic Fields (mf) or RF (emw) |
| Plasma fluid (ICP) | Plasma Module (ICP) |
| Bias electric field | Electrostatics (es) |
| Gas flow | Laminar Flow |
| Heat transfer | Heat Transfer in Fluids & Solids |
| Species transport | Transport of Diluted Species |

---

## 3. RF electromagnetics

### 3.1 RF power absorption

\[
Q_{RF} = \Re\{\mathbf{J}\cdot\mathbf{E}^\*\}
\]

Plasma conductivity:
\[
\sigma_p = \frac{n_e e^2}{m_e \nu_m}
\]

The RF solver provides `Q_RF(r,z)` to the plasma and thermal equations.

---

## 4. Plasma fluid equations (minimal but complete)

### 4.1 Electron continuity
\[
\frac{\partial n_e}{\partial t} + \nabla \cdot \Gamma_e = S_e
\]

\[
\Gamma_e = -\mu_e n_e \mathbf{E} - D_e \nabla n_e
\]

### 4.2 Ion continuity
\[
\frac{\partial n_i}{\partial t} + \nabla \cdot \Gamma_i = S_i
\]

### 4.3 Electron energy
\[
\frac{\partial}{\partial t}\left(\frac{3}{2}n_e k_B T_e\right)
+ \nabla\cdot\mathbf{q}_e
=
\mathbf{J}_e\cdot\mathbf{E}
- \sum_r R_r \epsilon_r
\]

---

## 5. Flow and heat transfer

### 5.1 Flow (laminar)
\[
\nabla \cdot (\rho \mathbf{u}) = 0
\]

### 5.2 Heat
\[
\rho c_p \frac{\partial T}{\partial t}
+ \rho c_p \mathbf{u}\cdot\nabla T
=
\nabla\cdot(k\nabla T)
+ Q_{RF} + Q_{chem}
\]

---

## 6. Species and chemistry (minimal set)

Tracked species:
- H₂
- H
- H₂O

Plasma dissociation source:
\[
R_{diss} = k_{diss}(T_e)\, n_e\, c_{H_2}
\]

Reduction sink (effective):
\[
R_{red} = k_{eff}\, c_H\, c_{MO}
\]

---

## 7. Flash physics (core)

### 7.1 Barrier balance

\[
\boxed{
\Delta B
=
k_{soft}\,\Delta G^0
-
\left(
n F E_{bias} r_{act}
+
W_{ph}
+
\Delta \mu_{chem}
\right)
}
\]

Flash condition:
\[
\Delta B \le 0
\]

### 7.2 Order parameter (χ smoothing)

\[
\chi = \frac{1}{1+\exp(\Delta B/B_s)}
\]

---

## 8. χ-dependent constitutive laws

Conductivity:
\[
\sigma = (1-\chi)\sigma_0 + \chi\sigma_{flash}
\]

Diffusion:
\[
D_{eff} = D_0 \exp\left(
\chi \frac{n F E_{bias} r_{act}}{k_B T}
\right)
\]

Kinetics:
\[
E_a^{eff} = E_a - \chi n F E_{bias} r_{act}
\]

\[
k_{eff} = k_0 \exp\left(-\frac{E_a^{eff}}{k_B T}\right)
\]

---

## 9. Plasma sheath (by reference)

The plasma sheath is **not resolved**.
Boundary conditions are defined exclusively in:

➡ **`PFR_BC_Sheath_Model.md`**

All solvers must use the same reduced sheath model.

---

## 10. Required outputs

Every valid run must produce:
- `E_bias`
- `Q_RF`
- `n_e`, `T_e`
- `T_gas`, `T_wall`
- `H2`, `H`, `H2O`
- `DeltaB`
- `chi`

Stored according to **`PFR_Data_Schema.md`**.

---

## 11. Validity regime

This model is valid when:
- plasma is collisional (fluid regime)
- pressure ≥ O(10 mbar)
- sheath thickness ≪ reactor scale
- no RF breakdown or multipactor

Outside this regime, kinetic validation is required.

---

## 12. Philosophy

This document defines **what the physics is**.  
**COMSOL enforces it.**  
**NVIDIA PhysicsNeMo learns from it.**  
Experiments validate it.

No ad-hoc physics is allowed outside this context.

---

**End of canonical context.**
