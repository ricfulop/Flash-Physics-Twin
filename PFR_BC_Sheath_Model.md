# PFR_BC_Sheath_Model.md
**Plasma Flash Reduction (PFR) — Canonical Reduced Sheath Boundary Model**

**Status:** canonical  
**Audience:** COMSOL modeling, MCP automation, NVIDIA PhysicsNeMo / ML, validation (PIC spot-checks)  
**Purpose:** Define a **single, solver-independent sheath model** to be used as boundary conditions (BCs) at all plasma–surface interfaces in PFR simulations.

This model is intentionally **unresolved** (no Debye-length meshing). It provides consistent:
- particle fluxes (ions/electrons) to surfaces  
- energy fluxes to surfaces  
- sheath potential drop (floating vs biased)  
- wall recombination (H radical losses)

This is the file that prevents **boundary-condition drift** across tools.

---

## 0) What this model is (and is not)

### 0.1 Model class
- **Collisionless / weakly collisional Bohm sheath** at the plasma edge
- **Unresolved sheath** represented only through **boundary flux conditions** and **algebraic sheath voltage**
- Compatible with fluid plasma solvers (COMSOL Plasma Module) and reduced ML / PINN / PhysicsNeMo models

### 0.2 Not included (by design, v1)
- RF-period sheath oscillations (time-resolved RF sheath)
- Full EEDF / kinetic distributions in the sheath
- Secondary electron emission cascades (SEE) unless explicitly enabled as a scalar correction
- Space-charge-limited emission, multipactor, breakdown modeling
- Surface charging of dielectrics as a dynamic PDE (may be added later)

---

## 1) Definitions and symbols

All quantities are SI unless stated.

### 1.1 Plasma-edge (“sheath edge”) quantities
- \(n_e\) : electron density at sheath edge (m⁻³)
- \(T_e\) : electron temperature at sheath edge (K)
- \(T_g\) : gas temperature near wall (K)
- \(m_i\) : ion mass (kg); for H plasmas typically use dominant ion (H⁺ or H₃⁺)
- \(e\) : elementary charge (C)
- \(k_B\) : Boltzmann constant
- \(V_p\) : plasma potential (V)
- \(V_w\) : wall/electrode potential (V)
- \(V_s\) : sheath drop (V), defined as:
  \[
  V_s \equiv V_p - V_w
  \]
  (positive \(V_s\) means wall is negative relative to plasma)

### 1.2 Fluxes and heat fluxes
- \(\Gamma_i\) : ion particle flux to wall (m⁻² s⁻¹)
- \(\Gamma_e\) : electron particle flux to wall (m⁻² s⁻¹)
- \(q_i\) : ion energy flux to wall (W m⁻²)
- \(q_e\) : electron energy flux to wall (W m⁻²)

### 1.3 Electron thermal speed (mean speed for Maxwellian)
\[
\bar{v}_e = \sqrt{\frac{8 k_B T_e}{\pi m_e}}
\]

### 1.4 Ion sound speed (Bohm speed)
\[
c_s = \sqrt{\frac{k_B T_e}{m_i}}
\]
(If multi-ion, use the dominant ion; if needed, define an effective \(m_i\).)

---

## 2) Core sheath boundary conditions (mandatory)

### 2.1 Ion flux to wall (Bohm criterion)
At all plasma-contacting surfaces:
\[
\boxed{\Gamma_i = n_e\,c_s}
\]
This is the canonical ion flux BC.

**Directional form** (for a boundary with outward normal \(\hat{n}\) pointing out of plasma domain):
\[
\hat{n}\cdot\Gamma_i^{vec} = -\Gamma_i
\]

---

### 2.2 Electron flux to wall (Boltzmann-suppressed)
Electron flux is Maxwellian to the sheath edge, suppressed by sheath potential drop:
\[
\boxed{
\Gamma_e =
\frac{1}{4} n_e \bar{v}_e
\exp\left(-\frac{e V_s}{k_B T_e}\right)
}
\]

Directional form:
\[
\hat{n}\cdot\Gamma_e^{vec} = -\Gamma_e
\]

**Notes**
- This assumes a Maxwellian electron population at the sheath edge.
- This is the correct v1 model for PFR unless kinetic validation indicates otherwise.

---

### 2.3 Floating wall sheath drop (if surface is electrically floating)
For an electrically floating surface, enforce **zero net current**:
\[
e\,\Gamma_i = e\,\Gamma_e
\]
which yields an implicit equation for \(V_s\).

In the common approximation (singly charged ions, Maxwellian electrons), the floating potential drop is:
\[
\boxed{
V_s^{float} \approx
\frac{k_B T_e}{e}
\ln\left(\sqrt{\frac{m_i}{2\pi m_e}}\right)
}
\]

Use this when:
- boundary is a dielectric or floating conductor
- no explicit bias is applied

---

### 2.4 Biased electrode sheath drop (if surface potential is imposed)
If a wall electrode potential \(V_w\) is known (bias applied), then:
\[
\boxed{
V_s = V_p - V_w
}
\]
In fluid models, \(V_p\) may be approximated by the plasma potential variable or derived from quasi-neutrality constraints.

**Implementation guidance**
- For COMSOL Plasma Module, use its built-in sheath BC options where possible.
- Otherwise treat \(V_s\) as a computed boundary variable, with \(V_p\) taken from plasma potential field and \(V_w\) from electrostatics.

---

## 3) Energy flux boundary conditions (mandatory)

### 3.1 Ion energy flux
Assume ions enter sheath at \(c_s\) and gain energy \(eV_s\) across the sheath:
\[
\boxed{
q_i = \Gamma_i \left(eV_s + \frac{1}{2}k_B T_e\right)
}
\]
(You may omit the \( \frac{1}{2}k_B T_e \) term if your plasma module already accounts for ion enthalpy at the sheath edge; be consistent across solvers.)

### 3.2 Electron energy flux
Electrons arriving at the wall carry approximately \(2k_BT_e\) per electron for a Maxwellian:
\[
\boxed{
q_e = \Gamma_e \left(2 k_B T_e\right)
}
\]

### 3.3 Net plasma-to-wall heat flux
\[
\boxed{
q_{plasma\rightarrow wall} = q_i + q_e
}
\]

This heat flux enters the wall heat-transfer BC:
\[
-k\nabla T\cdot\hat{n} = q_{plasma\rightarrow wall} + q_{chem,wall}
\]

---

## 4) Species boundary conditions (H radical recombination)

Wall recombination of H radicals is defined **exclusively** by:

➡ **`PFR_BC_Sheath_Model.md`**

Chemistry models must **not redefine** wall losses.

The effective wall sink is:
\[
\Gamma_H^{wall}
=
\gamma_H \frac{1}{4} c_H \bar{v}_{H}
\]

with material-dependent \(\gamma_H\).

**Default (v1):**
- quartz wall: \(\gamma_H = 0.01\)  
- metal wall/electrode: \(\gamma_H = 0.05\)

---

## 5) Optional: Secondary electron emission (SEE)

If needed (only after evidence), incorporate SEE as a multiplicative correction to electron current.

**Default (v1):** off (\(\delta_{eff}=0\))

---

## 6) Validity regime and kinetic escalation

This reduced sheath model is valid when:
- sheath thickness \(\lambda_D\) ≪ reactor scale
- plasma is quasi-neutral in the bulk
- pressure supports collisional fluid plasma
- no evidence of multipactor or breakdown

Trigger PIC / kinetic validation if:
- pressure < ~1 Pa
- extreme bias or pulsed operation
- arcing or SEE-dominated behavior

---

## 7) Data schema integration

Recommended boundary KPIs:
- mean/max \(V_s\)
- total wall heat load
- total H radical wall loss

These should be written to `kpis.json` per **`PFR_Data_Schema.md`**.

---

## 8) Philosophy

The sheath is where models silently diverge.  
This file exists so:
- COMSOL enforcement and NVIDIA PhysicsNeMo inversion remain consistent
- boundary physics is stable across runs
- learned parameters reflect physics, not BC artifacts

---

**End of sheath boundary model.**
