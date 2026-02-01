# COMSOL Templates for PFR Digital Twin

This directory contains COMSOL model templates for the PFR (Plasma Flash Reduction) digital twin.

## Templates

### pfr_coil_acdc_axisym.mph

**AC/DC Coil Model (2D Axisymmetric)**

A time-harmonic electromagnetic model for computing RF-induced fields from an inductive coil.

**Module:** AC/DC (Magnetic Fields)  
**Dimension:** 2D Axisymmetric  
**Study:** Frequency Domain at 13.56 MHz

**Key Features:**
- Quartz tube reactor geometry
- Multi-turn RF coil as external current source
- Gas region with parameterized effective conductivity
- Computes B_mag, E_mag, Q_RF fields

**Parameters:**
| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `I_coil` | 10 | A | Coil current amplitude |
| `f_RF` | 13.56e6 | Hz | RF frequency |
| `sigma_eff` | 0.01 | S/m | Effective plasma conductivity |
| `n_turns` | 5 | - | Number of coil turns |

**Outputs:**
- `/em/B_mag` - Magnetic field magnitude [T]
- `/em/E_mag` - Induced electric field magnitude [V/m]  
- `/em/Q_RF` - RF power deposition [W/m³]

**Usage:**
```python
from mcp.comsol_mcp.comsol_api import ComsolBackend

backend = ComsolBackend()
backend.open_or_create(
    run_path, 
    template_path="mcp/comsol_mcp/templates/pfr_coil_acdc_axisym.mph"
)
backend.apply_parameters(run_path, {"I_coil": 20, "sigma_eff": 0.1})
backend.run_study(run_path, "std1")
backend.export_em_fields(run_path)
```

## Creating New Templates

When creating a new COMSOL template:

1. Follow the PFR_Data_Schema.md for output field specifications
2. Use consistent parameter naming conventions
3. Export to the standard (r, z) grid for axisymmetric models
4. Include a specification comment block at the top of the .mph file
5. Test with `pfr_data_mcp.validate_outputs` before registering runs
