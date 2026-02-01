# PFR Workflows

This directory contains workflow scripts for the PFR (Plasma Flash Reduction) digital twin.

## Available Workflows

### run_em_coil.py

Runs an AC/DC coil electromagnetic simulation using COMSOL's AC/DC module (or synthetic equivalent).

**Usage:**
```bash
# Basic run with defaults
python workflows/run_em_coil.py

# Custom parameters
python workflows/run_em_coil.py --I_coil 20 --sigma_eff 0.1 --notes "high current test"

# Sweep example (bash)
for I in 5 10 20 50; do
    python workflows/run_em_coil.py --I_coil $I --notes "current sweep I=${I}A"
done
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--I_coil` | 10.0 | Coil current amplitude (A) |
| `--sigma_eff` | 0.01 | Effective plasma conductivity (S/m) |
| `--f_RF` | 13.56e6 | RF frequency (Hz) |
| `--notes` | "" | Run notes |

**Outputs:**
- `results/runs/<run_id>/outputs/fields.h5` with:
  - `/em/B_mag` - Magnetic field magnitude [T]
  - `/em/E_mag` - Induced electric field magnitude [V/m]
  - `/em/Q_RF` - RF power deposition [W/m³]
- `results/runs/<run_id>/outputs/kpis.json` with EM statistics

## Workflow Structure

All workflows follow the same pattern:

1. **Create Run** - Generate unique run ID and directory structure
2. **Snapshot Inputs** - Copy parameter files to run's `inputs/` directory
3. **Run Simulation** - Execute COMSOL model or synthetic equivalent
4. **Export Fields** - Write results to schema-compliant `fields.h5`
5. **Export KPIs** - Compute scalar metrics to `kpis.json`
6. **Validate Outputs** - Check against `PFR_Data_Schema.md`
7. **Register Run** - Mark run as complete in `run_card.json`

## Adding New Workflows

When creating a new workflow:

1. Follow the template in `run_em_coil.py`
2. Use the `ComsolBackend` for COMSOL operations
3. Ensure outputs comply with `PFR_Data_Schema.md`
4. Include proper error handling and status updates
5. Add documentation to this README
