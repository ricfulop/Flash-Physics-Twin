"""
Minimal Tidy3D simulation with a simple coil (helix) geometry.
Uses built-in Tidy3D geometries to approximate a coil structure.
"""

import numpy as np
import tidy3d as td

# ============================================================================
# Create a helical coil geometry using cylinders
# ============================================================================

def create_coil_segments(
    radius: float = 2.0,       # Radius of the helix (μm)
    pitch: float = 1.0,        # Vertical distance per turn (μm)
    wire_radius: float = 0.15, # Radius of the wire (μm)
    num_turns: float = 2,      # Number of turns
    segments_per_turn: int = 12,  # Number of cylinder segments per turn
):
    """
    Create a helical coil using tilted cylinders along the helix path.
    
    Each cylinder segment approximates a small arc of the helix.
    """
    geometries = []
    
    total_segments = int(num_turns * segments_per_turn)
    total_angle = num_turns * 2 * np.pi
    
    for i in range(total_segments):
        # Start and end angles for this segment
        t0 = i * total_angle / total_segments
        t1 = (i + 1) * total_angle / total_segments
        t_mid = (t0 + t1) / 2
        
        # Start and end points on helix
        x0 = radius * np.cos(t0)
        y0 = radius * np.sin(t0)
        z0 = pitch * t0 / (2 * np.pi)
        
        x1 = radius * np.cos(t1)
        y1 = radius * np.sin(t1)
        z1 = pitch * t1 / (2 * np.pi)
        
        # Midpoint (center of cylinder)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        cz = (z0 + z1) / 2
        
        # Length of segment
        length = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        
        # Create a cylinder for this segment
        # Tidy3D cylinders are axis-aligned, so we use a small sphere at each point
        # For a better approximation, use small boxes or spheres
        
        # Use a sphere at each sample point for smooth coil
        sphere = td.Sphere(center=(cx, cy, cz), radius=wire_radius)
        geometries.append(sphere)
    
    return geometries


print("Creating helical coil using sphere segments...")

# Coil parameters (in micrometers)
coil_radius = 2.0      # μm
coil_pitch = 1.0       # μm per turn
wire_radius = 0.15     # μm
num_turns = 2

coil_geometries = create_coil_segments(
    radius=coil_radius,
    pitch=coil_pitch,
    wire_radius=wire_radius,
    num_turns=num_turns,
    segments_per_turn=24,  # More segments = smoother coil
)

print(f"  Segments: {len(coil_geometries)}")

# Combine all spheres into a GeometryGroup
coil_geo = td.GeometryGroup(geometries=coil_geometries)

# ============================================================================
# Define materials
# ============================================================================

# Copper-like material for the coil (simplified as PEC for RF)
# For optical frequencies, we'd use a Drude model
copper = td.PECMedium()  # Perfect electrical conductor

# Alternatively, use a dielectric for optical simulation
dielectric_coil = td.Medium(permittivity=4.0)  # Example dielectric

# ============================================================================
# Create the structure
# ============================================================================

coil_structure = td.Structure(
    geometry=coil_geo,
    medium=dielectric_coil,  # Using dielectric for this demo
    name="coil"
)

# ============================================================================
# Define simulation domain
# ============================================================================

# Simulation size (needs to encompass the coil)
coil_height = coil_pitch * num_turns
sim_size = (
    2 * (coil_radius + wire_radius) + 2,  # x
    2 * (coil_radius + wire_radius) + 2,  # y
    coil_height + 4,                       # z (extra space for source/monitors)
)

# Simulation center
sim_center = (0, 0, coil_height / 2)

# Z bounds of simulation
z_min = sim_center[2] - sim_size[2] / 2
z_max = sim_center[2] + sim_size[2] / 2

# Wavelength for simulation
wavelength = 1.55  # μm (telecom wavelength)
freq0 = td.C_0 / wavelength

# ============================================================================
# Define source
# ============================================================================

# Plane wave source from below (inside simulation domain)
source = td.PlaneWave(
    source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0/10),
    center=(0, 0, z_min + 0.5),  # Just inside the bottom of domain
    size=(td.inf, td.inf, 0),
    direction="+",
    pol_angle=0,
    name="plane_wave"
)

# ============================================================================
# Define monitors
# ============================================================================

# Field monitor at center of coil
field_monitor = td.FieldMonitor(
    center=(0, 0, coil_height / 2),
    size=(sim_size[0] - 0.5, sim_size[1] - 0.5, 0),
    freqs=[freq0],
    name="field_xy"
)

# Flux monitor above the coil
flux_monitor = td.FluxMonitor(
    center=(0, 0, z_max - 0.5),  # Just inside the top of domain
    size=(sim_size[0] - 1, sim_size[1] - 1, 0),
    freqs=[freq0],
    name="flux_out"
)

# ============================================================================
# Create the simulation
# ============================================================================

sim = td.Simulation(
    size=sim_size,
    center=sim_center,
    structures=[coil_structure],
    sources=[source],
    monitors=[field_monitor, flux_monitor],
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=15, wavelength=wavelength),
    run_time=50 / freq0,  # ~50 optical cycles
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
)

print(f"\nSimulation created:")
print(f"  Size: {sim_size} μm")
print(f"  Wavelength: {wavelength} μm")
print(f"  Frequency: {freq0/1e12:.2f} THz")
print(f"  Run time: {sim.run_time*1e15:.1f} fs")

# Validate the simulation (pre-run check)
print("\nValidating simulation...")
try:
    sim.validate_pre_upload()
    print("  ✓ Simulation is valid!")
except Exception as e:
    print(f"  ✗ Validation error: {e}")

# ============================================================================
# Submit to Flexcompute cloud
# ============================================================================
print("\n" + "=" * 60)
print("SUBMITTING TO FLEXCOMPUTE CLOUD")
print("=" * 60)

import tidy3d.web as web

# Submit and run the simulation
task_name = "coil_simulation_test"
print(f"\nSubmitting task: {task_name}")

try:
    sim_data = web.run(
        simulation=sim,
        task_name=task_name,
        path="results/tidy3d/coil_sim_data.hdf5",
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    # ============================================================================
    # Extract and report monitor values
    # ============================================================================
    print("\n--- Field Monitor (field_xy) ---")
    field_data = sim_data["field_xy"]
    
    # Get field components
    if hasattr(field_data, 'Ex'):
        Ex = field_data.Ex.values
        print(f"  Ex shape: {Ex.shape}")
        print(f"  Ex magnitude: min={abs(Ex).min():.4e}, max={abs(Ex).max():.4e}, mean={abs(Ex).mean():.4e}")
    
    if hasattr(field_data, 'Ey'):
        Ey = field_data.Ey.values
        print(f"  Ey shape: {Ey.shape}")
        print(f"  Ey magnitude: min={abs(Ey).min():.4e}, max={abs(Ey).max():.4e}, mean={abs(Ey).mean():.4e}")
    
    if hasattr(field_data, 'Ez'):
        Ez = field_data.Ez.values
        print(f"  Ez shape: {Ez.shape}")
        print(f"  Ez magnitude: min={abs(Ez).min():.4e}, max={abs(Ez).max():.4e}, mean={abs(Ez).mean():.4e}")
    
    print("\n--- Flux Monitor (flux_out) ---")
    flux_data = sim_data["flux_out"]
    flux_values = flux_data.flux.values
    print(f"  Flux shape: {flux_values.shape}")
    print(f"  Flux at {wavelength} μm: {flux_values[0]:.6f}")
    
    # Compute transmission (normalized to source power)
    # For a plane wave, source power depends on domain size
    print(f"\n--- Summary ---")
    print(f"  Output flux: {flux_values[0]:.6f}")
    
except Exception as e:
    print(f"\n✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
