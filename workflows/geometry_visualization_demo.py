#!/usr/bin/env python3
"""
PFR Reactor Geometry & Field Visualization Demo
================================================

Creates publication-quality visualizations of the PFR reactor:
- 3D geometry rendering
- Electric field distribution
- Activation regions
- Cross-sectional views

For Haig - demonstrating the digital twin visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Wedge
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from pathlib import Path

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
RESULTS_DIR = Path(__file__).parent.parent / "results" / "geometry_demo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# REACTOR GEOMETRY PARAMETERS (mm)
# =============================================================================
GEOMETRY = {
    # Tube
    "tube_ID": 100.0,       # mm
    "tube_OD": 110.0,       # mm (5mm wall)
    "tube_length": 300.0,   # mm
    
    # Electrodes
    "pin_radius": 10.0,     # mm (a)
    "mesh_radius": 22.0,    # mm (b)
    "electrode_length": 250.0,  # mm
    
    # RF Coil
    "coil_radius": 55.0,    # mm
    "coil_turns": 8,
    "coil_pitch": 25.0,     # mm per turn
    "wire_radius": 3.0,     # mm
    "coil_z_start": 25.0,   # mm
    
    # Powder region
    "powder_r_min": 5.0,    # mm
    "powder_r_max": 35.0,   # mm
    "powder_z_start": 60.0, # mm
    "powder_z_end": 240.0,  # mm
}

# =============================================================================
# PHYSICS PARAMETERS
# =============================================================================
PHYSICS = {
    "V_bias": 400.0,        # V - applied voltage
    "lambda_onset": 60000,  # V/m - activation threshold
    "T_preheat": 1100,      # K
}


# =============================================================================
# ELECTRIC FIELD CALCULATIONS
# =============================================================================
def compute_E_field_coaxial(r, a, b, V):
    """
    Compute electric field in coaxial geometry.
    E(r) = V / (r * ln(b/a))
    """
    a_m = a / 1000  # convert to meters
    b_m = b / 1000
    r_m = np.maximum(r / 1000, a_m)  # clamp to avoid singularity
    r_m = np.minimum(r_m, b_m)
    
    E = V / (r_m * np.log(b_m / a_m))
    return E


def compute_activation_radius(V, lambda_onset, a, b):
    """
    Compute the radius where E = lambda_onset.
    r_onset = V / (lambda_onset * ln(b/a))
    """
    a_m = a / 1000
    b_m = b / 1000
    r_onset = V / (lambda_onset * np.log(b_m / a_m))
    r_onset_mm = r_onset * 1000
    return np.clip(r_onset_mm, a, b)


# =============================================================================
# 3D GEOMETRY VISUALIZATION
# =============================================================================
def create_cylinder_mesh(r, z_start, z_end, n_theta=50, n_z=20):
    """Create mesh for a cylinder surface."""
    theta = np.linspace(0, 2*np.pi, n_theta)
    z = np.linspace(z_start, z_end, n_z)
    theta, z = np.meshgrid(theta, z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def create_helix_points(r, z_start, pitch, turns, n_points_per_turn=50):
    """Create points along a helix for the RF coil."""
    n_total = int(turns * n_points_per_turn)
    t = np.linspace(0, turns * 2 * np.pi, n_total)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = z_start + pitch * t / (2 * np.pi)
    return x, y, z


def plot_3d_geometry():
    """Create 3D visualization of reactor geometry."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    g = GEOMETRY
    
    # Quartz tube (transparent)
    x, y, z = create_cylinder_mesh(g["tube_ID"]/2, 0, g["tube_length"])
    ax.plot_surface(x, y, z, alpha=0.15, color='lightblue', 
                   label='Quartz tube')
    
    # Pin electrode (solid)
    x, y, z = create_cylinder_mesh(g["pin_radius"], 25, 275, n_theta=30)
    ax.plot_surface(x, y, z, alpha=0.8, color='gray')
    
    # Mesh electrode (wireframe style - using lines)
    theta = np.linspace(0, 2*np.pi, 40)
    z_lines = np.linspace(25, 275, 20)
    for z_val in z_lines:
        x = g["mesh_radius"] * np.cos(theta)
        y = g["mesh_radius"] * np.sin(theta)
        z = np.ones_like(theta) * z_val
        ax.plot(x, y, z, 'b-', alpha=0.3, linewidth=0.5)
    
    # Vertical mesh lines
    for t in np.linspace(0, 2*np.pi, 20):
        x = g["mesh_radius"] * np.cos(t) * np.ones(20)
        y = g["mesh_radius"] * np.sin(t) * np.ones(20)
        z = np.linspace(25, 275, 20)
        ax.plot(x, y, z, 'b-', alpha=0.3, linewidth=0.5)
    
    # RF Coil (helix)
    x, y, z = create_helix_points(
        g["coil_radius"], 
        g["coil_z_start"], 
        g["coil_pitch"], 
        g["coil_turns"]
    )
    ax.plot(x, y, z, 'r-', linewidth=4, label='RF Coil')
    
    # Powder fall region (shaded volume indicator)
    # Draw as a translucent cylinder section
    theta = np.linspace(0, 2*np.pi, 30)
    for r in [g["powder_r_min"], g["powder_r_max"]]:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z_bottom = np.ones_like(theta) * g["powder_z_start"]
        z_top = np.ones_like(theta) * g["powder_z_end"]
        ax.plot(x, y, z_bottom, 'g--', alpha=0.5, linewidth=1)
        ax.plot(x, y, z_top, 'g--', alpha=0.5, linewidth=1)
    
    # Labels
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title('PFR Reactor 3D Geometry\nPin-Mesh Coaxial Configuration with RF Coil', 
                fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = g["tube_length"] / 2
    ax.set_xlim([-60, 60])
    ax.set_ylim([-60, 60])
    ax.set_zlim([0, g["tube_length"]])
    
    # Add legend annotations
    ax.text(0, 0, 290, 'Pin (cathode)', fontsize=10, ha='center')
    ax.text(25, 0, 290, 'Mesh (anode)', fontsize=10, ha='center', color='blue')
    ax.text(55, 0, 150, 'RF Coil', fontsize=10, ha='left', color='red')
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / '3d_reactor_geometry.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: 3d_reactor_geometry.png")


# =============================================================================
# 2D CROSS-SECTION WITH E-FIELD
# =============================================================================
def plot_rz_cross_section():
    """Create r-z cross-section with electric field colormap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    g = GEOMETRY
    p = PHYSICS
    
    # Create r-z grid
    r = np.linspace(0, g["tube_ID"]/2 + 5, 200)
    z = np.linspace(0, g["tube_length"], 300)
    R, Z = np.meshgrid(r, z)
    
    # Compute E-field (only in active region)
    E_field = np.zeros_like(R)
    
    # Active region mask
    active_mask = (R >= g["pin_radius"]) & (R <= g["mesh_radius"]) & \
                  (Z >= 25) & (Z <= 275)
    
    E_field[active_mask] = compute_E_field_coaxial(
        R[active_mask], g["pin_radius"], g["mesh_radius"], p["V_bias"]
    )
    
    # Activation radius
    r_onset = compute_activation_radius(
        p["V_bias"], p["lambda_onset"], g["pin_radius"], g["mesh_radius"]
    )
    
    # =========================================================================
    # Left panel: E-field magnitude
    # =========================================================================
    ax1 = axes[0]
    
    # E-field colormap
    E_plot = np.ma.masked_where(E_field == 0, E_field)
    im1 = ax1.pcolormesh(R, Z, E_plot/1000, cmap='hot', shading='auto',
                         vmin=0, vmax=150)
    
    # Draw geometry
    # Quartz tube walls
    ax1.axvline(x=g["tube_ID"]/2, color='cyan', linewidth=3, label='Quartz tube')
    ax1.axvline(x=g["tube_OD"]/2, color='cyan', linewidth=3)
    ax1.fill_betweenx([0, g["tube_length"]], g["tube_ID"]/2, g["tube_OD"]/2, 
                      color='lightcyan', alpha=0.5)
    
    # Pin electrode
    ax1.fill_betweenx([25, 275], 0, g["pin_radius"], color='gray', alpha=0.8)
    ax1.axvline(x=g["pin_radius"], color='black', linewidth=2, linestyle='-')
    
    # Mesh electrode
    ax1.axvline(x=g["mesh_radius"], color='blue', linewidth=2, linestyle='--', 
               label='Mesh electrode')
    
    # Activation threshold contour
    ax1.axvline(x=r_onset, color='lime', linewidth=3, linestyle='-',
               label=f'Activation boundary (r={r_onset:.1f}mm)')
    
    # Shade activated region
    ax1.fill_betweenx([25, 275], g["pin_radius"], r_onset, 
                      color='lime', alpha=0.2, label='χ > 0.5 region')
    
    # RF Coil positions
    for i in range(g["coil_turns"]):
        z_coil = g["coil_z_start"] + i * g["coil_pitch"]
        ax1.plot(g["coil_radius"], z_coil, 'ro', markersize=8)
        ax1.plot(g["coil_radius"], z_coil + g["coil_pitch"]/2, 'ro', markersize=8)
    
    ax1.set_xlabel('Radial position r (mm)', fontsize=12)
    ax1.set_ylabel('Axial position z (mm)', fontsize=12)
    ax1.set_title(f'Electric Field Distribution (V={p["V_bias"]}V)\nActivation threshold λ={p["lambda_onset"]/1000:.0f} kV/m', 
                 fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 65])
    ax1.set_ylim([0, g["tube_length"]])
    ax1.legend(loc='upper right', fontsize=9)
    
    cbar1 = plt.colorbar(im1, ax=ax1, label='E-field (kV/m)')
    
    # =========================================================================
    # Right panel: Activation fraction (χ)
    # =========================================================================
    ax2 = axes[1]
    
    # Compute χ using sigmoid
    B_s = 5000  # smoothing parameter
    DeltaG0 = 350e3  # J/mol
    n, F, r_act, k_soft = 2, 96485, 2e-6, 8.0
    
    # Barrier balance
    DeltaB = np.zeros_like(R)
    E_for_chi = np.zeros_like(R)
    E_for_chi[active_mask] = E_field[active_mask]
    
    # ΔB = k_soft * ΔG₀ - n*F*E*r_act
    DeltaB = k_soft * DeltaG0 - n * F * E_for_chi * r_act
    
    # χ = 1 / (1 + exp(ΔB/B_s))
    chi = 1 / (1 + np.exp(DeltaB / B_s))
    chi[~active_mask] = 0
    
    chi_plot = np.ma.masked_where(chi == 0, chi)
    im2 = ax2.pcolormesh(R, Z, chi_plot, cmap='RdYlGn', shading='auto',
                         vmin=0, vmax=1)
    
    # Draw geometry (same as left panel)
    ax2.fill_betweenx([0, g["tube_length"]], g["tube_ID"]/2, g["tube_OD"]/2, 
                      color='lightcyan', alpha=0.5)
    ax2.fill_betweenx([25, 275], 0, g["pin_radius"], color='gray', alpha=0.8)
    ax2.axvline(x=g["pin_radius"], color='black', linewidth=2)
    ax2.axvline(x=g["mesh_radius"], color='blue', linewidth=2, linestyle='--')
    ax2.axvline(x=r_onset, color='white', linewidth=2, linestyle='-')
    
    # Powder fall region
    ax2.axhline(y=g["powder_z_start"], color='green', linewidth=1, linestyle=':')
    ax2.axhline(y=g["powder_z_end"], color='green', linewidth=1, linestyle=':')
    ax2.axvline(x=g["powder_r_min"], color='green', linewidth=1, linestyle=':')
    ax2.axvline(x=g["powder_r_max"], color='green', linewidth=1, linestyle=':')
    
    ax2.set_xlabel('Radial position r (mm)', fontsize=12)
    ax2.set_ylabel('Axial position z (mm)', fontsize=12)
    ax2.set_title('Flash Activation Order Parameter χ\n(Green = Activated, Red = Inactive)', 
                 fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 65])
    ax2.set_ylim([0, g["tube_length"]])
    
    cbar2 = plt.colorbar(im2, ax=ax2, label='χ (activation)')
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'rz_cross_section_fields.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: rz_cross_section_fields.png")


# =============================================================================
# RADIAL E-FIELD PROFILE
# =============================================================================
def plot_radial_E_profile():
    """Plot radial E-field profile at reactor midplane."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    g = GEOMETRY
    p = PHYSICS
    
    # Radial positions
    r = np.linspace(g["pin_radius"], g["mesh_radius"], 200)
    
    # E-field profile
    E = compute_E_field_coaxial(r, g["pin_radius"], g["mesh_radius"], p["V_bias"])
    
    # Activation radius
    r_onset = compute_activation_radius(
        p["V_bias"], p["lambda_onset"], g["pin_radius"], g["mesh_radius"]
    )
    
    # =========================================================================
    # Left: E-field profile
    # =========================================================================
    ax1 = axes[0]
    
    ax1.fill_between(r, 0, E/1000, where=(r <= r_onset), 
                     color='lime', alpha=0.3, label='Activated region')
    ax1.fill_between(r, 0, E/1000, where=(r > r_onset), 
                     color='red', alpha=0.2, label='Inactive region')
    ax1.plot(r, E/1000, 'b-', linewidth=2.5)
    
    ax1.axhline(y=p["lambda_onset"]/1000, color='red', linestyle='--', 
               linewidth=2, label=f'λ_onset = {p["lambda_onset"]/1000:.0f} kV/m')
    ax1.axvline(x=r_onset, color='green', linestyle='-', linewidth=2,
               label=f'r_onset = {r_onset:.1f} mm')
    
    # Annotations
    ax1.annotate('Pin\n(cathode)', xy=(g["pin_radius"], E[0]/1000), 
                xytext=(g["pin_radius"]-3, E[0]/1000*0.7),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('Mesh\n(anode)', xy=(g["mesh_radius"], E[-1]/1000),
                xytext=(g["mesh_radius"]+3, E[-1]/1000*1.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax1.set_xlabel('Radial position r (mm)', fontsize=12)
    ax1.set_ylabel('Electric field E (kV/m)', fontsize=12)
    ax1.set_title(f'Radial E-Field Profile at z = L/2\nV = {p["V_bias"]}V, Pin-Mesh Geometry', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([g["pin_radius"]-2, g["mesh_radius"]+2])
    ax1.set_ylim([0, max(E/1000)*1.1])
    
    # =========================================================================
    # Right: Voltage sweep showing activation radius
    # =========================================================================
    ax2 = axes[1]
    
    V_range = np.linspace(100, 1200, 100)
    r_onset_range = []
    chi_fraction = []
    
    for V in V_range:
        r_act = compute_activation_radius(V, p["lambda_onset"], 
                                         g["pin_radius"], g["mesh_radius"])
        r_onset_range.append(r_act)
        
        # Activation fraction
        a, b = g["pin_radius"], g["mesh_radius"]
        frac = (r_act**2 - a**2) / (b**2 - a**2)
        chi_fraction.append(np.clip(frac, 0, 1))
    
    ax2_twin = ax2.twinx()
    
    line1, = ax2.plot(V_range, r_onset_range, 'b-', linewidth=2.5, label='r_onset')
    ax2.axhline(y=g["pin_radius"], color='gray', linestyle=':', label='Pin radius')
    ax2.axhline(y=g["mesh_radius"], color='gray', linestyle='--', label='Mesh radius')
    ax2.fill_between(V_range, g["pin_radius"], r_onset_range, alpha=0.2, color='lime')
    
    line2, = ax2_twin.plot(V_range, [c*100 for c in chi_fraction], 'r--', 
                          linewidth=2, label='χ>0.5 fraction')
    
    ax2.axvline(x=p["V_bias"], color='orange', linewidth=2, linestyle='-',
               label=f'Current V = {p["V_bias"]}V')
    
    ax2.set_xlabel('Bias Voltage V (V)', fontsize=12)
    ax2.set_ylabel('Activation radius r_onset (mm)', fontsize=12, color='blue')
    ax2_twin.set_ylabel('Activation fraction (%)', fontsize=12, color='red')
    ax2.set_title('Activation Radius vs Voltage\n(Design Curve)', 
                 fontsize=13, fontweight='bold')
    
    # Combined legend
    lines = [line1, line2]
    labels = ['r_onset (mm)', 'χ>0.5 fraction (%)']
    ax2.legend(lines, labels, loc='center right', fontsize=10)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([100, 1200])
    ax2.set_ylim([g["pin_radius"]-1, g["mesh_radius"]+5])
    ax2_twin.set_ylim([0, 105])
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'radial_E_profile.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: radial_E_profile.png")


# =============================================================================
# COMPREHENSIVE SCHEMATIC
# =============================================================================
def plot_reactor_schematic():
    """Create a labeled schematic diagram of the reactor."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    g = GEOMETRY
    
    # Scale factor (mm to plot units)
    scale = 0.02
    offset_x = 2
    offset_z = 1
    
    def s(val):
        return val * scale
    
    # Draw tube (as rectangle in side view)
    tube_rect = Rectangle(
        (offset_x - s(g["tube_OD"]/2), offset_z),
        s(g["tube_OD"]), s(g["tube_length"]),
        facecolor='lightcyan', edgecolor='cyan', linewidth=2, alpha=0.5
    )
    ax.add_patch(tube_rect)
    
    # Inner tube wall
    inner_left = Rectangle(
        (offset_x - s(g["tube_ID"]/2) - s(5), offset_z),
        s(5), s(g["tube_length"]),
        facecolor='cyan', edgecolor='cyan', linewidth=1, alpha=0.7
    )
    inner_right = Rectangle(
        (offset_x + s(g["tube_ID"]/2), offset_z),
        s(5), s(g["tube_length"]),
        facecolor='cyan', edgecolor='cyan', linewidth=1, alpha=0.7
    )
    ax.add_patch(inner_left)
    ax.add_patch(inner_right)
    
    # Pin electrode
    pin = Rectangle(
        (offset_x - s(g["pin_radius"]), offset_z + s(25)),
        s(g["pin_radius"]*2), s(250),
        facecolor='darkgray', edgecolor='black', linewidth=2
    )
    ax.add_patch(pin)
    
    # Mesh electrode (dashed outline)
    mesh_left = ax.axvline(x=offset_x - s(g["mesh_radius"]), 
                          ymin=(25+offset_z/scale)/g["tube_length"], 
                          ymax=(275+offset_z/scale)/g["tube_length"],
                          color='blue', linewidth=2, linestyle='--')
    mesh_right = ax.axvline(x=offset_x + s(g["mesh_radius"]), 
                           ymin=(25+offset_z/scale)/g["tube_length"], 
                           ymax=(275+offset_z/scale)/g["tube_length"],
                           color='blue', linewidth=2, linestyle='--')
    
    # Active region (E-field zone)
    active = Rectangle(
        (offset_x - s(g["mesh_radius"]), offset_z + s(25)),
        s(g["mesh_radius"] - g["pin_radius"]), s(250),
        facecolor='yellow', edgecolor='none', alpha=0.3
    )
    ax.add_patch(active)
    active_r = Rectangle(
        (offset_x + s(g["pin_radius"]), offset_z + s(25)),
        s(g["mesh_radius"] - g["pin_radius"]), s(250),
        facecolor='yellow', edgecolor='none', alpha=0.3
    )
    ax.add_patch(active_r)
    
    # RF Coil turns
    for i in range(g["coil_turns"]):
        z_coil = offset_z + s(g["coil_z_start"] + i * g["coil_pitch"])
        # Left coil section
        coil_l = Circle((offset_x - s(g["coil_radius"]), z_coil), 
                        s(g["wire_radius"]), facecolor='red', edgecolor='darkred')
        ax.add_patch(coil_l)
        # Right coil section
        coil_r = Circle((offset_x + s(g["coil_radius"]), z_coil), 
                        s(g["wire_radius"]), facecolor='red', edgecolor='darkred')
        ax.add_patch(coil_r)
    
    # Powder fall arrows
    for r_off in [-s(20), 0, s(20)]:
        ax.annotate('', xy=(offset_x + r_off, offset_z + s(80)),
                   xytext=(offset_x + r_off, offset_z + s(220)),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Gas flow arrows
    ax.annotate('', xy=(offset_x - s(40), offset_z + s(50)),
               xytext=(offset_x - s(55), offset_z + s(50)),
               arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.annotate('', xy=(offset_x + s(55), offset_z + s(50)),
               xytext=(offset_x + s(40), offset_z + s(50)),
               arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    
    # Labels with arrows
    ax.annotate('Pin Electrode\n(Cathode, a=10mm)', 
               xy=(offset_x, offset_z + s(150)),
               xytext=(offset_x - 1.5, offset_z + s(150)),
               fontsize=11, ha='right',
               arrowprops=dict(arrowstyle='->', color='black'))
    
    ax.annotate('Mesh Electrode\n(Anode, b=22mm)', 
               xy=(offset_x + s(g["mesh_radius"]), offset_z + s(200)),
               xytext=(offset_x + 1.8, offset_z + s(200)),
               fontsize=11, ha='left',
               arrowprops=dict(arrowstyle='->', color='blue'))
    
    ax.annotate('RF Induction Coil\n(8 turns, r=55mm)', 
               xy=(offset_x + s(g["coil_radius"]), offset_z + s(100)),
               xytext=(offset_x + 2.2, offset_z + s(100)),
               fontsize=11, ha='left', color='red',
               arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.annotate('Quartz Tube\n(ID=100mm)', 
               xy=(offset_x + s(g["tube_ID"]/2), offset_z + s(280)),
               xytext=(offset_x + 1.5, offset_z + s(320)),
               fontsize=11, ha='left', color='darkcyan',
               arrowprops=dict(arrowstyle='->', color='darkcyan'))
    
    ax.annotate('Powder\nFall', 
               xy=(offset_x, offset_z + s(150)),
               xytext=(offset_x + 0.6, offset_z + s(150)),
               fontsize=10, ha='left', color='green')
    
    ax.annotate('H₂ Flow', 
               xy=(offset_x - s(48), offset_z + s(50)),
               xytext=(offset_x - s(48), offset_z + s(30)),
               fontsize=10, ha='center', color='purple')
    
    ax.annotate('Active Region\n(E-field)', 
               xy=(offset_x + s(16), offset_z + s(100)),
               xytext=(offset_x + s(16), offset_z + s(100)),
               fontsize=9, ha='center', color='orange', fontweight='bold')
    
    # Dimension lines
    # Gap dimension
    ax.annotate('', xy=(offset_x + s(g["pin_radius"]), offset_z + s(-15)),
               xytext=(offset_x + s(g["mesh_radius"]), offset_z + s(-15)),
               arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(offset_x + s((g["pin_radius"] + g["mesh_radius"])/2), 
           offset_z + s(-25), 'gap=12mm', ha='center', fontsize=10)
    
    # Title and formatting
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 8])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('PFR Reactor Schematic (Cross-Section View)\nPlasma Flash Reduction System', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend box
    legend_elements = [
        Rectangle((0,0), 1, 1, facecolor='darkgray', edgecolor='black', label='Pin Electrode'),
        Rectangle((0,0), 1, 1, facecolor='none', edgecolor='blue', linestyle='--', label='Mesh Electrode'),
        Circle((0,0), 0.1, facecolor='red', label='RF Coil'),
        Rectangle((0,0), 1, 1, facecolor='yellow', alpha=0.3, label='Active Region'),
        Rectangle((0,0), 1, 1, facecolor='lightcyan', edgecolor='cyan', label='Quartz Tube'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'reactor_schematic.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: reactor_schematic.png")


# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================
def plot_summary_dashboard():
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 14))
    
    g = GEOMETRY
    p = PHYSICS
    
    # Grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # Panel 1: 3D-like perspective (pseudo-3D using matplotlib)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    
    # Simplified 3D view
    x, y, z = create_cylinder_mesh(g["tube_ID"]/2, 0, g["tube_length"], n_theta=30, n_z=10)
    ax1.plot_surface(x, y, z, alpha=0.1, color='lightblue')
    
    x, y, z = create_cylinder_mesh(g["pin_radius"], 25, 275, n_theta=20, n_z=10)
    ax1.plot_surface(x, y, z, alpha=0.9, color='gray')
    
    x, y, z = create_helix_points(g["coil_radius"], g["coil_z_start"], 
                                  g["coil_pitch"], g["coil_turns"])
    ax1.plot(x, y, z, 'r-', linewidth=4)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Reactor Geometry', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    
    # =========================================================================
    # Panel 2: Radial E-field profile
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    r = np.linspace(g["pin_radius"], g["mesh_radius"], 100)
    E = compute_E_field_coaxial(r, g["pin_radius"], g["mesh_radius"], p["V_bias"])
    r_onset = compute_activation_radius(p["V_bias"], p["lambda_onset"], 
                                        g["pin_radius"], g["mesh_radius"])
    
    ax2.fill_between(r, 0, E/1000, where=(r <= r_onset), color='lime', alpha=0.4)
    ax2.fill_between(r, 0, E/1000, where=(r > r_onset), color='red', alpha=0.2)
    ax2.plot(r, E/1000, 'b-', linewidth=2)
    ax2.axhline(y=p["lambda_onset"]/1000, color='r', linestyle='--', linewidth=2)
    ax2.axvline(x=r_onset, color='g', linestyle='-', linewidth=2)
    
    ax2.set_xlabel('r (mm)')
    ax2.set_ylabel('E (kV/m)')
    ax2.set_title(f'Radial E-Field (V={p["V_bias"]}V)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 3: Activation vs Voltage
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 2:4])
    
    V_range = np.linspace(100, 1200, 100)
    chi_frac = []
    for V in V_range:
        r_act = compute_activation_radius(V, p["lambda_onset"], 
                                         g["pin_radius"], g["mesh_radius"])
        frac = (r_act**2 - g["pin_radius"]**2) / (g["mesh_radius"]**2 - g["pin_radius"]**2)
        chi_frac.append(np.clip(frac, 0, 1) * 100)
    
    ax3.fill_between(V_range, 0, chi_frac, alpha=0.3, color='green')
    ax3.plot(V_range, chi_frac, 'g-', linewidth=2)
    ax3.axvline(x=p["V_bias"], color='orange', linewidth=2, linestyle='--')
    ax3.axhline(y=40, color='red', linewidth=1, linestyle=':')
    
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Activation %')
    ax3.set_title('Activation Fraction vs Bias', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # =========================================================================
    # Panel 4: r-z E-field heatmap
    # =========================================================================
    ax4 = fig.add_subplot(gs[2, 0:2])
    
    r = np.linspace(0, g["tube_ID"]/2, 100)
    z = np.linspace(0, g["tube_length"], 150)
    R, Z = np.meshgrid(r, z)
    
    E_field = np.zeros_like(R)
    mask = (R >= g["pin_radius"]) & (R <= g["mesh_radius"]) & (Z >= 25) & (Z <= 275)
    E_field[mask] = compute_E_field_coaxial(R[mask], g["pin_radius"], 
                                            g["mesh_radius"], p["V_bias"])
    
    E_plot = np.ma.masked_where(E_field == 0, E_field)
    im = ax4.pcolormesh(R, Z, E_plot/1000, cmap='hot', shading='auto')
    ax4.set_xlabel('r (mm)')
    ax4.set_ylabel('z (mm)')
    ax4.set_title('E-Field Distribution (kV/m)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    # =========================================================================
    # Panel 5: Key parameters table
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 2:4])
    ax5.axis('off')
    
    table_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Tube ID', f'{g["tube_ID"]:.0f}', 'mm'],
        ['Pin radius (a)', f'{g["pin_radius"]:.0f}', 'mm'],
        ['Mesh radius (b)', f'{g["mesh_radius"]:.0f}', 'mm'],
        ['Electrode gap', f'{g["mesh_radius"]-g["pin_radius"]:.0f}', 'mm'],
        ['Active length', f'{250:.0f}', 'mm'],
        ['RF Coil turns', f'{g["coil_turns"]}', '-'],
        ['Bias voltage', f'{p["V_bias"]:.0f}', 'V'],
        ['λ_onset', f'{p["lambda_onset"]/1000:.0f}', 'kV/m'],
        ['r_onset', f'{r_onset:.1f}', 'mm'],
        ['χ>0.5 fraction', f'{chi_frac[int(p["V_bias"]/12)-8]:.1f}', '%'],
    ]
    
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax5.set_title('Reactor Parameters', fontsize=12, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle('PFR Digital Twin: Geometry & Field Visualization\nFor Haig - System Overview', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'summary_dashboard.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: summary_dashboard.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("PFR REACTOR GEOMETRY & FIELD VISUALIZATION DEMO")
    print("=" * 70 + "\n")
    
    print("Generating visualizations...")
    
    # Generate all visualizations
    plot_3d_geometry()
    plot_rz_cross_section()
    plot_radial_E_profile()
    plot_reactor_schematic()
    plot_summary_dashboard()
    
    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}\n")
    
    # List generated files
    print("Generated files:")
    for f in sorted(RESULTS_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
