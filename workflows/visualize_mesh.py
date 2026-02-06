#!/usr/bin/env python3
"""
Mesh/Grid Visualization for PFR Digital Twin
=============================================

Extracts and visualizes the computational grid from COMSOL simulation results.
Shows the (r,z) mesh structure used for field calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.colors as mcolors
from pathlib import Path
import h5py

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
RESULTS_DIR = Path(__file__).parent.parent / "results" / "mesh_visualization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GEOMETRY PARAMETERS (from run)
# =============================================================================
GEOMETRY = {
    "tube_ID": 100.0,       # mm
    "pin_radius": 10.0,     # mm (a)
    "mesh_radius": 22.0,    # mm (b)  
    "electrode_length": 250.0,  # mm
}


def load_grid_from_h5(h5_path: Path) -> dict:
    """Load grid coordinates and field data from HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        data = {}
        
        # Load grid
        if 'grid/r' in f and 'grid/z' in f:
            data['r'] = f['grid/r'][:]
            data['z'] = f['grid/z'][:]
        
        # Load available fields
        for group in ['em', 'flash', 'thermal', 'flow']:
            if group in f:
                data[group] = {}
                for key in f[group].keys():
                    data[group][key] = f[group][key][:]
        
        # Get attributes
        if 'coordinate_system' in f.attrs:
            data['coord_system'] = f.attrs['coordinate_system']
        
    return data


def create_mesh_visualization(data: dict, run_name: str):
    """Create comprehensive mesh visualization."""
    
    r = data.get('r', np.linspace(0.005, 0.05, 50))
    z = data.get('z', np.linspace(0, 0.3, 100))
    
    # Convert to mm for display
    r_mm = r * 1000
    z_mm = z * 1000
    
    nr, nz = len(r_mm), len(z_mm)
    
    print(f"  Grid dimensions: {nr} x {nz} = {nr*nz:,} cells")
    print(f"  r range: {r_mm[0]:.1f} - {r_mm[-1]:.1f} mm")
    print(f"  z range: {z_mm[0]:.1f} - {z_mm[-1]:.1f} mm")
    
    # =========================================================================
    # FIGURE 1: Full Mesh Grid
    # =========================================================================
    fig1, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Full mesh view
    ax = axes[0]
    
    # Draw vertical lines (constant r)
    for ri in r_mm[::max(1, nr//30)]:  # Sample every ~30 lines
        ax.axvline(x=ri, color='blue', alpha=0.3, linewidth=0.5)
    
    # Draw horizontal lines (constant z)  
    for zi in z_mm[::max(1, nz//50)]:  # Sample every ~50 lines
        ax.axhline(y=zi, color='blue', alpha=0.3, linewidth=0.5)
    
    # Mark electrode boundaries
    ax.axvline(x=GEOMETRY['pin_radius'], color='red', linewidth=2, 
               linestyle='--', label=f"Pin (a={GEOMETRY['pin_radius']}mm)")
    ax.axvline(x=GEOMETRY['mesh_radius'], color='green', linewidth=2,
               linestyle='--', label=f"Mesh (b={GEOMETRY['mesh_radius']}mm)")
    ax.axvline(x=GEOMETRY['tube_ID']/2, color='black', linewidth=2,
               linestyle='-', label=f"Tube wall")
    
    ax.set_xlabel('Radial Position r (mm)', fontsize=12)
    ax.set_ylabel('Axial Position z (mm)', fontsize=12)
    ax.set_title(f'Computational Grid (r-z)\n{nr} × {nz} = {nr*nz:,} cells', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 55])
    ax.set_ylim([0, z_mm[-1]])
    ax.set_aspect('equal')
    ax.grid(False)
    
    # Right: Zoomed view of electrode region
    ax2 = axes[1]
    
    # Draw mesh cells as rectangles
    for i, ri in enumerate(r_mm[:-1]):
        for j, zi in enumerate(z_mm[:-1]):
            if ri < 35:  # Only in active region
                dr = r_mm[i+1] - ri
                dz = z_mm[j+1] - zi
                # Color by cell size (finer mesh = darker)
                cell_area = dr * dz
                alpha = min(0.8, 0.05 / (cell_area + 0.01))
                rect = Rectangle((ri, zi), dr, dz, 
                                 fill=False, edgecolor='blue', 
                                 alpha=0.5, linewidth=0.3)
                ax2.add_patch(rect)
    
    # Mark electrodes
    ax2.axvline(x=GEOMETRY['pin_radius'], color='red', linewidth=3, label='Pin')
    ax2.axvline(x=GEOMETRY['mesh_radius'], color='green', linewidth=3, label='Mesh electrode')
    
    # Highlight active region
    ax2.axhspan(60, 240, alpha=0.1, color='yellow', label='Powder region')
    
    ax2.set_xlabel('Radial Position r (mm)', fontsize=12)
    ax2.set_ylabel('Axial Position z (mm)', fontsize=12)
    ax2.set_title('Electrode Region Detail\n(Active Zone)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, 40])
    ax2.set_ylim([0, z_mm[-1]])
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'COMSOL Mesh Visualization: {run_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / 'mesh_grid.png', dpi=150)
    plt.close(fig1)
    print(f"  ✓ Saved: mesh_grid.png")
    
    # =========================================================================
    # FIGURE 2: Mesh Quality Metrics
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cell size distribution (radial)
    ax = axes2[0, 0]
    dr = np.diff(r_mm)
    ax.bar(range(len(dr)), dr, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Radial Cell Index', fontsize=11)
    ax.set_ylabel('Cell Width Δr (mm)', fontsize=11)
    ax.set_title('Radial Cell Size Distribution', fontsize=12, fontweight='bold')
    ax.axhline(y=np.mean(dr), color='red', linestyle='--', label=f'Mean: {np.mean(dr):.3f}mm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cell size distribution (axial)
    ax = axes2[0, 1]
    dz = np.diff(z_mm)
    ax.bar(range(len(dz)), dz, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Axial Cell Index', fontsize=11)
    ax.set_ylabel('Cell Height Δz (mm)', fontsize=11)
    ax.set_title('Axial Cell Size Distribution', fontsize=12, fontweight='bold')
    ax.axhline(y=np.mean(dz), color='red', linestyle='--', label=f'Mean: {np.mean(dz):.3f}mm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2D cell area map
    ax = axes2[1, 0]
    DR, DZ = np.meshgrid(dr, dz[:-1] if len(dz) > len(dr) else dz, indexing='ij')
    if DR.shape[0] > 0 and DR.shape[1] > 0:
        # Truncate to match
        min_r = min(DR.shape[0], len(r_mm)-1)
        min_z = min(DR.shape[1], len(z_mm)-1)
        cell_areas = DR[:min_r, :min_z] * DZ[:min_r, :min_z]
        
        im = ax.imshow(cell_areas.T, origin='lower', aspect='auto', cmap='viridis',
                      extent=[r_mm[0], r_mm[min_r], z_mm[0], z_mm[min_z]])
        plt.colorbar(im, ax=ax, label='Cell Area (mm²)')
        ax.set_xlabel('r (mm)', fontsize=11)
        ax.set_ylabel('z (mm)', fontsize=11)
        ax.set_title('Cell Area Distribution', fontsize=12, fontweight='bold')
    
    # Mesh statistics table
    ax = axes2[1, 1]
    ax.axis('off')
    
    stats = [
        ['Metric', 'Value'],
        ['Total Cells', f'{nr * nz:,}'],
        ['Radial Cells (nr)', f'{nr}'],
        ['Axial Cells (nz)', f'{nz}'],
        ['r range', f'{r_mm[0]:.1f} - {r_mm[-1]:.1f} mm'],
        ['z range', f'{z_mm[0]:.1f} - {z_mm[-1]:.1f} mm'],
        ['Min Δr', f'{np.min(dr):.4f} mm'],
        ['Max Δr', f'{np.max(dr):.4f} mm'],
        ['Min Δz', f'{np.min(dz):.4f} mm'],
        ['Max Δz', f'{np.max(dz):.4f} mm'],
        ['Aspect Ratio Range', f'{np.min(dr)/np.max(dz):.2f} - {np.max(dr)/np.min(dz):.2f}'],
    ]
    
    table = ax.table(cellText=stats, loc='center', cellLoc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    
    for j in range(2):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Mesh Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Mesh Quality Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'mesh_quality.png', dpi=150)
    plt.close(fig2)
    print(f"  ✓ Saved: mesh_quality.png")
    
    # =========================================================================
    # FIGURE 3: 3D Mesh Visualization (Axisymmetric Revolution)
    # =========================================================================
    fig3 = plt.figure(figsize=(14, 10))
    ax3d = fig3.add_subplot(111, projection='3d')
    
    # Create cylindrical mesh by revolving r-z grid
    n_theta = 60
    theta = np.linspace(0, 2*np.pi, n_theta)
    
    # Sample the grid
    r_sample = r_mm[::max(1, nr//15)]
    z_sample = z_mm[::max(1, nz//20)]
    
    # Draw radial lines (circles at constant z)
    for zi in z_sample:
        for ri in r_sample:
            x = ri * np.cos(theta)
            y = ri * np.sin(theta)
            z_line = np.full_like(theta, zi)
            ax3d.plot(x, y, z_line, color='blue', alpha=0.2, linewidth=0.3)
    
    # Draw axial lines
    for th in theta[::6]:
        for ri in r_sample:
            x = ri * np.cos(th) * np.ones(len(z_sample))
            y = ri * np.sin(th) * np.ones(len(z_sample))
            ax3d.plot(x, y, z_sample, color='blue', alpha=0.2, linewidth=0.3)
    
    # Draw electrode surfaces
    # Pin
    th_full = np.linspace(0, 2*np.pi, 50)
    z_full = np.linspace(20, 280, 20)
    TH, ZZ = np.meshgrid(th_full, z_full)
    X_pin = GEOMETRY['pin_radius'] * np.cos(TH)
    Y_pin = GEOMETRY['pin_radius'] * np.sin(TH)
    ax3d.plot_surface(X_pin, Y_pin, ZZ, color='red', alpha=0.4, label='Pin')
    
    # Mesh electrode
    X_mesh = GEOMETRY['mesh_radius'] * np.cos(TH)
    Y_mesh = GEOMETRY['mesh_radius'] * np.sin(TH)
    ax3d.plot_surface(X_mesh, Y_mesh, ZZ, color='green', alpha=0.3, label='Mesh')
    
    # Tube wall
    X_tube = (GEOMETRY['tube_ID']/2) * np.cos(TH)
    Y_tube = (GEOMETRY['tube_ID']/2) * np.sin(TH)
    ax3d.plot_surface(X_tube, Y_tube, ZZ, color='gray', alpha=0.2, label='Tube')
    
    ax3d.set_xlabel('X (mm)')
    ax3d.set_ylabel('Y (mm)')
    ax3d.set_zlabel('Z (mm)')
    ax3d.set_title('3D Axisymmetric Mesh\n(r-z grid revolved around axis)', 
                  fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = max(r_mm[-1], z_mm[-1]/2)
    ax3d.set_xlim([-max_range, max_range])
    ax3d.set_ylim([-max_range, max_range])
    ax3d.set_zlim([0, z_mm[-1]])
    
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / 'mesh_3d.png', dpi=150)
    plt.close(fig3)
    print(f"  ✓ Saved: mesh_3d.png")
    
    # =========================================================================
    # FIGURE 4: Field on Mesh (if available)
    # =========================================================================
    if 'em' in data and 'E_mag' in data['em']:
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        
        E_mag = data['em']['E_mag']
        R, Z = np.meshgrid(r_mm, z_mm, indexing='ij')
        
        # Clip for visualization
        E_plot = np.clip(E_mag, 0, 200000)
        
        im = ax4.pcolormesh(R, Z, E_plot/1000, shading='auto', cmap='hot')
        plt.colorbar(im, ax=ax4, label='|E| (kV/m)')
        
        # Overlay mesh lines (sparse)
        for ri in r_mm[::max(1, nr//20)]:
            ax4.axvline(x=ri, color='white', alpha=0.2, linewidth=0.3)
        for zi in z_mm[::max(1, nz//30)]:
            ax4.axhline(y=zi, color='white', alpha=0.2, linewidth=0.3)
        
        ax4.set_xlabel('r (mm)', fontsize=12)
        ax4.set_ylabel('z (mm)', fontsize=12)
        ax4.set_title('Electric Field |E| on Computational Mesh', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlim([0, 50])
        
        plt.tight_layout()
        fig4.savefig(RESULTS_DIR / 'field_on_mesh.png', dpi=150)
        plt.close(fig4)
        print(f"  ✓ Saved: field_on_mesh.png")


def main():
    print("\n" + "🔷 " + "=" * 60)
    print("   MESH VISUALIZATION FOR PFR DIGITAL TWIN")
    print("🔷 " + "=" * 60)
    
    # Find an HDF5 file with grid data
    h5_candidates = list(Path("/Users/ricfulop/Flash-Physics-Twin/Flash-Physics-Twin/results").rglob("fields.h5"))
    
    if not h5_candidates:
        print("\n⚠ No HDF5 field files found. Creating synthetic mesh for visualization.")
        # Create synthetic grid matching typical COMSOL output
        data = {
            'r': np.linspace(0.005, 0.05, 100),  # 5-50mm
            'z': np.linspace(0, 0.3, 200),       # 0-300mm
        }
        run_name = "Synthetic Grid"
    else:
        h5_path = h5_candidates[0]
        run_name = h5_path.parent.parent.name
        print(f"\n📂 Loading: {h5_path}")
        print(f"   Run: {run_name}")
        
        data = load_grid_from_h5(h5_path)
    
    print(f"\n📊 Creating mesh visualizations...")
    create_mesh_visualization(data, run_name)
    
    print(f"\n🎯 All outputs saved to: {RESULTS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
