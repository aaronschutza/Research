import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import brentq

# --- 1. PHYSICAL CONSTANTS ---
HBAR_C = 197.327     # MeV fm
M_PROTON = 938.272   # MeV
M_NEUTRON = 939.565  # MeV
M_REDUCED = (M_PROTON * M_NEUTRON) / (M_PROTON + M_NEUTRON)

# --- 2. SOLVER ENGINE ---
def get_energy_for_depth(v0, r_grid, width, target_energy=None):
    N = len(r_grid)
    dr = r_grid[1] - r_grid[0]
    KE_factor = (HBAR_C**2) / (2 * M_REDUCED * dr**2)
    V = -v0 * np.exp(-(r_grid / width)**2)
    
    # Hamiltonian Construction
    dim = N - 2
    main_diag = 2 * KE_factor + V[1:-1]
    off_diag = -KE_factor * np.ones(dim - 1)
    H = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    evals, evecs = np.linalg.eigh(H)
    E_ground = evals[0]
    
    if target_energy is not None:
        return E_ground - target_energy
    
    psi_inner = evecs[:, 0]
    psi_full = np.zeros(N)
    psi_full[1:-1] = psi_inner
    norm = np.sqrt(simpson(psi_full**2, x=r_grid))
    return E_ground, psi_full / norm

# --- 3. CALIBRATION & SIMULATION ---
print("--- Generating Vacuum Compression Plot ---")
r_grid = np.linspace(0.0, 15.0, 2000) 
r_width = 1.8 
target_E = -2.224

# Calibrate Potential
optimal_v0 = brentq(get_energy_for_depth, 30.0, 80.0, args=(r_grid, r_width, target_E))
E_final, u_final = get_energy_for_depth(optimal_v0, r_grid, r_width)
rho = u_final**2

# WTS Stiffness Profile (The "Compressive Force")
BETA = 1.91
# stiffness_profile represents the localized vacuum rigidity 
stiffness_profile = np.exp(-(r_grid / (1.0/BETA))**2) 

# --- 4. VISUALIZATION ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Density
ax.plot(r_grid, rho, 'k-', linewidth=3, label=r'Nucleon Density $\rho(r)$ (Compact)')
ax.fill_between(r_grid, rho, color='gray', alpha=0.1)

# Plot Stiffness Intensity
ax2 = ax.twinx()
ax2.plot(r_grid, stiffness_profile, 'r--', linewidth=2, label='Vacuum Stiffness Intensity')
ax2.set_ylabel("Geometric Stiffness Penalty", color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 1.2)

# Annotations
ax.text(3.5, max(rho)*0.6, "MODERN CONSENSUS\n$r_p \\approx 0.84$ fm", fontsize=12, fontweight='bold', color='black')
ax.annotate("Stiff Core Region\nPrevents Diffusion", 
            xy=(0.5, max(rho)), xytext=(2.0, max(rho)),
            arrowprops=dict(facecolor='red', shrink=0.05),
            fontsize=10, color='red')

# Styling
ax.set_title(f"Vacuum Compression: Stiffness Constrains the Nucleon\nTarget Binding: {E_final:.2f} MeV", fontsize=14)
ax.set_xlabel("Radial Distance $r$ (fm)", fontsize=12)
ax.set_ylabel("Probability Density", fontsize=12)
ax.set_xlim(0, 8)
ax.grid(True, alpha=0.3)

# Legend
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='center right')

plt.tight_layout()
plt.savefig("deuteron_compression.png")
print("Saved to deuteron_compression.png")
plt.show()