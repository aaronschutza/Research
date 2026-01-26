import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def solve_glueball_spectrum(beta, n_segments=100, tension=1.0, r_eq=1.0):
    """
    Solves the eigenmodes of a closed QCD flux tube (Glueball).
    
    Parameters:
    beta (float): Geometric Stiffness (1.91 for WTS, 1.0 for Standard)
    n_segments (int): Discretization of the loop
    tension (float): String tension (sigma)
    
    Returns:
    eigenvalues (mass_squared), eigenvectors
    """
    
    # 1. Initialize Stiffness Matrix (Hessian)
    # Dimension is N x N (Radial degrees of freedom for breathing/deformation)
    K = np.zeros((n_segments, n_segments))
    
    # The "Effective Spring Constant" of the vacuum radial potential
    # V(r) ~ r^beta  -->  k_eff = V''(r) ~ beta * (beta-1) * r^(beta-2)
    # However, for the linearized restoring force F = -k*x:
    # In the WTS derivation, the restoring force scales as F ~ r^(beta-1)
    # So the effective local stiffness is proportional to beta.
    k_radial = beta * (tension / r_eq) 
    
    # The String Tension coupling between neighbors
    # For a discrete ring, k_string ~ Tension / (ds^2)
    ds = (2 * np.pi * r_eq) / n_segments
    k_string = tension / (ds**2)

    for i in range(n_segments):
        # Diagonal Elements: Radial Stiffness + 2 Neighbors
        K[i, i] = k_radial + 2 * k_string
        
        # Off-Diagonal: Neighbor Coupling (Periodic BC)
        left = (i - 1) % n_segments
        right = (i + 1) % n_segments
        
        K[i, left] -= k_string
        K[i, right] -= k_string
        
    # 2. Solve Eigenvalue Problem
    # We solve K * v = w^2 * v
    # w (frequency) corresponds to Mass
    eigenvals, eigenvecs = eigh(K)
    
    # Return Mass (sqrt of eigenvalue), sorted
    # We ignore the zero-mode (rotation) if present, but radial stiffness 
    # makes all radial modes massive.
    masses = np.sqrt(np.abs(eigenvals))
    return masses

# --- MAIN ANALYSIS ---

# Constants
BETA_WTS = 1.90986   # 6/pi
BETA_STD = 1.0       # Standard String
M_SCALAR_EXP = 1710.0 # f0(1710) Candidate (MeV)

# Run Simulations
print("--- WTS GLUEBALL EIGENMODE SOLVER ---")
print(f"Target Scalar Mass: {M_SCALAR_EXP} MeV")

# 1. Standard String (Beta = 1)
raw_masses_std = solve_glueball_spectrum(BETA_STD)
# Normalize so ground state (Breathing Mode n=0) matches 1710
scale_std = M_SCALAR_EXP / raw_masses_std[0] 
masses_std = raw_masses_std * scale_std

# 2. WTS Stiff String (Beta = 1.91)
raw_masses_wts = solve_glueball_spectrum(BETA_WTS)
# Normalize so ground state matches 1710
scale_wts = M_SCALAR_EXP / raw_masses_wts[0]
masses_wts = raw_masses_wts * scale_wts

# --- EXTRACT KEY MODES ---
# Mode 0: Breathing (0++) - Monopole
# Mode 1: Dipole (1--) - Often forbidden/mixed in pure glue
# Mode 2: Quadrupole (2++) - The Tensor Glueball
# In a 1D ring simulation, the degeneracy is:
# k=0 (1 state), k=1 (2 states), k=2 (2 states)...

# Extracting the unique frequencies (handling degeneracy)
unique_wts = np.unique(masses_wts.round(1))
unique_std = np.unique(masses_std.round(1))

m0_wts = unique_wts[0] # 0++
m2_wts = unique_wts[1] # 2++ (First excited geometric mode)

m0_std = unique_std[0]
m2_std = unique_std[1]

print("\n--- RESULTS ---")
print(f"WTS PREDICTION (Beta={BETA_WTS:.2f}):")
print(f"  Scalar (0++): {m0_wts:.1f} MeV")
print(f"  Tensor (2++): {m2_wts:.1f} MeV")
print(f"  Ratio M(2)/M(0): {m2_wts/m0_wts:.3f}")

print(f"\nSTANDARD STRING PREDICTION (Beta={BETA_STD:.2f}):")
print(f"  Scalar (0++): {m0_std:.1f} MeV")
print(f"  Tensor (2++): {m2_std:.1f} MeV")
print(f"  Ratio M(2)/M(0): {m2_std/m0_std:.3f}")

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

# Plot Levels
plt.hlines(unique_wts[:4], 1, 2, colors='r', linewidth=3, label='WTS (Stiff)')
plt.hlines(unique_std[:4], 0, 1, colors='b', linewidth=3, linestyles='--', label='Standard (Floppy)')

# Labels
for m in unique_wts[:4]:
    plt.text(1.5, m + 50, f"{int(m)}", color='r', ha='center', fontweight='bold')

for m in unique_std[:4]:
    plt.text(0.5, m + 50, f"{int(m)}", color='b', ha='center')

# Annotations
plt.text(0.5, m0_std - 150, "Scalar (0++)", ha='center', color='gray')
plt.text(0.5, m2_std - 150, "Tensor (2++)", ha='center', color='gray')

plt.title(f"Glueball Mass Spectrum: Stiff Vacuum vs. Standard String\nNormalized to f0(1710)", fontsize=14)
plt.ylabel("Mass (MeV)")
plt.xticks([0.5, 1.5], ["Standard String\n(Beta=1)", "WTS Stiff String\n(Beta=1.91)"])
plt.xlim(0, 2)
plt.ylim(1500, 4500)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig("glueball_spectrum.png")
print("\nSpectrum plot saved to glueball_spectrum.png")
plt.show()