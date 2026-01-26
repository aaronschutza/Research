import numpy as np
import matplotlib.pyplot as plt

def yukawa_potential(r, g_sq=0.5, mass=0.14):
    """
    Standard meson-exchange potential (Pion exchange).
    V(r) ~ - (g^2 / r) * exp(-m * r)
    """
    # Avoid singularity
    r = np.maximum(r, 0.05) 
    return - (g_sq / r) * np.exp(-mass * r)

def wts_knot_potential(r, g_sq=0.5, mass=0.14, stiffness=1.91, r_knot=1.2):
    """
    WTS 'Crimped' Potential.
    Adds a Stiffness Barrier that represents the energy cost 
    to untie the geometric knot.
    """
    v_base = yukawa_potential(r, g_sq, mass)
    
    # The Stiffness Barrier (The "Crimp")
    # A Gaussian-like barrier at the dissociation radius r_knot
    # Height scales with Vacuum Stiffness (beta)
    barrier_height = 0.15 * stiffness
    barrier_width = 0.4
    
    barrier = barrier_height * np.exp(-((r - r_knot)**2) / (2 * barrier_width**2))
    
    return v_base + barrier

# --- CONFIGURATION ---
R_DOMAIN = np.linspace(0.1, 4.0, 500) # Fermis (fm)
M_PION = 0.14 # GeV (Exchange mass)
STIFFNESS_BETA = 1.91

# --- SIMULATION ---
print("--- Simulating Exotic Hadron Binding (X3872) ---")
v_std = yukawa_potential(R_DOMAIN, mass=M_PION)
v_wts = wts_knot_potential(R_DOMAIN, mass=M_PION, stiffness=STIFFNESS_BETA)

# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))

# Plot Standard Model
plt.plot(R_DOMAIN, v_std, 'b--', linewidth=2, label='Standard Molecular Model (Yukawa)')
# Shade the "Leaking" area
plt.fill_between(R_DOMAIN, v_std, 0, where=(R_DOMAIN > 1.5), color='blue', alpha=0.05)
plt.text(2.5, -0.05, "Easy Decay path\n(Broad Width)", color='blue', ha='center', fontsize=9)

# Plot WTS Model
plt.plot(R_DOMAIN, v_wts, 'r-', linewidth=3, label='WTS Geometric Knot (Crimped)')
# Shade the "Trapped" area
plt.fill_between(R_DOMAIN, v_wts, 0, where=(v_wts < 0), color='red', alpha=0.1)

# Annotations
# The Barrier
barrier_idx = np.argmax(v_wts)
r_barrier = R_DOMAIN[barrier_idx]
v_barrier = v_wts[barrier_idx]

plt.annotate('THE GEOMETRIC CRIMP\n(Stiffness Barrier)', 
             xy=(r_barrier, v_barrier), 
             xytext=(r_barrier + 0.5, v_barrier + 0.2),
             arrowprops=dict(facecolor='red', shrink=0.05),
             color='red', fontweight='bold')

plt.hlines(0, 0, 4, colors='k', linestyles='-', alpha=0.3)
plt.text(0.5, -0.4, "Bound State\nTrapped Here", color='red', fontweight='bold', ha='center')

plt.title("The Origin of the Narrow Width: X(3872) Stability\nStandard Molecule vs. WTS Geometric Knot", fontsize=14)
plt.xlabel("Separation Distance $r$ (fm)")
plt.ylabel("Effective Potential $V(r)$ (GeV)")
plt.ylim(-0.6, 0.4)
plt.xlim(0, 4)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("crimped_molecule.png")
print("Simulation complete. Visual saved to crimped_molecule.png")
plt.show()