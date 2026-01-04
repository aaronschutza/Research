# Research
My personal repository for research materials and scripts.  Any data or numerical code which appears in papers is included here.
# Axiomatic Physical Homeostasis (APH): A Computational Framework for Non-Associative Geometry

## Overview
This repository contains the primary manuscripts, derivation notes, and computational proofs for Axiomatic Physical Homeostasis (APH), a unified framework proposing that physical laws are emergent homeostatic control mechanisms on a $G_2$ manifold.

The core hypothesis challenges the standard perturbative approach to Quantum Chromodynamics (QCD). Instead of lattice discretization, we propose a Geometric Stiffness parameter ($\beta_{QCD} = 6/\pi \approx 1.91$) derived from the topology of Sedenion zero divisors, which stabilizes the vacuum against non-associative decay.

## Featured Research: The Dissertation Proposal
Title: *Non-Linear Stability of Effective Strings: A $G_2$ Geometric Stiffness Derivation of the Yang-Mills Mass Gap via Thin Filament Dynamics* January 4, 2026  
Status: Active Defense / Preprint  

### Abstract & Key Findings
Recent mathematical results (Reggiani, 2024) establish an isometry between the manifold of Sedenion zero divisors $\mathcal{Z}(\mathbb{S})$ and the exceptional Lie group $G_2$. We utilize the Thin Filament Code (TFC)—a verified non-linear stability solver originally designed for magnetospheric flux tubes—to model the QCD vacuum as a chaotic, non-associative medium.

By inputting the derived stiffness $\beta_{QCD}$ into the TFC solver, we recover the hadronic spectrum from first principles without free parameters.

Key Predictions:
* The Mass Gap: A strictly positive fundamental frequency $\omega_0 > 0$, providing a classical dynamical origin for the Yang-Mills Mass Gap.
* **Glueball Mass: Prediction of a scalar glueball mass at **1710 MeV**.
* Proton Spin: A computed proton spin contribution of $\Sigma \approx 0.34$, matching EMC measurements.


## 📚 Repository Contents

| File | Description |
| :--- | :--- |
| `Dissertation_Proposal.pdf` | Start Here. The formal definition of the $G_2$ Stiffness Hypothesis and QCD application. |
| `Flavor_from_Geometry.pdf` | The foundational manuscript deriving the 3-generation limit of fermions from $G_2$ geometry. |
| `The_Cosmic_Strata.pdf` | Chronology of the non-associative Big Bang and Sedenionic filtration events. |
| `Dynamical_Intelligence.pdf` | Architecture for Artificial General Intelligence (AGI) based on rank-3 filament networks. |
| `Rule_30.pdf` | Analysis of Wolfram's Rule 30 as the micro-state of the primordial vacuum. |
| `/TFC_Source` | (Coming Soon) The Fortran/C++ source code for the Thin Filament solver. |

---

## 🖊️ How to Cite
If you utilize the APH framework, the Geometric Stiffness parameter, or the TFC solver methodology in your research, please cite the foundational text:

```bibtex
@book{Schutza2026Flavor,
  title={The Flavor Hierarchy from Geometry: An Algebraic Framework in M-theory on G2 Manifolds and the Homeostatic Universe},
  author={Schutza, Aaron M.},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.XXXXXX},
  url={[https://zenodo.org/record/XXXXXX](https://zenodo.org/record/XXXXXX)}
}
