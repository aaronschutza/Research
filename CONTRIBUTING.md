# Contributing to WTS-RT

First off, welcome. If you are here, you probably saw the Reddit threads or the Bluesky demos.

**WTS-RT** is not a standard graphics project. It is a **Computational Physics** project applied to rendering. This distinction matters because "optimizing" the code often means "breaking the physics."

This document outlines how to contribute effectively without accidentally turning the Symplectic Solver back into a standard Gaussian Blur.

## 📉 The Core Philosophy: Physics First

We are porting the **Rice Convection Model (RCM)**—a magnetospheric plasma physics code—to solve the Rendering Equation.

* **Standard Denoiser:** "How do I smooth this pixel based on its neighbors?"
* **WTS-RT:** "How does the vacuum stiffness  constrain the flow of the photon fluid?"

**Rule #1:** If your PR removes the `momentum` term or the `symplectic` update step in favor of a standard `lerp()`, it will be rejected. The "Inertia" is the feature, not a bug.

## 🛠 Areas We Need Help With

We are currently in **Phase 1: Proof of Concept**. The Python/Taichi implementation is a toy model. We need help scaling this to a real engine.

### 1. Porting to C++ / HLSL / GLSL

The ultimate goal is a Compute Shader implementation that can be dropped into a Vulkan or DX12 engine.

* **Goal:** A standalone `.comp` or `.hlsl` shader that takes a `Texture2D<float4> NoisyInput` and `Texture2D<float4> GBuffer` and outputs `Texture2D<float4> RelaxedOutput`.
* **Constraint:** Must maintain the double-buffered state (Ping-Pong) to prevent race conditions.

### 2. The "Stiffness Heuristic"

Currently, we calculate  (Stiffness) using a naive check:

```python
beta = 0.1 # Vacuum
if hit_geometry: beta = 50.0 # Wall

```

This is crude. We need a robust heuristic that derives  from the **G-Buffer** (Depth, Normal, Roughness) automatically.

* **Idea:**  (Stiffness increases with geometric variance).
* **Help Wanted:** Experiment with different mapping functions in `wts_demo_rtx.py` and PR the one that preserves edges best.

### 3. JAX / Torch Optimization

The `solver.py` script is slow because it runs on CPU (mostly). A pure JAX implementation that runs fully on GPU would allow for massive batch testing of parameters.

## 🚫 The "Third Rail" (Do Not Touch)

Unless you have a specific derivation from the WTS Action, **do not change the Integrator Scheme.**

We use a **Damped Symplectic Integrator**:

1. `Force = Laplacian / Stiffness`
2. `Momentum = Momentum * Damping + Force * dt`
3. `Field = Field + Momentum`

**Why?**
Many contributors try to simplify this to:
`Field += (Target - Field) * Rate` (Successive Over-Relaxation)

**Do not do this.** SOR is purely diffusive (Parabolic PDE). It has no mass. It cannot model the "swing" of a shadow when a light source moves. WTS is Hyperbolic (Wave-like). It preserves temporal coherence via inertia.

## 🐛 Reporting Bugs

If the simulation "explodes" (turns black/NaNs), please include:

1. The Resolution used.
2. The `VACUUM_STIFFNESS` value.
3. A screenshot of the failure state.

(Low stiffness at High Resolution usually causes numerical explosions due to the massive forces involved. We use a `tanh` soft-clamp to mitigate this, but it's not perfect).

## 📜 License & Attribution

This project is MIT Licensed.
However, because this is derived from the **Wolf-Toffoletto-Schutza** framework, please maintain the reference to the original **Rice Convection Model** lineage in the headers.
