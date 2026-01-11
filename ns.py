import taichi as ti
import math

try:
    ti.init(arch=ti.vulkan, device_memory_GB=4.0, offline_cache=True)
except:
    ti.init(arch=ti.cpu)

# --- CONFIG ---
RES = (960, 540)
ASPECT = RES[0] / RES[1]
GRAVITY_CONST = 0.4 # Reduced to prevent instant collapse
DAMPING = 0.99
MAGNETIC_SPIN = 2.0 

# --- FIELDS ---
SIM_RES = (512, 512)
density = ti.field(dtype=float, shape=SIM_RES)    
velocity = ti.field(dtype=float, shape=SIM_RES)   
stiffness = ti.field(dtype=float, shape=SIM_RES)  
hazard = ti.field(dtype=float, shape=SIM_RES) 

# Visual Buffers
display_buffer = ti.Vector.field(3, dtype=float, shape=RES)
color_map = ti.Vector.field(3, dtype=float, shape=8) 

# Telemetry
radial_profile = ti.field(dtype=float, shape=128) 
radial_counts = ti.field(dtype=int, shape=128) # For averaging
time = ti.field(dtype=float, shape=())
telemetry_max_density = ti.field(dtype=float, shape=())
telemetry_max_stiffness = ti.field(dtype=float, shape=())
telemetry_torsion = ti.field(dtype=float, shape=())

# --- HELPERS ---
@ti.func
def clamp(x, mi, ma): return min(max(x, mi), ma)
@ti.func
def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
@ti.func
def mix(x, y, a): return x * (1.0 - a) + y * a
@ti.func
def map_coord(x, y):
    return ti.Vector([x / SIM_RES[0] * 2.0 - 1.0, y / SIM_RES[1] * 2.0 - 1.0])

@ti.kernel
def init_sim():
    color_map[0] = [0.0, 0.0, 0.02] # Space
    color_map[1] = [0.5, 0.1, 0.4]  # Accretion (Red-Shift)
    color_map[2] = [1.0, 0.4, 0.1]  # Crust (Hot)
    color_map[3] = [0.2, 0.8, 1.0]  # Atmosphere (Cyan)
    color_map[4] = [1.0, 1.0, 1.0]  # Core (White Hot)
    color_map[5] = [0.2, 0.6, 1.0]  # Jet

    # SEED: Large, Stable Neutron Star
    center_x = SIM_RES[0] // 2
    center_y = SIM_RES[1] // 2
    for i, j in density:
        dx = float(i - center_x)
        dy = float(j - center_y)
        dist = ti.sqrt(dx*dx + dy*dy)
        
        # Make the star visibly large (radius 60 pixels)
        if dist < 60.0:
            # Smooth density falloff
            d_val = 50.0 * ti.pow(1.0 - dist/60.0, 0.5) 
            density[i, j] = d_val
            stiffness[i, j] = 20.0 # Initial Rigidity

@ti.kernel
def update_hydrodynamics(dt: float):
    max_rho = 0.0
    max_beta = 0.0
    total_torsion = 0.0
    
    for i, j in density:
        uv = map_coord(i, j)
        r = uv.norm()
        
        rho = density[i, j]
        vel = velocity[i, j]
        beta = stiffness[i, j]
        haz = hazard[i, j]
        
        # 1. GRAVITY
        f_gravity = -GRAVITY_CONST / (r + 0.1) * rho
        
        # 2. PAULI EXCLUSION FORCE (Volume Preservation)
        # If density gets too high, stiffness goes to Infinity
        # This prevents the "Invisible Singularity" bug
        target_beta = 0.1 + ti.pow(rho, 2.0)
        if rho > 40.0: target_beta *= 10.0 # Hard core repulsion
        
        beta = mix(beta, target_beta, 0.2)
        stiffness[i, j] = beta
        
        # 3. WTS PRESSURE
        laplacian = 0.0
        if i > 0 and i < SIM_RES[0]-1 and j > 0 and j < SIM_RES[1]-1:
            laplacian = density[i+1,j] + density[i-1,j] + density[i,j+1] + density[i,j-1] - 4*rho
        
        f_pressure = beta * ti.tanh(laplacian * 1.0)
        
        # 4. TORSION
        f_torsion = 0.0
        angle = ti.atan2(uv.y, uv.x)
        if r < 0.25: f_torsion = 4.0 * dt # Spin the core
        
        # Integration
        accel = f_gravity + f_pressure
        vel += accel * dt
        if r < 0.5: vel += f_torsion * 0.2
        vel *= DAMPING
        rho += vel * dt
        
        # 5. ACCRETION DISK FEED
        # Keep the ring visible
        if r > 0.4 and r < 0.8:
            spiral = ti.sin(angle * 4.0 + time[None]*2.0)
            if spiral > 0.5: rho += 0.1 * dt
            
        # Clamps
        if rho < 0.0: rho = 0.0
        if rho > 80.0: rho = 80.0 # Ceiling
        
        density[i, j] = rho
        velocity[i, j] = vel
        
        if rho > max_rho: max_rho = rho
        if beta > max_beta: max_beta = beta
        total_torsion += ti.abs(vel)
        
    telemetry_max_density[None] = max_rho
    telemetry_max_stiffness[None] = max_beta
    telemetry_torsion[None] = total_torsion / (SIM_RES[0]*SIM_RES[1])

@ti.kernel
def update_profile_binned():
    # Clear bins
    for k in range(128):
        radial_profile[k] = 0.0
        radial_counts[k] = 0
        
    center_x = SIM_RES[0] // 2
    center_y = SIM_RES[1] // 2
    
    # Iterate ALL pixels to bin them by radius
    # This removes noise and gives a clean trend line
    for i, j in density:
        dx = float(i - center_x)
        dy = float(j - center_y)
        dist = ti.sqrt(dx*dx + dy*dy)
        
        # Map distance to bin index (0..127)
        # Max radius ~ 256 pixels
        bin_idx = int((dist / 256.0) * 128.0)
        if bin_idx >= 0 and bin_idx < 128:
            radial_profile[bin_idx] += density[i, j]
            radial_counts[bin_idx] += 1
            
    # Average
    for k in range(128):
        if radial_counts[k] > 0:
            radial_profile[k] /= float(radial_counts[k])

@ti.kernel
def render_star(t: float):
    cam_pos = ti.Vector([0.0, 4.0, -5.0]) 
    target = ti.Vector([0.0, 0.0, 0.0])
    up = ti.Vector([0.0, 1.0, 0.0])
    fwd = (target - cam_pos).normalized()
    right = fwd.cross(up).normalized()
    up_vec = right.cross(fwd)
    
    spin_axis = ti.Vector([ti.sin(t*0.5)*0.2, 1.0, ti.cos(t*0.5)*0.2]).normalized()
    
    for x, y in display_buffer:
        uv = (ti.Vector([x, y]) / RES) * 2.0 - 1.0
        uv.x *= ASPECT
        rd = (fwd + uv.x * right * 0.9 + uv.y * up_vec * 0.9).normalized()
        col = color_map[0] 
        
        # Raymarch
        oc = cam_pos
        b = oc.dot(rd)
        c = oc.dot(oc) - 25.0 # Radius 5 sphere (Larger)
        h = b*b - c
        
        if h > 0.0:
            t_min = -b - ti.sqrt(h)
            dist = max(0.0, t_min)
            step_size = 0.05 
            transmittance = 1.0
            
            for i in range(80): 
                p = cam_pos + rd * dist
                if p.norm() > 5.0: break 
                
                # Volumetric Mapping
                h_norm = p.y * 0.25 + 0.5 
                r_norm = ti.Vector([p.x, p.z]).norm() * 0.25 
                
                # Sim coordinates
                u_c = int(clamp(r_norm, 0.0, 1.0) * (SIM_RES[0]-1))
                v_c = int(clamp(h_norm, 0.0, 1.0) * (SIM_RES[1]-1))
                
                d = density[u_c, v_c]
                
                # Composite
                sample = ti.Vector([0.0, 0.0, 0.0])
                alpha = 0.0
                
                if d > 0.5:
                    # Star Core (Opaque)
                    sample = mix(color_map[2], color_map[4], smoothstep(10.0, 40.0, d))
                    if r_norm > 0.15: # Accretion Ring
                        sample = mix(sample, color_map[1], 0.7)
                        
                    alpha = min(1.0, d * 0.2) 
                
                # Jet Glow
                p_n = p.normalized()
                jet = ti.pow(ti.abs(p_n.dot(spin_axis)), 10.0)
                if jet > 0.1:
                    sample += color_map[5] * jet * 2.0
                    alpha += jet * 0.1
                
                col += sample * transmittance * alpha
                transmittance *= (1.0 - alpha)
                if transmittance < 0.01: break
                dist += step_size
                
        display_buffer[x, y] = col

@ti.kernel
def trigger_starquake():
    cx = SIM_RES[0] // 2
    cy = SIM_RES[1] // 2
    for i, j in density:
        if (i-cx)**2 + (j-cy)**2 < 800:
            density[i, j] += 60.0

init_sim()
gui = ti.GUI("APH Neutron Star: Calibrated", RES)

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.SPACE:
            trigger_starquake()
            
    for _ in range(4): update_hydrodynamics(0.05)
    update_profile_binned() # Using the new binned profile
    time[None] += 0.05
    
    render_star(time[None])
    gui.set_image(display_buffer)
    
    rho = telemetry_max_density[None]
    beta = telemetry_max_stiffness[None]
    tau = telemetry_torsion[None]
    
    gui.text(f"Core Density: {rho:.1f}", pos=(0.05, 0.95), font_size=18, color=0xFFFFFF)
    gui.text(f"Stiffness: {beta:.1f} GPa", pos=(0.05, 0.92), font_size=18, color=0x88AAFF)
    
    # Draw Graph
    gui.text("Radial Density Profile", pos=(0.05, 0.25), font_size=16, color=0xAAAAAA)
    points = []
    norm = max(1.0, rho)
    for i in range(128):
        val = radial_profile[i]
        h = 0.05 + (val / norm) * 0.2
        w = 0.05 + (i / 128.0) * 0.3
        points.append([w, h])
    
    import numpy as np
    pts = np.array(points)
    if len(pts) > 1:
        gui.lines(begin=pts[:-1], end=pts[1:], radius=2, color=0xFF0000)
    
    gui.show()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        