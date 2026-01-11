import taichi as ti
import math

try:
    ti.init(arch=ti.vulkan, device_memory_GB=4.0, offline_cache=True)
except:
    ti.init(arch=ti.cpu)

# --- CONFIG ---
RES = (1280, 720)
ASPECT = RES[0] / RES[1]
MAX_STEPS = 400
STEP_SIZE = 0.05

# --- FIELDS ---
display_buffer = ti.Vector.field(3, dtype=float, shape=RES)
disk_noise = ti.field(dtype=float, shape=(512, 512)) 

# Interactive Physics
bh_spin = ti.field(dtype=float, shape=())       
inclination = ti.field(dtype=float, shape=())   
time = ti.field(dtype=float, shape=())
visual_mode = ti.field(dtype=int, shape=())     

# --- HELPERS ---
@ti.func
def mix(x, y, a): return x * (1.0 - a) + y * a
@ti.func
def clamp(x, mi, ma): return min(max(x, mi), ma)
@ti.func
def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
@ti.func
def sign(x): return 1.0 if x >= 0 else -1.0

@ti.kernel
def init_sim():
    bh_spin[None] = 0.6        
    inclination[None] = 0.2    
    visual_mode[None] = 0
    
    # Generate High-Quality Noise
    for i, j in disk_noise:
        u = i / 512.0
        v = j / 512.0
        val = 0.0
        scale = 1.0
        # FBM Octaves
        for k in range(4):
            uv = ti.Vector([u, v]) * (10.0 * scale)
            val += (ti.sin(uv.x) * ti.cos(uv.y)) / scale
            scale *= 2.0
        disk_noise[i, j] = val * 0.5 + 0.5

@ti.func
def sample_disk_density(pos, r, t, isco):
    # Volumetric Density Profile: rho(r, z)
    # 1. Radial falloff (Power law)
    radial_dens = 100.0 / (r * r + 1.0)
    
    # 2. Vertical Structure (Gaussian Scale Height)
    # H(r) ~ 0.1 * r (Flared Disk)
    scale_height = 0.05 * r + 0.1
    z_profile = ti.exp(-(pos.y * pos.y) / (2.0 * scale_height * scale_height))
    
    # 3. Turbulence (Texture)
    # Differential rotation
    rot_speed = 8.0 / (r * ti.sqrt(r) + 0.1)
    angle = ti.atan2(pos.z, pos.x)
    phi = angle + t * rot_speed
    
    tex = disk_noise[int((r/25.0)*512)%512, int((phi/6.28)*512)%512]
    
    # Sharp cut at ISCO
    mask = smoothstep(isco, isco + 0.5, r)
    
    return radial_dens * z_profile * tex * mask

@ti.func
def blackbody_color(temp):
    # Approximation of Planckian Locus
    # Temp is normalized 0..1
    
    # Cold (Red) -> Hot (Blue)
    r = clamp(temp * 3.0, 0.0, 1.0)
    g = clamp(temp * 2.0, 0.0, 1.0)
    b = clamp(temp * 5.0 - 2.0, 0.0, 1.0)
    
    # Add intensity punch
    return ti.Vector([r, g, b]) + temp * 0.5

@ti.func
def calculate_isco(a):
    # Exact Bardeen-Petterson-Teukolsky ISCO
    Z1 = 1.0 + ti.pow(1.0 - a*a, 1.0/3.0) * (ti.pow(1.0 + a, 1.0/3.0) + ti.pow(1.0 - a, 1.0/3.0))
    Z2 = ti.sqrt(3.0 * a*a + Z1*Z1)
    term_root = ti.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    # ISCO radius
    r_isco = 3.0 + Z2 - sign(a) * term_root
    return r_isco

@ti.func
def get_volumetric_light(p, r, rd, spin, isco):
    # FIXED: No early returns allowed in Taichi functions called from kernels
    
    col = ti.Vector([0.0, 0.0, 0.0])
    opacity = 0.0
    
    # 1. DENSITY SAMPLE
    density = sample_disk_density(p, r, time[None], isco)
    
    # Only compute physics if density exists
    if density >= 0.01:
        # 2. RELATIVISTIC BEAMING
        omega = 1.0 / (ti.pow(r, 1.5) + spin) 
        vel_mag = r * omega
        if r < isco: vel_mag = 0.8 
        
        vel_vec = ti.Vector([-p.z, 0.0, p.x]).normalized()
        
        beta = vel_mag
        cos_theta = vel_vec.dot(rd)
        gamma = 1.0 / ti.sqrt(max(0.01, 1.0 - beta*beta))
        doppler = 1.0 / (gamma * (1.0 - beta * cos_theta))
        
        flux = density * ti.pow(doppler, 4.0)
        
        # 3. THERMODYNAMICS
        temp = ti.pow(isco / r, 0.75)
        
        col = blackbody_color(temp) * flux
        opacity = clamp(density * 0.5, 0.0, 1.0)
    
    return col, opacity

@ti.kernel
def render(t: float):
    spin = bh_spin[None]
    inc = inclination[None]
    isco = calculate_isco(spin)
    
    # Horizon Radius r+
    rh = 1.0 + ti.sqrt(max(0.0, 1.0 - spin*spin))
    
    # Camera
    cam_dist = 16.0
    cam_y = cam_dist * ti.sin(inc)
    cam_z = -cam_dist * ti.cos(inc)
    cam_pos = ti.Vector([0.0, cam_y, cam_z])
    
    fwd = (ti.Vector([0.0,0.0,0.0]) - cam_pos).normalized()
    up = ti.Vector([0.0, 1.0, 0.0])
    right = fwd.cross(up).normalized()
    up = right.cross(fwd)
    
    for x, y in display_buffer:
        uv = (ti.Vector([x, y]) / RES) * 2.0 - 1.0
        uv.x *= ASPECT
        
        rd = (fwd + uv.x * right * 0.8 + uv.y * up * 0.8).normalized()
        p = cam_pos
        
        acc_color = ti.Vector([0.0, 0.0, 0.0])
        transmittance = 1.0
        
        # RAYMARCH
        for i in range(MAX_STEPS):
            r2 = p.dot(p)
            r = ti.sqrt(r2)
            
            # Horizon Check
            if r < rh:
                transmittance = 0.0
                break
            
            # --- VOLUMETRIC INTEGRATION ---
            # Check bounding box of disk (Flat-ish cylinder)
            if ti.abs(p.y) < 3.0 and r < 25.0 and r > 1.0:
                # Sample Physics
                v_col, v_op = get_volumetric_light(p, r, rd, spin, isco)
                
                if v_op > 0.001:
                    # Alpha Blending
                    # Light += Color * Transmittance * Opacity
                    acc_color += v_col * transmittance * 0.2 # Step weight
                    transmittance *= (1.0 - v_op * 0.2)
                    
                    if transmittance < 0.01: break
            
            # --- GEODESIC STEP ---
            # F ~ -M/r^2
            g_dir = -p.normalized()
            bend = 1.5 / r2
            
            # Frame Dragging
            # d_vec = (-z, 0, x)
            d_vec = ti.Vector([-p.z, 0.0, p.x]).normalized()
            drag = (spin * 2.0) / (r2 * r)
            
            curvature = g_dir * bend + d_vec * drag
            rd = (rd + curvature * STEP_SIZE).normalized()
            
            p += rd * STEP_SIZE
            if r > 30.0: break
            
        # BACKGROUND STARS
        if transmittance > 0.0:
            # Map direction to starfield
            # High frequency noise for stars
            dir_u = ti.atan2(rd.z, rd.x) * 10.0
            dir_v = rd.y * 20.0
            star = ti.sin(dir_u) * ti.sin(dir_v)
            star = ti.pow(max(0.0, star), 20.0) # Sharpen
            
            acc_color += ti.Vector([1.0, 1.0, 1.0]) * star * transmittance
            
        display_buffer[x, y] = acc_color

init_sim()
window = ti.ui.Window("APH Kerr Lab: Scientific", RES)
canvas = window.get_canvas()
gui = window.get_gui()

while window.running:
    if window.is_pressed(ti.ui.LEFT): bh_spin[None] -= 0.01
    if window.is_pressed(ti.ui.RIGHT): bh_spin[None] += 0.01
    if window.is_pressed(ti.ui.UP): inclination[None] -= 0.01
    if window.is_pressed(ti.ui.DOWN): inclination[None] += 0.01
    
    # FIXED: Use Python min/max here, not Taichi clamp
    bh_spin[None] = min(max(bh_spin[None], -0.99), 0.99)
    inclination[None] = min(max(inclination[None], -1.5), 1.5)
    
    time[None] += 0.05
    render(time[None])
    canvas.set_image(display_buffer)
    
    with gui.sub_window("Scientific Controls", 0.05, 0.05, 0.3, 0.2):
        s = bh_spin[None]
        # Approximate display calc
        isco_approx = 6.0
        if abs(s) > 0.01:
             isco_approx = 2.0 if s > 0 else 9.0
        
        gui.text(f"Spin (a): {s:.3f}")
        gui.text(f"Inclination: {inclination[None]:.2f}")
        
        if s > 0: gui.text(f"Mode: Prograde (ISCO Shrinks)")
        else: gui.text(f"Mode: Retrograde (ISCO Expands)")
        
    window.show()