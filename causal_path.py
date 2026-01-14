import taichi as ti
import math
import random
import time as std_time

# --- INIT ---
try:
    ti.init(arch=ti.vulkan, device_memory_GB=4.0, offline_cache=True)
except:
    try:
        ti.init(arch=ti.opengl, device_memory_GB=3.5, offline_cache=True)
    except:
        ti.init(arch=ti.cpu)

# --- CONFIG ---
RES = (960, 540) # Optimized resolution
ASPECT = RES[0] / RES[1]
MAX_STARS = 256 
NUM_BIOMES = 11 # 0-9 Normal, 10 Black Hole

# --- PHYSICS ---
MOMENTUM_DAMPING = 0.94 
SYNERGEIA_SLOPE = 0.015 
FOV_SCALE = 0.60  

# --- FIELDS ---
color_buffer = ti.Vector.field(4, dtype=float, shape=RES) 
reflect_buffer = ti.Vector.field(4, dtype=float, shape=RES) 
hazard_buffer = ti.Vector.field(3, dtype=float, shape=RES)
momentum = ti.Vector.field(3, dtype=float, shape=RES) 
relaxed_buffer_A = ti.Vector.field(3, dtype=float, shape=RES)
relaxed_buffer_B = ti.Vector.field(3, dtype=float, shape=RES)
history_buffer = ti.Vector.field(3, dtype=float, shape=RES)
display_buffer = ti.Vector.field(3, dtype=float, shape=RES)
frame_parity = ti.field(dtype=int, shape=())

# Black Hole Fields
disk_noise = ti.field(dtype=float, shape=(512, 512))
bh_spin = ti.field(dtype=float, shape=())       
inclination = ti.field(dtype=float, shape=())   

# Flight State
camera_pos = ti.Vector.field(3, dtype=float, shape=())
camera_vel = ti.Vector.field(3, dtype=float, shape=())
camera_rot = ti.Vector.field(2, dtype=float, shape=()) 
camera_dir = ti.Vector.field(3, dtype=float, shape=()) 
active_stars = ti.field(dtype=int, shape=())

# Particles
star_pos = ti.Vector.field(3, dtype=float, shape=MAX_STARS)
star_vel = ti.Vector.field(3, dtype=float, shape=MAX_STARS)
star_props = ti.Vector.field(3, dtype=float, shape=MAX_STARS)

# Universe State
global_env = ti.Vector.field(3, dtype=float, shape=())
seeds = ti.Vector.field(3, dtype=float, shape=()) 
frame_counter = ti.field(dtype=int, shape=())
palette = ti.Vector.field(3, dtype=float, shape=8) 
genes = ti.field(dtype=float, shape=8) 
biome_id = ti.field(dtype=int, shape=()) 

# --- MATH ---
@ti.func
def fract(x): return x - ti.floor(x)
@ti.func
def clamp(x, mi, ma): return min(max(x, mi), ma)
@ti.func
def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
@ti.func
def mix(x, y, a): return x * (1.0 - a) + y * a
@ti.func
def rotate(v, a):
    c, s = ti.cos(a), ti.sin(a)
    return ti.Vector([c * v.x - s * v.y, s * v.x + c * v.y])
@ti.func
def sign(x): return 1.0 if x >= 0 else -1.0

@ti.func
def hash22(p):
    p3 = fract(ti.Vector([p.x, p.y, p.x]) * ti.Vector([0.1031, 0.1030, 0.0973]))
    p3 += p3.dot(p3.zxy + 33.33)
    return fract((p3.xx + p3.yz) * p3.zy)

@ti.func
def noise2(p):
    i = ti.floor(p); f = fract(p); u = f * f * (3.0 - 2.0 * f)
    a = hash22(i); b = hash22(i + ti.Vector([1.0, 0.0]))
    c = hash22(i + ti.Vector([0.0, 1.0])); d = hash22(i + ti.Vector([1.0, 1.0]))
    return mix(mix(a.x, b.x, u.x), mix(c.x, d.x, u.x), u.y)

@ti.func
def noise3(p):
    i = ti.floor(p); f = fract(p); u = f * f * (3.0 - 2.0 * f)
    return mix(noise2(p.xy), noise2(p.yz), u.z)

@ti.func
def sample_rayleigh(x, slope):
    return slope * x * ti.exp(-slope * x * x * 0.5)

@ti.func
def sd_cylinder(p, h, r):
    d_x = p.xz.norm() - r
    d_y = abs(p.y) - h
    return min(max(d_x, d_y), 0.0) + ti.Vector([max(d_x, 0.0), max(d_y, 0.0)]).norm()

@ti.func
def sd_sphere(p, r): return p.norm() - r

# --- BH LOGIC ---
@ti.kernel
def init_bh_noise():
    bh_spin[None] = 0.6
    inclination[None] = 0.2
    for i, j in disk_noise:
        u = i / 512.0
        v = j / 512.0
        val = 0.0
        scale = 1.0
        for k in range(4):
            uv = ti.Vector([u, v]) * (10.0 * scale)
            val += (ti.sin(uv.x) * ti.cos(uv.y)) / scale
            scale *= 2.0
        disk_noise[i, j] = val * 0.5 + 0.5

@ti.func
def sample_disk_density(pos, r, t, isco):
    radial_dens = 100.0 / (r * r + 1.0)
    scale_height = 0.05 * r + 0.1
    z_profile = ti.exp(-(pos.y * pos.y) / (2.0 * scale_height * scale_height))
    rot_speed = 8.0 / (r * ti.sqrt(r) + 0.1)
    angle = ti.atan2(pos.z, pos.x)
    phi = angle + t * rot_speed
    tex = disk_noise[int((r/25.0)*512)%512, int((phi/6.28)*512)%512]
    mask = smoothstep(isco, isco + 0.5, r)
    return radial_dens * z_profile * tex * mask

@ti.func
def blackbody_color(temp):
    r = clamp(temp * 3.0, 0.0, 1.0)
    g = clamp(temp * 2.0, 0.0, 1.0)
    b = clamp(temp * 5.0 - 2.0, 0.0, 1.0)
    return ti.Vector([r, g, b]) + temp * 0.5

@ti.func
def calculate_isco(a):
    Z1 = 1.0 + ti.pow(1.0 - a*a, 1.0/3.0) * (ti.pow(1.0 + a, 1.0/3.0) + ti.pow(1.0 - a, 1.0/3.0))
    Z2 = ti.sqrt(3.0 * a*a + Z1*Z1)
    term_root = ti.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    r_isco = 3.0 + Z2 - sign(a) * term_root
    return r_isco

@ti.func
def get_volumetric_light(p, r, rd, spin, isco, time):
    col = ti.Vector([0.0, 0.0, 0.0])
    opacity = 0.0
    density = sample_disk_density(p, r, time, isco)
    if density >= 0.01:
        omega = 1.0 / (ti.pow(r, 1.5) + spin) 
        vel_mag = r * omega
        if r < isco: vel_mag = 0.8 
        vel_vec = ti.Vector([-p.z, 0.0, p.x]).normalized()
        beta = vel_mag
        cos_theta = vel_vec.dot(rd)
        gamma = 1.0 / ti.sqrt(max(0.01, 1.0 - beta*beta))
        doppler = 1.0 / (gamma * (1.0 - beta * cos_theta))
        flux = density * ti.pow(doppler, 4.0)
        temp = ti.pow(isco / r, 0.75)
        col = blackbody_color(temp) * flux
        opacity = clamp(density * 0.5, 0.0, 1.0)
    return col, opacity

@ti.func
def render_bh_background(ro, rd, time):
    spin = bh_spin[None]
    isco = calculate_isco(spin)
    rh = 1.0 + ti.sqrt(max(0.0, 1.0 - spin*spin))
    
    p = ro
    acc_color = ti.Vector([0.0, 0.0, 0.0])
    transmittance = 1.0
    
    MAX_STEPS = 50 # Reduced for real-time integration
    STEP_SIZE = 0.2
    
    hit_horizon = False

    for i in range(MAX_STEPS):
        r2 = p.dot(p)
        r = ti.sqrt(r2)
        
        if r < rh:
            hit_horizon = True
            transmittance = 0.0
            break
        
        # Volumetric Integration
        if ti.abs(p.y) < 3.0 and r < 25.0 and r > 1.0:
            v_col, v_op = get_volumetric_light(p, r, rd, spin, isco, time)
            if v_op > 0.001:
                acc_color += v_col * transmittance * 0.2
                transmittance *= (1.0 - v_op * 0.2)
                if transmittance < 0.01: break
        
        # Geodesic Step (Approximated for perf)
        g_dir = -p.normalized()
        bend = 1.5 / r2
        d_vec = ti.Vector([-p.z, 0.0, p.x]).normalized()
        drag = (spin * 2.0) / (r2 * r)
        
        curvature = g_dir * bend + d_vec * drag
        rd = (rd + curvature * STEP_SIZE).normalized()
        p += rd * STEP_SIZE
        if r > 40.0: break
        
    if transmittance > 0.0:
        dir_u = ti.atan2(rd.z, rd.x) * 10.0
        dir_v = rd.y * 20.0
        star = ti.sin(dir_u) * ti.sin(dir_v)
        star = ti.pow(max(0.0, star), 20.0) 
        acc_color += ti.Vector([1.0, 1.0, 1.0]) * star * transmittance
    elif hit_horizon:
         acc_color = ti.Vector([0.0, 0.0, 0.0])

    return acc_color

# --- SDF SCENE ---
@ti.func
def get_attractor_positions(time):
    # NOTE: These must match the CPU side logic in the main loop!
    fractal_pos = ti.Vector([0.0, 1.0, 0.0])
    # Egg (Sphere) -> Past
    egg_pos = ti.Vector([3.5 * ti.sin(time*0.2), 2.5 + ti.cos(time*0.15), 3.5 * ti.cos(time*0.25)])
    # Clock -> Future
    clock_pos = ti.Vector([4.0 * ti.sin(-time*0.15 + 3.14), 2.0 + ti.sin(time*0.1), 4.0 * ti.cos(-time*0.15 + 3.14)])
    return fractal_pos, egg_pos, clock_pos

@ti.func
def sd_melting_clock(p, time):
    bend = 0.2; p.y += bend * p.x * p.x
    p.xz = rotate(p.xz, p.y * 0.2 + time * 0.1)
    return sd_cylinder(p, 0.05, 1.2) * 0.8 

@ti.func
def sd_fractal(pos, time):
    melt_strength = genes[0]; base_power = genes[1]; z = pos
    melt_offset = ti.sin(z.y * 1.5 - time * 0.3) * melt_strength
    z.x += melt_offset; z.z += melt_offset * 0.5
    dr = 1.0; r = 0.0; trap = 1e10
    power = base_power + ti.sin(time * 0.2) * 0.5
    for i in range(5):
        r = z.norm()
        if r > 4.0: break
        trap = min(trap, ti.abs(z.y) + 0.1/r) 
        theta = ti.acos(z.z / r); phi = ti.atan2(z.y, z.x)
        dr =  ti.pow(r, power - 1.0) * power * dr + 1.0
        zr = ti.pow(r, power)
        z = zr * ti.Vector([ti.sin(theta)*power*ti.cos(phi*power), ti.sin(phi*power)*ti.sin(theta*power), ti.cos(theta*power)])
        z += pos
        rot_xz = rotate(z.xz, time * 0.05 + z.y * 0.2)
        z.x = rot_xz.x; z.z = rot_xz.y
    return 0.5 * ti.log(r) * r / dr, trap

@ti.func
def get_terrain_height(xz):
    b_id = biome_id[None]
    h = -1.0
    
    # Biome-specific terrain logic
    if b_id == 0: # Dali Desert 
        checkers = smoothstep(0.4, 0.6, fract(xz.x*0.5 + 0.2*ti.sin(xz.y*0.2)) - fract(xz.y*0.5))
        h += noise2(xz * 0.05) * 1.5 + checkers * 0.1
    elif b_id == 1: # Mars 
        h += noise2(xz * 0.1) * 3.0 + noise2(xz * 0.5) * 0.5
    elif b_id == 2: # Ice 
        h += noise2(xz * 0.03) * 2.0 + max(0.0, noise2(xz * 0.2) - 0.6) * 5.0
    elif b_id == 3: # Toxic 
        h += noise2(xz * 0.1 + global_env[None].x * 0.1) * 1.0
    elif b_id == 4: # Cyber 
        grid = smoothstep(0.9, 0.95, abs(ti.sin(xz.x)) * abs(ti.sin(xz.y)))
        h += grid * 0.5
    elif b_id == 10: # Black Hole - No terrain
        h = -100.0
    else: # Default 
        h += noise2(xz * 0.08) * 4.0
        
    return h

@ti.func
def map_scene(p, time):
    fp, ep, cp = get_attractor_positions(time)
    d_obj, trap = sd_fractal(p - fp, time)
    d_egg = (p - ep).norm() - 0.9
    rel_clock = p - cp; rel_clock.yz = rotate(rel_clock.yz, 0.5); rel_clock.xz = rotate(rel_clock.xz, time * 0.15)
    d_clock = sd_melting_clock(rel_clock, time)
    d_scene = min(d_obj, min(d_egg, d_clock))
    final_trap = trap
    if d_egg < d_obj and d_egg < d_clock: final_trap = 1000.0 
    if d_clock < d_obj and d_clock < d_egg: final_trap = 2000.0 
    return d_scene, final_trap

@ti.func
def map_scene_render(p, time):
    d_scene, trap = map_scene(p, time)
    d_stars = 100.0; star_idx = 0
    count = active_stars[None]
    for i in range(count):
        d_s = sd_sphere(p - star_pos[i], 0.04) 
        if d_s < d_stars: d_stars = d_s; star_idx = i
    d_final = min(d_scene, d_stars)
    final_trap = trap
    if d_stars < d_scene: final_trap = 3000.0 + float(star_idx)
    return d_final, final_trap

@ti.func
def get_normal(p, time, id):
    e = 0.005; n = ti.Vector([0.0, 1.0, 0.0])
    if id == 1: 
        d0, _ = map_scene_render(p, time)
        n = ti.Vector([
            map_scene_render(p+ti.Vector([e,0,0]), time)[0]-d0,
            map_scene_render(p+ti.Vector([0,e,0]), time)[0]-d0,
            map_scene_render(p+ti.Vector([0,0,e]), time)[0]-d0
        ]).normalized()
    elif id == 2:
        h0 = get_terrain_height(p.xz)
        n = ti.Vector([
            h0 - get_terrain_height(p.xz + ti.Vector([0.1, 0])), 0.1,
            h0 - get_terrain_height(p.xz + ti.Vector([0, 0.1]))
        ]).normalized()
    return n

@ti.func
def get_gradient_physics(p, time):
    e = 0.005; d0, _ = map_scene(p, time)
    dx, _ = map_scene(p + ti.Vector([e, 0, 0]), time)
    dy, _ = map_scene(p + ti.Vector([0, e, 0]), time)
    dz, _ = map_scene(p + ti.Vector([0, 0, e]), time)
    return ti.Vector([dx-d0, dy-d0, dz-d0]).normalized()

@ti.func
def get_sky_background(rd, sun_dir):
    sky_t = smoothstep(-0.1, 0.3, rd.y)
    col = mix(palette[1], palette[0], sky_t)
    sun_dot = max(0.0, rd.dot(sun_dir))
    col += palette[6] * ti.pow(sun_dot, 80.0) * 2.0 
    
    # Biome Sky Mods
    b_id = biome_id[None]
    if b_id == 1: col += ti.Vector([0.2, 0.05, 0.0]) * 0.5 
    elif b_id == 2: col += ti.Vector([0.0, 0.1, 0.2]) * 0.5 
    elif b_id == 4: col *= 0.5; col.z += 0.2 
    
    # Atmosphere Density / Space Tunnel
    cam_y = camera_pos[None].y
    # Fade starts at 20, completely black space by 70
    atmosphere = clamp(1.0 - (cam_y - 20.0) / 50.0, 0.0, 1.0) 
    col *= atmosphere
    
    # Reveal stars as atmosphere thins
    if atmosphere < 0.9:
        dir_u = ti.atan2(rd.z, rd.x) * 10.0
        dir_v = rd.y * 20.0
        star = ti.sin(dir_u) * ti.sin(dir_v)
        star = ti.pow(max(0.0, star), 30.0)
        col += ti.Vector([1.0, 1.0, 1.0]) * star * (1.0 - atmosphere)
    
    return col

@ti.func
def get_material(id, p, n, trap, time):
    albedo = ti.Vector([0.0, 0.0, 0.0])
    rough = 0.5; metal = 0.0; emit = ti.Vector([0.0, 0.0, 0.0])
    
    if id == 1: # Objects / Stars
        if trap >= 3000.0: # Stars
            s_idx = int(trap - 3000.0)
            c_id = int(star_props[s_idx].z)
            col = ti.Vector([1.0, 1.0, 1.0])
            if c_id == 0: col = ti.Vector([1.0, 0.1, 0.1]) 
            elif c_id == 1: col = ti.Vector([0.1, 1.0, 0.1]) 
            elif c_id == 2: col = ti.Vector([0.2, 0.5, 1.0]) 
            elif c_id == 3: col = ti.Vector([1.0, 0.8, 0.2]) 
            elif c_id == 4: col = ti.Vector([1.0, 0.2, 0.8]) 
            emit = col * 30.0; rough = 0.0
        elif trap > 1500.0: # Clock (Future)
            _, _, clock_p = get_attractor_positions(time)
            local_p = p - clock_p
            local_p.yz = rotate(local_p.yz, 0.5); local_p.xz = rotate(local_p.xz, time * 0.15)
            dist = local_p.xz.norm(); angle = ti.atan2(local_p.z, local_p.x)
            albedo = ti.Vector([0.9, 0.9, 0.85]); rough = 0.3
            if dist > 1.1: 
                albedo = ti.Vector([0.8, 0.6, 0.2]); metal = 1.0; rough = 0.1
            else:
                marks = smoothstep(0.8, 0.9, ti.abs(ti.sin(angle * 6.0)))
                if dist > 0.9 and marks > 0.5: albedo = ti.Vector([0.1, 0.1, 0.1])
                m_ang = -(time * 2.0); m_dir = ti.Vector([ti.cos(m_ang), ti.sin(m_ang)])
                if local_p.xz.dot(m_dir) > dist * 0.95 and dist < 0.8: albedo = ti.Vector([0.0, 0.0, 0.0]) 
        elif trap > 500.0: # Egg (Past)
            albedo = ti.Vector([0.95, 0.95, 0.92]); rough = 0.1
            if noise3(p * 5.0) > 0.65: albedo = palette[7] 
        else: 
            if trap < 0.1: emit = palette[7] * (1.0/(trap+0.01)) * 0.05
    elif id == 2: # Terrain
        b_id = biome_id[None]
        grain = noise2(p.xz * 8.0) * 0.1 # High frequency texture detail
        
        if b_id == 0: # Dali
            check = smoothstep(0.4, 0.6, fract(p.x * 0.3 + 0.2 * ti.sin(p.z*0.1)) - fract(p.z * 0.3))
            dist_fade = smoothstep(20.0, 0.0, p.norm())
            base_col = mix(palette[2], mix(palette[2], palette[3], abs(check)), dist_fade)
            albedo = base_col + ti.Vector([grain, grain, grain])
            rough = 0.8 + grain # Sand-like roughness
            
        elif b_id == 1: # Mars
            base_col = ti.Vector([0.8, 0.4, 0.2]) * (0.5 + 0.5 * noise2(p.xz*0.5))
            albedo = base_col + ti.Vector([grain, grain, grain]) * 0.5
            rough = 0.7
            
        elif b_id == 2: # Ice
            albedo = ti.Vector([0.9, 0.95, 1.0])
            rough = 0.1 + noise2(p.xz)*0.2; metal = 0.5 # Glossy ice variation
            
        elif b_id == 4: # Cyber
            grid = smoothstep(0.95, 0.98, abs(ti.sin(p.x)) * abs(ti.sin(p.z)))
            albedo = ti.Vector([0.1, 0.1, 0.1]); emit = ti.Vector([0.0, 1.0, 0.5]) * grid
            metal = 0.8; rough = 0.3
        else:
            albedo = palette[2] + grain
            rough = 0.9
            
    elif id == 3: # Liquid
        albedo = palette[7]; rough = 0.04; metal = 0.9 # High reflectivity
        
    return albedo, rough, metal, emit

@ti.func
def calc_softshadow(ro, rd, tmin, tmax, k, dither, time):
    res = 1.0; t = tmin + dither * 0.05
    for i in range(12): 
        d1, _ = map_scene_render(ro + rd * t, time)
        if d1 < 0.001: res = 0.0; break
        res = min(res, k * d1 / t)
        t += d1; 
        if t > tmax: break
    return res

@ti.kernel
def update_physics(dt: float, time: float):
    frac_p, egg_p, clock_p = get_attractor_positions(time)
    scene_center = ti.Vector([0.0, 2.0, 0.0])
    count = active_stars[None]
    b_id = biome_id[None]
    
    for i in range(count):
        p = star_pos[i]; v = star_vel[i]
        force = ti.Vector([0.0, 0.0, 0.0])
        
        # Calculate N-Body Force Unconditionally
        # (Hoisting this loop out of the if/else fixes the Taichi IRVerifier error)
        nbody_force = ti.Vector([0.0, 0.0, 0.0])
        for j in range(count):
            if i != j:
                diff_b = p - star_pos[j]; dist_sq = diff_b.norm_sqr()
                if dist_sq < 0.8: nbody_force += diff_b.normalized() * (2.0 / (dist_sq + 0.1))

        if b_id == 10: # Black Hole Physics
             # Simple Newtonian Gravity Point Mass
             r_sq = p.norm_sqr()
             dir = -p.normalized()
             # F = G * M / r^2
             force += dir * (100.0 / (r_sq + 0.1))
             # Tangential Damping to simulate drag
             v *= 0.999
        else:
            # Normal Physics
            diff1 = frac_p - p; force += diff1.normalized() * (12.0 / (diff1.norm_sqr() + 0.2))
            diff2 = egg_p - p; force += diff2.normalized() * (10.0 / (diff2.norm_sqr() + 0.2))
            diff3 = clock_p - p; force += diff3.normalized() * (10.0 / (diff3.norm_sqr() + 0.2))
            d_org = (p - scene_center).norm()
            if d_org > 8.0: force += -(p - scene_center).normalized() * (d_org - 8.0) * 4.0
            
            # Apply N-body force here
            force += nbody_force
                    
        v += force * dt; v *= 0.995 
        p += v * dt
        
        # --- TERRAIN COLLISION ---
        if b_id != 10:
            if p.y < -3.5: p.y = -3.5; v.y *= -0.8
            terr_h = get_terrain_height(p.xz)
            if p.y < terr_h + 0.05: # Simple radius check
                t_normal = get_normal(p, time, 2)
                if v.dot(t_normal) < 0.0:
                    v = v - 1.5 * v.dot(t_normal) * t_normal # Bounce with restitution
                p.y = max(p.y, terr_h + 0.05)
            
            # Object collision (Only in normal biome)
            dist, _ = map_scene(p, time)
            if dist < 0.1:
                normal = get_gradient_physics(p, time)
                if v.dot(normal) < 0.0: v = (v - 1.8 * v.dot(normal) * normal) + normal * 0.5 
                p += normal * (0.12 - dist) 
        else:
             if p.norm() < 1.0: # Event Horizon for stars
                 # Reset star
                 p = ti.Vector([10.0, 0.0, 0.0]) + ti.Vector([ti.random(), ti.random(), ti.random()])
                 v = ti.Vector([0.0, 0.0, 1.0])
        # -------------------------
        
        star_pos[i] = p; star_vel[i] = v

@ti.kernel
def check_player_status() -> int:
    status = 0
    cam = camera_pos[None]
    b_id = biome_id[None]
    
    if b_id != 10:
        # Ground/Water Check
        h = get_terrain_height(cam.xz)
        water_h = genes[3]
        
        if cam.y < h + 0.5: status = 1 # Ground Hit
        elif cam.y < water_h + 0.2: status = 2 # Water Hit
        elif cam.y > 80.0: status = 3 # Space Hit
    return status

@ti.kernel
def spawn_star(pos: ti.types.vector(3, float), vel: ti.types.vector(3, float)):
    idx = active_stars[None]
    if idx < MAX_STARS:
        star_pos[idx] = pos; star_vel[idx] = vel
        col_id = float(int(ti.random() * 5.0))
        star_props[idx] = ti.Vector([0.0, 0.0, col_id])
        active_stars[None] = idx + 1

@ti.kernel
def render_pass(time: float, frame: int):
    yaw = camera_rot[None].x; pitch = camera_rot[None].y
    fwd = ti.Vector([ti.sin(yaw)*ti.cos(pitch), ti.sin(pitch), ti.cos(yaw)*ti.cos(pitch)]).normalized()
    right = ti.Vector([0,1,0]).cross(fwd).normalized()
    up = fwd.cross(right).normalized()
    camera_dir[None] = fwd 
    cam = camera_pos[None]
    sun_angle = 1.5 + time * genes[4] * 0.05
    sun_dir = ti.Vector([ti.cos(sun_angle), ti.sin(sun_angle) * 0.4 + 0.1, ti.sin(sun_angle)]).normalized()
    jit = ti.Vector([(float(frame%2)-0.5)*0.5, (float((frame//2)%2)-0.5)*0.5])
    dither = hash22(ti.Vector([float(frame%1024), 0.0])).x
    
    b_id = biome_id[None]

    for u, v in color_buffer:
        uv = (ti.Vector([u + jit.x, v + jit.y]) / ti.Vector([RES[0], RES[1]]) * 2.0 - 1.0) * ti.Vector([ASPECT, 1.0])
        rd = (fwd + uv.x * right * FOV_SCALE + uv.y * up * FOV_SCALE).normalized()
        t = 0.0; t_final = 10000.0; hit_id = 0; trap_final = 0.0
        
        # 1. Raymarch Objects (Stars/Props) - Always visible
        for i in range(48): 
            p = cam + rd * t
            d, trap = map_scene_render(p, time)
            if d < 0.002: t_final = t; hit_id = 1; trap_final = trap; break
            t += d
            if t > 70.0: break
        
        # 2. Raymarch Terrain (If not BH)
        if b_id != 10:
            if (hit_id == 0 or t > 70.0) and rd.y < 0.2:
                t = max(0.0, (4.0 - cam.y)/rd.y if rd.y < 0 else 0)
                for i in range(64):
                    p = cam + rd * t
                    h = get_terrain_height(p.xz); d = p.y - h
                    if d < 0.01 * t: 
                        if t < t_final: t_final = t; hit_id = 2; break
                    t += d * 0.6 
                    if t > 200.0: break
            
            water_h = genes[3]
            if rd.y < -0.001:
                t_w = (water_h - cam.y) / rd.y
                if t_w > 0.0 and t_w < t_final: t_final = t_w; hit_id = 3
            
        col = ti.Vector([0.0, 0.0, 0.0])
        if b_id != 10:
            col = get_sky_background(rd, sun_dir)
        else:
            # Black Hole Background Pass
            col = render_bh_background(cam, rd, time)

        refl = ti.Vector([0.0, 0.0, 0.0]); emit = ti.Vector([0.0, 0.0, 0.0]); fluid = 0.0
        
        if hit_id > 0:
            p = cam + t_final * rd
            norm = ti.Vector([0.0, 1.0, 0.0])
            if hit_id == 1: norm = get_normal(p, time, 1)
            elif hit_id == 2: norm = get_normal(p, time, 2)
            elif hit_id == 3:
                warp = momentum[u,v].x * 0.15; norm = ti.Vector([warp, 1.0, warp]).normalized()
                n_det = noise2(p.xz * 2.0 + time) * 0.05; norm = (norm + ti.Vector([n_det, 0, n_det])).normalized()
                fluid = 1.0

            alb, rough, met, em = get_material(hit_id, p, norm, trap_final, time)
            
            if hit_id == 1 and trap_final < 500.0: 
                # Transparent object logic
                ior = 0.75; ref_dir = rd 
                k = 1.0 - ior * ior * (1.0 - norm.dot(rd) * norm.dot(rd))
                if k < 0.0: ref_dir = ti.math.reflect(rd, norm)
                else: ref_dir = ti.math.refract(rd, norm, ior)
                env_ref = get_sky_background(ref_dir, sun_dir)
                if ref_dir.y < -0.01:
                    t_ground = (p.y - (-4.0)) / -ref_dir.y
                    if t_ground > 0.0:
                        p_g = p + ref_dir * t_ground
                        check = smoothstep(0.4, 0.6, fract(p_g.x*0.3 + 0.2*ti.sin(p_g.z*0.1)) - fract(p_g.z*0.3))
                        env_ref = mix(palette[2], palette[3], abs(check))
                fresnel = pow(1.0 - max(0.0, norm.dot(-rd)), 3.0)
                col = mix(env_ref * palette[4] * 1.5, get_sky_background(ti.math.reflect(rd, norm), sun_dir), fresnel)
                col += em 
            else:
                # --- Advanced Lighting & Reflections ---
                
                # 1. Main Sun Light
                NdotL = max(0.0, norm.dot(sun_dir))
                shadow = 1.0
                if hit_id != 3: # Skip shadow for water (optimized)
                    shadow = calc_softshadow(p + norm * 0.05, sun_dir, 0.1, 15.0, 12.0, dither, time)
                
                # 2. Star Lighting (Point Lights)
                star_acc = ti.Vector([0.0, 0.0, 0.0])
                # We now allow stars to light up everything except the transparent object itself
                count = active_stars[None]
                for i in range(count):
                    l_pos = star_pos[i]; to_l = l_pos - p; dist_sq = to_l.norm_sqr()
                    if dist_sq < 80.0: # Optimized light radius
                        dist = ti.sqrt(dist_sq); L = to_l / dist
                        atten = 1.0 / (1.0 + dist_sq * 0.6) # Sharper falloff
                        c_idx = int(star_props[i].z)
                        s_col = ti.Vector([1.0, 1.0, 1.0])
                        if c_idx == 0: s_col = ti.Vector([1.0, 0.1, 0.1])
                        elif c_idx == 1: s_col = ti.Vector([0.1, 1.0, 0.1])
                        elif c_idx == 2: s_col = ti.Vector([0.2, 0.5, 1.0])
                        elif c_idx == 3: s_col = ti.Vector([1.0, 0.8, 0.2])
                        elif c_idx == 4: s_col = ti.Vector([1.0, 0.2, 0.8])
                        
                        diff_factor = max(0.0, norm.dot(L))
                        star_acc += s_col * diff_factor * atten * 4.0 # Boosted star intensity
                
                # 3. Specular (Sun)
                H = (sun_dir - rd).normalized()
                spec = pow(max(0.0, norm.dot(H)), 64.0 * (1.0 - rough)) * shadow * 2.0
                
                # 4. Environment Reflection (PBR-lite)
                ref_dir = ti.math.reflect(rd, norm)
                env_ref = ti.Vector([0.0, 0.0, 0.0])
                if b_id != 10:
                    env_ref = get_sky_background(ref_dir, sun_dir)
                else:
                    env_ref = render_bh_background(p, ref_dir, time)
                    
                f0 = mix(ti.Vector([0.04, 0.04, 0.04]), alb, met) # Fresnel Base
                fresnel = f0 + (1.0 - f0) * ti.pow(1.0 - max(0.0, norm.dot(-rd)), 5.0)
                
                # 5. Combine
                ambient = 0.15 * palette[6] # Base ambient
                diffuse_light = ambient + NdotL * shadow * palette[6] + star_acc
                
                if hit_id == 3: # Water: Mix based on Fresnel
                    col = mix(alb * diffuse_light, env_ref, fresnel) + spec + em
                else: # Standard Dielectric/Metal
                    col = alb * diffuse_light + spec + em + env_ref * met * (1.0 - rough) * 0.5

                refl = env_ref * (1.0 - rough) # For fluid sim buffer
                emit = em 
            
            fog = 1.0 - ti.exp(-t_final * genes[5] * 0.01)
            col = mix(col, palette[1], fog)
            refl = mix(refl, ti.Vector([0.0,0.0,0.0]), fog)

        color_buffer[u, v] = ti.Vector([col.x, col.y, col.z, fluid])
        reflect_buffer[u, v] = ti.Vector([refl.x, refl.y, refl.z, t_final])
        hazard_buffer[u, v] = emit

@ti.kernel
def physics_wts_update(dt: float, parity: int):
    for i, j in relaxed_buffer_A:
        val = relaxed_buffer_A[i,j] if parity==0 else relaxed_buffer_B[i,j]
        if ti.random() < 0.008:
            momentum[i,j] += ti.Vector([0.0, sample_rayleigh(ti.random()*5.0, SYNERGEIA_SLOPE)*0.5, 0.0])
        mom = momentum[i,j] * MOMENTUM_DAMPING
        lap = ti.Vector([0.0,0.0,0.0])
        for x, y in ti.static([(-1,0), (1,0), (0,-1), (0,1)]):
            ix, iy = clamp(i+x, 0, RES[0]-1), clamp(j+y, 0, RES[1]-1)
            lap += relaxed_buffer_A[ix,iy] if parity==1 else relaxed_buffer_B[ix,iy]
        mom += (lap - 4.0 * val) * 4.0 * dt
        momentum[i,j] = mom
        new_val = (val + mom) * 0.99
        if parity==0: relaxed_buffer_B[i,j] = new_val
        else: relaxed_buffer_A[i,j] = new_val

@ti.kernel
def resolve(parity: int):
    for i, j in history_buffer:
        curr = color_buffer[i,j].xyz + reflect_buffer[i,j].xyz
        hist = history_buffer[i,j]
        curr = ti.min(curr, ti.Vector([5.0, 5.0, 5.0]))
        history_buffer[i,j] = mix(hist, curr, 0.85)

@ti.kernel
def composite():
    for i, j in display_buffer:
        col = history_buffer[i, j]
        bloom = ti.Vector([0.0, 0.0, 0.0])
        for x, y in ti.static([(-2,-2), (-2,2), (2,-2), (2,2), (0,0)]):
             ix, iy = clamp(i+x, 0, RES[0]-1), clamp(j+y, 0, RES[1]-1)
             bloom += hazard_buffer[ix, iy]
        col += bloom * 0.1
        col = col / (col + 1.0); col = ti.pow(col, 1.0/2.2)
        display_buffer[i, j] = col

@ti.kernel
def init_game_state(id: int):
    # Set default camera
    camera_pos[None] = ti.Vector([0.0, 4.0, -4.0])
    camera_vel[None] = ti.Vector([0.0, 0.0, 0.0]) 
    camera_rot[None] = ti.Vector([0.0, 0.0])
    active_stars[None] = 0
    biome_id[None] = id
    
    # Biome Palette Injection
    if id == 0: # Dali (Classic)
        palette[0] = [0.0,0.0,0.2]; palette[1] = [0.1,0.4,0.7]
        palette[2] = [0.8,0.6,0.3]; palette[3] = [0.05,0.1,0.3]
        palette[4] = [0.9,0.7,0.1]; palette[5] = [0.8,0.2,0.1]
        palette[6] = [1.0,0.95,0.8]; palette[7] = [0.1,0.2,0.9] # Sun, Liquid
    elif id == 1: # Mars
        palette[0] = [0.2,0.05,0.0]; palette[1] = [0.8,0.4,0.1]
        palette[2] = [0.6,0.3,0.1]; palette[3] = [0.4,0.2,0.1]
        palette[4] = [1.0,0.8,0.5]; palette[5] = [0.8,0.2,0.0]
        palette[6] = [1.0,0.8,0.6]; palette[7] = [0.8,0.2,0.1]
    elif id == 2: # Ice
        palette[0] = [0.0,0.1,0.3]; palette[1] = [0.8,0.9,1.0]
        palette[2] = [0.9,0.95,1.0]; palette[3] = [0.5,0.7,0.9]
        palette[4] = [0.0,1.0,1.0]; palette[5] = [0.0,0.5,1.0]
        palette[6] = [0.9,0.95,1.0]; palette[7] = [0.1,0.4,0.8]
    elif id == 3: # Toxic
        palette[0] = [0.0,0.2,0.0]; palette[1] = [0.1,0.8,0.2]
        palette[2] = [0.2,0.4,0.1]; palette[3] = [0.1,0.2,0.0]
        palette[4] = [0.5,1.0,0.0]; palette[5] = [0.2,0.8,0.0]
        palette[6] = [0.8,1.0,0.5]; palette[7] = [0.3,0.6,0.1]
    elif id == 4: # Cyber
        palette[0] = [0.0,0.0,0.1]; palette[1] = [0.0,0.0,0.3]
        palette[2] = [0.1,0.1,0.1]; palette[3] = [0.0,0.0,0.0]
        palette[4] = [0.0,1.0,0.8]; palette[5] = [1.0,0.0,0.8]
        palette[6] = [0.5,0.0,1.0]; palette[7] = [0.1,0.1,0.1]
    elif id == 5: # Random
        palette[0] = [0.0,0.1,0.1]; palette[1] = [0.0,0.1,0.3]
        palette[2] = [0.1,0.0,0.1]; palette[3] = [0.1,0.0,0.0]
        palette[4] = [0.0,1.0,0.8]; palette[5] = [1.0,0.1,0.8]
        palette[6] = [1.0,1.0,1.0]; palette[7] = [0.5,0.5,0.5]
    elif id == 6: # All
        palette[0] = [0.5,0.5,0.1]; palette[1] = [0.5,0.5,0.3]
        palette[2] = [0.5,0.5,0.1]; palette[3] = [0.0,0.5,0.0]
        palette[4] = [0.5,1.0,0.8]; palette[5] = [1.0,0.0,0.8]
        palette[6] = [1.0,1.0,1.0]; palette[7] = [0.5,0.5,0.5]
    elif id == 7: # Ohhhhh
        palette[0] = [0.1,0.1,0.1]; palette[1] = [0.1,0.1,0.1]
        palette[2] = [0.1,0.1,0.1]; palette[3] = [0.1,0.1,0.1]
        palette[4] = [0.1,1.0,0.1]; palette[5] = [0.1,0.1,0.1]
        palette[6] = [0.8,0.8,0.8]; palette[7] = [0.1,0.1,0.1]
    elif id == 8: # What
        palette[0] = [0.2,0.3,0.1]; palette[1] = [0.2,0.3,0.3]
        palette[2] = [0.2,0.3,0.1]; palette[3] = [0.3,0.4,0.2]
        palette[4] = [0.4,1.0,0.8]; palette[5] = [1.0,0.0,0.8]
        palette[6] = [1.0,0.8,0.5]; palette[7] = [0.2,0.5,0.2]
    elif id == 9: # Not
        palette[0] = [0.0,0.0,0.0]; palette[1] = [0.0,0.0,0.0]
        palette[2] = [0.0,0.0,0.0]; palette[3] = [0.0,0.0,0.0]
        palette[4] = [0.0,0.0,0.0]; palette[5] = [0.0,0.0,0.0]
        palette[6] = [0.0,0.0,0.0]; palette[7] = [0.0,0.0,0.0]

def load_biome(id):
    init_game_state(id)

load_biome(0)
init_bh_noise() # Initialize BH Noise
gui = ti.GUI("Dali Universe Explorer", RES)
sim_time = 0.0 
teleport_cooldown = 0 # Cooldown frames
last_biome = 0

while gui.running:
    events = gui.get_events(gui.PRESS)
    for e in events:
        if e.key == gui.SPACE:
            spawn_star(camera_pos[None] + camera_dir[None]*2.0, camera_dir[None]*6.0 + camera_vel[None])
        elif e.key == 'r': active_stars[None] = 0
            
    # FLIGHT PHYSICS
    acc = 0.015; drag = 0.985
    yaw = camera_rot[None].x; pitch = camera_rot[None].y
    fwd = ti.Vector([math.sin(yaw), math.sin(pitch), math.cos(yaw)]).normalized()
    right = ti.Vector([0, 1, 0]).cross(fwd).normalized()
    
    vel = camera_vel[None]
    if gui.is_pressed('w'): vel += fwd * acc
    if gui.is_pressed('s'): vel -= fwd * acc
    if gui.is_pressed('a'): vel -= right * acc
    if gui.is_pressed('d'): vel += right * acc
    if gui.is_pressed('q'): vel -= ti.Vector([0,1,0]) * acc 
    if gui.is_pressed('e'): vel += ti.Vector([0,1,0]) * acc 
    
    # Apply Gravity in BH Biome
    if biome_id[None] == 10:
        cam_p = camera_pos[None]
        r_sq = cam_p.norm_sqr()
        if r_sq > 0.1:
            gravity = -cam_p.normalized() * (120.0 / (r_sq + 1.0)) * 0.01 # Tuning
            vel += gravity
            
    vel *= drag
    camera_vel[None] = vel
    camera_pos[None] += vel
    
    rot_speed = 0.03
    if gui.is_pressed(gui.LEFT): camera_rot[None].x -= rot_speed
    if gui.is_pressed(gui.RIGHT): camera_rot[None].x += rot_speed
    if gui.is_pressed(gui.UP): camera_rot[None].y -= rot_speed   
    if gui.is_pressed(gui.DOWN): camera_rot[None].y += rot_speed 
    
    sim_time += 0.03; frame_counter[None] += 1
    global_env[None] = ti.Vector([sim_time, 0, 0])
    
    # --- GAMEPLAY LOGIC (CPU Side Collision) ---
    if teleport_cooldown > 0:
        teleport_cooldown -= 1
    else:
        # Check Player Status (Collision/Space)
        status = check_player_status()
        current_id = biome_id[None]

        if status == 1 or status == 2:
            # Respawn on hit
            print("Crashed! Respawning...")
            camera_pos[None] = ti.Vector([0.0, 4.0, -4.0])
            camera_vel[None] = ti.Vector([0,0,0])
            active_stars[None] = 0 # Reset stars on crash
        elif status == 3:
            # Space Warp to Black Hole
            if current_id != 10:
                last_biome = current_id
                load_biome(10)
                camera_pos[None] = ti.Vector([0.0, 5.0, -15.0])
                camera_vel[None] = ti.Vector([0.3, 0.0, 0.0])
                teleport_cooldown = 100
                print("ATMOSPHERE BREACH -> ORBITAL INSERTION")

        # Object Logic
        # Replicate object motion logic from get_attractor_positions
        t = sim_time
        # Egg (Sphere) -> Past (-1)
        egg_p = ti.Vector([3.5 * math.sin(t*0.2), 2.5 + math.cos(t*0.15), 3.5 * math.cos(t*0.25)])
        # Clock -> Future (+1)
        clock_p = ti.Vector([4.0 * math.sin(-t*0.15 + 3.14), 2.0 + math.sin(t*0.1), 4.0 * math.cos(-t*0.15 + 3.14)])
        # Fractal (Crystal) -> Black Hole Teleport
        fractal_p = ti.Vector([0.0, 1.0, 0.0])
        
        cam_p = camera_pos[None]
        
        # Distances
        d_egg = (cam_p - egg_p).norm()
        d_clock = (cam_p - clock_p).norm()
        d_fractal = (cam_p - fractal_p).norm()
        
        TRIGGER_DIST = 1.2 # Tolerance
        
        if current_id == 10:
            # Escape condition: Hit the horizon or fly VERY far away?
            # Let's say hit horizon -> escape to safety
            dist_to_center = cam_p.norm()
            if dist_to_center < 1.1: # Horizon radius approx
                load_biome(last_biome)
                camera_pos[None] = ti.Vector([0.0, 4.0, -10.0])
                camera_vel[None] = ti.Vector([0,0,0])
                teleport_cooldown = 100
                print("Escaped Event Horizon!")
        else:
            if d_clock < TRIGGER_DIST:
                last_biome = current_id
                next_id = (current_id + 1) % (NUM_BIOMES - 1) # Exclude 10 from rotation
                load_biome(next_id)
                camera_vel[None] = ti.Vector([0,0,0]) 
                teleport_cooldown = 100
                print(f"Time Warp -> Future: Biome {next_id}")
                
            elif d_egg < TRIGGER_DIST:
                last_biome = current_id
                prev_id = (current_id - 1) % (NUM_BIOMES - 1)
                load_biome(prev_id)
                camera_vel[None] = ti.Vector([0,0,0]) 
                teleport_cooldown = 100
                print(f"Time Warp -> Past: Biome {prev_id}")
            
            elif d_fractal < TRIGGER_DIST:
                last_biome = current_id
                load_biome(10) # Black Hole ID
                camera_pos[None] = ti.Vector([0.0, 5.0, -15.0]) # Safe distance start
                camera_vel[None] = ti.Vector([0.3, 0.0, 0.0]) # Orbital injection
                teleport_cooldown = 100
                print("WARP: BLACK HOLE SYSTEM")

    
    update_physics(0.05, sim_time)
    render_pass(sim_time, frame_counter[None])
    
    p = frame_parity[None]
    physics_wts_update(0.1, p) 
    frame_parity[None] = 1 - p
    
    resolve(frame_parity[None])
    composite()
    
    gui.set_image(display_buffer)
    count = active_stars.to_numpy()[()]
    b_id = biome_id.to_numpy()[()]
    
    label = f"Biome: {b_id}"
    if b_id == 10: label = "Biome: ERGOSPHERE"
    
    gui.text(f"{label} | Stars: {count}/{MAX_STARS}", pos=(0.05, 0.95), font_size=20, color=0xFFFFFF)
    if teleport_cooldown > 0:
        gui.text(f"JUMP COOLDOWN: {teleport_cooldown}", pos=(0.05, 0.90), font_size=20, color=0xFF0000)
    else:
        gui.text(f"PORTALS READY", pos=(0.05, 0.90), font_size=20, color=0x00FF00)

    gui.show()