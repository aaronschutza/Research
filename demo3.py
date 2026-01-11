import taichi as ti
import math
import random

# --- ROBUST INIT (Optimized) ---
try:
    ti.init(arch=ti.vulkan, device_memory_GB=4.0, offline_cache=True)
except:
    try:
        ti.init(arch=ti.opengl, device_memory_GB=3.5, offline_cache=True)
    except:
        ti.init(arch=ti.cpu)

# --- Configuration ---
RES = (1280, 720)
ASPECT = RES[0] / RES[1]
MOMENTUM_DAMPING = 0.93  
VACUUM_STIFFNESS = 0.1      
TERRAIN_STIFFNESS = 40.0
OBJECT_STIFFNESS = 80.0     
WATER_STIFFNESS = 20.0   
VEG_STIFFNESS = 50.0      
WATER_LEVEL = -2.5        
FOV_SCALE = 0.75          

# APH PHYSICS CONSTANTS
WEAK_BUFFER_LIMIT = 0.85  
CONSERVATION_LIMIT = 8.0  
NOISE_CEILING = 0.3        

# --- Fields ---
color_buffer = ti.Vector.field(4, dtype=float, shape=RES) 
reflect_buffer = ti.Vector.field(4, dtype=float, shape=RES) 
specular_buffer = ti.Vector.field(3, dtype=float, shape=RES)

stiffness_map = ti.field(dtype=float, shape=RES) 
momentum = ti.Vector.field(3, dtype=float, shape=RES) 
prev_momentum = ti.Vector.field(3, dtype=float, shape=RES)

relaxed_buffer_A = ti.Vector.field(3, dtype=float, shape=RES)
relaxed_buffer_B = ti.Vector.field(3, dtype=float, shape=RES)
history_buffer = ti.Vector.field(3, dtype=float, shape=RES)
display_buffer = ti.Vector.field(3, dtype=float, shape=RES)
frame_parity = ti.field(dtype=int, shape=())

camera_orbit = ti.Vector.field(2, dtype=float, shape=()) 
camera_pos = ti.Vector.field(3, dtype=float, shape=())
look_at = ti.Vector([0.0, 1.5, 0.0]) 

global_env = ti.Vector.field(3, dtype=float, shape=())
seeds = ti.Vector.field(3, dtype=float, shape=())
frame_counter = ti.field(dtype=int, shape=())

# --- OPTIMIZED MATH HELPERS ---
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
def sd_sphere(p, r): return p.norm() - r
@ti.func
def sd_torus(p, t):
    q = ti.Vector([ti.Vector([p.x, p.z]).norm() - t.x, p.y])
    return q.norm() - t.y

@ti.func
def hash22(p):
    p3 = fract(ti.Vector([p.x, p.y, p.x]) * ti.Vector([0.1031, 0.1030, 0.0973]))
    p3 += p3.dot(p3.zxy + 33.33)
    return fract((p3.xx + p3.yz) * p3.zy)

@ti.func
def noise2(p):
    i = ti.floor(p)
    f = fract(p)
    u = f * f * (3.0 - 2.0 * f)
    a = hash22(i); b = hash22(i + ti.Vector([1.0, 0.0]))
    c = hash22(i + ti.Vector([0.0, 1.0])); d = hash22(i + ti.Vector([1.0, 1.0]))
    res = mix(mix(a.x, b.x, u.x), mix(c.x, d.x, u.x), u.y)
    return res

@ti.func
def fbm2(p, octaves: ti.template()):
    v = 0.0; amp = 0.5; freq = 1.0 
    for _ in ti.static(range(octaves)): 
        v += noise2(p * freq) * amp
        amp *= 0.5; freq *= 2.0
    return v

# --- APH PROTOCOLS ---
@ti.func
def adaptive_energy_conservation(col):
    max_c = max(col.x, max(col.y, col.z))
    res = col
    if max_c > CONSERVATION_LIMIT:
        scale = (CONSERVATION_LIMIT + ti.log(1.0 + (max_c - CONSERVATION_LIMIT))) / max_c
        res = col * scale
    return res

@ti.func
def limit_noise_vector(v, limit):
    mag = v.norm()
    res = v
    if mag > limit:
        res = v.normalized() * limit
    return res

# --- UNIFIED VOLUMETRICS ---
@ti.func
def get_sky(rd, sun_dir, storm, flash):
    day_top = ti.Vector([0.1, 0.4, 0.90]) * 1.5
    day_hor = ti.Vector([0.6, 0.8, 0.95]) * 1.2
    night_top = ti.Vector([0.005, 0.005, 0.02])
    night_hor = ti.Vector([0.02, 0.02, 0.05])
    
    sun_h = sun_dir.y
    day_mix = smoothstep(-0.2, 0.2, sun_h)
    horizon_grad = smoothstep(0.0, 0.6, abs(rd.y))
    
    sky = mix(mix(night_hor, night_top, horizon_grad), mix(day_hor, day_top, horizon_grad), day_mix)
    
    if day_mix < 0.5:
        star_val = hash22(rd.xy * 200.0).x
        if star_val > 0.995:
             sky += (star_val - 0.995) * 400.0 * (1.0 - day_mix)
    
    sky = mix(sky, ti.Vector([0.05, 0.05, 0.06]), storm)
    
    cloud_p = rd.xz * (20.0 / (abs(rd.y) + 0.05)) + ti.Vector([global_env[None].x * 1.5, 0.0])
    den = smoothstep(0.4, 0.8, noise2(cloud_p * 0.03 + seeds[None].xy)) * smoothstep(0.0, 0.3, abs(rd.y))
    
    light_dir = sun_dir if sun_h > 0.0 else -sun_dir
    light_lit = max(0.0, rd.dot(light_dir))
    
    c_cloud = mix(ti.Vector([0.05, 0.05, 0.1]), ti.Vector([1.0,0.95,0.9]), day_mix)
    c_cloud = mix(c_cloud, ti.Vector([0.1,0.12,0.15]), storm) * (0.4 + 0.6*light_lit)
    
    disk_col = mix(ti.Vector([0.8, 0.9, 1.0]), ti.Vector([1.0, 0.9, 0.7]), day_mix)
    disk = smoothstep(0.999, 0.9995, light_lit) * 50.0 * disk_col * smoothstep(-0.05, 0.1, light_dir.y)
    
    return mix(sky, c_cloud, den) + disk

@ti.func
def get_volume_color(rd):
    depth_factor = smoothstep(0.0, 0.9, abs(rd.y))
    return mix(ti.Vector([0.0, 0.8, 0.9]), ti.Vector([0.0, 0.02, 0.2]), depth_factor)

# --- PROCEDURAL TEXTURING ---
@ti.func
def get_terrain_texture(p, normal, slope):
    c_grass = ti.Vector([0.05, 0.25, 0.05])
    c_rock  = ti.Vector([0.3, 0.3, 0.32])
    c_snow  = ti.Vector([0.95, 0.95, 1.0])
    base = mix(c_rock, c_grass, smoothstep(0.3, 0.6, slope))
    base = mix(base, c_snow, smoothstep(6.0, 9.0, p.y))
    grain = (hash22(p.xz * 10.0).x - 0.5) * 0.1
    return base + ti.Vector([grain, grain, grain])

# --- UNIFIED MATERIAL SYSTEM ---
@ti.func
def get_material_props(id, p, time, normal):
    albedo = ti.Vector([0.0,0.0,0.0])
    roughness = 1.0; metal = 0.0; trans = 0.0; ior = 1.0
    beta = VACUUM_STIFFNESS; fluidity = 0.0; conductivity = 0.0
    
    if id == 1: # Marble
        n = noise2(p.xz*4.0) 
        albedo = ti.Vector([0.9,0.9,0.95]) * (0.85 + 0.15*n)
        roughness = 0.5; beta = OBJECT_STIFFNESS; conductivity = 0.05
    elif id == 2: # Gold
        albedo = ti.Vector([0.9,0.7,0.2]) 
        roughness = 0.15; metal = 1.0; beta = OBJECT_STIFFNESS; conductivity = 0.01
    elif id == 3: # Crystal
        albedo = ti.Vector([0.02, 0.05, 0.25])
        roughness = 0.0; trans = 1.0; ior = 1.6; beta = OBJECT_STIFFNESS; conductivity = 0.05
    elif id == 4: # Terrain
        slope = 1.0 - abs(normal.y)
        albedo = get_terrain_texture(p, normal, slope)
        roughness = 0.9; beta = TERRAIN_STIFFNESS
        is_sand = smoothstep(WATER_LEVEL+1.5, WATER_LEVEL+0.2, p.y)
        conductivity = mix(0.2, 0.8, is_sand)
    elif id == 5: # Water
        albedo = ti.Vector([0.0,0.0,0.0]) 
        roughness = 0.05; trans = 1.0; ior = 1.33; beta = WATER_STIFFNESS
        fluidity = 1.0; conductivity = 1.0
        
    return albedo, roughness, metal, trans, ior, beta, fluidity, conductivity

# --- SCENE MAPPING ---
@ti.func
def get_terrain_height(xz):
    xz_s = xz + seeds[None].xy
    base = noise2(xz_s * 0.03) * 6.0 - 2.0
    mountains = smoothstep(0.4, 1.0, noise2(xz_s * 0.06 + seeds[None].z)) * 5.0
    raw_h = base + mountains
    r = xz.norm()
    safe_zone = smoothstep(14.0, 18.0, r) 
    return mix(-3.0, raw_h, safe_zone)

@ti.func
def map_scene(p):
    S = 2.5; center_y = -1.0 * S 
    
    plinth_top = center_y + 1.2 * S 
    q = abs(p - ti.Vector([0.0, -10.0, 0.0])) - ti.Vector([0.7*S, 100.0, 0.7*S])
    d_plinth = max(q.x, max(p.y - plinth_top, q.z))
    d_shaft = (ti.Vector([p.x, p.z])).norm() - 0.45*S
    d_shaft = max(d_shaft, abs(p.y - (center_y + 1.5*S)) - 0.5*S)
    
    d_col_body = min(d_plinth, d_shaft)
    d_gold = 100.0; d_sphere = 100.0
    
    if p.y > center_y + 1.0 * S:
        bowl_center = ti.Vector([0.0, center_y + 1.5*S - 0.1*S, 0.0])
        d_bowl = sd_sphere(p - bowl_center, 0.53*S) 
        d_ring = sd_torus(p - ti.Vector([0.0, center_y + 1.65*S, 0.0]), ti.Vector([0.55*S, 0.06*S]))
        d_gold = min(d_bowl, d_ring)
        d_sphere = sd_sphere(p - ti.Vector([0.0, center_y + 1.8*S, 0.0]), 0.5*S)
        
    d_obj = min(d_col_body, min(d_gold, d_sphere))
    return d_col_body, d_gold, d_sphere, d_obj

@ti.func
def get_obj_normal(p):
    e = 0.002; d0 = map_scene(p)[3]
    return ti.Vector([
        map_scene(p+ti.Vector([e,0.0,0.0]))[3]-d0,
        map_scene(p+ti.Vector([0.0,e,0.0]))[3]-d0,
        map_scene(p+ti.Vector([0.0,0.0,e]))[3]-d0
    ]).normalized()

@ti.func
def get_terrain_normal(p):
    e = 0.05; h0 = get_terrain_height(ti.Vector([p.x, p.z]))
    dx = get_terrain_height(ti.Vector([p.x+e, p.z])) - h0
    dz = get_terrain_height(ti.Vector([p.x, p.z+e])) - h0
    return ti.Vector([-dx, e, -dz]).normalized()

@ti.func
def calc_softshadow(ro, rd, tmin, tmax, k, dither):
    res = 1.0; t = tmin + dither * 0.05 
    for _ in range(4): 
        d1, d2, d3, h = map_scene(ro + rd * t)
        if h < 0.001: res = 0.0; break
        res = min(res, k * h / t)
        t += h * (1.2 + 0.3 * dither)
        if t > tmax: break
    return res

@ti.func
def calc_ao(p, n):
    occ = 0.0; sca = 1.0
    for i in ti.static(range(2)): 
        h = 0.1 + 0.25 * float(i)
        d = map_scene(p + h * n)[3]
        occ += (h - d) * sca
        sca *= 0.5
    return max(0.0, 1.0 - 1.5 * occ)

@ti.kernel
def render_pass(time: float, frame: int):
    sun_angle = (time * 0.03) + 2.2 
    sun_dir = ti.Vector([ti.cos(sun_angle), ti.sin(sun_angle), 0.4]).normalized()
    storm = global_env[None].y
    flash = global_env[None].z
    cam = camera_pos[None]
    fwd = (look_at - cam).normalized()
    right = fwd.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    up = right.cross(fwd)
    
    jit_x = (float(frame % 2) - 0.5) * 0.5
    jit_y = (float((frame // 2) % 2) - 0.5) * 0.5

    for u, v in color_buffer:
        uv_base = ti.Vector([u + jit_x, v + jit_y])
        uv = (uv_base / ti.Vector([RES[0], RES[1]]) * 2.0 - 1.0) * ti.Vector([ASPECT, 1.0])
        rd = (fwd + uv.x * right * FOV_SCALE + uv.y * up * FOV_SCALE).normalized()
        
        dither_val = hash22(ti.Vector([float(u)+time, float(v)])).x
        
        t_final = 10000.0; hit_id = 0; obj_id = 0
        t = 0.0
        
        for _ in range(32):
            p = cam + rd * t
            d1, d2, d3, d = map_scene(p)
            if d < 0.001:
                t_final = t; hit_id = 1
                if d == d1: obj_id = 1
                elif d == d2: obj_id = 2
                else: obj_id = 3
                break
            t += d
            if t > 60.0: break
            
        if (hit_id == 0 or t > 80.0) and rd.y < 0.1: 
            t = 0.0; t_start = 0.0
            if cam.y > 10.0 and rd.y < 0.0: t_start = (5.0 - cam.y)/rd.y
            t = max(0.0, t_start)
            
            for _ in range(48): 
                p = cam + rd * t
                h = get_terrain_height(ti.Vector([p.x, p.z]))
                d = p.y - h
                if d < 0.005 * t: 
                    if t < t_final: t_final = t; hit_id = 2
                    break
                t += d * 0.75
                if t > 250.0: break

        if rd.y < -0.001:
            t_w = (WATER_LEVEL - cam.y) / rd.y
            if t_w > 0.0 and t_w < t_final: t_final = t_w; hit_id = 3 

        mat_type_id = 0
        if hit_id == 1: mat_type_id = obj_id 
        elif hit_id == 2: mat_type_id = 4    
        elif hit_id == 3: mat_type_id = 5    
        
        c_matter = get_sky(rd, sun_dir, storm, flash)
        c_reflect = ti.Vector([0.0, 0.0, 0.0])
        c_spark = ti.Vector([0.0, 0.0, 0.0])
        
        beta_val = VACUUM_STIFFNESS
        fluidity_val = 0.0
        
        if mat_type_id > 0:
            p = cam + t_final * rd
            norm = ti.Vector([0.0, 1.0, 0.0])
            
            if hit_id == 1: norm = get_obj_normal(p)
            elif hit_id == 2: norm = get_terrain_normal(p)
            elif hit_id == 3: 
                view_angle = abs(rd.dot(ti.Vector([0.0, 1.0, 0.0]))) 
                aniso_damp = 1.0 / (1.0 + t_final * 0.05 + (1.0 - view_angle) * 5.0)
                t_noise = time - ti.floor(time / 100.0) * 100.0
                q = p.xz * 2.0 + t_noise * 0.5
                
                w1 = noise2(q); w2 = noise2(q * 2.1 - 1.0) * 0.5; w3 = noise2(q * 4.3 + 2.0) * 0.25
                wave = (w1 + w2 + w3) * 0.04 * aniso_damp
                
                phys_warp = momentum[u, v].x * 0.03
                wave += phys_warp
                wave_fade = smoothstep(120.0, 10.0, t_final) 
                d_wave = ti.Vector([wave, 1.0, wave]) 
                d_wave = mix(ti.Vector([0.0, 1.0, 0.0]), d_wave, wave_fade)
                norm = limit_noise_vector(d_wave, 1.0 + NOISE_CEILING).normalized()
            
            albedo, roughness, metal, trans, ior, beta, fluidity, conductivity = get_material_props(mat_type_id, p, time, norm)
            
            occ = 1.0
            if hit_id != 3: occ = calc_ao(p, norm)
            
            light_dir = sun_dir if sun_dir.y > 0.0 else -sun_dir
            
            shadow = 1.0
            if hit_id != 3: shadow = calc_softshadow(p + norm*0.05, light_dir, 0.1, 20.0, 4.0, dither_val)
            
            view_dot_n = max(0.0, -rd.dot(norm))
            f0 = mix(0.04, albedo.x, metal) 
            fresnel = f0 + (1.0 - f0) * pow(1.0 - view_dot_n, 5.0)
            
            r_dir = ti.math.reflect(rd, norm)
            ref_sky = get_sky(r_dir, sun_dir, storm, flash)
            if mat_type_id == 2: ref_sky *= albedo 
            if mat_type_id == 5: ref_sky *= ti.Vector([1.2, 1.25, 1.4]) 
            c_reflect = adaptive_energy_conservation(ref_sky) * fresnel
            if roughness > 0.2: c_reflect *= 0.0 
            
            h = (light_dir - rd).normalized()
            n_dot_h = max(0.0, norm.dot(h))
            eff_rough = roughness
            if mat_type_id == 5: 
                wave_mag = momentum[u,v].norm()
                eff_rough += clamp(wave_mag * 5.0, 0.0, 0.4) 
            
            spec_pow = mix(50.0, 300.0, 1.0-eff_rough)
            day_mix = smoothstep(-0.2, 0.2, sun_dir.y)
            sun_col = ti.Vector([1.0, 0.98, 0.95])
            moon_col = ti.Vector([0.3, 0.35, 0.5]) * 0.2
            l_col = mix(moon_col, sun_col, day_mix) * (1.0 - storm * 0.8)
            light_fade = smoothstep(-0.05, 0.1, abs(sun_dir.y)) 
            l_col *= light_fade
            if flash > 0.01: l_col += ti.Vector([0.8, 0.9, 1.0]) * flash * 6.0
            
            raw_spark = pow(n_dot_h, spec_pow) * (1.0 - eff_rough) * 2.0
            raw_spark = min(raw_spark, 8.0) 
            c_spark = l_col * raw_spark * shadow
            
            c_matter = albedo * (occ * 0.1 + l_col * max(0.0, norm.dot(light_dir)) * shadow)
            
            if trans > 0.0:
                refr_dir = ti.math.refract(rd, norm, 1.0/ior)
                vol_col = get_volume_color(refr_dir)
                if mat_type_id == 3: vol_col *= ti.Vector([0.95, 0.95, 1.1]) * 1.2
                
                phase_angle = max(0.0, rd.dot(light_dir))
                scatter_strength = min(pow(phase_angle, 8.0) * 2.0, 5.0) * light_fade
                sun_through = max(0.0, light_dir.dot(norm)) 
                scatter_light = ti.Vector([0.0, 0.4, 0.5]) * sun_through * 0.2 * light_fade
                scatter_light += ti.Vector([0.1, 0.8, 0.6]) * scatter_strength * sun_through
                
                c_matter = (vol_col + scatter_light) * (1.0 - fresnel) * trans
            
            beta_val = beta
            fluidity_val = fluidity
            
            if hit_id != 99: 
                fog_dist = t_final
                fog_amount = 1.0 - ti.exp(-fog_dist * 0.0035)
                fog_col = get_sky(ti.Vector([rd.x, 0.05, rd.z]).normalized(), sun_dir, storm, flash)
                c_matter = mix(c_matter, fog_col, fog_amount)
                c_reflect = mix(c_reflect, ti.Vector([0.0,0.0,0.0]), fog_amount)
                c_spark = mix(c_spark, ti.Vector([0.0,0.0,0.0]), fog_amount)

        c_matter_phys = ti.math.clamp(c_matter, 0.0, WEAK_BUFFER_LIMIT)
        color_buffer[u, v] = ti.Vector([c_matter_phys.r, c_matter_phys.g, c_matter_phys.b, fluidity_val])
        reflect_buffer[u, v] = ti.Vector([
            adaptive_energy_conservation(c_reflect).x, 
            adaptive_energy_conservation(c_reflect).y,
            adaptive_energy_conservation(c_reflect).z,
            t_final 
        ])
        specular_buffer[u, v] = c_spark
        stiffness_map[u, v] = beta_val

@ti.kernel
def wts_active_physics(dt: float, parity: int):
    storm = global_env[None].y
    
    for i, j in relaxed_buffer_A:
        center = relaxed_buffer_A[i, j]
        if parity == 1: center = relaxed_buffer_B[i, j]
        
        beta_c = stiffness_map[i, j]
        fluidity = color_buffer[i, j].w 
        
        if beta_c < 1.0: 
            val = color_buffer[i, j].xyz
            if parity == 0: relaxed_buffer_B[i, j] = val
            else: relaxed_buffer_A[i, j] = val
            continue
            
        mom = momentum[i, j]
        prev = prev_momentum[i, j]
        if mom.dot(prev) < 0.0: mom *= 0.5 
        prev_momentum[i, j] = mom
        mom *= 0.999 
        
        if fluidity > 0.5 and storm > 0.1: 
            if ti.random() > 0.995:
                impact = ti.random() * storm * 5.0
                mom += ti.Vector([impact, impact, impact])
        
        laplacian = ti.Vector([0.0, 0.0, 0.0])
        offsets = [ti.Vector([-1, 0]), ti.Vector([1, 0]), ti.Vector([0, -1]), ti.Vector([0, 1])]
        
        for k in ti.static(range(4)):
            off = offsets[k]
            ix, iy = clamp(i + off.x, 0, RES[0]-1), clamp(j + off.y, 0, RES[1]-1)
            
            n_val = relaxed_buffer_A[ix, iy]
            if parity == 1: n_val = relaxed_buffer_B[ix, iy]
            
            n_id = color_buffer[ix, iy].w
            
            if n_id < 0.5:
                n_val = center
                
            laplacian += n_val
        
        laplacian -= 4.0 * center
        force = ti.math.tanh(laplacian / max(0.01, beta_c) * 0.15) * 20.0 
        
        mom = mom * MOMENTUM_DAMPING + force * dt
        mom = ti.math.clamp(mom, -5.0, 5.0)
        
        momentum[i, j] = mom
        new_val = center + mom
        
        input_col = color_buffer[i, j].xyz 
        if fluidity > 0.5:
            gradient = (input_col - center).norm()
            base_feed = 0.30
            adaptive_feed = base_feed / (1.0 + gradient * 5.0) 
            new_val = ti.math.clamp(new_val, 0.0, WEAK_BUFFER_LIMIT)
            new_val = new_val * (1.0 - adaptive_feed) + input_col * adaptive_feed
        else:
            new_val = input_col
        
        if parity == 0: relaxed_buffer_B[i, j] = new_val
        else: relaxed_buffer_A[i, j] = new_val

@ti.kernel
def temporal_resolve(parity: int):
    sun_angle = (global_env[None].x * 0.03) + 2.2 
    sun_dir = ti.Vector([ti.cos(sun_angle), ti.sin(sun_angle), 0.4]).normalized()
    storm = global_env[None].y
    flash = global_env[None].z

    for i, j in history_buffer:
        curr_matter = relaxed_buffer_A[i, j]
        if parity == 1: curr_matter = relaxed_buffer_B[i, j]
        
        depth = reflect_buffer[i, j].w
        fluidity = color_buffer[i, j].w
        final_color = curr_matter + reflect_buffer[i, j].xyz + specular_buffer[i, j]
        
        if fluidity > 0.5: 
             fog_dist = depth * 2.5
             fog_amount = 1.0 - ti.exp(-fog_dist * 0.0035)
             fog_col = get_sky(ti.Vector([0.0, 0.05, 1.0]), sun_dir, storm, flash) 
             final_color = mix(final_color, fog_col, fog_amount)
        
        min_c = ti.Vector([9999.0, 9999.0, 9999.0])
        max_c = ti.Vector([-9999.0, -9999.0, -9999.0])
        
        for x in ti.static(range(-1, 2)):
            for y in ti.static(range(-1, 2)):
                if not (x == 0 and y == 0): # Fixed: Inverted logic to avoid continue
                    ix, iy = clamp(i+x, 0, RES[0]-1), clamp(j+y, 0, RES[1]-1)
                    
                    c_m = relaxed_buffer_A[ix, iy]
                    if parity == 1: c_m = relaxed_buffer_B[ix, iy]
                    
                    n_col = c_m + reflect_buffer[ix, iy].xyz + specular_buffer[ix, iy]
                    min_c = ti.min(min_c, n_col)
                    max_c = ti.max(max_c, n_col)
        
        hist = history_buffer[i, j]
        hist_clamped = ti.math.clamp(hist, min_c, max_c)
        
        diff = (final_color - hist_clamped).norm()
        blend = mix(0.08, 0.25, smoothstep(0.0, 0.5, diff))
        
        history_buffer[i, j] = mix(hist_clamped, final_color, blend)

@ti.kernel
def post_process_aces():
    for i, j in display_buffer:
        depth = reflect_buffer[i, j].w
        focal_plane = 10.0 
        
        coc = abs(depth - focal_plane) * 0.015 
        coc = clamp(coc, 0.0, 2.0) 
        
        blur_col = ti.Vector([0.0, 0.0, 0.0])
        total_w = 0.0
        
        for x in ti.static(range(-2, 3)):
            for y in ti.static(range(-2, 3)):
                ix, iy = clamp(i + x, 0, RES[0]-1), clamp(j + y, 0, RES[1]-1)
                
                spark = specular_buffer[ix, iy]
                if spark.norm() > 1.0:
                    dist = ti.sqrt(float(x*x + y*y))
                    blur_col += spark * (0.05 / (1.0 + dist))
                
                w = 1.0 
                if abs(float(x)) > 0.0 or abs(float(y)) > 0.0:
                     w = smoothstep(coc + 0.5, coc - 0.5, float(max(abs(x), abs(y))))
                
                blur_col += history_buffer[ix, iy] * w
                total_w += w
        
        base_col = blur_col / max(total_w, 0.001)
        
        uv = ti.Vector([float(i)/RES[0], float(j)/RES[1]])
        to_center = uv - 0.5
        dist_sq = to_center.dot(to_center)
        aberration = to_center * dist_sq * 2.0 
        r_off = ti.cast(aberration * RES[0] * 0.01, int)
        
        ir = clamp(i - r_off.x, 0, RES[0]-1)
        ib = clamp(i + r_off.x, 0, RES[0]-1)
        
        c_r = history_buffer[ir, j].x
        c_g = base_col.y
        c_b = history_buffer[ib, j].z
        
        color = ti.Vector([c_r, c_g, c_b])
        vignette = 1.0 - dist_sq * 0.5
        color *= vignette
        
        color *= 0.6 
        a = 2.51; b = 0.03; c = 2.43; d = 0.59; e = 0.14
        mapped = (color * (a * color + b)) / (color * (c * color + d) + e)
        mapped = ti.math.clamp(mapped, 0.0, 1.0)
        mapped = ti.pow(mapped, 1.0/2.2) 
        
        dither = (hash22(ti.Vector([float(i), float(j)])).x - 0.5) / 255.0
        display_buffer[i, j] = mapped + dither

@ti.kernel
def init_all():
    camera_orbit[None] = ti.Vector([0.0, 1.35]) 
    seeds[None] = ti.Vector([ti.random()*100, ti.random()*100, ti.random()*100])
    frame_counter[None] = 0

init_all()
gui = ti.GUI("WTS-RTX: Optimized APH", RES)
sim_time = 0.0 

while gui.running:
    camera_orbit[None].x += 0.029
    theta = camera_orbit[None].x
    phi = camera_orbit[None].y
    r = 9.0 
    camera_pos[None] = ti.Vector([r*ti.sin(phi)*ti.cos(theta), r*ti.cos(phi), r*ti.sin(phi)*ti.sin(theta)])
    
    sim_time += 0.033 
    frame_counter[None] += 1
    
    storm_cycle = (math.sin(sim_time * 0.15) + 1.0) * 0.5
    storm = 0.0
    if storm_cycle > 0.65: storm = (storm_cycle - 0.65) / 0.35
    
    flash = 0.0
    if storm > 0.5 and random.random() < 0.015: flash = 1.0
    prev_flash = global_env[None].z
    flash = max(flash, prev_flash * 0.8)
    global_env[None] = ti.Vector([sim_time, storm, flash])
    
    render_pass(sim_time, frame_counter[None])
    
    for _ in range(3): 
        p = frame_parity[None]
        wts_active_physics(0.2, p) 
        frame_parity[None] = 1 - p
        
    temporal_resolve(frame_parity[None])
    post_process_aces()
    
    gui.set_image(display_buffer)
    gui.show()