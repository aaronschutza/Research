import taichi as ti
import math

# --- ROBUST INIT ---
try:
    ti.init(arch=ti.cuda, device_memory_GB=1.0, offline_cache=False)
except:
    ti.init(arch=ti.cpu)

# --- Configuration ---
RES = (1280, 720)
ASPECT = RES[0] / RES[1]
MOMENTUM_DAMPING = 0.90   
VACUUM_STIFFNESS = 0.1    
TERRAIN_STIFFNESS = 40.0
OBJECT_STIFFNESS = 80.0   

# --- Fields ---
color_buffer = ti.Vector.field(4, dtype=float, shape=RES) 
stiffness_map = ti.field(dtype=float, shape=RES) 
momentum = ti.Vector.field(3, dtype=float, shape=RES) 

relaxed_buffer_A = ti.Vector.field(3, dtype=float, shape=RES)
relaxed_buffer_B = ti.Vector.field(3, dtype=float, shape=RES)
history_buffer = ti.Vector.field(3, dtype=float, shape=RES)
display_buffer = ti.Vector.field(3, dtype=float, shape=RES)
frame_parity = ti.field(dtype=int, shape=())

camera_orbit = ti.Vector.field(2, dtype=float, shape=()) 
camera_pos = ti.Vector.field(3, dtype=float, shape=())
look_at = ti.Vector([0.0, 0.5, 0.0]) 

# --- NOISE KERNELS ---
@ti.func
def fract(x): return x - ti.floor(x)

@ti.func
def random2(st):
    return fract(ti.sin(st.dot(ti.Vector([12.9898, 78.233]))) * 43758.5453123)

@ti.func
def noise2(st):
    i = ti.floor(st)
    f = fract(st)
    u = f * f * (3.0 - 2.0 * f)
    a = random2(i); b = random2(i + ti.Vector([1.0, 0.0]))
    c = random2(i + ti.Vector([0.0, 1.0])); d = random2(i + ti.Vector([1.0, 1.0]))
    return (a * (1.0 - u.x) + b * u.x) * (1.0 - u.y) + (c * (1.0 - u.x) + d * u.x) * u.y

@ti.func
def fbm2(p, octaves: ti.template()):
    v = 0.0; amp = 0.5; freq = 1.0 
    for _ in range(octaves):
        v += noise2(p * freq) * amp
        amp *= 0.5; freq *= 2.0
    return v

@ti.func
def random3(st):
    return fract(ti.sin(st.dot(ti.Vector([12.9898, 78.233, 45.543]))) * 43758.5453123)

@ti.func
def noise3(st):
    i = ti.floor(st)
    f = fract(st)
    u = f * f * (3.0 - 2.0 * f)
    n000 = random3(i + ti.Vector([0,0,0])); n100 = random3(i + ti.Vector([1,0,0]))
    n010 = random3(i + ti.Vector([0,1,0])); n110 = random3(i + ti.Vector([1,1,0]))
    n001 = random3(i + ti.Vector([0,0,1])); n101 = random3(i + ti.Vector([1,0,1]))
    n011 = random3(i + ti.Vector([0,1,1])); n111 = random3(i + ti.Vector([1,1,1]))
    nx00 = n000*(1-u.x) + n100*u.x; nx10 = n010*(1-u.x) + n110*u.x
    nx01 = n001*(1-u.x) + n101*u.x; nx11 = n011*(1-u.x) + n111*u.x
    nxy0 = nx00*(1-u.y) + nx10*u.y; nxy1 = nx01*(1-u.y) + nx11*u.y
    return nxy0*(1-u.z) + nxy1*u.z

@ti.func
def fbm3(p, octaves: ti.template()):
    v = 0.0; amp = 0.5; freq = 1.0
    for _ in range(octaves):
        v += noise3(p * freq) * amp
        amp *= 0.5; freq *= 2.0
    return v

# --- SDF SCENE ---
@ti.func
def sd_sphere(p, r): return p.norm() - r

@ti.func
def sd_box(p, b):
    q = abs(p) - b
    return max(q, 0.0).norm() + min(max(q.x, max(q.y, q.z)), 0.0)

@ti.func
def sd_cylinder(p, h, r):
    d = abs(ti.Vector([p.x, p.z]).norm()) - r
    return max(d, abs(p.y) - h)

@ti.func
def sd_torus(p, t):
    q = ti.Vector([ti.Vector([p.x, p.z]).norm() - t.x, p.y])
    return q.norm() - t.y

@ti.func
def map_scene(p):
    # 1. Classical Column Base (Deep Extrusion)
    # Extends from Y=-2.0 to Y=+0.2
    d_plinth = sd_box(p - ti.Vector([0.0, -1.0, 0.0]), ti.Vector([0.7, 1.2, 0.7]))
    
    d_base_ring = sd_torus(p - ti.Vector([0.0, 0.25, 0.0]), ti.Vector([0.55, 0.08]))
    
    d_shaft = sd_cylinder(p - ti.Vector([0.0, 0.5, 0.0]), 0.4, 0.5)
    
    d_column = min(d_plinth, min(d_base_ring, d_shaft))
    
    # 2. Gold Mount
    d_mount = sd_torus(p - ti.Vector([0.0, 0.91, 0.0]), ti.Vector([0.4, 0.04]))
    
    # 3. Crystal Sphere
    d_sphere = sd_sphere(p - ti.Vector([0.0, 1.3, 0.0]), 0.5)
    
    d_objs = min(d_column, min(d_mount, d_sphere))
    return d_column, d_mount, d_sphere, d_objs

# --- AMBIENT OCCLUSION ---
@ti.func
def calc_ao(p, n):
    occ = 0.0; sca = 1.0
    for i in range(5):
        h = 0.01 + 0.12 * float(i) / 4.0
        d, _, _, _ = map_scene(p + h * n)
        occ += (h - d) * sca
        sca *= 0.95
        if occ > 0.35: break 
    return max(0.0, 1.0 - 3.0 * occ)

@ti.func
def raymarch_terrain(ro, rd):
    t = 0.0; hit = 0
    if rd.y < 0.0:
        t = (0.5 - ro.y) / rd.y if ro.y > 0.5 else 0.0
        for _ in range(64):
            p = ro + rd * t
            h = fbm2(ti.Vector([p.x, p.z]) * 0.4, 2) * 2.0 - 1.5
            d = p.y - h
            if d < 0.005 * t: hit = 1; break
            t += d * 0.5
            if t > 50.0: break
    return hit, t

@ti.func
def get_sky_and_light(rd, sun_dir):
    sun_height = sun_dir.y
    is_day = sun_height > -0.1
    
    main_light_dir = sun_dir
    main_light_col = ti.Vector([1.0, 0.9, 0.8]) * 1.5 
    ambient_col = ti.Vector([0.1, 0.2, 0.4]) 
    
    if not is_day:
        main_light_dir = -sun_dir 
        main_light_dir.y = max(0.1, main_light_dir.y) 
        main_light_col = ti.Vector([0.2, 0.3, 0.5]) * 0.4
        ambient_col = ti.Vector([0.01, 0.01, 0.03])
    
    col_day = ti.Vector([0.1, 0.4, 0.8]) 
    col_sunset = ti.Vector([0.8, 0.4, 0.1])
    col_night = ti.Vector([0.005, 0.005, 0.02]) 
    
    sky_col = col_night
    if sun_height > 0.2:
        t_mix = 1.0 - sun_height * 2.0
        sky_col = col_day * (1.0 - t_mix) + col_sunset * t_mix
    elif sun_height > -0.2:
        t_mix = (0.2 - sun_height) * 2.5
        sky_col = col_sunset * (1.0 - t_mix) + col_night * t_mix
        
    fade = pow(1.0 - abs(rd.y), 4.0)
    horizon_col = ti.Vector([0.3, 0.3, 0.4]) * max(0.2, sun_height + 0.5)
    sky_col = sky_col * (1.0 - fade) + horizon_col * fade
    
    if sun_height < 0.1:
        s = random2(rd.xz * 150.0 + rd.xy * 80.0) 
        star_thresh = 0.985
        stars = max(0.0, s - star_thresh) * (1.0 / (1.0 - star_thresh))
        stars = pow(stars, 20.0) * 1.5
        star_vis = (1.0 - max(0.0, sun_height * 10.0)) * (1.0 - fade)
        sky_col += ti.Vector([1.0, 1.0, 1.0]) * stars * star_vis
        
        planet_dir = ti.Vector([0.5, 0.3, 0.8]).normalized()
        planet_spot = pow(max(0.0, rd.dot(planet_dir)), 2000.0)
        sky_col += ti.Vector([1.0, 0.9, 0.8]) * planet_spot * 3.0 * star_vis

    spot_power = 800.0 if is_day else 400.0
    spot = pow(max(0.0, rd.dot(main_light_dir)), spot_power)
    sky_col += main_light_col * spot * (2.0 if is_day else 5.0)

    if rd.y > 0.0:
        cloud_uv = (rd.xz / (rd.y + 0.15)) * 0.5 + ti.Vector([0.1, 0.0])
        density = fbm2(cloud_uv, 3)
        density = max(0.0, density - 0.45) * 1.5 
        
        if density > 0.0:
             v_dot_l = max(0.0, rd.dot(main_light_dir))
             scatter = 0.1 + 0.9 * pow(v_dot_l, 6.0) 
             c_light = main_light_col * scatter
             c_amb = ambient_col
             final_cloud = (c_amb + c_light) * density
             visibility = ti.exp(-density)
             sky_col = sky_col * visibility + final_cloud * (1.0 - visibility)

    return sky_col, main_light_dir, main_light_col, ambient_col

@ti.func
def intersect_objects(ro, rd):
    t = 0.0; obj_id = 0
    for _ in range(64):
        p = ro + rd * t
        d1, d2, d3, d = map_scene(p)
        if d < 0.001:
            if d == d1: obj_id = 1
            elif d == d2: obj_id = 2
            else: obj_id = 3
            break
        t += d
        if t > 20.0: break
    return obj_id, t

@ti.func
def get_obj_normal(p):
    e = 0.001
    d1, d2, d3, d0 = map_scene(p)
    nx = map_scene(p + ti.Vector([e, 0, 0]))[3] - d0
    ny = map_scene(p + ti.Vector([0, e, 0]))[3] - d0
    nz = map_scene(p + ti.Vector([0, 0, e]))[3] - d0
    return ti.Vector([nx, ny, nz]).normalized()

@ti.func
def get_terrain_normal(p):
    e = 0.01
    dx = fbm2((p + ti.Vector([e, 0, 0])).xz*0.4, 4) - fbm2((p - ti.Vector([e, 0, 0])).xz*0.4, 4)
    dz = fbm2((p + ti.Vector([0, 0, e])).xz*0.4, 4) - fbm2((p - ti.Vector([0, 0, e])).xz*0.4, 4)
    macro_norm = ti.Vector([-dx, e*2.0, -dz]).normalized()
    tex_dx = noise2((p.xz + ti.Vector([e, 0])) * 8.0) - noise2((p.xz - ti.Vector([e, 0])) * 8.0)
    tex_dz = noise2((p.xz + ti.Vector([0, e])) * 8.0) - noise2((p.xz - ti.Vector([0, e])) * 8.0)
    return (macro_norm + ti.Vector([-tex_dx, 0.0, -tex_dz]) * 0.3).normalized()

@ti.kernel
def init_camera():
    camera_orbit[None] = ti.Vector([0.0, 1.4]) 
    theta = 0.0; phi = 1.4; radius = 4.5
    x = radius * ti.sin(phi) * ti.cos(theta)
    y = radius * ti.cos(phi)
    z = radius * ti.sin(phi) * ti.sin(theta)
    camera_pos[None] = ti.Vector([x, y, z])

@ti.kernel
def update_camera(auto_pan: float):
    camera_orbit[None].x += auto_pan
    theta = camera_orbit[None].x
    phi = camera_orbit[None].y
    radius = 4.5
    x = radius * ti.sin(phi) * ti.cos(theta)
    y = radius * ti.cos(phi)
    z = radius * ti.sin(phi) * ti.sin(theta)
    camera_pos[None] = ti.Vector([x, y, z])

# --- RENDERER ---
@ti.kernel
def render_noisy_frame(time_seconds: float):
    cycle_duration = 20.0
    sun_angle = (time_seconds / cycle_duration) * 6.28318
    sun_dir = ti.Vector([ti.sin(sun_angle), ti.cos(sun_angle), 0.5]).normalized()

    cam = camera_pos[None]
    fwd = (look_at - cam).normalized()
    right = fwd.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    up = right.cross(fwd)

    for u, v in color_buffer:
        color_buffer[u, v] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        stiffness_map[u, v] = VACUUM_STIFFNESS 
        
        uv = ti.Vector([u / RES[0], v / RES[1]]) * 2.0 - 1.0
        uv.x *= ASPECT 

        FOV = 0.6
        rd = (fwd + uv.x * right * FOV + uv.y * up * FOV).normalized()
        
        sky_col, main_light_dir, main_light_col, ambient_col = get_sky_and_light(rd, sun_dir)
        
        obj_id, t_obj = intersect_objects(cam, rd)
        hit_terr, t_terr = raymarch_terrain(cam, rd)
        
        final_hit = 0; final_t = 10000.0
        if obj_id > 0 and t_obj < final_t: final_t = t_obj; final_hit = 1
        if hit_terr > 0 and t_terr < final_t: final_t = t_terr; final_hit = 2
        
        col = ti.Vector([0.0, 0.0, 0.0])
        beta = VACUUM_STIFFNESS
        
        if final_hit == 0:
            col = sky_col
            beta = VACUUM_STIFFNESS 
            color_buffer[u, v] = ti.Vector([col.r, col.g, col.b, 1.0]) 

        else:
            p = cam + final_t * rd
            norm = ti.Vector([0.0, 1.0, 0.0])
            base_col = ti.Vector([0.0, 0.0, 0.0])
            occ = 1.0
            
            if final_hit == 1:
                norm = get_obj_normal(p)
                occ = calc_ao(p, norm) 
                
                if obj_id == 1: # Marble
                    m_noise = fbm3(p * 6.0, 4)
                    m_val = pow(m_noise, 1.2) 
                    wrap = 0.5
                    diff = max(0.0, (norm.dot(main_light_dir) + wrap) / (1.0 + wrap))
                    sss_glow = ti.Vector([1.0, 0.8, 0.7]) * pow(diff, 2.0) * (1.0 - m_val) * 0.5
                    surface_col = ti.Vector([0.9, 0.9, 0.95]) * (0.8 + 0.2 * m_val) 
                    lit_marble = surface_col * diff + sss_glow
                    base_col = lit_marble * main_light_col + surface_col * ambient_col
                    beta = OBJECT_STIFFNESS
                    lit_col = base_col * occ
                    if ti.random() > 0.85: 
                         lit_col *= 4.0 
                         color_buffer[u, v] = ti.Vector([lit_col.r, lit_col.g, lit_col.b, 1.0])
                    stiffness_map[u, v] = beta
                    continue 

                elif obj_id == 2: # Gold
                    base_col = ti.Vector([0.6, 0.4, 0.1]) 
                    beta = OBJECT_STIFFNESS
                elif obj_id == 3: # Crystal
                    refract_dir = (rd + norm * 0.4).normalized() 
                    lens_col = ti.Vector([0.0, 0.0, 0.0])
                    if refract_dir.y < -0.05:
                        t_plane = (-1.5 - p.y) / refract_dir.y
                        if t_plane > 0.0:
                            p_virt = p + refract_dir * t_plane
                            virt_norm = get_terrain_normal(p_virt)
                            virt_diff = max(0.0, virt_norm.dot(main_light_dir))
                            virt_base = ti.Vector([0.05, 0.08, 0.05])
                            lens_col = virt_base * (ambient_col.norm() + virt_diff * 1.2)
                    else:
                        l_sky, _, _, _ = get_sky_and_light(refract_dir, sun_dir)
                        lens_col = l_sky
                    
                    fresnel = pow(1.0 - max(0.0, norm.dot(-rd)), 3.0)
                    reflect_dir = rd - 2.0 * norm.dot(rd) * norm
                    r_sky, _, _, _ = get_sky_and_light(reflect_dir, sun_dir)
                    base_col = lens_col * 0.8 + r_sky * fresnel * 0.5 + ti.Vector([0.02, 0.05, 0.1]) * 0.1
                    beta = 120.0 

            elif final_hit == 2: # Terrain
                norm = get_terrain_normal(p)
                beta = TERRAIN_STIFFNESS
                base_col = ti.Vector([0.05, 0.08, 0.05]) 

            diffuse = max(0.0, norm.dot(main_light_dir))
            s_obj, s_t = intersect_objects(p + norm*0.01, main_light_dir)
            if s_obj > 0: diffuse = 0.0
            
            lit_col = base_col * (ambient_col * occ + main_light_col * diffuse * occ)
            
            if ti.random() > 0.85: 
                 lit_col *= 4.0 
                 color_buffer[u, v] = ti.Vector([lit_col.r, lit_col.g, lit_col.b, 1.0]) 
            
            stiffness_map[u, v] = beta

@ti.kernel
def wts_relax_step(dt: float, parity: int):
    for i, j in relaxed_buffer_A:
        center = relaxed_buffer_A[i, j] if parity == 0 else relaxed_buffer_B[i, j]
        beta_c = stiffness_map[i, j]
        
        il, ir = max(0, i-1), min(RES[0]-1, i+1)
        jd, ju = max(0, j-1), min(RES[1]-1, j+1)
        
        n_l = relaxed_buffer_A[il, j] if parity==0 else relaxed_buffer_B[il, j]
        n_r = relaxed_buffer_A[ir, j] if parity==0 else relaxed_buffer_B[ir, j]
        n_u = relaxed_buffer_A[i, ju] if parity==0 else relaxed_buffer_B[i, ju]
        n_d = relaxed_buffer_A[i, jd] if parity==0 else relaxed_buffer_B[i, jd]
        
        b_l = stiffness_map[il, j]; b_r = stiffness_map[ir, j]
        b_u = stiffness_map[i, ju]; b_d = stiffness_map[i, jd]
        
        w_l = 1.0 if abs(beta_c - b_l) < 50.0 else 0.0
        w_r = 1.0 if abs(beta_c - b_r) < 50.0 else 0.0
        w_u = 1.0 if abs(beta_c - b_u) < 50.0 else 0.0
        w_d = 1.0 if abs(beta_c - b_d) < 50.0 else 0.0
        
        sum_neighbors = n_l * w_l + n_r * w_r + n_u * w_u + n_d * w_d
        total_weight = w_l + w_r + w_u + w_d
        
        stress = sum_neighbors - total_weight * center
        raw_force = stress / (max(0.01, beta_c))
        force = ti.math.tanh(raw_force * 0.1) * 15.0 
        
        mom = momentum[i, j]
        mom = mom * MOMENTUM_DAMPING + force * dt
        momentum[i, j] = mom
        
        new_val = center + mom
        new_val = ti.math.max(new_val, ti.Vector([0.0, 0.0, 0.0]))
        
        input_data = color_buffer[i, j]
        if input_data.w > 0.5:
             new_val = new_val * 0.6 + input_data.xyz * 0.4
        
        if parity == 0: relaxed_buffer_B[i, j] = new_val
        else: relaxed_buffer_A[i, j] = new_val

@ti.kernel
def temporal_accumulation(parity: int):
    for i, j in history_buffer:
        current = relaxed_buffer_A[i, j] if parity == 0 else relaxed_buffer_B[i, j]
        history = history_buffer[i, j]
        
        # AGGRESSIVE ANTI-GHOSTING
        # Only keep 20% of history to prevent tracers
        history_buffer[i, j] = history * 0.2 + current * 0.8

@ti.kernel
def bilateral_filter():
    for i, j in display_buffer:
        center_color = history_buffer[i, j]
        center_beta = stiffness_map[i, j]
        final_color = center_color
        total_weight = 1.0
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0: continue
                nx, ny = i + dx, j + dy
                if nx >= 0 and nx < RES[0] and ny >= 0 and ny < RES[1]:
                    neighbor_color = history_buffer[nx, ny]
                    neighbor_beta = stiffness_map[nx, ny]
                    beta_diff = abs(center_beta - neighbor_beta)
                    weight = ti.exp(-beta_diff * 0.8) 
                    final_color += neighbor_color * weight
                    total_weight += weight
                    
        filt = final_color / total_weight
        
        a = 2.51; b = 0.03; c = 2.43; d = 0.59; e = 0.14
        mapped = (filt * (a * filt + b)) / (filt * (c * filt + d) + e)
        mapped = ti.math.clamp(mapped, 0.0, 1.0)
        
        display_buffer[i, j] = ti.pow(mapped, 1.0/2.4)

# --- Run ---
init_camera()
gui = ti.GUI("WTS-RTX: Final No Tracers", RES)
current_sim_time = 0.0 

while gui.running:
    update_camera(0.002) 
    current_sim_time += 0.033 
    
    render_noisy_frame(current_sim_time)
    
    for _ in range(6): 
        p = frame_parity[None]
        wts_relax_step(0.2, p) 
        frame_parity[None] = 1 - p
        
    temporal_accumulation(frame_parity[None])
    bilateral_filter()
    
    gui.set_image(display_buffer)
    gui.show()