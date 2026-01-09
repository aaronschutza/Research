import taichi as ti
import math
import random

# --- ROBUST INIT ---
try:
    ti.init(arch=ti.vulkan, device_memory_GB=4.0, offline_cache=True)
except:
    try:
        ti.init(arch=ti.opengl, device_memory_GB=3.5, offline_cache=True)
    except:
        ti.init(arch=ti.cpu)

# --- CONFIGURATION ---
RES = (1280, 720)
ASPECT = RES[0] / RES[1]
N_STARS = 64 # High particle count

# --- PHYSICS CONSTANTS ---
MOMENTUM_DAMPING = 0.96 
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

camera_orbit = ti.Vector.field(2, dtype=float, shape=()) 
camera_pos = ti.Vector.field(3, dtype=float, shape=())
look_at = ti.Vector([0.0, 1.0, 0.0]) 

# Orbital fields
star_pos = ti.Vector.field(3, dtype=float, shape=N_STARS)
star_vel = ti.Vector.field(3, dtype=float, shape=N_STARS)
star_props = ti.Vector.field(3, dtype=float, shape=N_STARS)

global_env = ti.Vector.field(3, dtype=float, shape=())
seeds = ti.Vector.field(3, dtype=float, shape=())
frame_counter = ti.field(dtype=int, shape=())

# Genome
palette = ti.Vector.field(3, dtype=float, shape=8) 
genes = ti.field(dtype=float, shape=8) 

# --- MATH HELPERS ---
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
def hash22(p):
    p3 = fract(ti.Vector([p.x, p.y, p.x]) * ti.Vector([0.1031, 0.1030, 0.0973]))
    p3 += p3.dot(p3.zxy + 33.33)
    return fract((p3.xx + p3.yz) * p3.zy)

@ti.func
def hash33(p):
    p3 = fract(p * ti.Vector([0.1031, 0.1030, 0.0973]))
    p3 += p3.dot(p3.yxz + 33.33)
    return fract((p3.xxy + p3.yxx) * p3.zyx)

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
def noise3(p):
    i = ti.floor(p); f = fract(p); u = f * f * (3.0 - 2.0 * f)
    a = hash33(i + ti.Vector([0.0, 0.0, 0.0])); b = hash33(i + ti.Vector([1.0, 0.0, 0.0]))
    c = hash33(i + ti.Vector([0.0, 1.0, 0.0])); d = hash33(i + ti.Vector([1.0, 1.0, 0.0]))
    e = hash33(i + ti.Vector([0.0, 0.0, 1.0])); f_val = hash33(i + ti.Vector([1.0, 0.0, 1.0]))
    g = hash33(i + ti.Vector([0.0, 1.0, 1.0])); h = hash33(i + ti.Vector([1.0, 1.0, 1.0]))
    return mix(mix(mix(a.x, b.x, u.x), mix(c.x, d.x, u.x), u.y), mix(mix(e.x, f_val.x, u.x), mix(g.x, h.x, u.x), u.y), u.z)

@ti.func
def rotate(v, a):
    c, s = ti.cos(a), ti.sin(a)
    return ti.Vector([c * v.x - s * v.y, s * v.x + c * v.y])

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

# --- SCENE SETUP ---
@ti.func
def get_attractor_positions(time):
    fractal_pos = ti.Vector([0.0, 1.0, 0.0])
    egg_pos = ti.Vector([3.5 * ti.sin(time*0.2), 2.5 + ti.cos(time*0.15), 3.5 * ti.cos(time*0.25)])
    clock_pos = ti.Vector([4.0 * ti.sin(-time*0.15 + 3.14), 2.0 + ti.sin(time*0.1), 4.0 * ti.cos(-time*0.15 + 3.14)])
    return fractal_pos, egg_pos, clock_pos

@ti.func
def sd_melting_clock(p, time):
    bend = 0.2
    p.y += bend * p.x * p.x
    p.xz = rotate(p.xz, p.y * 0.2 + time * 0.1)
    return sd_cylinder(p, 0.05, 1.2)

@ti.func
def sd_fractal(pos, time):
    melt_strength = genes[0]
    base_power = genes[1]
    z = pos
    
    melt_offset = ti.sin(z.y * 1.5 - time * 0.3) * melt_strength
    z.x += melt_offset; z.z += melt_offset * 0.5
    
    dr = 1.0; r = 0.0; trap = 1e10
    power = base_power + ti.sin(time * 0.2) * 0.5
    
    for i in range(5):
        r = z.norm()
        if r > 4.0: break
        trap = min(trap, ti.abs(z.y) + 0.1/r) 
        theta = ti.acos(z.z / r)
        phi = ti.atan2(z.y, z.x)
        dr =  ti.pow(r, power - 1.0) * power * dr + 1.0
        zr = ti.pow(r, power)
        theta = theta * power; phi = phi * power
        z = zr * ti.Vector([ti.sin(theta)*ti.cos(phi), ti.sin(phi)*ti.sin(theta), ti.cos(theta)])
        z += pos
        rot_xz = rotate(z.xz, time * 0.05 + z.y * 0.2)
        z.x = rot_xz.x; z.z = rot_xz.y
        
    return 0.5 * ti.log(r) * r / dr, trap

@ti.func
def get_terrain_height(xz):
    roughness = genes[2]
    checkers = ti.abs(ti.sin(xz.x * 0.5)*ti.sin(xz.y * 0.5))
    h = -4.0
    if roughness > 0.5: h += noise2(xz * 0.2) * 3.0
    else: h += noise2(xz * 0.05) * 1.5 + checkers * 0.2
    return h

@ti.func
def map_scene(p, time):
    frac_p, egg_p, clock_p = get_attractor_positions(time)
    
    d_obj, trap = sd_fractal(p - frac_p, time)
    d_egg = (p - egg_p).norm() - 0.9
    
    rel_clock = p - clock_p
    rel_clock.yz = rotate(rel_clock.yz, 0.5) 
    rel_clock.xz = rotate(rel_clock.xz, time * 0.15) 
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
    for i in range(N_STARS):
        d_s = sd_sphere(p - star_pos[i], 0.04) 
        if d_s < d_stars: d_stars = d_s; star_idx = i
            
    d_final = min(d_scene, d_stars)
    final_trap = trap
    if d_stars < d_scene: final_trap = 3000.0 + float(star_idx)
    return d_final, final_trap

@ti.func
def get_normal(p, time, id):
    e = 0.005
    n = ti.Vector([0.0, 1.0, 0.0])
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
            h0 - get_terrain_height(p.xz + ti.Vector([0.1, 0])),
            0.1,
            h0 - get_terrain_height(p.xz + ti.Vector([0, 0.1]))
        ]).normalized()
    return n

@ti.func
def get_gradient_physics(p, time):
    e = 0.005
    d0, _ = map_scene(p, time)
    dx, _ = map_scene(p + ti.Vector([e, 0, 0]), time)
    dy, _ = map_scene(p + ti.Vector([0, e, 0]), time)
    dz, _ = map_scene(p + ti.Vector([0, 0, e]), time)
    return ti.Vector([dx-d0, dy-d0, dz-d0]).normalized()

# --- SKY ---
@ti.func
def get_sky_background(rd, sun_dir):
    sky_t = smoothstep(-0.1, 0.3, rd.y)
    col = mix(palette[1], palette[0], sky_t)
    sun_dot = max(0.0, rd.dot(sun_dir))
    col += palette[6] * ti.pow(sun_dot, 80.0) * 2.0 
    return col

# --- MATERIAL SHADER ---
@ti.func
def get_material(id, p, n, trap, time):
    albedo = ti.Vector([0.0, 0.0, 0.0])
    rough = 0.5; metal = 0.0; emit = ti.Vector([0.0, 0.0, 0.0])
    
    if id == 1: 
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

        elif trap > 1500.0: # Clock
            _, _, clock_p = get_attractor_positions(time)
            local_p = p - clock_p
            local_p.yz = rotate(local_p.yz, 0.5)
            local_p.xz = rotate(local_p.xz, time * 0.15)
            bend = 0.2; local_p.y -= bend * local_p.x * local_p.x
            dist = local_p.xz.norm(); angle = ti.atan2(local_p.z, local_p.x)
            
            albedo = ti.Vector([0.9, 0.9, 0.85]); rough = 0.3
            if dist > 1.1: 
                albedo = ti.Vector([0.8, 0.6, 0.2]); metal = 1.0; rough = 0.1
            else:
                marks = ti.abs(ti.sin(angle * 6.0))
                if dist > 0.9 and marks > 0.9: albedo = ti.Vector([0.1, 0.1, 0.1]) 
                m_ang = -(time * 2.0); m_dir = ti.Vector([ti.cos(m_ang), ti.sin(m_ang)])
                if local_p.xz.dot(m_dir) > dist * 0.95 and dist < 0.8: albedo = ti.Vector([0.0, 0.0, 0.0])

        elif trap > 500.0: # Egg
            albedo = ti.Vector([0.95, 0.95, 0.92]); rough = 0.1
            if noise3(p * 5.0) > 0.65: albedo = palette[7] 
        else: 
            mix_val = smoothstep(0.0, 1.5, trap)
            if trap < 0.1: emit = palette[7] * (1.0/(trap+0.01)) * 0.05

    elif id == 2: # Terrain
        check = smoothstep(0.4, 0.6, fract(p.x * 0.3 + 0.2 * ti.sin(p.z*0.1)) - fract(p.z * 0.3))
        dist_fade = smoothstep(20.0, 0.0, p.norm())
        albedo = mix(palette[2], mix(palette[2], palette[3], abs(check)), dist_fade)
        rough = 0.9
        
    elif id == 3: # Liquid
        albedo = palette[7]
        rough = 0.02; metal = 0.5
        
    return albedo, rough, metal, emit

@ti.func
def calc_softshadow(ro, rd, tmin, tmax, k, dither, time):
    res = 1.0; t = tmin + dither * 0.05
    for i in range(12): 
        d1, _ = map_scene_render(ro + rd * t, time)
        if d1 < 0.001: res = 0.0; break
        res = min(res, k * d1 / t)
        t += d1
        if t > tmax: break
    return res

@ti.kernel
def update_star_physics(dt: float, time: float):
    frac_p, egg_p, clock_p = get_attractor_positions(time)
    scene_center = ti.Vector([0.0, 2.0, 0.0])
    
    for i in range(N_STARS):
        p = star_pos[i]; v = star_vel[i]
        force = ti.Vector([0.0, 0.0, 0.0])
        
        # 1. Attractors (Gravity) - Radial only
        diff1 = frac_p - p
        force += diff1.normalized() * (12.0 / (diff1.norm_sqr() + 0.2))
        
        diff2 = egg_p - p
        force += diff2.normalized() * (10.0 / (diff2.norm_sqr() + 0.2))
        
        diff3 = clock_p - p
        force += diff3.normalized() * (10.0 / (diff3.norm_sqr() + 0.2))
        
        # 2. Confinement (Keep in bounds, softer)
        d_org = (p - scene_center).norm()
        if d_org > 8.0: force += -(p - scene_center).normalized() * (d_org - 8.0) * 4.0
        
        # 3. Boids Repulsion (Prevent Stacking but allow trails)
        for j in range(N_STARS):
            if i != j:
                diff_b = p - star_pos[j]
                dist_b_sq = diff_b.norm_sqr()
                if dist_b_sq < 0.8: # Short range repulsion
                    force += diff_b.normalized() * (2.0 / (dist_b_sq + 0.1))
            
        v += force * dt
        v *= 0.995 # Very low drag for orbital mechanics
        p += v * dt
        
        if p.y < -1.5: p.y = -1.5; v.y *= -0.8
        
        # --- COLLISION PHYSICS ---
        dist, _ = map_scene(p, time)
        if dist < 0.1:
            normal = get_gradient_physics(p, time)
            v_normal = v.dot(normal)
            if v_normal < 0.0:
                v = (v - 1.8 * v_normal * normal) + normal * 0.5 
            p += normal * (0.12 - dist) 
        
        star_pos[i] = p; star_vel[i] = v

@ti.kernel
def render_pass(time: float, frame: int):
    sun_angle = 1.5 + time * genes[4] * 0.05
    sun_dir = ti.Vector([ti.cos(sun_angle), ti.sin(sun_angle) * 0.4 + 0.1, ti.sin(sun_angle)]).normalized()
    
    cam = camera_pos[None]
    fwd = (look_at - cam).normalized()
    right = fwd.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    up = right.cross(fwd)
    
    jit = ti.Vector([(float(frame%2)-0.5)*0.5, (float((frame//2)%2)-0.5)*0.5])
    dither = hash22(ti.Vector([float(frame%1024), 0.0])).x

    for u, v in color_buffer:
        uv = (ti.Vector([u + jit.x, v + jit.y]) / ti.Vector([RES[0], RES[1]]) * 2.0 - 1.0) * ti.Vector([ASPECT, 1.0])
        rd = (fwd + uv.x * right * FOV_SCALE + uv.y * up * FOV_SCALE).normalized()
        
        t = 0.0; t_final = 10000.0; hit_id = 0; trap_final = 0.0
        
        for i in range(80):
            p = cam + rd * t
            d, trap = map_scene_render(p, time)
            if d < 0.002:
                t_final = t; hit_id = 1; trap_final = trap
                break
            t += d
            if t > 70.0: break
            
        if (hit_id == 0 or t > 70.0) and rd.y < 0.2:
            t = max(0.0, (4.0 - cam.y)/rd.y if rd.y < 0 else 0)
            for i in range(64):
                p = cam + rd * t
                h = get_terrain_height(p.xz)
                d = p.y - h
                if d < 0.01 * t: 
                    if t < t_final: t_final = t; hit_id = 2
                    break
                t += d * 0.6 
                if t > 200.0: break
                
        water_h = genes[3]
        if rd.y < -0.001:
            t_w = (water_h - cam.y) / rd.y
            if t_w > 0.0 and t_w < t_final: t_final = t_w; hit_id = 3
            
        col = get_sky_background(rd, sun_dir)
        refl = ti.Vector([0.0, 0.0, 0.0])
        emit = ti.Vector([0.0, 0.0, 0.0])
        fluid = 0.0
        
        if hit_id > 0:
            p = cam + t_final * rd
            norm = ti.Vector([0.0, 1.0, 0.0])
            
            if hit_id == 1: norm = get_normal(p, time, 1)
            elif hit_id == 2: norm = get_normal(p, time, 2)
            elif hit_id == 3:
                warp = momentum[u,v].x * 0.15
                norm = ti.Vector([warp, 1.0, warp]).normalized()
                n_det = noise2(p.xz * 2.0 + time) * 0.05
                norm = (norm + ti.Vector([n_det, 0, n_det])).normalized()
                fluid = 1.0

            alb, rough, met, em = get_material(hit_id, p, norm, trap_final, time)
            
            # --- FRACTAL REFRACTION ---
            if hit_id == 1 and trap_final < 500.0:
                ior = 0.75 
                ref_dir = rd 
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
                NdotL = max(0.0, norm.dot(sun_dir))
                shadow = 1.0
                if hit_id != 3: 
                    shadow = calc_softshadow(p + norm * 0.05, sun_dir, 0.1, 15.0, 12.0, dither, time)
                
                # --- LIGHTING FROM MANY STARS ---
                star_acc = ti.Vector([0.0, 0.0, 0.0])
                if hit_id != 3:
                    for i in range(N_STARS):
                        l_pos = star_pos[i]
                        to_l = l_pos - p
                        dist_sq = to_l.dot(to_l)
                        if dist_sq < 100.0:
                            dist = ti.sqrt(dist_sq)
                            L = to_l / dist
                            atten = 1.0 / (1.0 + dist_sq * 0.5)
                            c_idx = int(star_props[i].z)
                            s_col = ti.Vector([1.0, 1.0, 1.0])
                            if c_idx == 0: s_col = ti.Vector([1.0, 0.1, 0.1])
                            elif c_idx == 1: s_col = ti.Vector([0.1, 1.0, 0.1])
                            elif c_idx == 2: s_col = ti.Vector([0.2, 0.5, 1.0])
                            elif c_idx == 3: s_col = ti.Vector([1.0, 0.8, 0.2])
                            elif c_idx == 4: s_col = ti.Vector([1.0, 0.2, 0.8])
                            star_acc += s_col * max(0.0, norm.dot(L)) * atten

                H = (sun_dir - rd).normalized()
                spec = pow(max(0.0, norm.dot(H)), 60.0 * (1.0-rough)) * shadow
                
                col = alb * (0.2 + NdotL * shadow * palette[6] + star_acc) + spec + em
                refl = get_sky_background(ti.math.reflect(rd, norm), sun_dir) * (1.0 - rough) * met
                emit = em 
            
            fog = 1.0 - ti.exp(-t_final * genes[5] * 0.01)
            col = mix(col, palette[1], fog)
            refl = mix(refl, ti.Vector([0.0,0.0,0.0]), fog)

        color_buffer[u, v] = ti.Vector([col.x, col.y, col.z, fluid])
        reflect_buffer[u, v] = ti.Vector([refl.x, refl.y, refl.z, t_final])
        hazard_buffer[u, v] = emit

@ti.kernel
def physics_update(dt: float, parity: int):
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
        history_buffer[i,j] = mix(hist, curr, 0.08)

@ti.kernel
def composite():
    for i, j in display_buffer:
        col = history_buffer[i, j]
        bloom = ti.Vector([0.0, 0.0, 0.0])
        for x, y in ti.static([(-2,-2), (-2,2), (2,-2), (2,2), (0,0)]):
             ix, iy = clamp(i+x, 0, RES[0]-1), clamp(j+y, 0, RES[1]-1)
             bloom += hazard_buffer[ix, iy]
        col += bloom * 0.1
        col = col / (col + 1.0)
        col = ti.pow(col, 1.0/2.2)
        display_buffer[i, j] = col

@ti.kernel
def init_genome(p0: ti.types.vector(3, float), p1: ti.types.vector(3, float), p2: ti.types.vector(3, float), p3: ti.types.vector(3, float), p4: ti.types.vector(3, float), p5: ti.types.vector(3, float), p6: ti.types.vector(3, float), p7: ti.types.vector(3, float), g0: float, g1: float, g2: float, g3: float, g4: float, g5: float):
    palette[0] = p0; palette[1] = p1; palette[2] = p2; palette[3] = p3
    palette[4] = p4; palette[5] = p5; palette[6] = p6; palette[7] = p7
    genes[0] = g0; genes[1] = g1; genes[2] = g2; genes[3] = g3; genes[4] = g4; genes[5] = g5
    seeds[None] = ti.Vector([ti.random(), ti.random(), ti.random()])
    
    frac_p, _, _ = get_attractor_positions(0.0)
    for i in range(N_STARS):
        star_props[i] = ti.Vector([0.0, 0.0, float(i%5)]) # 5 colors
        
        # CHAOS INJECTION: Spawn in random cloud around center
        rand_dir = ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5]).normalized()
        radius = 2.0 + ti.random() * 4.0
        star_pos[i] = frac_p + rand_dir * radius
        
        # Random initial velocity
        star_vel[i] = ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5]) * 4.0

def generate_dream():
    theme = random.choice(['dali', 'dark', 'neon', 'pastel'])
    if theme == 'dali':
        init_genome([0.0,0.0,0.2], [0.1,0.4,0.7], [0.8,0.6,0.3], [0.05,0.1,0.3], [0.9,0.7,0.1], [0.8,0.2,0.1], [1.0,0.9,0.8], [0.1,0.5,1.0], 0.5, 8.0, 0.0, -2.5, 1.0, 1.0)
    elif theme == 'dark':
        init_genome([0.0,0.0,0.0], [0.1,0.0,0.0], [0.05,0.05,0.05], [0.3,0.0,0.0], [0.1,0.1,0.1], [0.5,0.5,0.5], [0.8,0.0,0.0], [1.0,0.0,0.0], 0.3, 6.0, 1.0, -2.0, 2.0, 0.8)
    elif theme == 'neon':
        init_genome([0.1,0.0,0.2], [0.0,0.5,0.5], [0.1,0.0,0.2], [0.1,0.0,0.3], [0.0,1.0,1.0], [1.0,0.0,1.0], [1.0,0.0,0.5], [0.0,1.0,0.0], 0.7, 9.0, 0.0, -3.0, 1.5, 0.5)
    else:
        init_genome([0.8,0.9,1.0], [1.0,0.8,0.8], [0.9,0.8,0.7], [0.8,0.9,1.0], [0.7,0.9,0.8], [0.9,0.7,0.8], [1.0,1.0,0.9], [1.0,1.0,1.0], 0.4, 7.0, 0.0, -2.0, 0.8, 1.2)

generate_dream()
gui = ti.GUI("Dali Orbital Chaos", RES)
sim_time = 0.0 

while gui.running:
    camera_orbit[None].x += 0.002
    theta = camera_orbit[None].x; phi = 1.35; r = 9.0 
    bob = math.sin(sim_time * 0.3) * 0.3
    camera_pos[None] = ti.Vector([r*math.sin(phi)*math.cos(theta), r*math.cos(phi) + bob + 1.0, r*math.sin(phi)*math.sin(theta)])
    
    sim_time += 0.03; frame_counter[None] += 1
    global_env[None] = ti.Vector([sim_time, 0, 0])
    
    update_star_physics(0.05, sim_time)
    render_pass(sim_time, frame_counter[None])
    
    p = frame_parity[None]
    physics_update(0.1, p) 
    frame_parity[None] = 1 - p
    
    resolve(frame_parity[None])
    composite()
    
    gui.set_image(display_buffer)
    gui.show()