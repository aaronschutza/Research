import taichi as ti
import math

# --- ROBUST INIT ---
try:
    ti.init(arch=ti.cuda)
except:
    try:
        ti.init(arch=ti.vulkan)
    except:
        ti.init(arch=ti.cpu)

# --- Configuration ---
RES = (1280, 720) 
MOMENTUM_DAMPING = 0.90   
VACUUM_STIFFNESS = 0.1    
GEOMETRY_STIFFNESS = 50.0 

# --- Fields ---
color_buffer = ti.Vector.field(4, dtype=float, shape=RES) 
stiffness_map = ti.field(dtype=float, shape=RES) 
momentum = ti.Vector.field(3, dtype=float, shape=RES) 

relaxed_buffer_A = ti.Vector.field(3, dtype=float, shape=RES)
relaxed_buffer_B = ti.Vector.field(3, dtype=float, shape=RES)
history_buffer = ti.Vector.field(3, dtype=float, shape=RES)
display_buffer = ti.Vector.field(3, dtype=float, shape=RES)

frame_parity = ti.field(dtype=int, shape=())

# --- Camera ---
camera_orbit = ti.Vector.field(2, dtype=float, shape=()) 
camera_pos = ti.Vector.field(3, dtype=float, shape=())
look_at = ti.Vector([0.0, 1.0, 0.0])

@ti.func
def intersect_box(ro, rd, box_min, box_max):
    t_min = (box_min - ro) / rd
    t_max = (box_max - ro) / rd
    t1 = min(t_min, t_max)
    t2 = max(t_min, t_max)
    t_near = max(t1.x, max(t1.y, t1.z))
    t_far = min(t2.x, min(t2.y, t2.z))
    hit = 0
    t = -1.0
    if t_near < t_far and t_far > 0:
        hit = 1
        t = t_near if t_near > 0 else t_far
    return hit, t

@ti.func
def get_box_normal(p, box_min, box_max):
    c = (box_min + box_max) * 0.5
    d = (p - c).normalized()
    n = ti.Vector([0.0, 0.0, 0.0])
    if abs(d.x) > abs(d.y) and abs(d.x) > abs(d.z):
        n = ti.Vector([1.0 if d.x > 0 else -1.0, 0.0, 0.0])
    elif abs(d.y) > abs(d.z):
        n = ti.Vector([0.0, 1.0 if d.y > 0 else -1.0, 0.0])
    else:
        n = ti.Vector([0.0, 0.0, 1.0 if d.z > 0 else -1.0])
    return n

@ti.kernel
def init_camera():
    camera_orbit[None] = ti.Vector([0.5, 1.0]) 
    theta = 0.5; phi = 1.0; radius = 6.0
    x = radius * ti.sin(phi) * ti.cos(theta)
    y = radius * ti.cos(phi)
    z = radius * ti.sin(phi) * ti.sin(theta)
    camera_pos[None] = ti.Vector([x, y, z])

@ti.kernel
def update_camera(mouse_dx: float, mouse_dy: float):
    camera_orbit[None].x -= mouse_dx * 5.0
    camera_orbit[None].y = max(0.01, min(1.5, camera_orbit[None].y + mouse_dy * 5.0))
    theta = camera_orbit[None].x; phi = camera_orbit[None].y; radius = 6.0
    x = radius * ti.sin(phi) * ti.cos(theta)
    y = radius * ti.cos(phi)
    z = radius * ti.sin(phi) * ti.sin(theta)
    camera_pos[None] = ti.Vector([x, y, z])

@ti.kernel
def render_noisy_frame(t: float):
    light_pos = ti.Vector([ti.sin(t * 0.5) * 3.0, 4.0, ti.cos(t * 0.5) * 3.0])
    cam = camera_pos[None]
    fwd = (look_at - cam).normalized()
    right = fwd.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    up = right.cross(fwd)

    for u, v in color_buffer:
        color_buffer[u, v] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        stiffness_map[u, v] = VACUUM_STIFFNESS 
        
        uv = ti.Vector([u / RES[0], v / RES[1]]) * 2.0 - 1.0
        rd = (fwd + uv.x * right * 0.8 + uv.y * up * 0.6).normalized()
        
        box_min = ti.Vector([-1.0, 0.0, -1.0]); box_max = ti.Vector([1.0, 2.0, 1.0])
        hit_box, t_box = intersect_box(cam, rd, box_min, box_max)
        t_floor = -cam.y / rd.y if rd.y < -0.001 else -1.0
        
        hit_obj = 0; final_t = 10000.0
        if t_floor > 0 and t_floor < final_t: final_t = t_floor; hit_obj = 2 
        if hit_box and t_box < final_t: final_t = t_box; hit_obj = 1 
            
        beta = VACUUM_STIFFNESS
        if hit_obj > 0:
            p = cam + final_t * rd
            norm = ti.Vector([0.0, 1.0, 0.0])
            col = ti.Vector([0.0, 0.0, 0.0])
            
            if hit_obj == 1: # BOX
                norm = get_box_normal(p, box_min, box_max)
                beta = GEOMETRY_STIFFNESS
                col = ti.Vector([0.9, 0.2, 0.1])
            elif hit_obj == 2: # FLOOR
                beta = 5.0
                if (int(p.x*2) + int(p.z*2)) % 2 == 0: col = ti.Vector([0.8, 0.8, 0.8])
                else: col = ti.Vector([0.4, 0.4, 0.4])

            l_dir = (light_pos - p).normalized()
            s_hit, s_t = intersect_box(p + norm * 0.01, l_dir, box_min, box_max)
            diffuse = max(0.0, norm.dot(l_dir))
            if s_hit and s_t < (light_pos - p).norm(): diffuse = 0.0
            
            # Sparse Sampling (85% Noise)
            if ti.random() > 0.85: 
                 final_col = col * (diffuse + 0.05) * 6.0
                 color_buffer[u, v] = ti.Vector([final_col.r, final_col.g, final_col.b, 1.0]) 
        
        stiffness_map[u, v] = beta

@ti.kernel
def wts_relax_step(dt: float, parity: int):
    # Standard WTS Step
    for i, j in relaxed_buffer_A:
        center = ti.Vector([0.0, 0.0, 0.0])
        if parity == 0: center = relaxed_buffer_A[i, j]
        else: center = relaxed_buffer_B[i, j]
            
        il, ir = max(0, i-1), min(RES[0]-1, i+1)
        jd, ju = max(0, j-1), min(RES[1]-1, j+1)
        
        sum_neighbors = ti.Vector([0.0, 0.0, 0.0])
        if parity == 0:
             sum_neighbors = relaxed_buffer_A[il, j] + relaxed_buffer_A[ir, j] + \
                             relaxed_buffer_A[i, ju] + relaxed_buffer_A[i, jd]
        else:
             sum_neighbors = relaxed_buffer_B[il, j] + relaxed_buffer_B[ir, j] + \
                             relaxed_buffer_B[i, ju] + relaxed_buffer_B[i, jd]
        
        stress = sum_neighbors - 4.0 * center
        beta = stiffness_map[i, j]
        
        # --- FIXED LINE: Define raw_force ---
        raw_force = stress / (max(0.01, beta))
        
        # Soft Clamp to prevent explosion
        force = ti.math.tanh(raw_force * 0.1) * 10.0 
        
        mom = momentum[i, j]
        mom = mom * MOMENTUM_DAMPING + force * dt
        momentum[i, j] = mom
        new_val = center + mom
        new_val = ti.math.max(new_val, ti.Vector([0.0, 0.0, 0.0]))
        
        input_data = color_buffer[i, j]
        if input_data.w > 0.5:
             new_val = new_val * 0.7 + input_data.xyz * 0.3
        
        if parity == 0: relaxed_buffer_B[i, j] = new_val
        else: relaxed_buffer_A[i, j] = new_val

@ti.kernel
def temporal_accumulation(parity: int):
    for i, j in history_buffer:
        current = ti.Vector([0.0, 0.0, 0.0])
        if parity == 0: current = relaxed_buffer_A[i, j]
        else: current = relaxed_buffer_B[i, j]
            
        history = history_buffer[i, j]
        # Blend
        new_history = history * 0.8 + current * 0.2
        history_buffer[i, j] = new_history

@ti.kernel
def bilateral_filter():
    for i, j in display_buffer:
        center_color = history_buffer[i, j]
        center_beta = stiffness_map[i, j]
        
        final_color = center_color
        total_weight = 1.0
        
        # 3x3 Kernel
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0: continue
                
                nx, ny = i + dx, j + dy
                if nx >= 0 and nx < RES[0] and ny >= 0 and ny < RES[1]:
                    neighbor_color = history_buffer[nx, ny]
                    neighbor_beta = stiffness_map[nx, ny]
                    
                    beta_diff = abs(center_beta - neighbor_beta)
                    weight = ti.exp(-beta_diff * 0.5) 
                    
                    final_color += neighbor_color * weight
                    total_weight += weight
        
        display_buffer[i, j] = final_color / total_weight

# --- Run ---
init_camera()
gui = ti.GUI("WTS-RTX Hybrid (Denoised)", RES)
frame_id = 0
prev_mouse = gui.get_cursor_pos()

while gui.running:
    curr_mouse = gui.get_cursor_pos()
    if gui.is_pressed(ti.GUI.LMB):
        dx = curr_mouse[0] - prev_mouse[0]
        dy = curr_mouse[1] - prev_mouse[1]
        update_camera(dx, dy)
    prev_mouse = curr_mouse

    render_noisy_frame(frame_id * 0.02)
    
    for _ in range(2): 
        p = frame_parity[None]
        wts_relax_step(0.2, p) 
        frame_parity[None] = 1 - p
        
    temporal_accumulation(frame_parity[None])
    bilateral_filter()
    
    gui.set_image(display_buffer)
    gui.show()
    frame_id += 1