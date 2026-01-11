import taichi as ti
import math

try:
    ti.init(arch=ti.vulkan, device_memory_GB=4.0, offline_cache=True)
except:
    ti.init(arch=ti.cpu)

# --- CONFIG ---
RES = (1200, 600)
DT = 0.005 
STEPS = 10 
G = 9.81
L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0

SLR_BETA = 1.91 
SLR_GAIN = 1.2 

# [theta1, theta2, omega1, omega2]
state = ti.Vector.field(4, dtype=float, shape=2) 
trail_ptr = ti.field(dtype=int, shape=2)
trail_buffer = ti.Vector.field(2, dtype=float, shape=(2, 1024))
force_visual = ti.Vector.field(2, dtype=float, shape=2) # For debugging gravity

display_buffer = ti.Vector.field(3, dtype=float, shape=RES)

@ti.kernel
def init_sim():
    # START HORIZONTAL (90 degrees) to fix "Falling Up" confusion
    # They should clearly fall DOWN from here.
    start_theta = 1.57 # Pi/2
    
    for i in range(2):
        state[i] = ti.Vector([start_theta, 0.0, 0.0, 0.0])
        trail_ptr[i] = 0

@ti.func
def get_accel(s, mode: int):
    t1 = s[0]
    t2 = s[1]
    w1 = s[2]
    w2 = s[3]
    
    # --- GRAVITY FORCE CALC ---
    # Standard: F ~ -sin(t)
    gravity_force = -G * (2 * M1 + M2) * ti.sin(t1)
    
    if mode == 1:
        # APH SLR: F ~ -|t|^1.91 * sign(t)
        # This force pulls towards 0 much harder at large angles
        sign1 = 1.0 if t1 >= 0 else -1.0
        # Clamp to prevent numeric explosion at extreme angles
        mag = ti.min(ti.pow(ti.abs(t1), SLR_BETA), 50.0) 
        gravity_force = -G * (2 * M1 + M2) * sign1 * mag * SLR_GAIN
    
    # Store for visualization
    if mode == 0: force_visual[0] = ti.Vector([gravity_force * 0.01, 0.0])
    if mode == 1: force_visual[1] = ti.Vector([gravity_force * 0.01, 0.0])

    num1 = gravity_force
    num2 = -M2 * G * ti.sin(t1 - 2 * t2) 
    num3 = -2 * ti.sin(t1 - t2) * M2 * (w2*w2 * L2 + w1*w1 * L1 * ti.cos(t1 - t2))
    den = L1 * (2 * M1 + M2 - M2 * ti.cos(2 * t1 - 2 * t2))
    
    a1 = (num1 + num2 + num3) / den

    # Bob 2
    num1_2 = 2 * ti.sin(t1 - t2)
    num2_2 = (w1*w1 * L1 * (M1 + M2))
    g_term2 = G * (M1 + M2) * ti.cos(t1)
    num3_2 = w2*w2 * L2 * M2 * ti.cos(t1 - t2)
    den_2 = L2 * (2 * M1 + M2 - M2 * ti.cos(2 * t1 - 2 * t2))
    
    a2 = (num1_2 * (num2_2 + g_term2 + num3_2)) / den_2
    
    return ti.Vector([a1, a2])

@ti.kernel
def update(dt: float):
    for i in range(2):
        s = state[i]
        acc = get_accel(s, i)
        s[2] += acc[0] * dt
        s[3] += acc[1] * dt
        s[2] *= 0.999 
        s[3] *= 0.999
        s[0] += s[2] * dt
        s[1] += s[3] * dt
        state[i] = s
        
        if int(ti.floor(dt * 1000)) % 4 == 0:
            ptr = trail_ptr[i]
            x1 = L1 * ti.sin(s[0])
            y1 = -L1 * ti.cos(s[0])
            x2 = x1 + L2 * ti.sin(s[1])
            y2 = y1 - L2 * ti.cos(s[1])
            trail_buffer[i, ptr % 1024] = ti.Vector([x2, y2])
            trail_ptr[i] = ptr + 1

@ti.kernel
def render():
    for x, y in display_buffer:
        display_buffer[x, y] = ti.Vector([0.05, 0.05, 0.1])
        if abs(x - RES[0]//2) < 2: display_buffer[x, y] = 0.3

    for i in range(2):
        s = state[i]
        cx = RES[0] * 0.25 if i == 0 else RES[0] * 0.75
        cy = RES[1] * 0.5
        scale = 100.0
        
        # Trail
        ptr = trail_ptr[i]
        for j in range(1024):
            idx = (ptr - j - 1) % 1024
            if idx < 0: break
            pos = trail_buffer[i, idx]
            tx = cx + scale * pos[0]
            ty = cy + scale * pos[1]
            intensity = 1.0 - float(j) / 1024.0
            
            # Simple pixel draw
            if tx > 0 and tx < RES[0] and ty > 0 and ty < RES[1]:
                col = ti.Vector([1.0, 0.5, 0.2]) if i == 0 else ti.Vector([0.2, 1.0, 0.5])
                display_buffer[int(tx), int(ty)] = col * intensity

init_sim()
gui = ti.GUI("APH Chaos Control: Standard vs SLR", RES)

while gui.running:
    for _ in range(STEPS): update(DT)
    render()
    img = display_buffer.to_numpy()
    gui.set_image(img)
    
    for i in range(2):
        s = state[i].to_numpy()
        cx = 0.25 if i == 0 else 0.75
        cy = 0.5
        scale = 100.0 / RES[0]
        asp = RES[0]/RES[1]
        
        o = [cx, cy]
        p1 = [cx + L1 * math.sin(s[0]) * scale, cy + L1 * math.cos(s[0]) * scale * asp]
        p2 = [p1[0] + L2 * math.sin(s[1]) * scale, p1[1] + L2 * math.cos(s[1]) * scale * asp]
        
        col = 0xFF8844 if i == 0 else 0x44FF88
        gui.line(o, p1, radius=4, color=0xFFFFFF)
        gui.line(p1, p2, radius=4, color=0xFFFFFF)
        gui.circle(p1, radius=8, color=col)
        gui.circle(p2, radius=10, color=col)
        
        # Draw Force Vector (Yellow)
        # Shows which way Gravity is pulling Theta1
        fv = force_visual[i].to_numpy()
        # Visualize torque as a horizontal arrow from the pivot
        f_end = [cx + fv[0], cy + 0.05]
        gui.arrow([cx, cy + 0.05], f_end, radius=2, color=0xFFFF00)

    gui.text("Standard: Falls Slowly", pos=(0.15, 0.9), font_size=20, color=0xFF8844)
    gui.text("APH SLR: Snaps Down", pos=(0.65, 0.9), font_size=20, color=0x44FF88)
    gui.show()