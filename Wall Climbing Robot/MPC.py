import mujoco
import mujoco.viewer
import numpy as np
import time
import casadi as ca
import matplotlib.pyplot as plt

# LOAD MODEL
model = mujoco.MjModel.from_xml_path("robot.xml")
data  = mujoco.MjData(model)
dt = float(model.opt.timestep)

# IDs
ls = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tip")
rs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_tip")

jid_hipR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_R")
jid_hipL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")

qpos_hipR = int(model.jnt_qposadr[jid_hipR])
qvel_hipR = int(model.jnt_dofadr[jid_hipR])
qpos_hipL = int(model.jnt_qposadr[jid_hipL])
qvel_hipL = int(model.jnt_dofadr[jid_hipL])

ACT_L, ACT_R = 0, 1

# ADHESION MODEL
k, d = 6000.0, 350.0
reattach_time = 0.1

def site_velocity(site_id):
    v = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, v, 0
    )
    return v[:3]

def apply_soft_weld(site_id, anchor, alpha):
    pos = data.site_xpos[site_id]
    vel = site_velocity(site_id)
    F = alpha * (k * (anchor - pos) - d * vel)
    mujoco.mj_applyFT(
        model, data, F, np.zeros(3), pos,
        int(model.site_bodyid[site_id]),
        data.qfrc_applied
    )

def project_to_wall(xyz, wall_x=1.0):
    return np.array([wall_x, xyz[1], xyz[2]])

anchor_L = np.array([1.0, 0.0, 0.9])
anchor_R = np.array([1.0, 0.0, 1.1])

# NMPC
def build_nmpc(dt, N):

    U = ca.SX.sym("U", 2*N)
    x0     = ca.SX.sym("x0", 2)
    th_ref = ca.SX.sym("th_ref")
    u_prev = ca.SX.sym("u_prev", 2)

    w_th  = ca.SX.sym("w_th")
    w_u   = ca.SX.sym("w_u")
    w_du  = ca.SX.sym("w_du")
    w_tv  = ca.SX.sym("w_tv")

    Ieff = 0.1
    b    = 20
    mgl  = 0.3
    GEAR = 100.0

    cost = 0
    th, thd = x0[0], x0[1]
    uL_prev, uR_prev = u_prev[0], u_prev[1]

    for k in range(N):
        uL = U[2*k]
        uR = U[2*k+1]
        tau = GEAR * (uL - uR)

        thdd = (tau - b*thd - mgl*ca.sin(th)) / Ieff
        thd += dt * thdd
        th  += dt * thd

        cost += (
            w_th * (th - th_ref)**2 +
            w_u  * (uL**2 + uR**2) +
            w_du * ((uL - uL_prev)**2 + (uR - uR_prev)**2)
        )

        uL_prev, uR_prev = uL, uR

    cost += w_tv * ca.fmax(0, ca.fabs(thd) - 0.5)**2

    return ca.nlpsol(
        "solver", "ipopt",
        {"x": U, "f": cost,
         "p": ca.vertcat(x0, th_ref, u_prev, w_th, w_u, w_du, w_tv)},
        {"ipopt.print_level": 0, "print_time": 0}
    )

# MPC PARAMETERS
N = 20
solver = build_nmpc(dt, N)

W_TH, W_U, W_DU, W_TV = 18.39,3.26, 90.18, 2.84
THETA_UP, THETA_DOWN = 0.73, 0.24

u_prev = np.zeros(2)
u_sol  = np.zeros(2*N)
u_mag_ref = np.zeros(2)

phase = 1
MAX_CYCLES = 10
cycle_count = 0
prev_left_on = prev_right_on = True
reattach_L = reattach_R = reattach_time

# ================= METRICS (ADDED) =================
t_log, z_log = [], []
E_log = []
E_cum = 0.0

cycle_energy = []
cycle_height = []
cycle_velocity = []

E_cycle = 0.0
z_start = None
z_cycle_start = None
t_cycle_start = None
# ==================================================

# SIMULATION
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0

        # -------- FSM --------
        if phase == 1:
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.5)
            phase = 2

        elif phase == 2:
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.05)

            theta = float(data.qpos[qpos_hipR])
            theta_dot = float(data.qvel[qvel_hipR])

            sol = solver(
                x0=u_sol, lbx=-1, ubx=1,
                p=[theta, theta_dot, THETA_UP,
                   u_prev[0], u_prev[1],
                   W_TH, W_U, W_DU, W_TV]
            )

            u_sol = np.array(sol["x"]).flatten()
            u_prev = u_sol[:2]
            data.ctrl[:] = u_prev

            if abs(theta) >= THETA_UP:
                u_mag_ref[:] = np.abs(u_prev)
                phase, prev_right_on = 3, False

        elif phase == 3:
            if not prev_right_on:
                anchor_R = project_to_wall(data.site_xpos[rs])
                prev_right_on = True
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.5)
            phase = 4

        elif phase == 4:
            apply_soft_weld(rs, anchor_R, 0.5)
            apply_soft_weld(ls, anchor_L, 0.05)

            theta = float(data.qpos[qpos_hipL])
            theta_dot = float(data.qvel[qvel_hipL])

            sol = solver(
                x0=u_sol, lbx=-1, ubx=1,
                p=[theta, theta_dot, -THETA_DOWN,
                   u_prev[0], u_prev[1],
                   W_TH, W_U, W_DU, W_TV]
            )

            u_sol = np.array(sol["x"]).flatten()
            u_prev = u_sol[:2]

            data.ctrl[ACT_L] = np.sign(u_prev[0]) * u_mag_ref[0]
            data.ctrl[ACT_R] = np.sign(u_prev[1]) * u_mag_ref[1]

            if abs(theta) <= THETA_DOWN:
                phase, prev_left_on = 5, False

        elif phase == 5:
            if not prev_left_on:
                anchor_L = project_to_wall(data.site_xpos[ls])
                prev_left_on = True

            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.5)

            dz = data.qpos[1] - z_cycle_start
            dt_cycle = data.time - t_cycle_start

            cycle_energy.append(E_cycle)
            cycle_height.append(dz)
            cycle_velocity.append(dz / max(dt_cycle, 1e-6))

            cycle_count += 1
            if cycle_count >= MAX_CYCLES:
                break

            phase = 1
            u_prev[:] = 0
            u_sol[:] = 0
            E_cycle = 0.0
            z_cycle_start = data.qpos[1]
            t_cycle_start = data.time

        mujoco.mj_step(model, data)
        viewer.sync()

        # -------- ENERGY + VELOCITY LOGGING --------
        if z_start is None:
            z_start = data.qpos[1]
            z_cycle_start = data.qpos[1]
            t_cycle_start = data.time

        tau_L, tau_R = data.actuator_force[ACT_L], data.actuator_force[ACT_R]
        omega_L, omega_R = data.qvel[qvel_hipL], data.qvel[qvel_hipR]

        dE = (abs(tau_L * omega_L) + abs(tau_R * omega_R)) * dt
        E_cum += dE
        E_cycle += dE

        t_log.append(data.time)
        z_log.append(data.qpos[1])
        E_log.append(E_cum)

        reattach_L += dt
        reattach_R += dt
        time.sleep(dt)

# ================= FINAL METRICS =================
total_height = z_log[-1] - z_log[0]
total_time   = t_log[-1] - t_log[0]

avg_speed = total_height / max(total_time, 1e-6)
total_energy = E_log[-1]
energy_per_meter = total_energy / max(total_height, 1e-6)

print("\n========== PERFORMANCE METRICS ==========")
print(f"Total Height Climbed   : {total_height:.3f} m")
print(f"Total Time             : {total_time:.3f} s")
print(f"Average Climbing Speed : {avg_speed:.3f} m/s")
print("-----------------------------------------")
print(f"Total Energy Used      : {total_energy:.3f} J")
print(f"Energy per Meter       : {energy_per_meter:.3f} J/m")
print("=========================================")


# ---- LINE: HEIGHT VS TIME ----
plt.figure(figsize=(7,5))
plt.plot(t_log, z_log, 'r-', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.title("CO-Design Fixed MPC")
plt.grid(True)
plt.tight_layout()

plt.savefig("height_vs_time.png", dpi=300, bbox_inches="tight")
plt.show()
