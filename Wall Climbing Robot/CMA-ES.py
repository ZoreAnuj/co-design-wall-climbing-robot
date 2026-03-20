import numpy as np
import mujoco
import casadi as ca
import cma
import time

# ================= USER SETTINGS =================
XML_PATH = "robot.xml"
N = 20
MAX_CYCLES = 10
MAX_SIM_TIME = 15.0
SEED = 7

THETA_UP, THETA_DOWN = 0.74, 0.24

MIN_HEIGHT_OK = 0.5
LOW_HEIGHT_PENALTY = 200.0

# LOG SCALE FOR WEIGHTS
LOG_CENTER = np.log([50.0, 1.2, 35.5, 2.5])
LOG_SCALE  = np.array([1.0, 1.0, 1.0, 1.0])

W_MIN = np.array([10.0, 0.1, 5.0, 0.1])
W_MAX = np.array([300.0, 20.0, 300.0, 200.0])

def action_to_weights(a):
    w = np.exp(LOG_CENTER + np.clip(a, -1, 1) * LOG_SCALE)
    return tuple(np.clip(w, W_MIN, W_MAX))

# ================= MPC =================
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
    mgl  = 0.5
    GEAR = 100.0

    cost = 0
    th, thd = x0[0], x0[1]
    uL_prev, uR_prev = u_prev[0], u_prev[1]

    for k in range(N):
        uL = U[2*k]
        uR = U[2*k+1]

        tau = GEAR * (uL - uR)
        thdd = (tau - b*thd - mgl*ca.sin(th)) / Ieff
        thd = thd + dt * thdd
        th  = th  + dt * thd

        cost += (
            w_th * (th - th_ref)**2 +
            w_u  * (uL**2 + uR**2) +
            w_du * ((uL - uL_prev)**2 + (uR - uR_prev)**2)
        )

        uL_prev, uR_prev = uL, uR

    vel_tol = 0.005
    cost += w_tv * ca.fmax(0, ca.fabs(thd) - vel_tol)**2

    return ca.nlpsol(
        "solver", "ipopt",
        {
            "x": U,
            "f": cost,
            "p": ca.vertcat(x0, th_ref, u_prev, w_th, w_u, w_du, w_tv)
        },
        {"ipopt.print_level": 0, "print_time": 0}
    )

# ================= EPISODE =================
def run_episode(W, solver):

    W_TH, W_U, W_DU, W_TV = W

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt = model.opt.timestep

    ls = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tip")
    rs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_tip")

    jidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_R")
    jidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")

    qposR, qvelR = model.jnt_qposadr[jidR], model.jnt_dofadr[jidR]
    qposL, qvelL = model.jnt_qposadr[jidL], model.jnt_dofadr[jidL]

    k, d = 6000.0, 350.0
    reattach_time = 0.1

    def site_velocity(site_id):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, v, 0)
        return v[:3]

    def apply_soft_weld(site_id, anchor, alpha):
        pos = data.site_xpos[site_id]
        vel = site_velocity(site_id)
        F = alpha * (k * (anchor - pos) - d * vel)
        mujoco.mj_applyFT(
            model, data, F, np.zeros(3), pos,
            int(model.site_bodyid[site_id]), data.qfrc_applied
        )

    def project_to_wall(xyz):
        return np.array([1.0, xyz[1], xyz[2]])

    anchor_L = np.array([1.0, 0.0, 0.9])
    anchor_R = np.array([1.0, 0.0, 1.1])

    u_prev = np.zeros(2)
    u_sol  = np.zeros(2*N)
    u_mag_ref = np.zeros(2)

    phase = 1
    cycles = 0
    prev_left_on = prev_right_on = True

    E = 0.0
    z0 = None

    while data.time < MAX_SIM_TIME:

        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0

        if phase == 1:
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.5)
            phase = 2

        elif phase == 2:
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.05)

            theta = float(data.qpos[qposR])
            theta_dot = float(data.qvel[qvelR])

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
            phase = 4

        elif phase == 4:
            apply_soft_weld(rs, anchor_R, 0.5)
            apply_soft_weld(ls, anchor_L, 0.05)

            data.ctrl[0] = np.sign(u_prev[0]) * u_mag_ref[0]
            data.ctrl[1] = np.sign(u_prev[1]) * u_mag_ref[1]

            if abs(float(data.qpos[qposL])) <= THETA_DOWN:
                phase, prev_left_on = 5, False

        elif phase == 5:
            cycles += 1
            if cycles >= MAX_CYCLES:
                break
            phase = 1
            u_prev[:] = 0
            u_sol[:] = 0

        mujoco.mj_step(model, data)

        if z0 is None:
            z0 = data.qpos[1]

        tauL, tauR = data.actuator_force[:2]
        wL, wR = data.qvel[qvelL], data.qvel[qvelR]
        E += (abs(tauL*wL) + abs(tauR*wR)) * dt

    dz = max(data.qpos[1] - z0, 1e-6)
    return E/dz, dz

# ================= CMA-ES =================
if __name__ == "__main__":

    np.random.seed(SEED)
    dt = mujoco.MjModel.from_xml_path(XML_PATH).opt.timestep
    solver = build_nmpc(dt, N)

    x0 = np.zeros(4)
    es = cma.CMAEvolutionStrategy(
        x0, 0.6,
        {"bounds":[-np.ones(4), np.ones(4)], "popsize":6}
    )

    bestE = 1e9
    bestW = None

    while not es.stop() and es.countiter < 60:
        X = es.ask()
        F = []
        for x in X:
            f, h = run_episode(action_to_weights(x), solver)
            if h < MIN_HEIGHT_OK:
                f += LOW_HEIGHT_PENALTY * (MIN_HEIGHT_OK - h)
            F.append(f)

            if f < bestE:
                bestE = f
                bestW = action_to_weights(x)
                print(f"[NEW BEST] E/m={f:.3f}, W={bestW}")

        es.tell(X, F)
        es.disp()

    print("\n===== BEST WEIGHTS (CMA-ES) =====")
    print("W_TH =", bestW[0])
    print("W_U  =", bestW[1])
    print("W_DU =", bestW[2])
    print("W_TV =", bestW[3])
