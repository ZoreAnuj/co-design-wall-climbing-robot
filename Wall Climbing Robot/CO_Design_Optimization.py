import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import casadi as ca
import cma

# =========================================================
# CO-DESIGN WRAPPED FUNCTION
# =========================================================
def run_co_design(HIP_RANGE, LINK_LENGTH, GEAR_RATIO):

    SIM_TIMEOUT = 15.0
    start_wall_time = time.time()

    def compute_torso_x(L):
        return 1.0 - L - 0.1

    def generate_robot_xml():
        torso_x = compute_torso_x(LINK_LENGTH)
        xml = f"""
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="12 12 0.1" rgba="0.9 0.85 0.2 1"/>
    <geom type="box" size="0.05 12 12" pos="1 0 1"
          rgba="1 1 1 1" friction="0.5 0.01 0.001"/>

    <body name="torso" pos="{torso_x} 0 1">
      <joint type="slide" axis="1 0 0" damping="5"/>
      <joint type="slide" axis="0 0 1" damping="5"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.85 0.2 0.2 1"/>

      <body name="left_leg" pos="0 0 -0.05">
        <joint name="hip_L" type="hinge" axis="0 1 0"
               range="-{HIP_RANGE} {HIP_RANGE}" damping="20"/>
        <geom type="capsule" fromto="0 0 0 {LINK_LENGTH} 0 0"
              size="0.02" rgba="0.2 0.4 0.9 1"/>
        <site name="left_tip" pos="{LINK_LENGTH} 0 0" size="0.01"/>
      </body>

      <body name="right_leg" pos="0 0 0.05">
        <joint name="hip_R" type="hinge" axis="0 1 0"
               range="-{HIP_RANGE} {HIP_RANGE}" damping="20"/>
        <geom type="capsule" fromto="0 0 0 {LINK_LENGTH} 0 0"
              size="0.02" rgba="0.2 0.8 0.3 1"/>
        <site name="right_tip" pos="{LINK_LENGTH} 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="hip_L" gear="{GEAR_RATIO}" ctrlrange="-1 1"/>
    <motor joint="hip_R" gear="{GEAR_RATIO}" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
        open("Co_Design_Robot.xml", "w").write(xml)

    generate_robot_xml()

    model = mujoco.MjModel.from_xml_path("Co_Design_Robot.xml")
    data  = mujoco.MjData(model)
    dt = model.opt.timestep

    # IDs
    ls = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tip")
    rs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_tip")
    jid_hipR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_R")
    jid_hipL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")

    qpos_hipR = model.jnt_qposadr[jid_hipR]
    qvel_hipR = model.jnt_dofadr[jid_hipR]
    qpos_hipL = model.jnt_qposadr[jid_hipL]
    qvel_hipL = model.jnt_dofadr[jid_hipL]

    ACT_L, ACT_R = 0, 1

    # Adhesion
    k, d = 6000.0, 350.0
    reattach_time = 0.1

    def site_velocity(site_id):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(model, data,
                                 mujoco.mjtObj.mjOBJ_SITE,
                                 site_id, v, 0)
        return v[:3]

    def apply_soft_weld(site_id, anchor, alpha):
        pos = data.site_xpos[site_id]
        vel = site_velocity(site_id)
        F = alpha * (k * (anchor - pos) - d * vel)
        mujoco.mj_applyFT(model, data, F, np.zeros(3),
                          pos, model.site_bodyid[site_id],
                          data.qfrc_applied)

    def project_to_wall(x):
        return np.array([1.0, x[1], x[2]])

    anchor_L = np.array([1.0, 0.0, 0.9])
    anchor_R = np.array([1.0, 0.0, 1.1])

    # =========================================================
    # NEW MPC (REPLACED)
    # =========================================================
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
        b    = 20.0
        mgl  = 0.5
        GEAR = GEAR_RATIO

        th, thd = x0[0], x0[1]
        uL_prev, uR_prev = u_prev[0], u_prev[1]

        cost = 0
        for k in range(N):
            uL = U[2*k]
            uR = U[2*k+1]

            tau  = GEAR * (uL - uR)
            thdd = (tau - b*thd - mgl*ca.sin(th)) / Ieff
            thd  = thd + dt * thdd
            th   = th  + dt * thd

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
            {"x": U,
             "f": cost,
             "p": ca.vertcat(x0, th_ref, u_prev, w_th, w_u, w_du, w_tv)},
            {"ipopt.print_level": 0, "print_time": 0}
        )

    N = 20
    solver = build_nmpc(dt, N)

    W_TH, W_U, W_DU, W_TV = 18.39, 3.26, 86.38, 2.3
    THETA_UP   = math.radians(HIP_RANGE) - 0.03
    THETA_DOWN = 0.25

    u_prev = np.zeros(2)
    u_sol  = np.zeros(2*N)
    u_mag_ref = np.zeros(2)

    phase = 1
    MAX_CYCLES = 10
    cycle_count = 0
    prev_left_on = prev_right_on = True
    reattach_L = reattach_R = reattach_time

    z_start = None
    E_cum = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            if time.time() - start_wall_time > SIM_TIMEOUT:
                viewer.close()
                return {"cycles":0,"height":0.0,"energy":1e6}

            data.ctrl[:] = 0
            data.qfrc_applied[:] = 0

            if phase == 1:
                apply_soft_weld(ls, anchor_L, 0.5)
                apply_soft_weld(rs, anchor_R, 0.5)
                phase = 2

            elif phase == 2:
                apply_soft_weld(ls, anchor_L, 0.5)
                apply_soft_weld(rs, anchor_R, 0.05)

                th  = float(data.qpos[qpos_hipR])
                thd = float(data.qvel[qvel_hipR])

                sol = solver(
                    x0=u_sol, lbx=-1, ubx=1,
                    p=[th, thd, THETA_UP,
                       u_prev[0], u_prev[1],
                       W_TH, W_U, W_DU, W_TV]
                )

                u_sol  = np.array(sol["x"]).flatten()
                u_prev = u_sol[:2]

                data.ctrl[ACT_L] = u_prev[0]
                data.ctrl[ACT_R] = u_prev[1]

                if abs(th) >= THETA_UP:
                    u_mag_ref[:] = np.abs(u_prev)
                    phase, prev_right_on = 3, False

            elif phase == 3:
                if not prev_right_on:
                    anchor_R = project_to_wall(data.site_xpos[rs])
                    reattach_R = 0
                    prev_right_on = True
                apply_soft_weld(ls, anchor_L, 0.5)
                apply_soft_weld(rs, anchor_R, min(reattach_R/reattach_time,0.5))
                phase = 4

            elif phase == 4:
                apply_soft_weld(rs, anchor_R, 0.5)
                apply_soft_weld(ls, anchor_L, 0.05)

                th  = float(data.qpos[qpos_hipL])
                thd = float(data.qvel[qvel_hipL])

                sol = solver(
                    x0=u_sol, lbx=-1, ubx=1,
                    p=[th, thd, -THETA_DOWN,
                       u_prev[0], u_prev[1],
                       W_TH, W_U, W_DU, W_TV]
                )

                u_sol  = np.array(sol["x"]).flatten()
                u_prev = u_sol[:2]

                data.ctrl[ACT_L] = np.sign(u_prev[0]) * u_mag_ref[0]
                data.ctrl[ACT_R] = np.sign(u_prev[1]) * u_mag_ref[1]

                if abs(th) <= THETA_DOWN:
                    phase, prev_left_on = 5, False

            elif phase == 5:
                if not prev_left_on:
                    anchor_L = project_to_wall(data.site_xpos[ls])
                    reattach_L = 0
                    prev_left_on = True

                apply_soft_weld(ls, anchor_L, min(reattach_L/reattach_time,0.5))
                apply_soft_weld(rs, anchor_R, 0.5)

                cycle_count += 1
                if cycle_count >= MAX_CYCLES:
                    break

                phase = 1
                u_prev[:] = 0
                u_sol[:]  = 0

            mujoco.mj_step(model, data)
            viewer.sync()

            if z_start is None:
                z_start = data.qpos[1]

            tau_L, tau_R = data.actuator_force
            E_cum += (abs(tau_L*data.qvel[qvel_hipL]) +
                      abs(tau_R*data.qvel[qvel_hipR])) * dt

            reattach_L += dt
            reattach_R += dt
            time.sleep(dt)

    return {"cycles":MAX_CYCLES,
            "height":data.qpos[1]-z_start,
            "energy":E_cum}

# CMA-ES
def objective(x):

    HIP_RANGE   = float(x[0])
    LINK_LENGTH = float(x[1])
    GEAR_RATIO  = float(x[2])

    penalty = 0.0
    if HIP_RANGE < 30 or HIP_RANGE > 55:
        penalty += 1e3 * abs(HIP_RANGE - np.clip(HIP_RANGE, 30, 55))
    if LINK_LENGTH < 0.3 or LINK_LENGTH > 0.6:
        penalty += 1e3 * abs(LINK_LENGTH - np.clip(LINK_LENGTH, 0.3, 0.6))
    if GEAR_RATIO < 50 or GEAR_RATIO > 120:
        penalty += 1e3 * abs(GEAR_RATIO - np.clip(GEAR_RATIO, 50, 120))

    res = run_co_design(HIP_RANGE, LINK_LENGTH, GEAR_RATIO)

    if res["cycles"] < 10:
        penalty += 500 * (10 - res["cycles"])
    if res["height"] < 3.5:
        penalty += 500 * (3.5 - res["height"])

    return res["energy"] / max(res["height"],1e-6) + penalty


x0 = [42.5, 0.45, 85.0]
es = cma.CMAEvolutionStrategy(
    x0, 10.0,
    {"bounds":[[30,0.3,50],
               [55,0.6,120]],
     "popsize":6}
)

MAX_ITERS = 30
while not es.stop() and es.countiter < MAX_ITERS:
    X = es.ask()
    es.tell(X, [objective(x) for x in X])
    es.disp()

best = es.result.xbest
print("\n===== BEST CO-DESIGN =====")
print("HIP_RANGE :", best[0])
print("LINK_LEN  :", best[1])
print("GEAR_RATIO:", best[2])
print("=========================")
