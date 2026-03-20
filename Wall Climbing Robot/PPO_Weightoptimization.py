# PPO → EXACT MPC (LINE-BY-LINE) WEIGHT OPTIMIZATION
# BEST WEIGHTS PRINTED AT END

import numpy as np
import mujoco
import casadi as ca
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# USER SETTINGS
XML_PATH = "robot.xml"
N = 20
MAX_CYCLES = 10
MAX_SIM_TIME = 15.0
SEED = 7

THETA_UP, THETA_DOWN = 0.74, 0.24

TOTAL_EPISODES = 200

MIN_HEIGHT_OK = 0.5
LOW_HEIGHT_PENALTY = 200.0

# LOG-SCALE WEIGHTS
LOG_CENTER = np.log([50.0, 1.2, 35.5, 2.5])
LOG_SCALE  = np.array([1.0, 1.0, 1.0, 1.0])

W_MIN = np.array([10.0, 0.1, 5.0, 0.1])
W_MAX = np.array([300.0, 20.0, 300.0, 200.0])

def action_to_weights(a):
    w = np.exp(LOG_CENTER + np.clip(a, -1, 1) * LOG_SCALE)
    return tuple(np.clip(w, W_MIN, W_MAX))


# MPC 

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

#  EPISODE 

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

    ACT_L, ACT_R = 0, 1

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
    reattach_L = reattach_R = reattach_time

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

        reattach_L += dt
        reattach_R += dt

    dz = max(data.qpos[1] - z0, 1e-6)
    return E/dz, dz, data.time, False

# PPO ENV 

class PPOWeightEnv(gym.Env):

    def __init__(self, solver):
        self.solver = solver
        self.action_space = spaces.Box(-1,1,(4,),np.float32)
        self.observation_space = spaces.Box(0,1,(1,),np.float32)
        self.bestE = 1e9
        self.bestW = None

    def reset(self, seed=None, options=None):
        return np.zeros(1,np.float32), {}

    def step(self, action):
        W = action_to_weights(action)
        Epm, h, _, _ = run_episode(W, self.solver)

        reward = -Epm
        if h < MIN_HEIGHT_OK:
            reward -= LOW_HEIGHT_PENALTY * (MIN_HEIGHT_OK - h)

        if Epm < self.bestE and h > 1e-3:
            self.bestE = Epm
            self.bestW = W
            print(f"[NEW BEST] E/m={Epm:.3f}, W={W}")

        return np.zeros(1,np.float32), reward, True, False, {}


# TRAIN 

if __name__ == "__main__":

    np.random.seed(SEED)
    dt = mujoco.MjModel.from_xml_path(XML_PATH).opt.timestep
    solver = build_nmpc(dt, N)

    env = DummyVecEnv([lambda: PPOWeightEnv(solver)])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=10,
        batch_size=10,
        gamma=1.0,
        ent_coef=0.02,
        verbose=1,
        seed=SEED
    )

    model.learn(total_timesteps=TOTAL_EPISODES)
    best_env = env.envs[0]
    print("BEST WEIGHTS FOUND BY PPO")
    print("W_TH =", best_env.bestW[0])
    print("W_U  =", best_env.bestW[1])
    print("W_DU =", best_env.bestW[2])
    print("W_TV =", best_env.bestW[3])
