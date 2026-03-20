# Co-Design Wall Climbing Robot

Simulation and control framework for joint optimization of robot morphology, actuation parameters, and control policies. Uses MuJoCo for physics simulation with CMA-ES evolutionary optimization and PPO reinforcement learning.

## Architecture

- **Co-Design Optimizer** (`CO_Design_Optimization.py`) - Joint morphology + controller optimization loop using CMA-ES to tune physical parameters alongside control gains
- **MPC Controller** (`MPC.py`) - Nonlinear model predictive control for trajectory tracking with configurable horizons and constraint handling
- **PPO Weight Optimizer** (`PPO_Weightoptimization.py`) - RL-based weight tuning that learns optimal MPC gains through environment interaction
- **CMA-ES** (`CMA-ES.py`) - Covariance Matrix Adaptation for black-box optimization of robot design parameters

## Simulation

MuJoCo XML models (`Co_Design_Robot.xml`, `robot.xml`) define the wall-climbing robot with configurable joint limits, actuator parameters, and surface adhesion properties.

## Stack

Python / MuJoCo / CMA-ES / PPO (Stable Baselines3) / CasADi

## Results

Comparison across optimization strategies: manual MPC tuning vs. PPO-tuned MPC vs. CMA-ES tuned vs. full co-design optimization. See `Figures/` for performance plots.
