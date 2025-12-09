# Reinforcement Learning Environment for 6-DOF Kinova Gen3 Lite Manipulator

## 📌 Abstract

This project presents a fully redesigned reinforcement learning (RL) environment for controlling a 6-DOF Kinova Gen3 Lite robotic manipulator in PyBullet. The work began by evaluating an existing UR5 environment (https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py), which relied heavily on inverse kinematics (IK), non-Markovian observations, and an action space that bypassed robot dynamics—making it unsuitable for genuine RL control learning.  
To address these issues, I developed a physics-grounded, continuous joint-space environment with a complete 18-dimensional observation space, curriculum-based goal sampling, and a carefully structured reward function that promotes smooth, stable, and collision-free motion. The resulting environment enables RL algorithms (e.g., SAC, PPO, TD3) to learn **true multi-joint control policies**, rather than merely supplying IK targets. This provides a realistic research-grade benchmark for robotic reaching and forms a foundation for future manipulation and sim-to-real extensions.

---

# 📁 Project Overview

This repository contains:

- A fully rewritten **Gymnasium-compatible RL environment** for the Kinova Gen3 Lite robotic arm  
- A **joint-space continuous action interface**  
- A **18-dimensional Markov observation space**  
- A **multi-stage curriculum learning mechanism**  
- A **physics-based reward function**  
- Modular robot classes for motion control, collision checking, and gripper actuation  
- Extensible structure for adding grasping, vision, domain randomization, etc.

---

# 📌 Why Redesign the UR5 Environment?

The original UR5 environment served as a helpful demonstration but contained several severe limitations:

### ❌ 1. Action space bypassed robot dynamics
The agent output end-effector positions (x, y), which were passed directly to an internal IK solver.  
➡️ **IK—not RL—controlled the robot.**

### ❌ 2. Observation space was non-Markovian
No joint states, velocities, EE pose, or contact information were provided.  
➡️ No real dynamics learning could occur.

### ❌ 3. Reward reflected IK convergence, not behavior
Because IK solved the motion, rewards did not depend on meaningful policy decisions.

---

# ✔️ Key Improvements in This Project

## 1. True Continuous-Control Actions (Joint-Space Δq)

- RL agent outputs **6 joint-angle increments**  
- Robot executes joint-space position control  
- Entire IK pipeline is removed from RL loop  

➡️ Enables learning *real multi-joint coordinated motion.*

---

## 2. 18-Dimensional Markov Observation Space

State vector includes:

```
[q(6), dq(6), end_effector_position(3), goal - end_effector_position(3)]
```

➡️ Provides full robot & task information needed for dynamic decision-making.

---

## 3. Physics-Grounded Reward Function

Includes:

- Distance shaping  
- Collision penalties  
- Joint-limit proximity penalties  
- Smoothness penalties (‖action‖² & ‖Δaction‖²)  
- Success reward + time bonus  

➡️ Encourages safe, human-like, and efficient behavior.

---

## 4. Curriculum Learning Framework

Stages (0 → 4):

From easy, near-center targets → fully random workspace sampling.

Thresholds:

```
self.curriculum_thresholds = [50, 100, 150, 200]
```

➡️ Stabilizes learning and improves final success rates.

---

# 🚀 Usage Instructions

### Step 1: Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
```

### Step 2: Navigate into the repository

```bash
cd <your-repo>
```

### Step 3: Run the RL script

```bash
python3 main_rl_kinova.py
```

To run the **test loop**, open `main_rl_kinova.py` and uncomment:

```python
test_algo()
```

To **train a new RL policy**, uncomment:

```python
train_algo()
```

---

# 📌 Roadmap

- [ ] Add end-effector orientation control  
- [ ] Implement grasping via gripper actions and contact rewards  
- [ ] Add RGB-D or LiDAR-based observations  
- [ ] Introduce domain randomization for sim-to-real transfer  
- [ ] Deploy trained policies on physical Kinova Gen3 Lite hardware  
- [ ] Provide training benchmarks for SAC, PPO, and TD3  

---

# 📄 Citation

If you use this environment in academic work:

```
@misc{kinova_gen3_rl_env,
  author       = {Manas Dixit},
  title        = {Reinforcement Learning Environment for 6-DOF Kinova Gen3 Manipulator},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-username>/<your-repo>}}
}
```

---

# 📬 Contact

**Manas Dixit**  
M.S. Robotics, University of Minnesota  
📧 **manasdixit13@gmail.com**
