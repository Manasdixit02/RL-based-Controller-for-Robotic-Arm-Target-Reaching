import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from collections import namedtuple
import math
import random


class kinovaGen3Env(gym.Env):
    def __init__(self):
        super(kinovaGen3Env, self).__init__()

        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Set the simulation time step
        p.setTimeStep(1 / 300)

        # Action space: joint deltas (radians per step) for 6 joints
        self.action_space = spaces.Box(
            low=np.array([-0.05] * 6),
            high=np.array([0.05] * 6),
            dtype=np.float64
        )

        # Observation space: [q(6), dq(6), eef_pos(3), goal - eef_pos(3)] = 18
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(6 + 6 + 3 + 3,),
            dtype=np.float64
        )

        # Load environment objects
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            [0.5, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        self.tray_id = p.loadURDF(
            "tray/tray.urdf",
            [0.5, 0.9, 0.6],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        self.cube_id2 = p.loadURDF(
            "cube.urdf",
            [0.5, 0.9, 0.3],
            p.getQuaternionFromEuler([0, 0, 0]),
            globalScaling=0.6,
            useFixedBase=True
        )

        # Set GUI viewing angle
        self.set_gui_view()

        # Load the robot
        self.robot = kinovaGen3([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # Initialize cube
        self.cube_id = None

        # Episode/step settings
        self.max_steps = 100
        self.current_step = 0
        self.gripper_range = [0, 0.085]  # [fully closed, fully open]

        # Curriculum-related state
        self.episode_idx = 0            # how many episodes have started
        self.unlocked_stage = 0         # hardest stage unlocked so far (0..4)
        self.curriculum_stage = 0       # stage used for the current episode
        # thresholds in terms of episode index at which we unlock the next stage
        # 0-49: max stage 0, 50-99: max stage 1, 100-149: max stage 2,
        # 150-199: max stage 3, 200+: max stage 4
        #self.curriculum_thresholds = [50, 300, 500, 700]
        self.curriculum_thresholds = [0, 0, 0, 0]
        # For action smoothness penalty
        self.prev_action = np.zeros(6)

    # ----------------- Curriculum helpers -----------------

    def _update_unlocked_stage(self):
        """
        Update self.unlocked_stage based on episode_idx and thresholds.
        Stages:
          0: y=0, x ~ center
          1: y=0, x in lower region
          2: y in [-0.3, 0], x in [0.4, 0.7]
          3: y in [0, 0.3], x in [0.4, 0.7]
          4: y in [-0.3, 0.3], x in [0.4, 0.7] (fully random in workspace)
        """
        if self.episode_idx < self.curriculum_thresholds[0]:
            self.unlocked_stage = 0
        elif self.episode_idx < self.curriculum_thresholds[1]:
            self.unlocked_stage = 1
        elif self.episode_idx < self.curriculum_thresholds[2]:
            self.unlocked_stage = 2
        elif self.episode_idx < self.curriculum_thresholds[3]:
            self.unlocked_stage = 3
        else:
            self.unlocked_stage = 4

    def set_curriculum_stage(self, stage: int):
        """
        Optional: manually override the curriculum stage for debugging/eval.
        This forces the active stage for the next reset() call.
        """
        self.unlocked_stage = int(stage)
        self.curriculum_stage = int(stage)

    # ----------------- Observation -----------------

    def _obs(self):
        q = np.array(
            [p.getJointState(self.robot.id, j)[0]
             for j in self.robot.arm_controllable_joints]
        )
        dq = np.array(
            [p.getJointState(self.robot.id, j)[1]
             for j in self.robot.arm_controllable_joints]
        )
        eef_pos = np.array(
            p.getLinkState(self.robot.id, self.robot.eef_id)[4]
        )  # worldLinkFramePosition
        goal = self.target_pos
        return np.concatenate([q, dq, eef_pos, goal - eef_pos], axis=0)

    # ----------------- Visualization helpers -----------------

    def set_gui_view(self):
        """
        Set the GUI camera view (not actual camera capture)
        """
        camera_distance = 1.1
        camera_yaw = 90
        camera_pitch = -45
        camera_target = [0.5, 0, 0.6]  # Look-at point (center of table)

        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target
        )

    def draw_boundary(self, x_range, y_range, z_height):
        """
        Draw a boundary box for the specified x and y ranges.
        """
        corners = [
            [x_range[0], y_range[0], z_height],  # Bottom-left
            [x_range[1], y_range[0], z_height],  # Bottom-right
            [x_range[1], y_range[1], z_height],  # Top-right
            [x_range[0], y_range[1], z_height],  # Top-left
        ]

        for i in range(len(corners)):
            p.addUserDebugLine(
                corners[i],
                corners[(i + 1) % len(corners)],
                [1, 0, 0],
                lineWidth=2
            )

    # ----------------- Reset -----------------

    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        Uses a probabilistic curriculum:
          - Mostly samples from the hardest unlocked stage
          - Sometimes samples from earlier stages (to avoid forgetting)
        """
        # Gymnasium's seeding
        if seed is not None:
            super().reset(seed=seed)

        self.current_step = 0
        self.prev_action = np.zeros(6)

        # Bookkeeping
        self.episode_idx += 1
        self._update_unlocked_stage()

        # Sample the active curriculum stage for this episode
        max_stage = self.unlocked_stage
        if max_stage == 0:
            stage_for_ep = 0
        else:
            # e.g. 70% of episodes use the hardest unlocked stage,
            # 30% randomly sample an earlier stage [0, max_stage-1]
            if np.random.rand() < 0.9:
                stage_for_ep = max_stage
            else:
                stage_for_ep = np.random.randint(0, max_stage)

        self.curriculum_stage = stage_for_ep

        # Reset robot to a neutral pose
        self.robot.orginal_position(self.robot)

        # Workspace bounds
        x_min, x_max = 0.4, 0.7
        y_min, y_max = -0.3, 0.3

        # Curriculum-based cube placement for this episode
        if self.curriculum_stage == 0:
            # Stage 0: cube near center, y = 0, x minimally varying
            x = 0.55 + np.random.uniform(-0.02, 0.02)
            x = float(np.clip(x, x_min, x_max))
            y = 0.0

        elif self.curriculum_stage == 1:
            # Stage 1: move towards lower x at y = 0
            x = np.random.uniform(0.4, 0.5)
            y = 0.0

        elif self.curriculum_stage == 2:
            # Stage 2: random y in [-0.3, 0], x in [0.4, 0.7]
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(-0.3, 0.0)

        elif self.curriculum_stage == 3:
            # Stage 3: random y in [0, 0.3], x in [0.4, 0.7]
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(0.0, 0.3)

        else:
            # Stage 4: fully random in workspace y ∈ [-0.3, 0.3], x ∈ [0.4, 0.7]
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

        cube_start_pos = [x, y, 0.63]

        # Draw workspace boundary (purely visual)
        self.draw_boundary([x_min, x_max], [y_min, y_max], 0.63)

        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        if self.cube_id:
            p.resetBasePositionAndOrientation(
                self.cube_id,
                cube_start_pos,
                cube_start_orn
            )
        else:
            self.cube_id = p.loadURDF(
                "./urdf/cube_blue.urdf",
                cube_start_pos,
                cube_start_orn
            )

        self.initial_cube_pos = np.array(cube_start_pos[:2])
        self.target_pos = np.array([cube_start_pos[0], cube_start_pos[1], 1.0])

        observation = self._obs()
        info = {
            "curriculum_stage": self.curriculum_stage,
            "unlocked_stage": self.unlocked_stage,
            "episode_idx": self.episode_idx,
        }
        return observation, info

    # ----------------- Optional: gripper (unused for now) -----------------

    def gripper_close(self):
        grip_value = self.gripper_range[1]

        while True:
            contact_point = p.getContactPoints(bodyA=self.robot.id)

            force = {}
            if len(contact_point) > 0:
                for i in contact_point:
                    link_index = i[2]
                    if force.get(link_index) is None:
                        force[link_index] = {17: 0, 12: 0}
                    if i[3] == 17:
                        if i[9] > force[link_index][17]:
                            force[link_index][17] = i[9]
                    elif i[3] == 12:
                        if i[9] > force[link_index][12]:
                            force[link_index][12] = i[9]

            for link_index in force:
                if force[link_index][17] > 3 and force[link_index][12] > 3:
                    print(f"[Grasped] Link {link_index}: "
                          f"joint 17 = {force[link_index][17]:.2f}, "
                          f"joint 12 = {force[link_index][12]:.2f}")
                    return True

            for link_index in force:
                for joint in [17, 12]:
                    if force[link_index][joint] > 0:
                        print(f"Link {link_index}, joint {joint} force: "
                              f"{force[link_index][joint]:.2f}")

            if grip_value <= self.gripper_range[0]:
                break

            grip_value -= 0.001
            self.robot.move_gripper(grip_value)

            for _ in range(60):
                p.stepSimulation()

        return False

    # ----------------- Step -----------------

    def step(self, action):
        """
        Perform an action in the environment.
        Action: joint deltas (rad) for 6 joints.
        """
        self.current_step += 1

        # Current joint positions
        q = np.array(
            [p.getJointState(self.robot.id, j)[0]
             for j in self.robot.arm_controllable_joints]
        )

        dq_cmd = action  # per-step delta (rad)
        q_des = np.clip(q + dq_cmd,
                        self.robot.lower_limits,
                        self.robot.upper_limits)
        self.robot.move_arm(q_des)

        # Step simulation
        for _ in range(100):
            p.stepSimulation()

        # EE pose and distance to target
        eef_state = self.robot.get_current_ee_position()
        eef_position = np.array(eef_state[0])[:3]
        distance_to_target = float(np.linalg.norm(eef_position - self.target_pos))

        collision_pts = self.robot.check_collision()

        # --- Reward shaping params ---
        ALPHA_DIST_FAR = 5.0
        ALPHA_DIST_CLOSE = 1.0
        DEADBAND = 0.05        # inside this, no distance penalty
        COLLISION_PEN = 20.0
        SUCCESS_THRESH = 0.02  # 2 cm
        SUCCESS_BONUS = 100.0
        TIME_BONUS_W = 1.0
        BASE_SELF_CNT = 31

        ACT_L2_W = 0.01
        ACT_DELTA_W = 0.02

        # --- Distance shaping ---
        if distance_to_target > DEADBAND:
            if distance_to_target > 0.05:
                dist_penalty = -ALPHA_DIST_FAR * distance_to_target
            else:
                dist_penalty = -ALPHA_DIST_CLOSE * distance_to_target
        else:
            dist_penalty = 0.0

        reward = dist_penalty

        # --- Collision penalty (non-terminal) ---
        meaningful_collisions = max(0, collision_pts - BASE_SELF_CNT)
        if meaningful_collisions > 0:
            reward -= COLLISION_PEN

        # --- Limit proximity penalty ---
        margin = 0.05
        q_next = q + dq_cmd
        dist_low = q_next - np.array(self.robot.lower_limits)
        dist_high = np.array(self.robot.upper_limits) - q_next
        violation_amount = np.sum(
            np.clip(margin - dist_low, 0, margin) +
            np.clip(margin - dist_high, 0, margin)
        )
        reward -= violation_amount * 1.0

        # --- Action cost (smoothness) ---
        if self.prev_action is not None:
            act_l2 = float(np.mean(np.square(action)))
            act_delta = float(np.mean(np.square(action - self.prev_action)))
            reward -= ACT_L2_W * act_l2
            reward -= ACT_DELTA_W * act_delta
        self.prev_action = action.copy()

        # --- Success / termination ---
        done = False
        truncated = False
        is_successful = False

        if distance_to_target <= SUCCESS_THRESH:
            steps_left = max(0, self.max_steps - self.current_step)
            reward += SUCCESS_BONUS + TIME_BONUS_W * steps_left
            done = True
            is_successful = True
        elif self.current_step >= self.max_steps:
            done = True
            truncated = True
            reward -= 10.0 * distance_to_target

        print(f"reward: {reward}")
        print(f"Distance to target: {distance_to_target}")
        observation = self._obs()
        info = {
            "curriculum_stage": self.curriculum_stage,
            "unlocked_stage": self.unlocked_stage,
            "episode_idx": self.episode_idx,
            "is_successful": is_successful,
        }

        return observation, reward, done, truncated, info

    # ----------------- Extra helpers -----------------

    def lift_object_slowly(self, start_pos, end_z, eef_orientation,
                            steps=30, sim_steps_per_move=5, sleep_time=0.005):
        """
        Smooth lifting sequence (unused for RL right now).
        """
        for i in range(steps):
            intermediate_z = start_pos[2] + (end_z - start_pos[2]) * (i + 1) / steps
            lift_pos = np.array([start_pos[0], start_pos[1], intermediate_z])
            self.robot.move_arm_ik(lift_pos, eef_orientation)

            for _ in range(sim_steps_per_move):
                p.stepSimulation()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def close(self):
        p.disconnect()


class kinovaGen3:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 2

        # Hard-coded joint limits for 6 DOF arm (must match URDF)
        self.lower_limits = np.array([-2.69, -2.69, -2.69, -2.59, -2.57, -2.59])
        self.upper_limits = np.array([ 2.69,  2.69,  2.69,  2.59,  2.57,  2.59])

    def load(self):
        self.id = p.loadURDF(
            './urdf/gen3_lite.urdf',
            self.base_pos,
            self.base_ori,
            useFixedBase=True
        )
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        jointInfo = namedtuple(
            'jointInfo',
            ['id', 'name', 'type', 'lowerLimit', 'upperLimit',
             'maxForce', 'maxVelocity', 'controllable']
        )
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(
                    jointID, jointName, jointType,
                    jointLowerLimit, jointUpperLimit,
                    jointMaxForce, jointMaxVelocity,
                    controllable
                )
            )

        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [
            ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)
        ]

    def __setup_mimic_joints__(self):
        mimic_parent_name = 'right_finger_bottom_joint'
        mimic_children_names = {
            'right_finger_tip_joint': -0.676,
            'left_finger_bottom_joint': 1,
            'left_finger_tip_joint': -0.676
        }
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints if joint.name in mimic_children_names
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id, self.mimic_parent_id,
                self.id, joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            p.changeConstraint(
                c, gearRatio=-multiplier, maxForce=100, erp=1
            )

    def check_collision(self):
        pts = p.getClosestPoints(
            bodyA=self.id,
            bodyB=self.id,
            distance=0.0
        )
        print(len(pts))
        return len(pts)

    def move_gripper(self, open_length):
        open_length = max(self.gripper_range[0],
                          min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(
            self.id,
            self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=open_angle,
            force=50,
            maxVelocity=self.joints[self.mimic_parent_id].maxVelocity
        )

    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(
            self.id, self.eef_id, target_pos, target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                joint_poses[i],
                maxVelocity=self.max_velocity
            )

    def move_arm(self, target_pos):
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                target_pos[i],
                maxVelocity=self.max_velocity
            )

    def get_current_ee_position(self):
        return p.getLinkState(self.id, self.eef_id)

    def orginal_position(self, robot):
        target_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(
                robot.id,
                joint_id,
                p.POSITION_CONTROL,
                target_joint_positions[i]
            )
        for _ in range(100):
            p.stepSimulation()
        self.move_gripper(0.085)
        for _ in range(3500):
            p.stepSimulation()

