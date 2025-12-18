# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import os
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    JointVelocityProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    MotorCurrentProcessorStep,
    Numpy2TorchActionProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
)
from lerobot.processor.converters import identity_transition
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
    so100_leader,
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

try:
    from lerobot.robots.mkrobot.mk_robot import MKRobotConfig
    from lerobot.teleoperators.gamepad.gamepad_ik_teleop import GamepadIKTeleopConfig
    print("✅ 已成功注册 MKRobot 和 GamepadIK")
except ImportError as e:
    print(f"⚠️ 注册 MKRobot/GamepadIK 失败 (如果不是用这两个硬件可忽略): {e}")

#导入我们刚写的安全处理器
# 注意：如果没有这个文件，请确保你已经完成了上一步新建 safety_processor.py 的操作
try:
    from lerobot.processor.safety_processor import MKArmSafetyProcessorStep
except ImportError:
    MKArmSafetyProcessorStep = None
    print("⚠️ Warning: MKArmSafetyProcessorStep not found. Safety checks will be disabled.")

logging.basicConfig(level=logging.WARNING)
logging.getLogger("lerobot.src.lerobot.rl.learner").setLevel(logging.WARNING)
logging.getLogger("lerobot.src.lerobot.rl.learner_service").setLevel(logging.WARNING)


# 屏蔽 MKRobot 驱动的读取耗时日志 对应文件: src/lerobot/robots/mkrobot/follower_mkarm.py
logging.getLogger("lerobot.robots.mkrobot.follower_mkarm").setLevel(logging.INFO)
logging.getLogger("lerobot.robots.mkrobot").setLevel(logging.INFO)

# 屏蔽 OpenCV 摄像头的读取耗时日志 对应文件: src/lerobot/cameras/opencv/camera_opencv.py
logging.getLogger("lerobot.cameras.opencv.camera_opencv").setLevel(logging.INFO)

# --- 🛡️ 配置区域：Policy 安全屋 (训练活动范围) ---
# 这里的范围应该比 mk_robot.py 里的物理硬限位要小 (建议 80%~90%)
# 确保 Policy 不会把机械臂扭成 IK 算不出来的麻花姿态，方便人工随时接管
POLICY_SAFE_LIMITS = {
    # 关节索引: (最小弧度, 最大弧度)
    0: (-1.0, 1.0), # Base
    1: (0.74, 1.70), # Shoulder (限制不要倒地)
    2: (-1.0, -0.42), # Elbow
    3: (-1.7, 1.2), # Wrist 1
    4: (-0.4, 0.4), # Wrist 2
    5: (-2.0, 2.0), # Wrist 3
}

@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray) -> None:
    """Reset robot arm to target position using smooth trajectory."""
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict], dtype=np.float32
    )
    trajectory = torch.from_numpy(
        np.linspace(current_position, target_position, 50)
    )  # NOTE: 30 is just an arbitrary number
    for pose in trajectory:
        action_dict = dict(zip(current_position_dict, pose, strict=False))
        robot_arm.bus.sync_write("Goal_Position", action_dict)
        busy_wait(0.015)


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        self._joint_names = list(self.robot.bus.motors.keys())
        self._raw_joint_positions = None

        #用于存储上一步的平滑动作，实现滤波
        self.last_policy_action = None

        #状态机变量
        # 模式: "IDLE" (发呆/保持), "EXPLORE" (RL探索), "ZEROING" (自动归零)
        self.rl_mode = "IDLE" 

        self.last_policy_action = None # 用于平滑滤波

        self._setup_spaces()

    def _get_observation(self) -> dict[str, Any]:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])

        images = {key: obs_dict[key] for key in self._image_keys}

        return {"agent_pos": joint_positions, "pixels": images, **raw_joint_joint_position}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            agent_pos = current_observation["agent_pos"]

            observation_spaces[OBS_STATE] = gym.spaces.Box(
                #low=0,
                #high=10,
                low=-6.28, 
                high=6.28,
                shape=agent_pos.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        #  Action Space 改为直接对应关节数量.这里是根据mkrobot改掉了，可能不适应so101了
        #action_dim = 3
        action_dim = len(self._joint_names)

        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        ## (删除原本关于 use_gripper 的 if/else 判断，因为 joint_names 里已经包含了 gripper)
        # if self.use_gripper:
        #     action_dim += 1
        #     bounds["min"] = np.concatenate([bounds["min"], [0]])
        #     bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        #Reset 时不要重置 rl_mode，保持用户的控制状态
        # 比如用户正在 EXPLORE，回合结束 reset 后应该继续 EXPLORE，不需要重新按 Y
        # 除非处于归零状态，归零完成后会自动切回 IDLE
        
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, np.array(self.reset_pose))
            log_say("Reset the environment done.", play_sounds=True)

        busy_wait(self.reset_time_s - (time.perf_counter() - start_time))
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.episode_data = None
        
        self.last_policy_action = None
        
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        # [修正] 直接将 action 数组传给 Robot
        # MKRobot 的 send_action 接收数组，并在内部处理 Sim->Real 转换和字典打包
        # 之前的代码手动打包成了字典，导致 MKRobot 内部对字典切片报错
        self.robot.send_action(action)

        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render()

        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            #{TeleopEvents.IS_INTERVENTION: False},
            {},
        )
    # def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
    #     """Execute one environment step with given action."""
    #     joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}

    #     self.robot.send_action(joint_targets_dict)

    #     obs = self._get_observation()

    #     self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

    #     if self.display_cameras:
    #         self.render()

    #     self.current_step += 1

    #     reward = 0.0
    #     terminated = False
    #     truncated = False

    #     return (
    #         obs,
    #         reward,
    #         terminated,
    #         truncated,
    #         {TeleopEvents.IS_INTERVENTION: False},
    #     )

    def render(self) -> None:
        """Display robot camera feeds."""
        #import cv2

        current_observation = self._get_observation()
        # if current_observation is not None:
        #     image_keys = [key for key in current_observation if "image" in key]

        #     for key in image_keys:
        #         cv2.imshow(key, cv2.cvtColor(current_observation[key].numpy(), cv2.COLOR_RGB2BGR))
        #         cv2.waitKey(1)

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions


def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"

    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_pose = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None

    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
    )

    return env, teleop_device


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ]

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    # Full processor pipeline for real robot environment
    # Get robot and motor information for kinematics
    motor_names = list(env.robot.bus.motors.keys())

    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None:
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=motor_names,
        )

    env_pipeline_steps = [VanillaObservationProcessorStep()]

    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:
            env_pipeline_steps.append(JointVelocityProcessorStep(dt=1.0 / cfg.fps))
        if cfg.processor.observation.add_current_to_observation:
            env_pipeline_steps.append(MotorCurrentProcessorStep(robot=env.robot))

    if kinematics_solver is not None:
        env_pipeline_steps.append(
            ForwardKinematicsJointsToEEObservation(
                kinematics=kinematics_solver,
                motor_names=motor_names,
            )
        )

    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )

    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        # --- 👇 新增调试打印 👇 ---
        limit_steps = int(cfg.processor.reset.control_time_s * cfg.fps)
        print(f"\n🛑 [PROCESSOR CHECK] Creating TimeLimitProcessorStep:")
        print(f"   👉 Raw control_time_s: {cfg.processor.reset.control_time_s}")
        print(f"   👉 Raw FPS: {cfg.fps}")
        print(f"   👉 Final Max Episode Steps: {limit_steps} (Will run for this many frames)")
        # --- 👆 新增结束 👆 ---
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )

    # Add gripper penalty processor if gripper config exists and enabled
    if cfg.processor.gripper is not None and cfg.processor.gripper.use_gripper:
        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    env_pipeline_steps.append(AddBatchDimensionProcessorStep())
    env_pipeline_steps.append(DeviceProcessorStep(device=device))

    if (
        cfg.processor.reward_classifier is not None
        and cfg.processor.reward_classifier.pretrained_path is not None
    ):
        env_pipeline_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
                terminate_on_success=terminate_on_success,
            )
        )

    #动态解析 URDF 路径
    # 我们直接从 teleop 配置中读取路径，因为那里是你定义的真实硬件路径
    urdf_path = None
    if cfg.teleop and hasattr(cfg.teleop, "urdf_path"):
        raw_path = cfg.teleop.urdf_path
        # 将相对路径转换为绝对路径，确保 pinocchio 能找到它
        urdf_path = os.path.abspath(raw_path)
        print(f"🛡️ Safety Processor will use URDF: {urdf_path}")

    action_pipeline_steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
        InterventionActionProcessorStep(
            use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
            terminate_on_success=terminate_on_success,
        ),
    ]

    # 我们不再把 SafetyProcessor 放在 pipeline 里，因为它会错误地处理 Delta 值
    # # [新增] 如果路径存在且类已加载，则添加安全拦截器
    # if MKArmSafetyProcessorStep is not None and urdf_path is not None:
    #     action_pipeline_steps.append(
    #         MKArmSafetyProcessorStep(
    #             urdf_path=urdf_path, 
    #             min_z=0.26  # 你的安全高度限制
    #         )
    #     )
    # else:
    #     print("⚠️ Skipping SafetyProcessor: URDF path missing or class not imported.")

    #
    # # Replace InverseKinematicsProcessor with new kinematic processors
    # if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
    #     # Add EE bounds and safety processor
    #     inverse_kinematics_steps = [
    #         MapTensorToDeltaActionDictStep(
    #             use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
    #         ),
    #         MapDeltaActionToRobotActionStep(),
    #         EEReferenceAndDelta(
    #             kinematics=kinematics_solver,
    #             end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
    #             motor_names=motor_names,
    #             use_latched_reference=False,
    #             use_ik_solution=True,
    #         ),
    #         EEBoundsAndSafety(
    #             end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
    #         ),
    #         GripperVelocityToJoint(
    #             clip_max=cfg.processor.max_gripper_pos,
    #             speed_factor=1.0,
    #             discrete_gripper=True,
    #         ),
    #         InverseKinematicsRLStep(
    #             kinematics=kinematics_solver, motor_names=motor_names, initial_guess_current_joints=False
    #         ),
    #     ]
    #     action_pipeline_steps.extend(inverse_kinematics_steps)
    #     action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=motor_names))

    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    safety_helper: Any = None,
) -> EnvTransition:
    """
    使用处理器管道执行一步环境交互。
    包含 HIL-SERL 核心的状态机逻辑 (Zeroing / Idle / Explore / Intervention)。
    """
    # Create action transition
    transition[TransitionKey.ACTION] = action

    # 1. 执行 Action Pipeline (包含 InterventionProcessor 和 SafetyProcessor)
    # 这里的 processed_action 可能是 Policy 动作，也可能是被替换后的人类动作
    #    输入：Policy的原始输出 (Normalized Delta)
    #    输出：真实的物理增量 (Real Delta, e.g. 0.05 rad)
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    # Clone 一份出来操作
    robot_action = processed_action.clone()

    # 获取当前机械臂的物理关节位置 (Current Pos)
    raw_joints = env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    if TransitionKey.OBSERVATION not in transition or not isinstance(transition[TransitionKey.OBSERVATION], dict):
        transition[TransitionKey.OBSERVATION] = {}
    transition[TransitionKey.OBSERVATION].update(raw_joints)
    joint_names = list(env.robot.bus.motors.keys()) 
    current_pos_list = [raw_joints[f"{name}.pos"] for name in joint_names]
    current_pos_tensor = torch.tensor(current_pos_list, device=robot_action.device, dtype=robot_action.dtype)
    
    # -------------------------------------------------------------------------
    # 🎮 状态机控制逻辑 (State Machine Control)
    # -------------------------------------------------------------------------
    
    # 1. 获取事件 (由底层驱动层处理好的意图)
    info = processed_action_transition.get(TransitionKey.INFO, {})
    
    # A 键 (Start): 对应 SUCCESS 事件
    is_start_signal = info.get(TeleopEvents.SUCCESS, False)
    # X 键 (Reset): 对应 RERECORD 事件 (底层已处理长按)
    is_stop_signal = info.get(TeleopEvents.RERECORD_EPISODE, False)
    # RB 键 (Intervention): 对应 INTERVENTION 事件
    is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)

    # 2. 状态切换逻辑 (App 层决策)
    
    # [IDLE -> EXPLORE] 收到启动信号 (A)
    if is_start_signal:
        if env.rl_mode != "EXPLORE":
            env.rl_mode = "EXPLORE"
            print("\n🚀 [System] ACTIVATED: Policy Exploration Started! (A pressed)")
            # 这里可以重置平滑器，防止上次的残留
            env.last_policy_action = None

    # [ANY -> ZEROING] 收到停止信号 (X)
    if is_stop_signal:
        if env.rl_mode != "ZEROING":
            env.rl_mode = "ZEROING"
            print("\n🛑 [System] STOPPED: Returning to ZERO... (X pressed)")

    # 3. 根据当前模式修正 robot_action
    # [Case 1: 人工介入中]
    # 无论处于什么模式，只要按下了介入键，就听人类的。
    # 关键：在这里同步 Policy 的平滑器记忆，保证松手时 0 跳变。
    if is_intervention:
        #(我觉得没必要) # 如果有 safety_helper，也可以在这里检查一下手柄的输入是否安全
        # if safety_helper is not None:
        #     # 构造检查用的 transition (传入绝对位置)
        #     check_t = transition.copy()
        #     check_t[TransitionKey.ACTION] = robot_action
        #     # 显式调用 Helper
        #     res = safety_helper(check_t) 
        #     robot_action = res[TransitionKey.ACTION]

        # 同步 Policy 的平滑器记忆，防止接管结束时飞车
        if robot_action.ndim == 2:
            env.last_policy_action = robot_action[:, :6].clone()
        else:
            env.last_policy_action = robot_action[:6].clone()

    # [Case 2: 自动归零模式]
    elif env.rl_mode == "ZEROING":
        ZERO_SPEED = 0.05
        target = torch.zeros_like(current_pos_tensor)
        # 仅归零手臂(前6轴)，夹爪保持不动
        if robot_action.ndim == 2: # [1, 7]
            target = target.unsqueeze(0)
            target[:, 6] = current_pos_tensor[6] 
            delta = target[:, :6] - current_pos_tensor[:6]
            delta = torch.clamp(delta, -ZERO_SPEED, ZERO_SPEED)
            robot_action[:, :6] = current_pos_tensor[:6] + delta
            
            if torch.abs(current_pos_tensor[:6]).max() < 0.05:
                env.rl_mode = "IDLE"
                print("✅ [System] Zeroed. Entering IDLE mode.")
        else:
            target[6] = current_pos_tensor[6]
            delta = target[:6] - current_pos_tensor[:6]
            delta = torch.clamp(delta, -ZERO_SPEED, ZERO_SPEED)
            robot_action[:6] = current_pos_tensor[:6] + delta
            
            if torch.abs(current_pos_tensor[:6]).max() < 0.05:
                env.rl_mode = "IDLE"
                print("✅ [System] Zeroed. Entering IDLE mode.")

    # [Case 3: IDLE 模式] 锁死不动
    elif env.rl_mode == "IDLE":
        # 强制动作等于当前位置
        if robot_action.ndim == 2:
            robot_action = current_pos_tensor.unsqueeze(0)
        else:
            robot_action = current_pos_tensor

    # [Case 4: EXPLORE 模式] Policy 控制 (带滤波和软限位)
    elif env.rl_mode == "EXPLORE":
        # 如果没有介入，这里的 robot_action 是 Policy 的原始输出 (范围 -1 ~ 1)
        # === 修改开始: 改为 Delta Control ===
        
        # 定义 Action Scale: Policy 输出 1.0 代表一步移动多少弧度？
        # 建议设小一点，保证动作细腻。例如 0.05 rad/step
        ACTION_SCALE = 0.05 
        # 0.04 弧度约等于 2.3度。这意味着每秒最大转速约 35度 (2.3 * 15Hz)
        MAX_DELTA_PER_STEP = 0.04  
        
        # 提取数据
        policy_output_delta = None
        arm_current = None
        
        if robot_action.ndim == 2: 
            policy_output_delta = robot_action[:, :6].clone()
            arm_current = current_pos_tensor[:6].unsqueeze(0)
        elif robot_action.ndim == 1:
            policy_output_delta = robot_action[:6].clone()
            arm_current = current_pos_tensor[:6]
            
        if policy_output_delta is not None:
            # 1. 计算目标位置: Target = Current + (Policy_Output * Scale)
            # 这样如果 Policy 输出 0 (未知/犹豫)，机械臂就会停在原地，而不是归零
            delta = policy_output_delta * ACTION_SCALE
            # 强制限制每一帧的最大移动量（单位：弧度）
            delta = torch.clamp(delta, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP)
            target_raw = arm_current + delta
            
            EMA_ALPHA = 0.15  # 平滑系数 (0.1~0.3)。越小越顺滑，但延迟越高；越大反应越快但越抖。
            # 如果是刚开始（或刚结束人工介入），初始化记忆为当前位置，防止飞车
            if env.last_policy_action is None:
                env.last_policy_action = arm_current.clone()
            # 确保维度对齐 (处理 batch 维度 [1, 6] vs [6])
            if env.last_policy_action.ndim != target_raw.ndim:
                if target_raw.ndim == 2: 
                    env.last_policy_action = env.last_policy_action.unsqueeze(0)
                else:
                    env.last_policy_action = env.last_policy_action.squeeze(0)

            # 执行滤波公式: Smoothed = alpha * New + (1-alpha) * Old
            # 这样新的目标位置只占 15% 的权重，旧的位置占 85%，强迫电机慢慢动
            arm_target = EMA_ALPHA * target_raw + (1 - EMA_ALPHA) * env.last_policy_action
            
            # 更新记忆供下一帧使用
            #env.last_policy_action = arm_target.detach()            
                    
            #安全检查 (Safety Check with Helper)
            if safety_helper is not None:
                # 构造包含夹爪的 7维 向量用于检查
                full_action_check = robot_action.clone()
                if robot_action.ndim == 2:
                    full_action_check[:, :6] = arm_target # 填入计算好的绝对位置
                else:
                    full_action_check[:6] = arm_target    # 填入计算好的绝对位置
                # 构造临时 Transition
                check_transition = transition.copy()
                check_transition[TransitionKey.ACTION] = full_action_check
                # 🛡️ 显式调用 Helper 进行安检！
                # Helper 内部只会看到绝对位置，它的 internal state 也是绝对位置。
                # 如果触发回滚，它回滚的也是绝对位置。完美！
                result_transition = safety_helper(check_transition)
                # 获取安检后的动作
                safe_full_action = result_transition[TransitionKey.ACTION]
                # 如果安检后的动作和安检前差别很大，说明被 Safety 拦截并回滚了
                diff = (safe_full_action - full_action_check).abs().max().item()
                if diff > 1e-4:
                    print(f"🛡️ [SAFETY BLOCK] Request denied! Diff: {diff:.4f}. Rolling back.")

                # 拆分回 arm_target
                if robot_action.ndim == 2:
                    arm_target = safe_full_action[:, :6]
                else:
                    arm_target = safe_full_action[:6]

            # 3. [软限位] Policy Safe Limits
            # 限制 Target 不要超出安全范围
            for i in range(6):
                min_lim, max_lim = POLICY_SAFE_LIMITS.get(i, (-3.14, 3.14))
                if robot_action.ndim == 2:
                    arm_target[:, i] = torch.clamp(arm_target[:, i], min_lim, max_lim)
                else:
                    arm_target[i] = torch.clamp(arm_target[i], min_lim, max_lim)

            #所有检查做完后，再更新记忆！✅
            # 这样记忆里存的永远是合法的、实际发送给机器人的位置
            env.last_policy_action = arm_target.detach()

            # 4. [最终赋值]
            # 因为我们是基于 Current 算的 Delta，所以不需要再做额外的限速 (ACTION_SCALE 就是限速)
            if robot_action.ndim == 2:
                robot_action[:, :6] = arm_target
            else:
                robot_action[:6] = arm_target

            if env.current_step % 5 == 0: # 提高频率观察
                #观察前几个关节运动幅度
                j0_curr = arm_current.squeeze()[0].item()
                j0_targ = arm_target.squeeze()[0].item()
                raw_d0 = policy_output_delta.squeeze()[0].item()
                print(f"⚠️ [Step {env.current_step}] J0(Base): {j0_curr:.5f} -> {j0_targ:.5f} (Delta_Raw: {raw_d0:.5f})")
                j1_curr = arm_current.squeeze()[1].item()
                j1_targ = arm_target.squeeze()[1].item()
                raw_d1 = policy_output_delta.squeeze()[1].item()
                print(f"⚠️ [Step {env.current_step}] J1(Base): {j1_curr:.5f} -> {j1_targ:.5f} (Delta_Raw: {raw_d1:.5f})")
                j2_curr = arm_current.squeeze()[2].item()
                j2_targ = arm_target.squeeze()[2].item()
                raw_d2 = policy_output_delta.squeeze()[2].item()
                print(f"⚠️ [Step {env.current_step}] J2(Base): {j2_curr:.5f} -> {j2_targ:.5f} (Delta_Raw: {raw_d2:.5f})")
                
                # 获取前3轴的数据 (通常撞相机的是 Base, Shoulder 或 Elbow)
                # 检查是否贴近限位边界
                for i in range(3): # 只检查前3个主要关节
                    min_lim, max_lim = POLICY_SAFE_LIMITS.get(i, (-99, 99))
                    curr_val = arm_current.squeeze()[i].item()
                    if curr_val < min_lim + 0.05 or curr_val > max_lim - 0.05:
                        print(f"  🚨 DANGER ZONE: Joint {i} at {curr_val:.3f} is near limit {min_lim}~{max_lim}!")
    # -------------------------------------------------------------------------
    # 格式转换回 Numpy 发送给 Robot
    if isinstance(robot_action, torch.Tensor):
        robot_action = robot_action.cpu().numpy()
    
    if robot_action.ndim > 1:
        robot_action = robot_action.squeeze(0)

    # 发送动作
    obs, reward, terminated, truncated, info = env.step(robot_action)

    # 组装 Transition (存入 Buffer)
    # 注意：Buffer 中存储的是 processed_action (即 Policy 原始输出 或者 人类动作)
    # 而不是经过滤波、限速后的 robot_action。这符合 SERL 逻辑：存意图，而非执行结果。
    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    
    #new_info = processed_action_transition[TransitionKey.INFO].copy()
    #new_info.update(info)
    env_info = info.copy()
    # 移除 env_info 中的硬编码 False，避免覆盖
    if TeleopEvents.IS_INTERVENTION in env_info:
        del env_info[TeleopEvents.IS_INTERVENTION]
    new_info = processed_action_transition[TransitionKey.INFO].copy()
    new_info.update(env_info)    

    new_transition = create_transition(
        observation=obs,
        action=processed_action, 
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    return new_transition

# def step_env_and_process_transition(
#     env: gym.Env,
#     transition: EnvTransition,
#     action: torch.Tensor,
#     env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
#     action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
# ) -> EnvTransition:
#     """
#     使用处理器管道执行一步环境交互。
#     """
#     # Create action transition
#     transition[TransitionKey.ACTION] = action
    
#     raw_joints = env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
#     if TransitionKey.OBSERVATION not in transition or not isinstance(transition[TransitionKey.OBSERVATION], dict):
#         transition[TransitionKey.OBSERVATION] = {}
#     transition[TransitionKey.OBSERVATION].update(raw_joints)

#     processed_action_transition = action_processor(transition)
#     processed_action = processed_action_transition[TransitionKey.ACTION]

#     # 使用 clone() 创建副本，避免直接修改 Buffer 中存储的原始 Policy 动作
#     robot_action = processed_action.clone()
    
#     # =================================================================
#     # 🛡️ 三重安全逻辑: 滤波(Smoothing) + 限位(Safe Zone) + 限速(Speed Limit)
#     # =================================================================
    
#     # 1. 检查是否有人工介入
#     is_intervention = False
#     if TransitionKey.INFO in processed_action_transition:
#         info = processed_action_transition[TransitionKey.INFO]
#         is_rb_pressed = info.get(TeleopEvents.IS_INTERVENTION, False)
#         is_success_pressed = info.get(TeleopEvents.SUCCESS, False)
#         is_failure_pressed = info.get(TeleopEvents.FAILURE, False)
#         is_rerecord_pressed = info.get(TeleopEvents.RERECORD_EPISODE, False)
        
#         if is_success_pressed: print("💡 User Signal: SUCCESS (Y)")
#         if is_rerecord_pressed: print("💡 User Signal: RERECORD/RESET (X)")
            
#         is_intervention = is_rb_pressed or is_success_pressed or is_failure_pressed or is_rerecord_pressed
    
#     # 如果介入了，清空 Policy 平滑器的记忆，避免下次接管时跳变
#     if is_intervention:
#         env.last_policy_action = None

#     # 2. 如果是 Policy 控制 (非介入状态)，执行平滑和限制
#     if not is_intervention and isinstance(robot_action, torch.Tensor):
#         POLICY_MAX_STEP = 0.04  # 速度上限
#         EMA_ALPHA = 0.2         # 平滑系数 (0.1~1.0)，越小越顺滑但延迟越高
        
#         joint_names = list(env.robot.bus.motors.keys()) 
#         current_pos_list = [raw_joints[f"{name}.pos"] for name in joint_names]
        
#         current_pos_tensor = torch.tensor(
#             current_pos_list, 
#             device=robot_action.device, 
#             dtype=robot_action.dtype
#         )
        
#         # [A] 提取关节目标 & 增加 Batch 维度
#         arm_target = None
#         arm_current = None
        
#         if robot_action.ndim == 2: # [Batch, 7]
#             arm_target = robot_action[:, :6] 
#             arm_current = current_pos_tensor[:6].unsqueeze(0)
#         elif robot_action.ndim == 1: # [7]
#             arm_target = robot_action[:6]
#             arm_current = current_pos_tensor[:6]
            
#         if arm_target is not None:
#             # [B] EMA 平滑滤波 (Anti-Jitter)
#             # ----------------------------------------------------
#             last_action = env.last_policy_action
            
#             # 如果没有历史记录（刚开始或刚结束介入），用当前真实位置初始化
#             # 这样保证从静止开始启动，不会突变
#             if last_action is None:
#                 last_action = arm_current.clone()
            
#             # 确保维度匹配 (处理 Batch 广播)
#             if last_action.ndim != arm_target.ndim:
#                 if arm_target.ndim == 2: last_action = last_action.unsqueeze(0)
                
#             # 执行滤波公式: Smoothed = alpha * New + (1-alpha) * Old
#             arm_target_smoothed = EMA_ALPHA * arm_target + (1 - EMA_ALPHA) * last_action
            
#             # 更新记忆
#             env.last_policy_action = arm_target_smoothed.detach() # detach防止梯度累积
#             # ----------------------------------------------------

#             # [C] Policy 安全屋 (使用平滑后的目标)
#             for i in range(6):
#                 min_lim, max_lim = POLICY_SAFE_LIMITS.get(i, (-3.14, 3.14))
#                 if robot_action.ndim == 2:
#                     arm_target_smoothed[:, i] = torch.clamp(arm_target_smoothed[:, i], min_lim, max_lim)
#                 else:
#                     arm_target_smoothed[i] = torch.clamp(arm_target_smoothed[i], min_lim, max_lim)

#             # [D] 速度限制 (基于平滑后的目标计算 Delta)
#             delta = arm_target_smoothed - arm_current
#             delta_clipped = torch.clamp(delta, -POLICY_MAX_STEP, POLICY_MAX_STEP)
            
#             # [E] 写回 robot_action
#             if robot_action.ndim == 2:
#                 robot_action[:, :6] = arm_current + delta_clipped
#             else:
#                 robot_action[:6] = arm_current + delta_clipped

#     # =================================================================

#     if isinstance(robot_action, torch.Tensor):
#         robot_action = robot_action.cpu().numpy()
    
#     if robot_action.ndim > 1:
#         robot_action = robot_action.squeeze(0)

#     obs, reward, terminated, truncated, info = env.step(robot_action)

#     reward = reward + processed_action_transition[TransitionKey.REWARD]
#     terminated = terminated or processed_action_transition[TransitionKey.DONE]
#     truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
#     complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
#     new_info = processed_action_transition[TransitionKey.INFO].copy()
#     new_info.update(info)

#     new_transition = create_transition(
#         observation=obs,
#         action=processed_action,
#         reward=reward,
#         done=terminated,
#         truncated=truncated,
#         info=new_info,
#         complementary_data=complementary_data,
#     )
    
#     new_transition = env_processor(new_transition)

#     return new_transition



def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Press [A]: START Exploration")  # <--- 改这个
    print("- Long Press [X] (1s): STOP & Return to ZERO")
    print("- Hold [RB]: Manual Intervention")
    print(f"Current Mode: {env.rl_mode}")

    obs, info = env.reset()
    complementary_data = (
        {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
    )
    env_processor.reset()
    action_processor.reset()

    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    transition = env_processor(data=transition)

    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        action_features = teleop_device.action_features
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_success_frames = 0
    episode_start_time = time.perf_counter()

    current_joints = env.get_raw_joint_positions()
    joint_names = list(env.robot.bus.motors.keys())
    neutral_action = torch.tensor([current_joints[f"{k}.pos"] for k in joint_names], dtype=torch.float32)

    #手动初始化安全助手 (Safety Helper)，这里的Safety Helper还是用了之前safety_processor.py中的代码，只是没有按照管道来用
    safety_helper = None
    try:
        # 获取 URDF 路径 (兼容之前的逻辑)
        urdf_path = None
        if hasattr(cfg.env, "teleop") and cfg.env.teleop and hasattr(cfg.env.teleop, "urdf_path"):
             urdf_path = os.path.abspath(cfg.env.teleop.urdf_path)
        
        if MKArmSafetyProcessorStep is not None and urdf_path:
            logging.info(f"🛡️ [Helper] Safety Processor Initialized manually. (Min Z: 0.22)")
            # 实例化它，但不放入 pipeline，只作为一个普通对象使用
            safety_helper = MKArmSafetyProcessorStep(
                urdf_path=urdf_path, 
                min_z=0.26  # <--- 在这里设置你的安全高度
            )
    except Exception as e:
        logging.warning(f"⚠️ SafetyProcessor init failed: {e}")


    while episode_idx < cfg.dataset.num_episodes_to_record:
        step_start_time = time.perf_counter()

        if not isinstance(neutral_action, torch.Tensor):
             neutral_action = torch.from_numpy(neutral_action).float()

        transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            env_processor=env_processor,
            action_processor=action_processor,
            safety_helper=safety_helper,
        )

        # [Anti-Windup Logic] 每次循环后，重置 neutral_action 为当前真实位置
        # 解决 Policy 或 手柄 操作后的位置偏差
        obs_dict = transition[TransitionKey.OBSERVATION]
        current_joint_vals = []
        for name in joint_names:
             key = f"{name}.pos"
             val = obs_dict[key]
             if hasattr(val, "item"):
                 val = val.item()
             current_joint_vals.append(val)
        neutral_action = torch.tensor(current_joint_vals, dtype=torch.float32)

        # Print Info
        reward_val = transition[TransitionKey.REWARD]
        reward_val = reward_val.item() if hasattr(reward_val, "item") else reward_val
        print(f"Epi: {episode_idx} | Reward: {reward_val:.4f} | Steps: {episode_step}", end="\r")

        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        if reward_val > 0.0:
            episode_success_frames += 1

        if cfg.mode == "record":
            observations = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }
            action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "teleop_action", transition[TransitionKey.ACTION]
            )
            frame = {
                **observations,
                ACTION: action_to_record.cpu(),
                REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                DONE: np.array([terminated or truncated], dtype=bool),
            }
            if use_gripper:
                discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
                frame["complementary_info.discrete_penalty"] = np.array([discrete_penalty], dtype=np.float32)

            if dataset is not None:
                frame["task"] = cfg.dataset.task
                dataset.add_frame(frame)

        episode_step += 1

        if terminated or truncated:
            episode_time = time.perf_counter() - episode_start_time
            # 检测是否是因为用户手动重置
            is_rerecord = transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False)
            if is_rerecord:
                logging.info(f"\n🔄 Episode {episode_idx} RESET by USER (Button X). Reward={reward_val}")
            else:
                logging.info(
                    f"\n✅ Episode {episode_idx} finished. Steps: {episode_step}. Reward: {reward_val}"
                )

            episode_step = 0
            episode_success_frames = 0
            episode_idx += 1

            if dataset is not None:
                if is_rerecord:
                    logging.info(f"Re-recording episode {episode_idx}")
                    dataset.clear_episode_buffer()
                    episode_idx -= 1
                else:
                    logging.info(f"Saving episode {episode_idx}")
                    dataset.save_episode()

            obs, info = env.reset()
            env_processor.reset()
            action_processor.reset()

            current_joints = env.get_raw_joint_positions()
            neutral_action = torch.tensor([current_joints[f"{k}.pos"] for k in joint_names], dtype=torch.float32)

            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)
            episode_start_time = time.perf_counter()

        busy_wait(dt - (time.perf_counter() - step_start_time))

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def replay_trajectory(
    env: gym.Env, action_processor: DataProcessorPipeline, cfg: GymManipulatorConfig
) -> None:
    """Replay recorded trajectory on robot environment."""
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.replay_episode)
    actions = episode_frames.select_columns(ACTION)

    _, info = env.reset()

    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        transition = action_processor(transition)
        env.step(transition[TransitionKey.ACTION])
        busy_wait(1 / cfg.env.fps - (time.perf_counter() - start_time))


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """Main entry point for gym manipulator script."""
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    if cfg.mode == "replay":
        replay_trajectory(env, action_processor, cfg)
        exit()

    control_loop(env, env_processor, action_processor, teleop_device, cfg)


if __name__ == "__main__":
    main()
