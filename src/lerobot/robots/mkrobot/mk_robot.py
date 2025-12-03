import numpy as np
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, Any

from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCamera

try:
    from lerobot.robots.mkrobot.follower_mkarm import MKFollower, MKFollowerConfig 
except ImportError:
    # 允许在没有底层库的机器上被导入（主要防报错，实际运行还是需要的）
    MKFollower = None
    MKFollowerConfig = None

logger = logging.getLogger(__name__)

# --- 硬件方向修正 ---
# Sim (URDF) <-> Real (Motor)
HARDWARE_DIR = np.array([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0]) # 前6轴

class MKBusAdapter:
    """
    伪装成 DynamixelBus，为 gym_manipulator 提供 sync_read/write 接口。
    同时确保复位操作经过 MKRobot 的坐标转换，保证方向安全。
    """
    def __init__(self, mk_robot):
        self.mk_robot = mk_robot # 持有 MKRobot 实例以便调用其 send_action/get_observation
        self.names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]

    @property
    def motors(self):
        # 返回电机字典，用于获取键名列表
        return self.mk_robot.robot.motors

    def sync_read(self, prop):
        if prop == "Present_Position":
            # 使用 MKRobot.get_observation 获取经过 Sim 坐标转换后的状态
            obs = self.mk_robot.get_observation()
            state = obs['observation.state'].cpu().numpy()
            
            # 将数组重新映射回字典 {joint_name: value}
            return {name: float(val) for name, val in zip(self.names, state)}
        return {}

    def sync_write(self, prop, values):
        if prop == "Goal_Position":
            # values 是 {joint_name: val} (Sim 坐标系)
            # 转换为数组并调用 MKRobot.send_action (它会自动处理 Sim->Real 转换)
            target = np.zeros(7, dtype=np.float32)
            for i, name in enumerate(self.names):
                if name in values:
                    target[i] = values[name]
            
            self.mk_robot.send_action(target)

@RobotConfig.register_subclass("mk_robot")
@dataclass
class MKRobotConfig(RobotConfig):
    type: str = "mk_robot"
    port: str = "/dev/ttyACM0"
    joint_velocity_scaling: float = 1.0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

class MKRobot(Robot):
    config_class = MKRobotConfig
    name = "mk_robot"

    def __init__(self, config: MKRobotConfig):
        super().__init__(config)
        self.config = config
        
        #手动初始化相机列表
        self.cameras = {}
        for name, cam_config in config.cameras.items():
            # 这里的 cam_config 已经是通过 draccus 解析好的配置对象
            if cam_config.type == "opencv":
                self.cameras[name] = OpenCVCamera(cam_config)
            else:
                logger.warning(f"⚠️ MKRobot 目前仅显式支持 'opencv' 类型相机，跳过: {name} ({cam_config.type})")

        if MKFollowerConfig is None:
            raise ImportError("Could not import follower_mkarm. Please ensure it is in the python path.")

        # 初始化底层驱动
        self.follower_config = MKFollowerConfig(
            port=config.port,
            joint_velocity_scaling=config.joint_velocity_scaling,
            disable_torque_on_disconnect=True
        )
        self.robot = MKFollower(self.follower_config)
        self.is_connected_flag = False

        self._bus_adapter = MKBusAdapter(self)

    def connect(self):
        if not self.is_connected_flag:
            logger.info(f"🔗 MKRobot: Connecting to {self.config.port}...")
            self.robot.connect()
            #连接所有摄像头
            for name, cam in self.cameras.items():
                logger.info(f"📷 Connecting camera: {name}")
                cam.connect()

            self.is_connected_flag = True
            logger.info("✅ MKRobot: Connected!")

    def disconnect(self):
        if self.is_connected_flag:
            #断开所有摄像头
            for name, cam in self.cameras.items():
                cam.disconnect()

            self.robot.disconnect()
            self.is_connected_flag = False

    @property
    def is_connected(self):
        return self.is_connected_flag

    @property
    def bus(self):
        # 返回适配器而不是底层驱动
        return self._bus_adapter

    @property
    def is_calibrated(self):
        """假定电机已校准好，或者不需要校准"""
        return True

    def calibrate(self):
        """校准流程（空实现）"""
        pass

    def configure(self, config):
        """配置流程（空实现）"""
        pass

    @property
    def action_features(self):
        """定义动作空间的数据结构"""
        return {
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            }
        }

    @property
    def observation_features(self):
        """定义观测空间的数据结构"""
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            },
            # 如果你也返回速度，可以在这里加上
            # "observation.velocity": { ... }
        }

    def capture_images(self) -> Dict[str, Any]:
        """读取所有已连接摄像头的图像"""
        images = {}
        for name, camera in self.cameras.items():
            # 优先尝试异步读取以提高帧率，如果不支持则使用普通读取
            if hasattr(camera, "async_read"):
                images[name] = camera.async_read()
            else:
                images[name] = camera.read()
        return images

    # =========================================================
    # 🕹️ 核心收发逻辑
    # =========================================================

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        接收 Sim 坐标系动作 (URDF) -> 转换为 Real 动作 -> 发送
        """
        if not self.is_connected: return action

        # 1. 转换格式 (Tensor -> Numpy)
        if isinstance(action, torch.Tensor):
            q_sim = action.cpu().numpy()
        else:
            q_sim = action

        # 2. 关节角度映射 (Sim -> Real)
        # 前6轴乘系数
        q_real_joints = q_sim[:6] * HARDWARE_DIR
        
        # 3. 夹爪映射 (Sim -> Real)
        # 假设 Teleop 输出的是归一化 0.0(Open)~1.0(Close)
        # 如果你的真机是 1.0=Close, 0.0=Open，则直接用
        g_real = np.clip(q_sim[6], 0.0, 1.0)

        # 4. 组装字典发送
        command = {
            "joint_1.pos": q_real_joints[0],
            "joint_2.pos": q_real_joints[1],
            "joint_3.pos": q_real_joints[2],
            "joint_4.pos": q_real_joints[3],
            "joint_5.pos": q_real_joints[4],
            "joint_6.pos": q_real_joints[5],
            "gripper.pos": g_real
        }
        self.robot.send_action(command)
        
        return action

    def get_observation(self) -> Dict[str, Any]:
        """
        读取 Real 状态 -> 转换为 Sim 坐标系 (URDF) -> 返回
        """
        if not self.is_connected:
            # 返回空或零值，防止崩溃
            return {"observation.state": torch.zeros(7)}

        raw_obs = self.robot.get_observation()
        
        # 1. 解析并转换关节 (Real -> Sim)
        q_sim = np.zeros(7)
        q_sim[0] = raw_obs.get('joint_1.pos', 0) * HARDWARE_DIR[0]
        q_sim[1] = raw_obs.get('joint_2.pos', 0) * HARDWARE_DIR[1]
        q_sim[2] = raw_obs.get('joint_3.pos', 0) * HARDWARE_DIR[2]
        q_sim[3] = raw_obs.get('joint_4.pos', 0) * HARDWARE_DIR[3]
        q_sim[4] = raw_obs.get('joint_5.pos', 0) * HARDWARE_DIR[4]
        q_sim[5] = raw_obs.get('joint_6.pos', 0) * HARDWARE_DIR[5]

        # 2. 解析并转换夹爪 (Real -> Sim)
        # 假设真机返回 0.0~1.0
        g_real = raw_obs.get('gripper.pos', 0)
        q_sim[6] = g_real
        
        images = self.capture_images()

        # 必须包含: 
        #   - 独立的 joint_x.pos (供 GymEnv 读取)
        #   - 图像 (供 GymEnv 读取)
        #   - observation.state (供 Policy 使用)
        
        obs_dict = {
            "observation.state": torch.from_numpy(q_sim).float(),
            # 显式填入 GymEnv 需要的键名
            "joint_1.pos": q_sim[0],
            "joint_2.pos": q_sim[1],
            "joint_3.pos": q_sim[2],
            "joint_4.pos": q_sim[3],
            "joint_5.pos": q_sim[4],
            "joint_6.pos": q_sim[5],
            "gripper.pos": q_sim[6],
        }

        # 合并图像数据到字典中
        obs_dict.update(images)

        return obs_dict