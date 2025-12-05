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

# --- 🛡️ 安全配置：物理关节限位 (单位: 弧度) ---
# 请根据您的 dk2.SLDASM.urdf 文件中的 limit lower/upper 进行核对修正
# 这里提供的是一组相对安全的默认值
JOINT_LIMITS = {
    # 关节索引: (最小弧度, 最大弧度)
    0: (-3.0, 3.0),  # Joint 1: 底座旋转 (通常范围很大)
    1: (0.0, 3.0),  # Joint 2: 大臂 (注意避免撞地)
    2: (0.0, 3.0),  # Joint 3: 肘部
    3: (-1.7, 1.2),  # Joint 4: 腕部旋转
    4: (-0.4, 0.4),  # Joint 5: 腕部弯曲
    5: (-2.0, 2.0),  # Joint 6: 法兰旋转
}

# # 真实机械臂的物理限位 (用于发送指令前的安全截断)
# REAL_JOINT_LIMITS = {
#     "joint_1": [-3.0, 3.0],
#     "joint_2": [-0.3, 3.0],
#     "joint_3": [0.0, 3.0],   # 注意：这是正值区间
#     "joint_4": [-1.7, 1.2],
#     "joint_5": [-0.4, 0.4],  # 范围较窄
#     "joint_6": [-2.0, 2.0]
# }

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
    # 0.15 rad ≈ 8.6度。在30Hz下允许最大角速度约 4.5 rad/s。
    # 这既能跟上 Reset 指令，又能防止 RL 策略输出 3.14 时的飞车事故。
    max_step_rad: float = 0.15
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
        # 安全相关：记录上一次的目标位置，用于平滑处理
        self.last_target_joints = None

    def connect(self):
        if not self.is_connected_flag:
            logger.info(f"🔗 MKRobot: Connecting to {self.config.port}...")
            self.robot.connect()
            #连接所有摄像头
            for name, cam in self.cameras.items():
                logger.info(f"📷 Connecting camera: {name}")
                cam.connect()

            self.is_connected_flag = True

            # 连接时读取当前位置作为初始目标，防止一上电就跳变
            init_obs = self.robot.get_observation()
            if init_obs:
                q_real = np.zeros(6)
                for i in range(6):
                    q_real[i] = init_obs.get(f'joint_{i+1}.pos', 0)
                self.last_target_joints = q_real

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
            # if hasattr(camera, "async_read"):
            #     images[name] = camera.async_read()
            # else:
            images[name] = camera.read()
        return images

    # =========================================================
    # 🛡️ 核心安全逻辑：速度平滑 + 绝对位置硬限位
    # =========================================================

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected: return action

        # 1. 格式转换
        if isinstance(action, torch.Tensor):
            q_sim_target = action.cpu().numpy()
        else:
            q_sim_target = action

        # 2. 映射到 Real 坐标系
        q_real_target = q_sim_target[:6] * HARDWARE_DIR
        g_real_target = np.clip(q_sim_target[6], 0.0, 1.0)

        # 3. 读取当前真实位置
        current_obs = self.robot.get_observation()
        q_real_current = np.zeros(6)
        for i in range(6):
            q_real_current[i] = current_obs.get(f'joint_{i+1}.pos', 0)

        # ---------------------------------------------------
        # 🛡️ 优化后的安全逻辑: 先位置截断，再速度截断
        # ---------------------------------------------------
        
        q_real_safe = np.zeros(6)
        
        for i in range(6):
            # A. 获取限位
            min_lim, max_lim = JOINT_LIMITS.get(i, (-3.14, 3.14))
            
            # B. 【关键】先将目标强行限制在物理限位内
            # 这样无论 Policy 想要去多远的地方，我们只把它当做想要去边界
            target_clamped = np.clip(q_real_target[i], min_lim, max_lim)
            
            # C. 计算 真实位置 -> 边界 的距离
            delta = target_clamped - q_real_current[i]
            
            # D. 对这个距离进行限速 (平滑处理)
            # 即使 current 在边界外 (例如 2.0, limit 1.6), delta 是 -0.4
            # 也会被平滑限制为 -0.15, 从而安全地慢慢退回，而不是剧烈跳变
            max_step = self.config.max_step_rad
            delta_safe = np.clip(delta, -max_step, max_step)
            
            # E. 最终指令
            q_real_safe[i] = q_real_current[i] + delta_safe

        # 4. 发送最终的安全指令
        command = {
            "joint_1.pos": q_real_safe[0],
            "joint_2.pos": q_real_safe[1],
            "joint_3.pos": q_real_safe[2],
            "joint_4.pos": q_real_safe[3],
            "joint_5.pos": q_real_safe[4],
            "joint_6.pos": q_real_safe[5],
            "gripper.pos": g_real_target
        }
        self.robot.send_action(command)
        
        return action

    # # =========================================================
    # # 🕹️ 核心收发逻辑
    # # =========================================================

    # def send_action(self, action: torch.Tensor) -> torch.Tensor:
    #     """
    #     接收 Sim 坐标系动作 (URDF) -> 转换为 Real 动作 -> 发送
    #     """
    #     if not self.is_connected: return action

    #     # 1. 转换格式 (Tensor -> Numpy)
    #     if isinstance(action, torch.Tensor):
    #         q_sim = action.cpu().numpy()
    #     else:
    #         q_sim = action

    #     # 2. 关节角度映射 (Sim -> Real)
    #     # 前6轴乘系数
    #     q_real_target = q_sim[:6] * HARDWARE_DIR
        
    #     # 3. 夹爪映射 (Sim -> Real)
    #     # 假设 Teleop 输出的是归一化 0.0(Open)~1.0(Close)
    #     # 如果你的真机是 1.0=Close, 0.0=Open，则直接用
    #     g_real = np.clip(q_sim[6], 0.0, 1.0)

    #     # --- 🛡️ 安全限速核心代码 START ---
    #     # 读取当前真实的电机位置
    #     current_obs = self.robot.get_observation()
    #     q_real_current = np.zeros(6)
    #     for i in range(6):
    #         q_real_current[i] = current_obs.get(f'joint_{i+1}.pos', 0)

    #     # 计算这一帧想移动的量 (Target - Current)
    #     delta = q_real_target - q_real_current
        
    #     # 强制截断：每帧最大只能移动 config.max_step_rad (默认0.05)
    #     # 这样即使策略输出 3.14，也只会移动 0.05，变成平滑的运动
    #     max_step = self.config.max_step_rad
    #     delta_clipped = np.clip(delta, -max_step, max_step)
        
    #     # 计算出实际发送给电机的安全目标
    #     q_real_safe = q_real_current + delta_clipped
        
    #     # 更新 gripper (夹爪通常不需要平滑，或者可以给大一点的阈值)
    #     # 这里直接通过
    #     # --- 🛡️ 安全限速核心代码 END ---


    #     # 4. 组装字典发送
    #     command = {
    #         "joint_1.pos": q_real_safe[0],
    #         "joint_2.pos": q_real_safe[1],
    #         "joint_3.pos": q_real_safe[2],
    #         "joint_4.pos": q_real_safe[3],
    #         "joint_5.pos": q_real_safe[4],
    #         "joint_6.pos": q_real_safe[5],
    #         "gripper.pos": g_real
    #     }
    #     self.robot.send_action(command)
        
    #     return action

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