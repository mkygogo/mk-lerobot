import pygame
import torch
import numpy as np
import logging
from dataclasses import dataclass, field
import time
from typing import Dict, Optional

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.utils import TeleopEvents
from .mk_arm_ik_core import MKArmIKCore

logger = logging.getLogger(__name__)

@TeleoperatorConfig.register_subclass("gamepad_ik")
@dataclass
class GamepadIKTeleopConfig(TeleoperatorConfig):
    type: str = "gamepad_ik"
    urdf_path: str = ""
    mesh_dir: str = ""
    fps: int = 60
    visualize: bool = True
    inverse_kinematics: Optional[Dict] = field(default_factory=dict)
    trans_speed: float = 0.002
    rot_speed: float = 0.02

class GamepadIKTeleop(Teleoperator):
    def __init__(
        self,
        config: GamepadIKTeleopConfig = None,
        urdf_path: str = None,
        mesh_dir: str = None,
        fps: int = 60,
        visualize: bool = True,
        inverse_kinematics: dict = None
    ):
        
        if config is not None:
            self.config = config
        else:
            if urdf_path is None or mesh_dir is None:
                raise ValueError("GamepadIKTeleop: If 'config' is not provided, 'urdf_path' and 'mesh_dir' are required.")
            
            self.config = GamepadIKTeleopConfig(
                type="gamepad_ik",
                urdf_path=urdf_path,
                mesh_dir=mesh_dir,
                fps=fps,
                visualize=visualize,
                inverse_kinematics=inverse_kinematics or {},
                # 记得加上这两个默认值，防止报错
                trans_speed=0.002, 
                rot_speed=0.02
            )
        
        super().__init__(config=config)

        # 初始化 Core
        self.core = MKArmIKCore(self.config.urdf_path, 
            self.config.mesh_dir, 
            self.config.visualize, 
            ik_config=self.config.inverse_kinematics)
        
        self.x_press_start_time = None # 用于长按计时
        self.BTN_X = 2 # Xbox 手柄 X键通常是 ID 2，请根据你的实际情况调整
        #RB 键和安全锁状态
        self.BTN_RB = 5  # Xbox 手柄 RB 键通常是 5，根据实际情况调整
        self.rb_safety_lock = False # 防止归位后立刻误触发

        #用于记录上一帧 RB 状态，实现上升沿检测
        self.prev_rb_state = False

        #状态标志位，用于 get_teleop_events
        self.is_active = False

        self.joystick = None
        self._init_pygame()

        #启动同步标志位
        # 只要这个是 False，说明还没有根据真机状态初始化过
        self.has_synced_startup = False

    def _init_pygame(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logger.info(f"🎮 Teleop: Connected to {self.joystick.get_name()}")
        else:
            logger.warning("⚠️ Teleop: No Joystick found!")
            self.joystick = None

    # --- 映射逻辑 (参考 SixDofSim._get_inputs) ---
    def _get_inputs(self):
        xyz_delta = np.zeros(3)
        manual = {'j4':0, 'j5':0, 'j6':0, 'gripper':0}
        
        if not self.joystick: 
            return xyz_delta, manual

        # 死区过滤
        def filter_stick(val):
            return 0.0 if abs(val) < 0.15 else val

        # 读取轴 (Xbox Mapping)
        # 注意：这里请根据你实际手柄 ID 调整，参考你的原脚本
        lx = filter_stick(self.joystick.get_axis(0)) # AXIS_LX
        ly = filter_stick(self.joystick.get_axis(1)) # AXIS_LY
        rx = filter_stick(self.joystick.get_axis(3)) # AXIS_RX
        ry = filter_stick(self.joystick.get_axis(4)) # AXIS_RY
        hat = self.joystick.get_hat(0)

        # 你的控制方向定义
        # 'IK_X': -1.0, 'IK_Y': 1.0, 'IK_Z': -1.0
        # TRANS_SPEED 已经在 Core 里定义了，这里我们传 Normalized 值?
        # 不，你的 Arm.update 期望的是 delta 距离。
        # 所以这里要乘速度。
        
        # 为了保持一致，我们在 Core 里没有把 TRANS_SPEED 变成 global 常量，
        # 而是 Arm.update 接收 xyz_delta。
        # 我们可以把 TRANS_SPEED 定义在 Core 的 global 里，或者这里硬编码。
        TRANS_SPEED = self.config.trans_speed
        
        xyz_delta[0] = -1.0 * lx * TRANS_SPEED # IK_X
        xyz_delta[1] =  1.0 * ly * TRANS_SPEED # IK_Y
        xyz_delta[2] = -1.0 * ry * TRANS_SPEED # IK_Z
        
        manual['j4'] = -hat[1]
        manual['j5'] = -rx
        manual['j6'] = -hat[0]
        
        # 夹爪
        lt_val = (self.joystick.get_axis(2) + 1) / 2
        rt_val = (self.joystick.get_axis(5) + 1) / 2
        if rt_val > 0.1: 
            manual['gripper'] = 1
        elif lt_val > 0.1: 
            manual['gripper'] = -1
        
        return xyz_delta, manual

    # --- LeRobot 接口 ---
    @property
    def name(self) -> str: 
        return self.config.type
    
    def connect(self): 
        if not self.joystick: 
            self._init_pygame()
    
    def disconnect(self): 
        pygame.quit()
    
    @property
    def is_connected(self) -> bool: 
        return self.joystick is not None
    
    @property
    def is_calibrated(self) -> bool: 
        return True

    def calibrate(self): 
        pass

    def configure(self, config): 
        pass
    
    @property
    def action_features(self):
        return {"action": {"dtype": "float32", "shape": (7,), "names": ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","gripper"]}}
    
    @property
    def feedback_features(self): 
        return None
    
    def send_feedback(self, feedback): 
        pass

    def get_teleop_events(self):
        """
        返回当前遥操作事件状态。
        gym_manipulator 必须调用此方法来判断是否处于人工干预模式。
        """
        return {
            TeleopEvents.IS_INTERVENTION: self.is_active
        }

    # def get_action(self, observation: dict) -> torch.Tensor:
    #     pygame.event.pump()
        
    #     #启动时的首帧强制同步 (接口层安全保障)
    #     # 这确保了无论什么脚本调用，第一帧永远是“吸附”在真机当前位置的，绝对不会跳变
    #     if "observation.state" in observation:
    #         current_state = observation["observation.state"]
    #         if isinstance(current_state, torch.Tensor):
    #             current_state = current_state.cpu().numpy()

    #         if not self.has_synced_startup:
    #             self.core.set_state_from_hardware(current_state)
    #             self.has_synced_startup = True
    #             logger.info("🛡️ Safety: Teleop first-frame synced with hardware.")
    #             # 直接返回当前状态，跳过后续所有计算，确保绝对静止
    #             return torch.from_numpy(current_state).float()

    #     # ========================================================
    #     # 1. 状态监测与安全锁处理 (Deadman Switch & Safety Lock)
    #     # ========================================================
    #     # 获取物理按键状态
    #     phys_rb_pressed = (self.joystick.get_button(self.BTN_RB) == 1)
        
    #     # 处理安全锁：如果锁着，必须先松手才能解锁
    #     if self.rb_safety_lock:
    #         if not phys_rb_pressed:
    #             self.rb_safety_lock = False # 解锁
    #             logger.info("🔓 Safety Lock Disengaged (RB Released)")
    #         # 锁定期强制视为没按
    #         self.is_active = False
    #     else:
    #         self.is_active = phys_rb_pressed

    #     # ========================================================
    #     # 2. X键 长按归位检测 (最高优先级)
    #     # ========================================================
    #     if self.joystick.get_button(self.BTN_X):
    #         if self.x_press_start_time is None:
    #             self.x_press_start_time = time.time()
    #         elif time.time() - self.x_press_start_time > 2.0: 
    #             self.core.start_homing()
    #     else:
    #         self.x_press_start_time = None

    #     # ========================================================
    #     # 3. 归位模式执行 (Homing Mode)
    #     # ========================================================
    #     if self.core.is_homing:
    #         action_array = self.core.step_homing()
            
    #         # [关键] 检测归位是否刚刚结束
    #         # 如果这一步跑完，Core 里的标志位变 False 了，说明刚结束 -> 上锁
    #         if not self.core.is_homing:
    #             self.rb_safety_lock = True
    #             logger.info("🔒 Safety Lock Engaged (Homing Complete)")
                
    #         return torch.from_numpy(action_array).float()

    #     # ========================================================
    #     # 4. 常规控制模式 (HIL-SERL)
    #     # ========================================================
        
    #     # 获取手柄输入
    #     xyz_delta, manual = self._get_inputs()
        
    #     # [逻辑修改] 真机模式下，必须按住 RB 才算介入 (Active)，否则为同步 (Passive)
    #     # 纯仿真模式下 (没有 observation)，总是视为 Active
        
    #     if "observation.state" in observation:
    #         # --- 真机 / Gym 环境 ---
    #         current_state = observation["observation.state"]
    #         if isinstance(current_state, torch.Tensor):
    #             current_state = current_state.cpu().numpy()

    #         if self.is_active:
    #             #刚按下 RB 的瞬间，同步一次真机位置，防止跳变
    #             if not self.prev_rb_state:
    #                 self.core.set_state_from_hardware(current_state)
    #                 logger.info("🎮 Active Control Engaged: Synced with Hardware")
    #             # [主动控制] 按住了 RB -> 允许 IK 计算和移动
    #             # 即使摇杆不动，这里也应该调用 step，保持 IK 目标点稳定（Hold）
    #             action_array = self.core.step(xyz_delta, manual)
    #         else:
    #             # 没按 RB
    #             # 旧代码：self.core.set_state_from_hardware(current_state) -> 导致震荡发热
    #             # 新代码：发送全0的 delta，让 IK Core 保持输出上一次的稳定目标值
    #             action_array = self.core.step(np.zeros(3), {})
            
    #         self.prev_rb_state = self.is_active # 更新状态
    #     else:
    #         # --- 纯仿真模式 (Sim Only) ---
    #         # 这种模式下通常没有 observation，我们允许直接控制，不需要按 RB
    #         action_array = self.core.step(xyz_delta, manual)

    #     return torch.from_numpy(action_array).float()

    def get_action(self, observation: dict) -> torch.Tensor:
        pygame.event.pump()
        
        # --- 1. 处理观测数据 (处理 Batch 和 Tensor) ---
        current_state = None
        if "observation.state" in observation:
            raw_state = observation["observation.state"]
            
            # 统一转为 Numpy
            if isinstance(raw_state, torch.Tensor):
                raw_state = raw_state.cpu().numpy()
            elif not isinstance(raw_state, np.ndarray):
                raw_state = np.array(raw_state)

            # [核心修复] 强制压平数组 (flatten)，彻底解决 (1,14) vs (14,) 的问题
            current_state = raw_state.flatten()

        # --- 2. 启动同步 (Startup Sync) ---
        if current_state is not None:
            if not self.has_synced_startup:
                self.core.set_state_from_hardware(current_state)
                self.has_synced_startup = True
                logger.info("🛡️ Safety: Teleop first-frame synced with hardware.")
                
                # 返回对应长度的动作 (防止越界)
                n_joints = 7 # 假设7轴
                action_out = current_state[:n_joints] if len(current_state) >= n_joints else current_state
                return torch.from_numpy(action_out).float()

        # ========================================================
        # 3. 状态监测与安全锁处理 (Deadman Switch & Safety Lock)
        # ========================================================
        phys_rb_pressed = (self.joystick.get_button(self.BTN_RB) == 1)
        
        if self.rb_safety_lock:
            if not phys_rb_pressed:
                self.rb_safety_lock = False
                logger.info("🔓 Safety Lock Disengaged (RB Released)")
            self.is_active = False
        else:
            self.is_active = phys_rb_pressed

        # ========================================================
        # 4. X键 长按归位检测
        # ========================================================
        if self.joystick.get_button(self.BTN_X):
            if self.x_press_start_time is None:
                self.x_press_start_time = time.time()
            elif time.time() - self.x_press_start_time > 2.0: 
                self.core.start_homing()
        else:
            self.x_press_start_time = None

        # ========================================================
        # 5. 归位模式执行
        # ========================================================
        if self.core.is_homing:
            action_array = self.core.step_homing()
            if not self.core.is_homing:
                self.rb_safety_lock = True
                logger.info("🔒 Safety Lock Engaged (Homing Complete)")
            return torch.from_numpy(action_array).float()

        # ========================================================
        # 6. 常规控制模式 (HIL-SERL)
        # ========================================================
        xyz_delta, manual = self._get_inputs()
        
        if current_state is not None:
            # --- 真机模式 ---
            if self.is_active:
                if not self.prev_rb_state:
                    self.core.set_state_from_hardware(current_state)
                    logger.info("🎮 Active Control Engaged: Synced with Hardware")
                action_array = self.core.step(xyz_delta, manual)
            else:
                # 没按 RB -> 保持 IK 目标不变，不吸附真机
                action_array = self.core.step(np.zeros(3), {})
            
            self.prev_rb_state = self.is_active
        else:
            # --- 纯仿真模式 ---
            action_array = self.core.step(xyz_delta, manual)

        return torch.from_numpy(action_array).float()
