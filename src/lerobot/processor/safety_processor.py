import numpy as np
import torch
import pinocchio as pin
from dataclasses import dataclass
import logging
import os

from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.teleoperators.utils import TeleopEvents

logger = logging.getLogger(__name__)

@dataclass
@ProcessorStepRegistry.register("mk_arm_safety_processor")
class MKArmSafetyProcessorStep(ProcessorStep):
    """
    MK Arm 安全拦截器 (HIL-SERL 专用版)
    
    安全逻辑：
    1. 针对 Joint 4 (Wrist) 进行 FK 高度校验 (Z > min_z)。
    2. 如果检测到人工介入 (Teleop)，则**无条件放行**并更新安全状态。
    3. 如果 Policy 违规，则保持在上一帧的安全位置 (Hold)。
    """
    urdf_path: str
    min_z: float = 0.25  # Joint 4 (Link 4) 的最小高度，防止腕部撞桌子
    max_radius: float = 0.5 # 工作半径限制
    
    def __post_init__(self):
        # 加载 Pinocchio 模型

        self.model = pin.buildModelFromXML(open(self.urdf_path).read())
        self.data = self.model.createData()
        
        # ⚠️ 关键修正：根据用户指示，IK只算到 J3，高度限制作用于 Joint 4
        # 因此我们必须获取 Link 4 的 Frame ID
        target_link = "link4" 
        if self.model.existFrame(target_link):
            self.check_frame_id = self.model.getFrameId(target_link)
            logger.info(f"🛡️ Safety Target: {target_link} (ID={self.check_frame_id}) | Min Z: {self.min_z}m")
        else:
            # 如果找不到 link4，回退到 link3 并发出警告
            fallback = "link3"
            self.check_frame_id = self.model.getFrameId(fallback)
            logger.warning(f"⚠️ Link4 not found! Fallback to safety check on: {fallback}")
            
        self.last_safe_action = None

    def transform_features(self, features):
        # 安全处理器只修改数值，不改变 Tensor 的形状或类型，所以直接返回原特征
        return features

    def get_config(self):
        # 返回配置参数，用于序列化保存
        return {
            "urdf_path": self.urdf_path, 
            "min_z": self.min_z,
            "max_radius": self.max_radius
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        info = transition.get(TransitionKey.INFO, {})
        
        # 1. 人工介入检查 (HIL 核心逻辑)
        # 如果是人工在操作，我们假设人类知道自己在做什么，无条件放行
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)

        # 格式转换 (Tensor -> Numpy)
        if isinstance(action, torch.Tensor):
            q = action.cpu().numpy().flatten()
            device = action.device
            dtype = action.dtype
        else:
            q = np.array(action).flatten()
            device = "cpu"
            dtype = torch.float32

        # 如果是人工操作，直接更新历史记录并返回
        if is_intervention:
            self.last_safe_action = q.copy()
            return transition

        # -------------------------------------------------------------
        # 以下逻辑仅针对 Policy (自动驾驶) 状态
        # -------------------------------------------------------------

        # 2. 维度适配 Pinocchio (补齐8轴)
        model_nq = self.model.nq
        q_pin = np.zeros(model_nq)
        n_copy = min(len(q), model_nq)
        q_pin[:n_copy] = q[:n_copy]
        
        # 3. 计算 FK (针对 Link 4)
        pin.framesForwardKinematics(self.model, self.data, q_pin)
        curr_pos = self.data.oMf[self.check_frame_id].translation # [x, y, z]
        
        is_unsafe = False
        reason = ""

        # 4. 安全规则检查
        # 规则 A: Joint 4 高度限制
        if curr_pos[2] < self.min_z:
            is_unsafe = True
            reason = f"Link4 Low Z ({curr_pos[2]:.3f} < {self.min_z})"
            
        # 规则 B: 工作半径限制 (XY平面)
        dist_xy = np.linalg.norm(curr_pos[:2])
        if dist_xy > self.max_radius: 
            is_unsafe = True
            reason = f"Max Radius ({dist_xy:.3f} > {self.max_radius})"

        # 5. 处置逻辑
        if is_unsafe:
            if self.last_safe_action is not None:
                # 触发保护：回滚到上一次的安全动作 (Hold Position)
                # 这比置零更安全，防止机械臂突然掉下来
                # logger.warning(f"🛡️ Safety Triggered: {reason} -> Holding Position") # 可选：减少日志刷屏
                
                safe_action_tensor = torch.from_numpy(self.last_safe_action).to(device).type(dtype)
                
                # 恢复 Batch 维度
                if isinstance(action, torch.Tensor) and action.ndim > 1:
                     safe_action_tensor = safe_action_tensor.unsqueeze(0)
                    
                transition[TransitionKey.ACTION] = safe_action_tensor
            else:
                logger.warning(f"🛡️ Safety Triggered: {reason} -> No history, passing through (Critical!)")
        else:
            # 记录当前安全动作
            self.last_safe_action = q.copy()

        return transition

    def reset(self):
        self.last_safe_action = None
        logger.info("🛡️ Safety Processor reset.")