import numpy as np
import torch
import pinocchio as pin
from dataclasses import dataclass
import logging

# 从正确的模块导入基类
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

logger = logging.getLogger(__name__)

@dataclass
@ProcessorStepRegistry.register("mk_arm_safety_processor")
class MKArmSafetyProcessorStep(ProcessorStep):
    """
    MK Arm 安全拦截器：使用 Pinocchio 进行 FK 校验，防止 Policy 输出危险动作。
    """
    urdf_path: str
    # 默认限制 (参考你的 mk_arm_ik_core.py)
    min_z: float = 0.227
    max_radius: float = 0.5
    
    def __post_init__(self):
        # 加载 Pinocchio 模型用于 FK 计算
        self.model = pin.buildModelFromXML(open(self.urdf_path).read())
        self.data = self.model.createData()
        
        # 获取末端 Frame ID (假设是 link4)
        if self.model.existFrame("link4"):
            self.ee_frame_id = self.model.getFrameId("link4")
        else:
            self.ee_frame_id = self.model.getFrameId("link3") # Fallback
            
        self.last_safe_action = None
        logger.info(f"🛡️ Safety Processor initialized. Model nq={self.model.nq}, Target Frame ID={self.ee_frame_id}")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        
        # 1. 格式转换 (Tensor -> Numpy)
        if isinstance(action, torch.Tensor):
            q = action.cpu().numpy().flatten()
        else:
            q = np.array(action).flatten()
            
        # 2. 维度适配 (Critical Fix)
        # Pinocchio 期望 model.nq (通常是8)，而我们只有 7 个关节值
        model_nq = self.model.nq
        q_pin = np.zeros(model_nq) # 创建全 0 向量
        
        # 将我们的 7 个值填入前 7 位
        n_copy = min(len(q), model_nq)
        q_pin[:n_copy] = q[:n_copy]
        
        # 3. 计算 FK
        # 使用补齐后的 q_pin 进行计算
        pin.framesForwardKinematics(self.model, self.data, q_pin)
        curr_pos = self.data.oMf[self.ee_frame_id].translation # [x, y, z]
        
        is_unsafe = False
        reason = ""

        # 4. 安全检查逻辑
        # 检查高度 Z
        if curr_pos[2] < self.min_z:
            is_unsafe = True
            reason = f"Low Z ({curr_pos[2]:.3f} < {self.min_z})"
            
        # 检查工作半径 (防止伸太远或撞到底座)
        # 只检查 XY 平面半径往往更实用，或者全距离
        dist_xy = np.linalg.norm(curr_pos[:2])
        if dist_xy > self.max_radius: 
            is_unsafe = True
            reason = f"Max Radius ({dist_xy:.3f} > {self.max_radius})"

        # 5. 处置逻辑
        if is_unsafe:
            if self.last_safe_action is not None:
                # 触发保护：用上一次的安全动作覆盖当前动作 (Hold)
                if logger.isEnabledFor(logging.WARNING):
                     # 限制日志频率，防止刷屏 (可选)
                     logger.warning(f"🛡️ Safety Triggered: {reason} -> Holding Position")
                
                # 还原为 Tensor 并保持与原始 action 相同的设备和类型
                safe_action_tensor = torch.from_numpy(self.last_safe_action)
                if isinstance(action, torch.Tensor):
                    safe_action_tensor = safe_action_tensor.to(action.device).type(action.dtype)
                
                # 恢复 Batch 维度
                if isinstance(action, torch.Tensor) and action.ndim > 1:
                     safe_action_tensor = safe_action_tensor.unsqueeze(0)
                    
                transition[TransitionKey.ACTION] = safe_action_tensor
            else:
                logger.warning(f"🛡️ Safety Triggered: {reason} -> No history, passing through (Dangerous!)")
        else:
            # 记录当前安全动作 (保存为原始维度的副本)
            self.last_safe_action = q.copy()

        return transition

    def transform_features(self, features):
        return features
    
    def get_config(self):
        return {"urdf_path": self.urdf_path, "min_z": self.min_z}
    
    #在回合结束时清空记忆
    def reset(self):
        self.last_safe_action = None
        logger.info("🛡️ Safety Processor reset: History cleared.")