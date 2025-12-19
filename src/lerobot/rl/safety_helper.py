import numpy as np
import torch
import pinocchio as pin
import logging

# 如果您不想依赖 lerobot 的 logger，可以直接用 print 或标准的 logging
logger = logging.getLogger("SafetyHelper")

class MKArmSafetyHelper:
    """
    简化的 MK 机械臂安全助手 (Standalone Version)
    不依赖 LeRobot Processor 框架，直接用于手动调用。
    """
    def __init__(self, urdf_path: str, min_z: float = 0.20, max_radius: float = 0.5):
        self.urdf_path = urdf_path
        self.min_z = min_z
        self.max_radius = max_radius
        
        logger.info(f"🔧 初始化 Safety Helper... (URDF: {urdf_path})")
        
        # 1. 加载 Pinocchio 模型
        try:
            with open(self.urdf_path, 'r') as f:
                urdf_str = f.read()
            self.model = pin.buildModelFromXML(urdf_str)
            self.data = self.model.createData()
        except Exception as e:
            logger.error(f"❌ URDF 加载失败: {e}")
            raise e

        # 2. 查找关键 Link ID
        target_link = "link4" 
        if self.model.existFrame(target_link):
            self.check_frame_id = self.model.getFrameId(target_link)
            logger.info(f"✅ Safety Target: {target_link} | Min Z: {self.min_z}m | Max R: {self.max_radius}m")
        else:
            fallback = "link3"
            if self.model.existFrame(fallback):
                self.check_frame_id = self.model.getFrameId(fallback)
                logger.warning(f"⚠️ Link4 not found, fallback to {fallback}")
            else:
                raise ValueError(f"Critical: Neither {target_link} nor {fallback} found in URDF!")

    def check_and_correct(self, action_tensor, current_q_tensor=None):
        """
        核心检查函数。
        输入: 
            - action_tensor: Policy 输出的目标关节角度 (Tensor)
            - current_q_tensor: (可选) 机械臂当前实际关节角度 (Tensor)，用于判断趋势
        输出:
            - safe_action_tensor: 经过安全修正后的动作
            - is_modified: 布尔值，是否被修改/拦截
        """
        # 1. 数据格式转换 (Tensor -> Numpy)
        device = action_tensor.device
        dtype = action_tensor.dtype
        
        if action_tensor.ndim > 1:
            q_next = action_tensor[0].detach().cpu().numpy()
        else:
            q_next = action_tensor.detach().cpu().numpy()

        q_curr = None
        if current_q_tensor is not None:
            if current_q_tensor.ndim > 1:
                q_curr = current_q_tensor[0].detach().cpu().numpy()
            else:
                q_curr = current_q_tensor.detach().cpu().numpy()
        
        # 如果没有当前位置，就只能用目标位置自己和自己比（无法判断趋势）
        if q_curr is None:
            q_curr = q_next

        # 2. 运动学计算 (FK)
        z_next, rad_next = self._compute_fk(q_next)
        z_curr, rad_curr = self._compute_fk(q_curr)

        # 3. 违规判定
        violation_z = z_next < self.min_z
        violation_rad = rad_next > self.max_radius
        
        is_safe = not (violation_z or violation_rad)

        if is_safe:
            return action_tensor, False

        # 4. 智能自救判定 (Smart Rescue)
        allow_rescue = True
        reasons = []

        if violation_z:
            # 如果在抬高 (给 1mm 容差)，允许
            if z_next > z_curr + 0.001:
                pass 
            else:
                allow_rescue = False
                reasons.append(f"Low Z ({z_next:.3f}<{self.min_z}) & Not Rising")

        if violation_rad:
            # 如果在收缩，允许
            if rad_next < rad_curr - 0.001:
                pass
            else:
                allow_rescue = False
                reasons.append(f"Far Radius ({rad_next:.3f}>{self.max_radius}) & Not Retracting")

        if allow_rescue:
            # logger.info("🛡️ 允许自救动作")
            return action_tensor, False
        
        # 5. 拦截处理 (Block)
        #logger.warning(f"🛡️ [BLOCK] {', '.join(reasons)}. Holding Position.")
        print(f"🛡️ [BLOCK] {', '.join(reasons)}. Holding Position.")
        
        # 策略：返回"当前位置"作为安全动作 (原地不动)
        # 注意：这里需要把 q_curr 转回 Tensor
        safe_q_numpy = q_curr
        safe_action = torch.from_numpy(safe_q_numpy).to(device).type(dtype)
        
        if action_tensor.ndim > 1:
            safe_action = safe_action.unsqueeze(0)
            
        return safe_action, True

    def _compute_fk(self, q):
        """内部工具：计算给定关节角的 Z 和 半径"""
        model_nq = self.model.nq
        q_pin = np.zeros(model_nq)
        n_copy = min(len(q), model_nq)
        q_pin[:n_copy] = q[:n_copy]
        
        pin.framesForwardKinematics(self.model, self.data, q_pin)
        pos = self.data.oMf[self.check_frame_id].translation
        
        z = pos[2]
        radius = np.linalg.norm(pos[:2])
        return z, radius