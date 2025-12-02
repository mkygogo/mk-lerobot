import numpy as np
import pinocchio as pin
import logging
import os
import sys
from pathlib import Path
import meshcat.geometry as g
from datetime import datetime

# --- 移植过来的全局配置 (保持你的参数) ---
TRANS_SPEED = 0.002   
JOINT_SPEED = 0.02    
GRIPPER_SPEED = 0.002 
MAX_RADIUS = 0.5      
MIN_RADIUS_XY = 0.05 
MIN_JOINT4_Z = 0.227    
MAX_Y = -0.05 

# -------------------------------------------------------------------------
# 硬件方向修正 (Hardware Direction Correction)
# J2=1.0, 其他=-1.0
# -------------------------------------------------------------------------
HARDWARE_DIR = {
    "joint_1": -1.0, 
    "joint_2":  1.0, 
    "joint_3": -1.0, 
    "joint_4": -1.0, 
    "joint_5": -1.0, 
    "joint_6": -1.0,
    "gripper":  1.0 
}

# -------------------------------------------------------------------------
# 手柄控制方向 (Joystick Control Direction)
# -------------------------------------------------------------------------
CONTROL_DIR = {
    # 关节直控模式
    'CTRL_J1': -1.0, 'CTRL_J2': -1.0, 'CTRL_J3': -1.0, 
    'CTRL_J4':  1.0, 'CTRL_J5':  1.0, 'CTRL_J6':  1.0,

    # IK 模式 (Sim移动方向)
    'IK_X':  -1.0, 'IK_Y': 1.0, 'IK_Z':  -1.0,
    'IK_J4': 1.0, 'IK_J5': 1.0, 'IK_J6': 1.0
}


LOG_DIR = "logs"
file_handler = None  # 用于强制刷新

def setup_custom_logging():
    global file_handler
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger("MKArmLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # File Handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.info(f"📝 Log file created at: {log_path}")
    return logger

def force_flush_log():
    """强制将缓冲区写入硬盘，防止崩溃时丢失日志"""
    global file_handler
    if file_handler:
        file_handler.flush()
        try:
            os.fsync(file_handler.stream.fileno())
        except:
            pass

# 初始化日志
logger = setup_custom_logging()


# ==========================================
# 1. IK 解算器 (原封不动移植)
# ==========================================
class ThreeDofIKSolver:
    def __init__(self, model, data, frame_id, joint_limits):
        self.model = model
        self.data = data
        self.frame_id = frame_id
        self.joint_limits = joint_limits 
        
        self.max_iter = 15
        self.tol = 1e-3
        self.w_bias = 0.05
        # 仿真中的舒适姿态 (Sim坐标系：J3为负)
        self.q_ref_3dof = np.array([0.0, 1.5, -1.0]) 

    def solve(self, target_pos, q_current, dt=0.1):
        q = q_current.copy()
        debug_info = ""
        cond = 1.0
        final_err = 0.0
        success = False
        
        for i in range(self.max_iter):
            pin.framesForwardKinematics(self.model, self.data, q)
            current_pos = self.data.oMf[self.frame_id].translation
            
            err = target_pos - current_pos
            final_err = np.linalg.norm(err)
            
            if final_err < self.tol:
                success = True
                debug_info = "✅ Reached"
                break
            
            J = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_sub = J[:3, :3] 
            
            cond = np.linalg.cond(J_sub)
            damp = 1e-3 + 0.001 * (max(0, cond - 30))**2
            damp = min(damp, 0.1)

            H = J_sub.dot(J_sub.T) + damp * np.eye(3)
            v = J_sub.T.dot(np.linalg.solve(H, err))
            
            bias_force = self.w_bias * (self.q_ref_3dof - q[:3])
            v += bias_force * 0.1
            v = np.clip(v, -0.5, 0.5) 
            q[:3] += v * dt
            
            for k in range(3):
                q[k] = max(self.joint_limits[k][0], min(q[k], self.joint_limits[k][1]))
                
        if final_err > 0.05:
            debug_info = f"⛔ Diverged (Err:{final_err*100:.1f}cm)"
            success = False
        elif debug_info == "":
            debug_info = "✅ Reached"
            success = True
            
        return q, debug_info, cond, success, final_err

# ==========================================
# 2. 6自由度仿真臂 (移植，微调路径加载)
# ==========================================
class SixDofArm:
    def __init__(self, urdf_path, mesh_dir, ik_config=None):
        # 路径处理：确保是绝对路径
        self.urdf_path = str(Path(urdf_path).resolve()) if Path(urdf_path).exists() else urdf_path
        self.mesh_dir = str(Path(mesh_dir).resolve()) if Path(mesh_dir).exists() else mesh_dir
        
        self.model, self.collision_model, self.visual_model = self._load_model(self.urdf_path, self.mesh_dir)
        self.data = self.model.createData()
        
        # 解析 IK 配置 (矩形限位)
        self.ee_bounds_min = None
        self.ee_bounds_max = None
        target_frame_name = "link4" # 默认

        if ik_config:
            target_frame_name = ik_config.get("target_frame_name", "link4")
            if "end_effector_bounds" in ik_config:
                bounds = ik_config["end_effector_bounds"]
                self.ee_bounds_min = np.array(bounds.get("min", [-10, -10, -10]))
                self.ee_bounds_max = np.array(bounds.get("max", [10, 10, 10]))
                logger.info(f"📦 EE Bounds Set: Min={self.ee_bounds_min}, Max={self.ee_bounds_max}")

        # joints限位
        self.joint_limits = [
            [-3.0, 3.0],   # J1
            [-0.3, 3.0],   # J2
            [-3.0, 0.0],   # J3 (Sim坐标系)
            [-1.7, 1.2],   # J4
            [-0.4, 0.4],   # J5 
            [-2.0, 2.0],   # J6
            [0.0, 0.04],   # Gripper
        ]
        
        if self.model.existFrame("link4"):
            self.ik_frame_id = self.model.getFrameId("link4")
            logger.info(f"🎯 IK Target Frame: {target_frame_name} (ID: {self.ik_frame_id})")
        else:
            logger.warning(f"⚠️ Frame '{target_frame_name}' not found! Using 'link3'.")
            self.ik_frame_id = self.model.getFrameId("link3")
            
        self.ik_solver = ThreeDofIKSolver(self.model, self.data, self.ik_frame_id, self.joint_limits[:3])
        
        # 初始化姿态 (保持你的初始值)
        # 自动适配关节数量 (防止8轴报错)
        self.q = pin.neutral(self.model)
        init_vals = [0.020, 1.671, -0.670, -1.20, 0.0, 0.0]
        n_copy = min(len(init_vals), self.model.nq)
        self.q[:n_copy] = init_vals[:n_copy]
        
        self.in_zero_mode = False

        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy() 

    def _load_model(self, urdf_path, mesh_dir):
        # 你的原版加载逻辑，略微增强健壮性
        abs_urdf_path = os.path.abspath(urdf_path)
        abs_mesh_dir = os.path.abspath(mesh_dir)
        
        # 尝试寻找 meshes 文件夹
        if os.path.exists(os.path.join(abs_mesh_dir, "meshes")):
            meshes_folder_abs = os.path.join(abs_mesh_dir, "meshes")
        else:
            meshes_folder_abs = abs_mesh_dir # 如果 mesh_dir 本身就是 meshes

        with open(abs_urdf_path, 'r') as f: urdf_content = f.read()
        # 你的替换逻辑
        urdf_content = urdf_content.replace('filename="package://dk2.SLDASM/meshes/', f'filename="{meshes_folder_abs}/')
        urdf_content = urdf_content.replace('filename="../meshes/', f'filename="{meshes_folder_abs}/')
        
        import tempfile
        # 创建临时文件加载
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.urdf', delete=False) as tmp:
            tmp.write(urdf_content)
            tmp_urdf_path = tmp.name
        
        try:
            model = pin.buildModelFromXML(urdf_content)
            visual_model = pin.buildGeomFromUrdf(model, tmp_urdf_path, pin.GeometryType.VISUAL, package_dirs=abs_mesh_dir)
            collision_model = pin.buildGeomFromUrdf(model, tmp_urdf_path, pin.GeometryType.COLLISION, package_dirs=abs_mesh_dir)
        except Exception as e:
            logger.error(f"Failed to load URDF: {e}")
            raise e
        finally:
            if os.path.exists(tmp_urdf_path):
                os.remove(tmp_urdf_path)
        return model, collision_model, visual_model

    def set_state_from_hardware(self, q_real):
        """ SYNC 模式：q_real 已经是 SixDofRealArm 转换过的 Sim 坐标系数据 """
        n = min(len(self.q), len(q_real))
        self.q[:n] = q_real[:n]
        
        #夹爪单位换算 (归一化 0~1 -> 物理单位 0~0.04)
        # 如果不乘这个系数，仿真器会认为夹爪在“几米”远的地方，导致归位时动作长时间卡在最大值。
        if n > 6:
            # 假设 q[6] 是夹爪，且最大物理行程是 0.04 (与 step() 中的 clip 对应)
            self.q[6] = q_real[6] * 0.04

        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.ik_frame_id].translation.copy()
        self.valid_target_pos = self.target_pos.copy()
        
        self.ik_solver.q_ref_3dof = self.q[:3].copy()
        
        # [🚨 严重错误修复] 原来是 False，导致瞬间触发限位跳变
        # 改为 True，表示"当前状态是受信任的初始状态，暂时忽略限位检查"
        # 只有当用户推摇杆(has_input)时，update() 才会自动将其设为 False 并开始限位
        self.in_zero_mode = True

    def update(self, xyz_delta, manual_controls, dt=0.1):
        """ 完全保留你的 update 逻辑 (包含 Safety Clamping, Smoothing, IK) """
        has_input = np.linalg.norm(xyz_delta) > 1e-6 or any(val != 0 for val in manual_controls.values())
        if has_input:
            self.in_zero_mode = False

        # 1. 关节控制 (J4-J6 & Gripper)
        if manual_controls.get('j4', 0) != 0:
            self.q[3] += manual_controls['j4'] * JOINT_SPEED
            self.q[3] = np.clip(self.q[3], self.joint_limits[3][0], self.joint_limits[3][1])
        if manual_controls.get('j5', 0) != 0:
            self.q[4] += manual_controls['j5'] * JOINT_SPEED
            self.q[4] = np.clip(self.q[4], self.joint_limits[4][0], self.joint_limits[4][1])
        if manual_controls.get('j6', 0) != 0:
            self.q[5] += manual_controls['j6'] * JOINT_SPEED
            self.q[5] = np.clip(self.q[5], self.joint_limits[5][0], self.joint_limits[5][1])
        
        # 夹爪控制
        if manual_controls.get('gripper', 0) != 0:
            delta = manual_controls['gripper'] * GRIPPER_SPEED
            # 假设 q[6] 是左指, q[7] 是右指 (适配你的代码)
            if len(self.q) > 6:
                self.q[6] += delta 
                self.q[6] = np.clip(self.q[6], self.joint_limits[6][0], self.joint_limits[6][1])
            if len(self.q) > 7:
                self.q[7] -= delta 
                self.q[7] = np.clip(self.q[7], -self.joint_limits[6][1], self.joint_limits[6][0])

        # 2. XYZ IK 解算
        old_safe_pos = self.valid_target_pos.copy()
        self.target_pos += xyz_delta
        clamped_msg = ""
        ideal_pos = self.target_pos.copy()

        if not self.in_zero_mode:
            # 矩形盒限位 (新增)
            if self.ee_bounds_min is not None and self.ee_bounds_max is not None:
                ideal_pos = np.clip(ideal_pos, self.ee_bounds_min, self.ee_bounds_max)
                if not np.array_equal(ideal_pos, self.target_pos):
                    clamped_msg = "🔒 BoxLimit"

            #原有的球形和圆柱限位
            if ideal_pos[1] > MAX_Y: 
                ideal_pos[1] = MAX_Y
            if ideal_pos[2] < MIN_JOINT4_Z: 
                ideal_pos[2] = MIN_JOINT4_Z
            
            xy_dist = np.linalg.norm(ideal_pos[:2])
            if xy_dist < MIN_RADIUS_XY:
                if xy_dist < 1e-6: 
                    ideal_pos[:2] = [0, -MIN_RADIUS_XY]
                else: 
                    ideal_pos[:2] *= (MIN_RADIUS_XY / xy_dist)
            
            dist = np.linalg.norm(ideal_pos)
            if dist > MAX_RADIUS:
                ideal_pos *= (MAX_RADIUS / dist)
        else:
            clamped_msg = "⚠️ Zero Mode"

        # --- 你的平滑修正逻辑 ---
        SAFETY_SNAP_SPEED = 0.002 
        diff = ideal_pos - self.target_pos
        dist_err = np.linalg.norm(diff)
        
        if dist_err > 1e-6:
            clamped_msg = "🔒 SmoothClamp"
            if dist_err > SAFETY_SNAP_SPEED:
                self.target_pos += (diff / dist_err) * SAFETY_SNAP_SPEED
            else:
                self.target_pos = ideal_pos        

        # --- 调用你的 IK Solver ---
        q_new, debug_msg, cond, success, err = self.ik_solver.solve(self.target_pos, self.q)
        
        if not success:
            if not self.in_zero_mode: 
                self.target_pos = old_safe_pos.copy()
                debug_msg += " -> BLOCKED"
        else:
            self.q = q_new
            if err < 0.02:
                self.valid_target_pos = self.target_pos.copy()
                
        return debug_msg, cond, clamped_msg, success

# ==========================================
# 3. 封装给 LeRobot 使用的接口类
# ==========================================
class MKArmIKCore:
    """
    这个类作为 'SixDofSim' 的替代品。
    它负责初始化 Arm，处理 Meshcat，并提供 step() 接口。
    """
    def __init__(self, urdf_path, mesh_dir, visualize=True, ik_config=None):
        self.arm = SixDofArm(urdf_path, mesh_dir)
        self.visualize = visualize
        self.viz = None
        
        self.log_counter = 0 # 计数器
        # 强制刷新一次日志，确立文件头
        force_flush_log()

        if self.visualize:
            self._init_visualizer()

        self.is_homing = False
        self.HOMING_SPEED = 0.005 # 归位速度 (rad/step)，约 0.3 rad/s

    def _init_visualizer(self):
        try:
            from pinocchio.visualize import MeshcatVisualizer
            self.viz = MeshcatVisualizer(self.arm.model, self.arm.collision_model, self.arm.visual_model)
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
            self.viz.display(self.arm.q)
            
            self.viz.viewer["target"].set_object(g.Sphere(0.04), g.MeshBasicMaterial(color=0xff0000, opacity=0.8))
            self.viz.viewer["workspace_outer"].set_object(g.Sphere(MAX_RADIUS), 
                                            g.MeshBasicMaterial(color=0xffffff, opacity=1, wireframe=True))
            cyl_geom = g.Cylinder(0.4, MIN_RADIUS_XY, MIN_RADIUS_XY)
            self.viz.viewer["workspace_inner"].set_object(cyl_geom, 
                                            g.MeshBasicMaterial(color=0xff0000, opacity=1, wireframe=False))
            self.viz.viewer["workspace_inner"].set_transform(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0.2],[0,0,0,1]]))
            
            # 可视化矩形安全盒
            if self.arm.ee_bounds_min is not None:
                center = (self.arm.ee_bounds_min + self.arm.ee_bounds_max) / 2
                dims = self.arm.ee_bounds_max - self.arm.ee_bounds_min
                self.viz.viewer["safety_box"].set_object(g.Box(dims), g.MeshBasicMaterial(color=0x00ff00, opacity=0.1, wireframe=True))
                self.viz.viewer["safety_box"].set_transform(pin.SE3(np.eye(3), center).homogeneous)

            logger.info("✨ Meshcat Initialized")
        except Exception as e:
            logger.warning(f"Meshcat Init Failed: {e}")

    def step(self, xyz_delta, manual_controls):
        """
        相当于你 Sim 循环中的一次迭代。
        """
        # 调用 arm.update
        debug_msg, cond, clamp_msg, success = self.arm.update(xyz_delta, manual_controls)
        
        # 更新可视化
        if self.viz:
            self.viz.display(self.arm.q)
            self.viz.viewer["target"].set_transform(pin.SE3(np.eye(3), self.arm.target_pos).homogeneous)
        
        # 返回 LeRobot 需要的格式 [j1...j6, gripper]
        # 注意：这里我们返回的是 Simulation 坐标系的 q
        # LeRobot 录制的数据通常就是 Policy 的 Action。
        # 你的 Robot 类 (mk_robot.py) 负责把这个 q 转成硬件指令 (Hardware Dir 修正)。
        
        self.log_counter += 1
        # 构造信息字符串
        info_str = (f"🎮 Teleop | {debug_msg} {clamp_msg} | "
                    f"Tgt:[{self.arm.target_pos[0]:.3f}, {self.arm.target_pos[1]:.3f}, {self.arm.target_pos[2]:.3f}] | "
                    f"J:[{self.arm.q[0]:.2f}, {self.arm.q[1]:.2f}, {self.arm.q[2]:.2f}, "
                    f"{self.arm.q[3]:.2f}, {self.arm.q[4]:.2f}, {self.arm.q[5]:.2f}]")
        
        # 智能日志过滤逻辑
        # 1. 总是打印: 发生发散(Diverged) 或 卡死(BLOCKED)
        # 2. 正常打印: 不在 Zero Mode 时，每 20 帧打印一次 (保持原频率)
        # 3. 静默模式: 在 Zero Mode 时，每 600 帧 (约20秒) 才打印一次心跳，避免刷屏
        is_error = "Diverged" in debug_msg or "BLOCKED" in debug_msg
        is_active = not self.arm.in_zero_mode
        should_log = is_error or \
                     (is_active and self.log_counter % 20 == 0) or \
                     (not is_active and self.log_counter % 600 == 0)

        if should_log:
            logger.info(info_str)
            force_flush_log()
            print(info_str, end='\r')   

        # 提取前 6 个关节
        action = self.arm.q[:6].copy()
        
        gripper_raw = self.arm.q[6]
        gripper_norm = np.clip(gripper_raw / 0.04, 0.0, 1.0) # 0=Close, 1=Open? 
        # 你的代码里：q[6] += delta (Open方向)
        
        return np.append(action, gripper_norm)
    

    # 暴露同步接口
    def set_state_from_hardware(self, q_sim_array):
        """
        强制覆盖 IK 内部状态，使其与真机同步。
        q_sim_array: 必须是已经转换到 Sim 坐标系的 numpy 数组 [7]
        """
        # 调用 SixDofArm 的 set_state_from_hardware (你原脚本里已经写好了逻辑)
        self.arm.set_state_from_hardware(q_sim_array)
        
        # 顺便更新一下 Meshcat，让你看到同步效果
        if self.viz:
            self.viz.display(self.arm.q)
            self.viz.viewer["target"].set_transform(pin.SE3(np.eye(3), self.arm.target_pos).homogeneous)


    #启动归位模式
    def start_homing(self):
        if not self.is_homing:
            self.is_homing = True
            logger.info("🚀 Starting Homing to ZERO...")

    def step_homing(self):
        """
        让所有关节平滑地向 0.0 插值移动
        """
        is_done = True
        # 1. 计算插值
        for i in range(len(self.arm.q)):
            diff = 0.0 - self.arm.q[i]
            if abs(diff) > 1e-4:
                step = np.sign(diff) * min(abs(diff), self.HOMING_SPEED)
                self.arm.q[i] += step
                is_done = False
        
        # 2. 更新 FK 和 Target Pos
        pin.framesForwardKinematics(self.arm.model, self.arm.data, self.arm.q)
        self.arm.target_pos = self.arm.data.oMf[self.arm.ik_frame_id].translation.copy()
        self.arm.valid_target_pos = self.arm.target_pos.copy()

        # 3. 更新可视化
        if self.viz:
            self.viz.display(self.arm.q)
            self.viz.viewer["target"].set_transform(pin.SE3(np.eye(3), self.arm.target_pos).homogeneous)

        if is_done:
            self.is_homing = False
            # 必须设置为 True！
            # 这会告诉 SixDofArm.update()：我现在在 0 位，不要用 MIN_RADIUS_XY 检查我。
            # 只有当你下次推摇杆时，in_zero_mode 才会自动变为 False。
            self.arm.in_zero_mode = True 
            
            # 顺便把 IK 的“舒适姿态偏置”也归零，防止 IK 试图把它拉歪
            self.arm.ik_solver.q_ref_3dof = np.array([0.0, 0.0, 0.0])
            
            logger.info("✅ Homing Complete. Entered Zero Mode.")

        # 5. 返回动作
        action = self.arm.q[:6].copy()
        gripper_raw = self.arm.q[6]
        gripper_norm = np.clip(gripper_raw / 0.04, 0.0, 1.0)
        
        return np.append(action, gripper_norm)