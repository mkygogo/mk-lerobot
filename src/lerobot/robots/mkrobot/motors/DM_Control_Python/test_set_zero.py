#!/usr/bin/env python3
"""
test_set_zero.py
测试代码：控制TRLC-DK1机械臂前6个电机，上电时设置当前位置为零位
"""

import time
import logging
from DM_CAN import Motor, MotorControl, DM_Motor_Type, Control_Type, DM_variable
import serial

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleArmController:
    def __init__(self, port="/dev/ttyACM0"):
        self.port = port
        self.serial_device = None
        self.control = None
        
        # 只定义前6个电机（关节1-6）
        self.motors = {
            "joint_1": Motor(DM_Motor_Type.DM4340, 0x01, 0x11),
            "joint_2": Motor(DM_Motor_Type.DM4340, 0x02, 0x12),
            "joint_3": Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
            "joint_4": Motor(DM_Motor_Type.DM4310, 0x04, 0x14),
            "joint_5": Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
            "joint_6": Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
        }
        
        self.connected = False

    def connect(self):
        """连接机械臂"""
        try:
            logger.info(f"连接串口: {self.port}")
            self.serial_device = serial.Serial(self.port, 921600, timeout=0.5)
            time.sleep(0.5)
            
            self.control = MotorControl(self.serial_device)
            self.connected = True
            logger.info("串口连接成功")
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            raise

    def initialize_motors(self):
        """初始化电机并设置零位"""
        if not self.connected:
            raise RuntimeError("未连接机械臂")
            
        logger.info("开始初始化电机...")
        
        # 第一步：添加电机并设置控制模式
        for key, motor in self.motors.items():
            logger.info(f"初始化 {key}...")
            self.control.addMotor(motor)
            
            # 检查通信
            for i in range(3):
                self.control.refresh_motor_status(motor)
                time.sleep(0.01)
            
            # 验证通信
            ctrl_mode = self.control.read_motor_param(motor, DM_variable.CTRL_MODE)
            if ctrl_mode is not None:
                logger.info(f"{key} 通信正常, 控制模式: {ctrl_mode}")
            else:
                logger.error(f"{key} 通信失败")
                continue
                
            # 设置控制模式但不启用
            success = self.control.switchControlMode(motor, Control_Type.POS_VEL)
            if success:
                logger.info(f"{key} 控制模式设置成功")
            else:
                logger.warning(f"{key} 控制模式设置可能失败")

        # 第二步：显示当前位置
        logger.info("当前位置读取:")
        positions = {}
        for key, motor in self.motors.items():
            self.control.refresh_motor_status(motor)
            current_pos = motor.getPosition()
            positions[key] = current_pos
            logger.info(f"  {key}: {current_pos:.3f} rad")

        # 第三步：安全检查
        if self._safety_check(positions):
            # 第四步：设置零位
            logger.info("开始设置电机零位...")
            for key, motor in self.motors.items():
                try:
                    logger.info(f"设置 {key} 零位...")
                    self.control.set_zero_position(motor)
                    time.sleep(0.1)
                    
                    # 验证零位设置
                    self.control.refresh_motor_status(motor)
                    new_pos = motor.getPosition()
                    logger.info(f"  {key}: 设零后位置 = {new_pos:.3f} rad")
                    
                except Exception as e:
                    logger.error(f"设置 {key} 零位失败: {e}")
        else:
            logger.warning("安全检查未通过，跳过设置零位")

        # 第五步：启用所有电机
        logger.info("启用所有电机...")
        for key, motor in self.motors.items():
            try:
                self.control.enable(motor)
                logger.info(f"{key} 已启用")
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"启用 {key} 失败: {e}")

        logger.info("电机初始化完成")

    def _safety_check(self, positions):
        """安全检查"""
        logger.info("进行安全检查...")
        
        # 定义各关节的安全范围
        safety_limits = {
            "joint_1": 3.0,  # 基座
            "joint_2": 2.0,  # 肩部
            "joint_3": 2.0,  # 肘部
            "joint_4": 3.0,  # 腕部1
            "joint_5": 3.0,  # 腕部2
            "joint_6": 4.0,  # 腕部3
        }
        
        all_safe = True
        
        for key, pos in positions.items():
            limit = safety_limits.get(key, 3.0)
            if abs(pos) > limit:
                logger.warning(f"  {key}: 位置 {pos:.3f} 超出安全范围 ±{limit}")
                all_safe = False
            else:
                logger.info(f"  {key}: 位置安全")
        
        if all_safe:
            logger.info("✓ 所有关节位置安全")
            return True
        else:
            logger.warning("✗ 部分关节位置可能不安全")
            # 询问是否继续
            try:
                response = input("是否继续设置零位? (y/N): ").strip().lower()
                return response in ['y', 'yes']
            except:
                return False

    def get_positions(self):
        """获取所有电机位置"""
        positions = {}
        for key, motor in self.motors.items():
            self.control.refresh_motor_status(motor)
            positions[key] = motor.getPosition()
        return positions

    def move_to_position(self, target_positions, duration=3.0):
        """移动到目标位置"""
        logger.info(f"移动到目标位置，持续时间: {duration}秒")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            for key, target_pos in target_positions.items():
                if key in self.motors:
                    # 使用较低的速度
                    self.control.control_Pos_Vel(self.motors[key], target_pos, 1.0)
            time.sleep(0.02)
        
        # 等待稳定
        time.sleep(0.5)
        
        # 读取最终位置
        final_positions = self.get_positions()
        logger.info("最终位置:")
        for key, pos in final_positions.items():
            logger.info(f"  {key}: {pos:.3f} rad")

    def disconnect(self):
        """断开连接"""
        if self.connected:
            logger.info("断开连接...")
            # 禁用所有电机
            for motor in self.motors.values():
                try:
                    self.control.disable(motor)
                except:
                    pass
            
            if self.serial_device and self.serial_device.is_open:
                self.serial_device.close()
            
            self.connected = False
            logger.info("已断开连接")

def main():
    """主函数"""
    # 修改为你的实际串口设备
    PORT = "/dev/ttyACM0"  
    
    controller = None
    try:
        # 创建控制器
        controller = SimpleArmController(port=PORT)
        
        # 连接
        controller.connect()
        
        # 初始化并设置零位
        controller.initialize_motors()
        
        # 显示设置零位后的位置
        print("\n" + "="*50)
        print("零位设置完成后的位置:")
        positions = controller.get_positions()
        for key, pos in positions.items():
            print(f"  {key}: {pos:.3f} rad")
        
        time.sleep(3)
        # 简单的移动测试
        print("\n进行简单移动测试...")
        
        # 测试位置1：稍微移动每个关节
        move_step = 6.28/5
        test_positions_1 = {
            "joint_1": move_step,
            "joint_2": move_step,
            "joint_3": move_step,
            "joint_4": move_step,
            "joint_5": move_step,
            "joint_6": move_step,
        }
        controller.move_to_position(test_positions_1, duration=2.0)
        
        # 返回零位
        #print("\n返回零位...")
        #zero_positions = {key: 0.0 for key in controller.motors.keys()}
        #controller.move_to_position(zero_positions, duration=2.0)
        
        print("\n测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        
    finally:
        # 确保断开连接
        if controller:
            controller.disconnect()

if __name__ == "__main__":
    main()