#!/usr/bin/env python3
"""
test_check_motor.py
检测电机状态：温度、电压、版本信息等
"""

import time
import logging
from DM_CAN import Motor, MotorControl, DM_Motor_Type, Control_Type, DM_variable
import serial

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class MotorStatusChecker:
    def __init__(self, port="/dev/ttyACM0"):
        self.port = port
        self.serial_device = None
        self.control = None
        
        # 定义所有电机
        self.motors = {
            "joint_1": Motor(DM_Motor_Type.DM4340, 0x01, 0x11),
            "joint_2": Motor(DM_Motor_Type.DM4340, 0x02, 0x12),
            "joint_3": Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
            "joint_4": Motor(DM_Motor_Type.DM4310, 0x04, 0x14),
            "joint_5": Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
            "joint_6": Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
            "gripper": Motor(DM_Motor_Type.DM4310, 0x07, 0x17),
        }
        
        # 要监控的参数及其描述
        self.monitor_params = {
            DM_variable.OT_Value: "温度 (°C)",
            DM_variable.UV_Value: "电压 (V)", 
            DM_variable.OC_Value: "过流标志",
            DM_variable.hw_ver: "硬件版本",
            DM_variable.sw_ver: "软件版本",
            DM_variable.SN: "序列号",
            DM_variable.CTRL_MODE: "控制模式",
            DM_variable.TMAX: "最大扭矩 (Nm)",
            DM_variable.VMAX: "最大速度 (rad/s)",
            DM_variable.PMAX: "最大位置 (rad)",
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
            logger.info("串口连接成功\n")
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            raise

    def initialize_motors(self):
        """初始化电机通信"""
        if not self.connected:
            raise RuntimeError("未连接机械臂")
            
        logger.info("初始化电机通信...")
        
        for key, motor in self.motors.items():
            self.control.addMotor(motor)
            
            # 检查通信
            for i in range(3):
                self.control.refresh_motor_status(motor)
                time.sleep(0.01)
            
            # 验证通信
            ctrl_mode = self.control.read_motor_param(motor, DM_variable.CTRL_MODE)
            if ctrl_mode is not None:
                logger.info(f"✓ {key} 通信正常")
            else:
                logger.error(f"✗ {key} 通信失败")
        
        logger.info("")

    def read_motor_status(self, motor, motor_name):
        """读取单个电机的状态"""
        status = {}
        
        # 读取基本状态（位置、速度、扭矩）
        self.control.refresh_motor_status(motor)
        status['位置'] = f"{motor.getPosition():.3f} rad"
        status['速度'] = f"{motor.getVelocity():.3f} rad/s" 
        status['扭矩'] = f"{motor.getTorque():.3f} Nm"
        
        # 读取监控参数
        for param_id, param_desc in self.monitor_params.items():
            try:
                value = self.control.read_motor_param(motor, param_id)
                if value is not None:
                    # 根据参数类型进行格式化
                    if param_id == DM_variable.OT_Value:  # 温度
                        status[param_desc] = f"{value:.1f} °C"
                    elif param_id == DM_variable.UV_Value:  # 电压
                        status[param_desc] = f"{value:.1f} V"
                    elif param_id in [DM_variable.hw_ver, DM_variable.sw_ver, DM_variable.SN]:
                        status[param_desc] = f"0x{int(value):08X}"
                    elif param_id == DM_variable.CTRL_MODE:  # 控制模式
                        mode_names = {
                            1: "MIT模式", 
                            2: "位置速度模式", 
                            3: "速度模式", 
                            4: "扭矩位置模式"
                        }
                        status[param_desc] = mode_names.get(int(value), f"未知({int(value)})")
                    else:
                        status[param_desc] = f"{value:.3f}"
                else:
                    status[param_desc] = "读取失败"
            except Exception as e:
                status[param_desc] = f"错误: {e}"
                
        return status

    def check_all_motors_status(self):
        """检查所有电机状态"""
        if not self.connected:
            raise RuntimeError("未连接机械臂")
            
        logger.info("=" * 60)
        logger.info("开始检测所有电机状态")
        logger.info("=" * 60)
        
        all_status = {}
        
        for motor_name, motor in self.motors.items():
            logger.info(f"\n📊 检测电机: {motor_name}")
            logger.info("-" * 40)
            
            status = self.read_motor_status(motor, motor_name)
            all_status[motor_name] = status
            
            # 显示关键状态
            for key, value in status.items():
                logger.info(f"  {key:.<15} {value}")
        
        return all_status

    def monitor_realtime_status(self, duration=30, interval=1.0):
        """实时监控电机状态"""
        if not self.connected:
            raise RuntimeError("未连接机械臂")
            
        logger.info(f"\n🔄 开始实时监控，持续时间: {duration}秒，间隔: {interval}秒")
        logger.info("按 Ctrl+C 停止监控")
        logger.info("=" * 60)
        
        start_time = time.time()
        iteration = 0
        
        try:
            while time.time() - start_time < duration:
                iteration += 1
                logger.info(f"\n📈 监控周期 #{iteration} - 时间: {time.time()-start_time:.1f}s")
                logger.info("-" * 40)
                
                for motor_name, motor in self.motors.items():
                    # 只显示关键实时参数
                    self.control.refresh_motor_status(motor)
                    
                    # 读取温度
                    temp = self.control.read_motor_param(motor, DM_variable.OT_Value)
                    voltage = self.control.read_motor_param(motor, DM_variable.UV_Value)
                    
                    status_line = f"  {motor_name:.<10}"
                    status_line += f"位置:{motor.getPosition():6.2f}rad "
                    status_line += f"速度:{motor.getVelocity():6.2f}rad/s "
                    status_line += f"扭矩:{motor.getTorque():5.2f}Nm "
                    
                    if temp is not None:
                        status_line += f"温度:{temp:5.1f}°C "
                    
                    if voltage is not None:
                        status_line += f"电压:{voltage:5.1f}V"
                    
                    logger.info(status_line)
                
                # 温度警告检查
                self._check_temperature_warning()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️  用户停止监控")
        
        logger.info("实时监控结束")

    def _check_temperature_warning(self):
        """检查温度警告"""
        high_temp_motors = []
        
        for motor_name, motor in self.motors.items():
            temp = self.control.read_motor_param(motor, DM_variable.OT_Value)
            if temp is not None and temp > 60:  # 温度超过60°C警告
                high_temp_motors.append((motor_name, temp))
        
        if high_temp_motors:
            logger.warning("🔥 高温警告!")
            for motor_name, temp in high_temp_motors:
                logger.warning(f"  {motor_name} 温度过高: {temp:.1f}°C")

    def check_motor_versions(self):
        """检查电机版本信息"""
        logger.info("\n🔍 电机版本信息")
        logger.info("=" * 40)
        
        for motor_name, motor in self.motors.items():
            hw_ver = self.control.read_motor_param(motor, DM_variable.hw_ver)
            sw_ver = self.control.read_motor_param(motor, DM_variable.sw_ver)
            sn = self.control.read_motor_param(motor, DM_variable.SN)
            
            logger.info(f"\n{motor_name}:")
            if hw_ver is not None:
                logger.info(f"  硬件版本: 0x{int(hw_ver):08X}")
            if sw_ver is not None:
                logger.info(f"  软件版本: 0x{int(sw_ver):08X}") 
            if sn is not None:
                logger.info(f"  序列号: 0x{int(sn):08X}")

    def disconnect(self):
        """断开连接"""
        if self.connected:
            logger.info("\n断开连接...")
            if self.serial_device and self.serial_device.is_open:
                self.serial_device.close()
            self.connected = False
            logger.info("已断开连接")

def main():
    """主函数"""
    # 修改为你的实际串口设备
    PORT = "/dev/ttyACM0"  # 根据实际情况修改
    
    checker = None
    try:
        # 创建状态检查器
        checker = MotorStatusChecker(port=PORT)
        
        # 连接
        checker.connect()
        
        # 初始化电机通信
        checker.initialize_motors()
        
        # 1. 检查版本信息
        checker.check_motor_versions()
        
        # 2. 全面状态检测
        checker.check_all_motors_status()
        
        # 3. 实时监控（可选）
        print("\n" + "="*60)
        response = input("是否开始实时监控? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            duration = input("监控时长(秒) [默认30]: ").strip()
            try:
                duration = float(duration) if duration else 30.0
            except:
                duration = 30.0
            checker.monitor_realtime_status(duration=duration, interval=1.0)
        
    except Exception as e:
        logger.error(f"检测过程中出错: {e}")
        
    finally:
        # 确保断开连接
        if checker:
            checker.disconnect()

if __name__ == "__main__":
    main()