"""
DynamixelのXM540 W-270の設定を行うクラス
"""
from dynamixel_sdk import *

class DynamixelXM:
   def __init__(self, port_name, baudrate, dxl_id):
      """
      コンストラクタ
      
      Parameters
      ------------
      port_name : str
         ポート番号
      baudrate : int
         ボートレート
      dxl_id : int
         モータのid
      """
      self.ADDR_MX_OPERATING_MODE = 11
      self.ADDR_TORQUE_ENABLE = 64
      self.ADDR_GOAL_POSITION = 116
      self.ADDR_PRESENT_POSITION = 132
      self.ADDR_MOVING_SPEED = 32
      self.ADDR_MOVING = 46
      self.ADDR_CW_ANGLE_LIMIT = 6
      self.ADDR_CCW_ANGLE_LIMIT = 8

      self.TORQUE_ENABLE = 1
      self.TORQUE_DISABLE = 0
      self.MOTOR_MODE_POSITION = 3


      # プロトコルバージョン
      self.PROTOCOL_VERSION = 2.0

      # デフォルト設定
      self.DXL_ID = dxl_id
      self.BAUDRATE = baudrate
      self.PORT_NAME = port_name

      # portHandler
      self.port_handler = PortHandler(self.PORT_NAME)

      # PacketHandler
      self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

      # Open port
      if not self.port_handler.openPort():
         raise Exception("Faled to open the port")
      
      # Set port baudrate
      if not self.port_handler.setBaudRate(self.BAUDRATE):
         raise Exception("Failed to chang the baudrate")
      
      # Enable Dynamixel Torque
      self.enable_torque()

   def __del__(self):
      """
      destructor
      """
      self.disable_torque()
      self.close_port()
      
   def enable_torque(self):
      dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
      if dxl_comm_result != COMM_SUCCESS:
         raise Exception(f"Failed to enable torque: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
      elif dxl_error != 0:
         raise Exception(f"Dynamixel error: {self.packet_handler.getRxPacketError(dxl_error)}")

   def disable_torque(self):
      dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
      if dxl_comm_result != COMM_SUCCESS:
         raise Exception(f"Failed to disable torque: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
      elif dxl_error != 0:
         raise Exception(f"Dynamixel error: {self.packet_handler.getRxPacketError(dxl_error)}")
    
    # 関節モードの設定
   def setting_position_mode(self):
        dxl_comm_result, dxl_error =self.packet_handler.write1ByteTxRx(self.port_handler, self.DXL_ID, self.ADDR_MX_OPERATING_MODE, self.MOTOR_MODE_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to set operating mode: {self.packet_handler.getRxPacketError(dxl_error)}")
            return
        
   def set_goal_position(self, position):
      """
      関節モードでの目標位置を設定
      """
      result, error = self.packet_handler.write4ByteTxRx(self.port_handler, self.DXL_ID, self.ADDR_GOAL_POSITION, position)
      if result != COMM_SUCCESS:
         print(f"Failed to set goal position: {self.packet_handler.getTxRxResult(result)}")
      elif error != 0:
         print(f"Error: {self.packet_handler.getRxPacketError(error)}")

   def read_position(self):
      """
      現在の位置を読み取り
      """
      position, result, error = self.packet_handler.read4ByteTxRx(self.port_handler, self.DXL_ID, self.ADDR_PRESENT_POSITION)
      if result != COMM_SUCCESS:
         print(f"Failed to read position: {self.packet_handler.getTxRxResult(result)}")
      elif error != 0:
         print(f"Error: {self.packet_handler.getRxPacketError(error)}")
      return position
   
   def angle2pos(self, angle):
      """
      角度からpositionに変化
      """
      position = angle * 4096/360
      return int(position)
   
   def set_speed(self, speed):
      """
      モータの移動速度を設定する
      """
      result, error = self.packet_handler.write2ByteTxRx(self.port_handler, self.DXL_ID, self.ADDR_MOVING_SPEED, speed)
      if result != COMM_SUCCESS:
         print(f"Failed to set speed: {self.packet_handler.getTxRxResult(result)}")
      elif error != 0:
         print(f"Error: {self.packet_handler.getRxPacketError(error)}")

   def is_moving(self):
      """
      モータが動いているどうか
      """
      moving, result, error = self.packet_handler.read1ByteTxRx(self.port_handler, self.DXL_ID, self.ADDR_MOVING)
      if result != COMM_SUCCESS:
         print("%s" % self.packet_handler.getTxRxResult(result))
         return False
      elif error != 0:
         print("%s" % self.packet_handler.getRxPacketError(error))
         return False
      return moving == 1

   def init_position(self):
      """
      初期位置(2048)にする
      """
      self.set_goal_position(0)
      print(0)

   def cw_rotate_90(self):
      """
      CW方向に90度回転する
      """
      self.set_goal_position(579)
    
   def cw_rotate(self, cm):
      """
      cw方向に〇cm移動
      """
      position = self.angle2pos(35.87 * cm)
      print(position)
      self.set_goal_position(position)

   def ccw_rotate(self, cm):
      """
      ccw方向に〇cm移動
      """
      position = self.angle2pos(35.87 * (-cm))
      print(position)
      self.set_goal_position(position)

   def ccw_rotate_90(self):
      """
      CCW方向に90度回転する
      """
      self.set_goal_position(3517)

   def cleaner_wipe(self):
      self.setting_position_mode()
      self.enable_torque()
      time.sleep(1)
      self.ccw_rotate(13)
      time.sleep(3)
      #motor.init_position()
      self.cw_rotate(13)
      time.sleep(3)
      self.disable_torque()
      self.close_port()
   
   def move_by_state(self, state):
      try:
         if state == -1:
            print("Adjusting to initial position")
            self.init_position()
         elif state == 1:
            move_cm = 7
            print(f"Adjusting to {move_cm}cm (CW)")
            self.ccw_rotate(move_cm)
         elif state == 2:
            move_cm = 13
            print(f"Adjusting to {move_cm}cm (CW)")
            self.ccw_rotate(7)
         elif state == 3:
            move_cm = 19
            print(f"Adjusting to {move_cm}cm (CW)")
            self.ccw_rotate(move_cm)
      except AttributeError:
         pass
   
   def is_current_same_as_destination(self, state):
      """
      # TODO: read_positionの許容する誤差範囲を決める
      """
      move_cm_list = [7, 13, 19]
      if state == -1:
         return True
      if  self.read_position() == move_cm_list[state - 1]:
         return True
      else:
         return False
   
   def close_port(self):
      self.port_handler.closePort()




