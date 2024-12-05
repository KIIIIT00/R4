from pynput import keyboard
from dynamixel_sdk import *

class DynamixelXM:
    def __init__(self, port_name, baudrate, dxl_id):
        self.ADDR_MX_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_MOVING_SPEED = 32
        self.ADDR_MOVING = 46

        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0
        self.MOTOR_MODE_POSITION = 3

        self.PROTOCOL_VERSION = 2.0
        self.DXL_ID = dxl_id
        self.BAUDRATE = baudrate
        self.PORT_NAME = port_name

        self.port_handler = PortHandler(self.PORT_NAME)
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise Exception("Failed to open the port")

        if not self.port_handler.setBaudRate(self.BAUDRATE):
            raise Exception("Failed to change the baudrate")

        self.enable_torque()

    def enable_torque(self):
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            raise Exception(f"Failed to enable torque: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            raise Exception(f"Dynamixel error: {self.packet_handler.getRxPacketError(dxl_error)}")

    def set_goal_position(self, position):
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, self.DXL_ID, self.ADDR_GOAL_POSITION, position)
        if result != COMM_SUCCESS:
            print(f"Failed to set goal position: {self.packet_handler.getTxRxResult(result)}")
        elif error != 0:
            print(f"Error: {self.packet_handler.getRxPacketError(error)}")

    def angle2pos(self, angle):
        position = angle * 4096 / 360
        return int(position)

    def ccw_rotate(self, cm):
      """
      ccw方向に15cm移動
      """
      position = self.angle2pos(35.87 * (-cm))
      print(position)
      self.set_goal_position(position)

    def init_position(self):
        """
        初期位置(0度)に設定
        """
        self.set_goal_position(2048)
        print("Moved to initial position: 2048")

    def close_port(self):
        self.port_handler.closePort()


def main():
    motor = DynamixelXM(port_name="/dev/cu.usbserial-FT4TCJC1", baudrate=115200, dxl_id=4)

    def on_press(key):
        try:
            if key.char == '0':
                print("Adjusting to initial position")
                motor.init_position()
            elif key.char == '1':
                print("Adjusting to 3cm (CW)")
                motor.ccw_rotate(3)
            elif key.char == '2':
                print("Adjusting to 7cm (CW)")
                motor.ccw_rotate(7)
            elif key.char == '3':
                print("Adjusting to 13cm (CW)")
                motor.ccw_rotate(13)
        except AttributeError:
            pass

    def on_release(key):
        if key == keyboard.Key.esc:
            print("Exiting...")
            return False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        print("Listening for keyboard inputs (0, 1, 2, 3). Press 'Esc' to exit.")
        listener.join()

    motor.close_port()


if __name__ == "__main__":
    main()
