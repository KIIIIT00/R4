import serial
import time
import crcmod.predefined
import math

class DDSM115:
    def __init__(self, serial_port="/dev/cu.usbserial-FT4TCJC1", baudrate=115200):
        self.crc8 = crcmod.predefined.mkPredefinedCrcFun('crc-8-maxim')
        self.ser = serial.Serial(serial_port, baudrate)
        
    def serial_write(self, id, vel):
        rpm = int((vel*60)/(10*math.pi))
        rpm = (65536 + rpm) % 65536
        senddata = [id,0x64,rpm//256,rpm%256,0x00,0x00,0x05,0x00,0x00]
        senddata.append(self.crc8(bytes(senddata)))
        send_binary = bytes(senddata)
        #print(send_binary)
        self.ser.write(send_binary)
        time.sleep(0.005) #【必須】最低でも2[ms]以上の間隔をあける
    
    def move(self, key):
        """
        車輪を動かす
        
        Parameter:
            key : キーボード入力
        """
        if key == ord('q'): # 停止
            print("Pressed 'q' key") 
            self.serial_write(1,0)
            self.serial_write(2,0)
        elif key == ord('w'): # 前進
            print("Pressed 'w' key") 
            self.serial_write(1,2)
            self.serial_write(2,-2)
        elif key == ord('s'): # 後退
            print("Pressed 's' key") 
            self.serial_write(1,20)
            self.serial_write(2,-20)
        elif key == ord('a'): # 左
            print("Pressed 'a' key") 
            self.serial_write(1,-20)
            self.serial_write(2,10)
        elif key == ord('d'): # 右
            print("Pressed 'd' key")  
            self.serial_write(1,-10)
            self.serial_write(2,20)
    
    def close(self):
        self.ser.close()

        