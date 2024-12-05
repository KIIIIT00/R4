#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import time
import crcmod.predefined
import math


crc8 = crcmod.predefined.mkPredefinedCrcFun('crc-8-maxim')
ser = serial.Serial("/dev/cu.usbserial-FT4TCJC1", 115200)
key = 0

# ddsm115を制御するための関数
# velは[cm/s]
def serial_write(id, vel):
    rpm = int((vel*60)/(10*math.pi))
    rpm = (65536 + rpm) % 65536
    senddata = [id,0x64,rpm//256,rpm%256,0x00,0x00,0x05,0x00,0x00]
    senddata.append(crc8(bytes(senddata)))
    send_binary = bytes(senddata)
    #print(send_binary)
    ser.write(send_binary)
    time.sleep(0.005) #【必須】最低でも2[ms]以上の間隔をあける

while True:
    key = input()
    #print(key)
    #key= "w"
    if key == "q":
        print("Pressed 'q' key")
        serial_write(1,0)
        serial_write(2,0)
    elif key == "w": # 前進
        print("Pressed 'w' key")
        serial_write(1,2)
        serial_write(2,-2)
    elif key == "a": # 左
        print("Pressed 'a' key")
        serial_write(1,-20)
        serial_write(2,10)
    elif key == "s":
        print("Pressed 's' key") 
        serial_write(1,0)
        serial_write(2,0)
    elif key == "d": # 右
        print("Pressed 'd' key") 
        serial_write(1,-10)
        serial_write(2,20)
    elif key == "x": # 後退
        serial_write(1,20)
        serial_write(2,-20)

serial_write(1,0)
serial_write(2,0)
 
ser.close()
