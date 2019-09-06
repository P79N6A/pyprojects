#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

import socket
import fcntl
import struct


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])

if __name__ == '__main__':
    cur_ip = get_ip_address('eth0')
    print(cur_ip)
    if cur_ip == "10.23.63.92":
        os.system("sh bin/gatling.sh -s computerdatabase.RoomsTest.RoomsTest")
    elif cur_ip == "10.23.63.93":
        os.system("sh bin/gatling.sh -s computerdatabase.AudienceRoomTest.AudienceRoomTest")
    elif cur_ip == "10.23.63.94":
        os.system("sh bin/gatling.sh -s computerdatabase.GetMsgTest.GetMsgTest")
    elif cur_ip == "10.23.63.95":
        os.system("sh bin/gatling.sh -s computerdatabase.RanklistTest.RanklistTest")
    elif cur_ip == "10.23.63.98":
        os.system("sh bin/gatling.sh -s computerdatabase.ImChatTest.ImChatTest")