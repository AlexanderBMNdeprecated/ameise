#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
import threading
import time as t
import os

class SensorMonitor:
    def __init__(self, topic_name, sensor_name):
        self.topic_name = topic_name
        self.sensor_name = sensor_name
        self.sensor_active = False
        self.last_message_time = rospy.Time().now()

    def callback(self, data):
        self.last_message_time = rospy.Time.now()

    def check_sensor_status(self):
        # Prüfe, ob während der letzten 5 Sekunden eine Nachricht empfangen wurde
        if rospy.Time.now() - self.last_message_time <= rospy.Duration(2):
            self.sensor_active = True
        else:
            self.sensor_active = False

    def run(self):
        if "OS" in self.sensor_name:
            rospy.Subscriber(self.topic_name, PointCloud2, self.callback)
        else:
            rospy.Subscriber(self.topic_name, Image, self.callback)

        rate = rospy.Rate(1)  # Überprüfe den Status alle 1 Sekunde

        while not rospy.is_shutdown():
            self.check_sensor_status()
            rate.sleep()

def printMonitorMessage():
    monitor_output_old = ""
    while not rospy.is_shutdown():
        monitor_output = "---------------------------\n" \
                         "Sensor Overview:\n" \
                         "---------------------------"
        for sensor in sensor_objects:
            monitor_output += f"\n{sensor.sensor_name} {sensor.sensor_active}"

        if not monitor_output == monitor_output_old:
            os.system('clear')
            print(monitor_output)
            monitor_output_old = monitor_output
        t.sleep(1)

if __name__ == '__main__':
    rospy.init_node('sensor_monitor')

    sensor_list = [['/lidar/OS0_left/points', 'OS0_Left'],
               ['/lidar/OS1_top/points', 'OS1_Top'],
               ['/lidar/OS0_right/points', 'OS0_Right'],
               ['/camera/mono_left/image_raw', 'Mono_Left'],
               ['/camera/stereo_left/image_raw', 'Stereo_Left'],
               ['/camera/stereo_right/image_raw', 'Stereo_Right'],
               ['/camera/mono_right/image_raw', 'Mono_Right']]
    sensor_objects = []

    monitor_threads = []
    for topic_name, sensor_name in sensor_list:
        sensor_monitor = SensorMonitor(topic_name, sensor_name)
        sensor_objects.append(sensor_monitor)

        thread = threading.Thread(target=sensor_monitor.run)
        thread.daemon = True
        thread.start()
        monitor_threads.append(thread)

    t.sleep(3)

    printMonitorMessage()
