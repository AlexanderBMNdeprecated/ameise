#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyt


def scale_img(img, scale_percent=30):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


# Callback-Funktion für den linken Kamera-Topic
def left_image_callback(msg):
    global left_image
    left_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

# Callback-Funktion für den rechten Kamera-Topic
def right_image_callback(msg):
    global right_image
    right_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


rospy.init_node('image_proc_node')

# Initialisiere den Bildbrückenkonverter
bridge = CvBridge()


# Variablen für die linken und rechten Bilder
left_image = None
right_image = None

# Abonniere die Kamera-Topics
rospy.Subscriber('/camera/stereo_left/image_rect', Image, left_image_callback)
rospy.Subscriber('/camera/stereo_right/image_rect', Image, right_image_callback)

while not rospy.is_shutdown():
    pass
    # Do Something
