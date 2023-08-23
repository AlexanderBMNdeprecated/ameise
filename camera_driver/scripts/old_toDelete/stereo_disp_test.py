#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from stereo_msgs.msg import DisparityImage

# Callback-Funktion für den linken Kamera-Topic
def left_image_callback(msg):
    global left_image, timestamp
    left_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    timestamp = msg.header.stamp

# Callback-Funktion für den rechten Kamera-Topic
def right_image_callback(msg):
    global right_image
    right_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


rospy.init_node('disparity_map_node')

# Initialisiere den Bildbrückenkonverter
bridge = CvBridge()

# Variablen für die linken und rechten Bilder
left_image = None
right_image = None
timestamp = None

# Abonniere die Kamera-Topics
rospy.Subscriber('/camera/stereo_left/image_rect', Image, left_image_callback)
rospy.Subscriber('/camera/stereo_right/image_rect', Image, right_image_callback)
disp_image_pub = rospy.Publisher('/camera/stereo/disparity', DisparityImage, queue_size=10)
disp_points_pub = rospy.Publisher('/camera/stereo/points', PointCloud2, queue_size=10)

# Setting parameters for StereoSGBM algorithm
numDisparities = 64
blockSize = 23
preFilterType = 1
preFilterSize = 81
preFilterCap = 47
textureThreshold = 71
uniquenessRatio = 12
speckleRange = 5
speckleWindowSize = 11
disp12MaxDiff = 5
minDisparity = 0

# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoBM_create(numDisparities=numDisparities,
                               blockSize=blockSize)

stereo.setPreFilterType(preFilterType)
stereo.setPreFilterSize(preFilterSize)
stereo.setPreFilterCap(preFilterCap)
stereo.setTextureThreshold(textureThreshold)
stereo.setUniquenessRatio(uniquenessRatio)
stereo.setSpeckleRange(speckleRange)
stereo.setSpeckleWindowSize(speckleWindowSize)
stereo.setDisp12MaxDiff(disp12MaxDiff)
stereo.setMinDisparity(minDisparity)

while not rospy.is_shutdown():
    # Warte auf Bilder von beiden Kamera-Topics
    if left_image is not None and right_image is not None:
        # Konvertiere die Bilder in Graustufen
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("left.png", left_image)
        # cv2.imwrite("right.png", right_image)

        # Calculating disparith using the StereoSGBM algorithm
        disp = stereo.compute(left_gray, right_gray).astype(np.float32)
        disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
        #print(disp)
        try:
            disp_msg = DisparityImage()
            disp_msg.image = CvBridge().cv2_to_imgmsg(disp,  encoding="passthrough")
            disp_msg.header.frame_id = "disparity_image"
            disp_msg.header.stamp = timestamp
            disp_msg.image.header.frame_id = "disparity_image"
            disp_msg.image.header.stamp = timestamp
            disp_image_pub.publish(disp_msg)

            disp_points_msg = PointCloud2()
            disp_points_msg.header.frame_id = "stereo_points"
            disp_points_msg.header.stamp = timestamp
            points = cv2.reprojectImageTo3D(disp)



        except CvBridgeError as e:
            print(e)