#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import datetime

# Klasse zum Speichern der Kamerainformationen
class CamInfo:
    def __init__(self, cam_info_msg):
        self.frame_id = cam_info_msg.header.frame_id
        # Extrahiere die Höhe und Breite des Bildes
        self.height = cam_info_msg.height
        self.width = cam_info_msg.width

        # Extrahiere die Kameramatrix und Verzerrungskoeffizienten
        self.cam_matrix = np.array(cam_info_msg.K).reshape(3, 3)
        self.dist_coeff = np.array(cam_info_msg.D)

        # Extrahiere die Rektifizierungsmatrix und die Projektionsmatrix
        self.rect_matrix = np.array(cam_info_msg.R).reshape(3, 3)
        self.proj_matrix = np.array(cam_info_msg.P).reshape(3, 4)

        # Extrahiere das Region of Interest (ROI)-Rechteck
        self.roi = cam_info_msg.roi


# Callback-Funktion für das "image_raw"-Topic
def image_callback(image_msg):
    # Konvertiere das Bild von ROS-Nachricht in das OpenCV-Format
    image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")

    # Bild rectifizieren
    rectified_image = rectify_image(image)

    # Konvertiere das rectifizierte Bild in eine ROS-Nachricht
    rectified_image_msg = bridge.cv2_to_imgmsg(rectified_image, encoding="bgr8")
    rectified_image_msg.header.frame_id = image_msg.header.frame_id
    rectified_image_msg.header.stamp = image_msg.header.stamp

    # Veröffentliche das rectifizierte Bild auf einem neuen ROS-Topic
    rectified_image_publisher.publish(rectified_image_msg)


# Callback-Funktion für das "camera_info"-Topic
def camera_info_callback(camera_info_msg):
    global cam_info
    # Speichere die Kamerainformationen in der globalen Variable
    cam_info = CamInfo(camera_info_msg)
    if publish_new_cam_info:
        camera_info_msg.D = []
        rectified_info_publisher.publish(camera_info_msg)


# Funktion zur Bildrectifizierung
def rectify_image(image):
    # Initialisiere die rectifizierten Abbildungs-Maps für die Bildrectifizierung
    mapx, mapy = cv.initUndistortRectifyMap(cam_info.cam_matrix, cam_info.dist_coeff,
                                            cam_info.rect_matrix, cam_info.proj_matrix,
                                            (cam_info.width, cam_info.height), cv.CV_16SC2)

    # Wende die Abbildungs-Maps auf das Bild an, um es zu rectifizieren
    rectified_image = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)

    # Schneide das Bild gemäß dem definierten ROI-Rechteck aus

    if cut_image:
        x, y, w, h = cam_info.roi.x_offset, cam_info.roi.y_offset, cam_info.roi.width, cam_info.roi.height
        rectified_image = rectified_image[y:y + h, x:x + w]

    return rectified_image


if __name__ == '__main__':
    # ROS-Initialisierung
    camera_name = sys.argv[1]
    cut_image = rospy.get_param(f'/{camera_name}/image_rect/cut_image')
    publish_new_cam_info = rospy.get_param(f'/{camera_name}/image_rect/pub_rect_info')
    # camera_name = 'stereo_left'
    bridge = CvBridge()

    rospy.init_node(f'{camera_name}_image_rect', anonymous=True)

    # Initialisiere die Variable für die Kamerainformationen
    cam_info = None

    # Erstelle Subscriber für "camera_info" und "image_raw" Topics
    camera_info_subscriber = rospy.Subscriber(f'/camera/{camera_name}/camera_info', CameraInfo, camera_info_callback)
    image_subscriber = rospy.Subscriber(f'/camera/{camera_name}/image_raw', Image, image_callback)

    # Erstelle Publisher für das rectifizierte Bild
    rectified_image_publisher = rospy.Publisher(f'/camera/{camera_name}/image_rect', Image, queue_size=10)
    if publish_new_cam_info:
        rectified_info_publisher = rospy.Publisher(f'/camera/{camera_name}/image_rect/camera_info', CameraInfo, queue_size=10)

    # Schleife für die ROS-Spin-Funktion
    while not rospy.is_shutdown():
        rospy.spin()