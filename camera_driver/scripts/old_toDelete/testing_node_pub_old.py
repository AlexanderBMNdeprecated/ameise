#!/usr/bin/env python3

from pypylon import pylon

import os
import numpy as np
import yaml
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo


class CameraInfoObject:
    def __init__(self, frame_id, yaml_file):
        with open(yaml_file, "r") as stream:
            calib = yaml.safe_load(stream)
            self.width = calib["image_width"]
            self.height = calib["image_height"]
            self.distortion_model = calib["distortion_model"]
            self.K = calib["camera_matrix"]["data"]
            self.frame_id = frame_id
            self.D = calib["distortion_coefficients"]["data"]
            self.R = calib["rectification_matrix"]["data"]
            self.P = calib["projection_matrix"]["data"]



def main():
    os.chdir("/home/ameise/catkin_ws/src/pano_pkg/scripts/")
    confFile = "cam_config.pfs"
    rospy.init_node('cam_node_pub', anonymous=True)
    cam_left_info = CameraInfoObject("alex_isses", "../config/mono_right.yaml")
    """
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "alex_isses"
    static_transformStamped.child_frame_id = "oh_yes"
    
    static_transformStamped.transform.translation.x = 0
    static_transformStamped.transform.translation.y = 0
    static_transformStamped.transform.translation.z = 0
    quat = tf.transformations.quaternion_from_euler(0, 0, 0)
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]
    broadcaster.sendTransform(static_transformStamped)
    """

    cam_name = 'mono_right'
    for i in pylon.TlFactory.GetInstance().EnumerateDevices():
        if i.GetUserDefinedName() == cam_name:
            cam_info = i
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cam_info))
    camera.Open()
    pylon.FeaturePersistence.Load(confFile, camera.GetNodeMap(), True)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImages)
    while True:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            img = grabResult.Array
            pub_image(img, cam_name, cam_left_info)


def pub_image(img, name, info):
    image_pub = rospy.Publisher("Alex_hat_rieeeeeesig/image", Image, queue_size=1)
    info_pub = rospy.Publisher("Alex_hat_rieeeeeesig/camera_info", CameraInfo, queue_size=1)
    bridge = CvBridge()
    try:
        image_msg = bridge.cv2_to_imgmsg(np.asarray(img), "bgr8")
        image_msg.header.frame_id = "alex_isses"
        image_msg.header.stamp = rospy.Time.now()
        image_pub.publish(image_msg)
        camera_info = CameraInfo()
        camera_info.header.stamp = rospy.Time.now()
        camera_info.header.frame_id = info.frame_id
        camera_info.height = info.height
        camera_info.width = info.width
        camera_info.K = info.K
        camera_info.distortion_model = info.distortion_model
        camera_info.D = info.D
        camera_info.R = info.R
        camera_info.P = info.P
        info_pub.publish(camera_info)
    except CvBridgeError as e:
        print(e)




if __name__ == '__main__':
    main()
