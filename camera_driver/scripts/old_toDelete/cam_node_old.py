#!/usr/bin/env python3

from pypylon import pylon
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import yaml
import rospy
import time
import os

def load_camera_information(frame_id, calib_file):
    camera_info = CameraInfo()
    with open(calib_file, "r") as stream:
        calib = yaml.safe_load(stream)
        # camera_info.header.stamp = rospy.Time.now()
        camera_info.header.frame_id = frame_id
        camera_info.height = calib["image_height"]
        camera_info.width = calib["image_width"]
        camera_info.distortion_model = calib["distortion_model"]
        camera_info.K = calib["camera_matrix"]["data"]
        camera_info.D = calib["distortion_coefficients"]["data"]
        camera_info.R = calib["rectification_matrix"]["data"]
        camera_info.P = calib["projection_matrix"]["data"]
    return camera_info

class CameraNode:
    def __init__(self, camera, config_file, calib_file, frame_id):
        self.frame_id = frame_id
        self.camera = camera
        self.camera_name = frame_id
        self.config_file = config_file
        self.image_raw_pub = rospy.Publisher(f'/camera/{self.camera_name}/image_raw', Image, queue_size=1)
        self.info_pub = rospy.Publisher(f'/camera/{self.camera_name}/camera_info', CameraInfo, queue_size=1)
        self.raw_image = None
        self.camera_info = load_camera_information(frame_id, calib_file)
        self.__camera_startup()

    def __camera_startup(self):
        self.camera.Open()
        # camera configuration
        pylon.FeaturePersistence.Load(self.config_file, self.camera.GetNodeMap(), True)
        if "stereo" in self.camera_name:
            self.camera.ReverseX.SetValue(True)
            self.camera.ReverseY.SetValue(True)
        # use PTP
        self.__init_ptp()

    def __init_ptp(self):
        # PTP setup
        self.camera.PtpEnable.SetValue(False)
        self.camera.BslPtpPriority1.SetValue(128)
        self.camera.BslPtpProfile.SetValue("DelayRequestResponseDefaultProfile")
        self.camera.BslPtpNetworkMode.SetValue("Multicast")
        self.camera.BslPtpManagementEnable.SetValue(False)
        self.camera.BslPtpTwoStep.SetValue(True)
        self.camera.PtpEnable.SetValue(True)
        self.camera.PtpDataSetLatch.Execute()

    def __check_ptp(self):
        while self.camera.PtpStatus.GetValue() != 'Slave':
            self.camera.PtpDataSetLatch.Execute()
            time.sleep(0.5)
        # print(self.camera_name + " PTP sync: " + str(self.camera.PtpStatus.GetValue()))
        # print(self.camera_name + str(self.camera.PtpOffsetFromMaster.GetValue()))
        # print(self.camera_name + str(self.camera.PtpParentClockID.GetValue()))

    def publish_images(self):
        self.__check_ptp()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while not rospy.is_shutdown():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            if grab_result.IsValid():
                self.raw_image = grab_result.Array
                seconds = int(grab_result.TimeStamp // 1e9)
                nanoseconds = int(grab_result.TimeStamp % 1e9)
                timestamp = rospy.Time(seconds, nanoseconds)
                try:
                    image_raw_msg = CvBridge().cv2_to_imgmsg(np.asarray(self.raw_image), "rgb8")
                    image_raw_msg.header.frame_id = self.frame_id
                    image_raw_msg.header.stamp = timestamp
                    self.image_raw_pub.publish(image_raw_msg)
                    self.camera_info.header.stamp = timestamp
                    self.info_pub.publish(self.camera_info)
                except CvBridgeError as e:
                    print(e)
            else:
                while not self.camera.GetGrabResultWaitObject().Wait(0):
                    time.sleep(0.01)


def setCameraObject(cam_name):
    for i in pylon.TlFactory.GetInstance().EnumerateDevices():
        if i.GetUserDefinedName() == cam_name:
            cam_object = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(i))
            break
    return cam_object

def main():
    # Define workspace
    os.chdir("/basler_cam/scripts/config")
    camera_name = rospy.get_param('/cam_name')
    #camera_name = 'stereo_left'
    config_file = 'cam_config.pfs'
    calib_file = f'{camera_name}.yaml'
    rospy.init_node(f'{camera_name}_node')

    camera = setCameraObject(camera_name)

    camera_topic = CameraNode(camera,
                             config_file,
                             calib_file,
                             camera_name)

    camera_topic.publish_images()


if __name__ == '__main__':
    main()
