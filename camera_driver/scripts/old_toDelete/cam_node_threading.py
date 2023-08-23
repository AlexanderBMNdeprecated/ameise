#!/usr/bin/env python3

from pypylon import pylon
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

import os
import numpy as np
import yaml
import rospy
import threading
import cv2
import time

publish_raw = False
calib_path = "../config/"
camera_param_name = rospy.get_param("cam_name")


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
    def __init__(self, camera, config_file, calib_file, frame_id, publish_raw=False):
        self.frame_id = frame_id
        self.camera = camera
        self.camera_name = frame_id
        self.config_file = config_file
        self.image_pub = rospy.Publisher(self.camera_name + "/image_mono", Image, queue_size=1)
        self.info_pub = rospy.Publisher(self.camera_name + "/camera_info", CameraInfo, queue_size=1)
        self.publish_raw = publish_raw
        if publish_raw:
            self.raw_image_pub = rospy.Publisher(self.camera_name + "/raw_image", Image, queue_size=1)
        self.cv_bridge = CvBridge()
        self.raw_image = None
        self.rect_image = None
        self.mtx = None
        self.dist = None
        self.new_camera_mtx = None
        self.roi = None
        self.camera_info = load_camera_information(frame_id, calib_file)
        self.__camera_startup()

    def __camera_startup(self):
        self.camera.Open()
        # camera configuration
        pylon.FeaturePersistence.Load(self.config_file, self.camera.GetNodeMap(), True)
        self.MaxNumBuffer = 15
        if "stereo" in self.camera_name:
            self.camera.ReverseX.SetValue(True)
            self.camera.ReverseY.SetValue(True)
        # use PTP
        self.__init_ptp()
        # camera calibration
        self.__calibration()

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
        print(self.camera_name + " PTP sync: " + str(self.camera.PtpStatus.GetValue()))
        # print(self.camera_name + str(self.camera.PtpOffsetFromMaster.GetValue()))
        # print(self.camera_name + str(self.camera.PtpParentClockID.GetValue()))

    def __calibration(self):
        self.mtx = np.array(self.camera_info.K).reshape((3, 3))
        dist_mtx = self.camera_info.D
        distortion_coefficient = np.zeros((1, 14))
        n_dst_yaml = len(dist_mtx)
        distortion_coefficient[0, :n_dst_yaml] = dist_mtx
        self.dist = distortion_coefficient
        self.new_camera_mtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx,
                                                                      self.dist,
                                                                      (self.camera_info.width, self.camera_info.height),
                                                                      1,
                                                                      (self.camera_info.width, self.camera_info.height))

    def __undistort(self):
        dst = cv2.undistort(self.raw_image, self.mtx, self.dist, None, self.new_camera_mtx)
        # crop the image
        x, y, w, h = self.roi
        self.rect_image = dst[y:y + h, x:x + w]
        return self.rect_image

    def publish_images(self):
        self.__check_ptp()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while True:
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            if grab_result.IsValid():
                self.raw_image = grab_result.Array
                timestamp = rospy.Time(int(str(grab_result.TimeStamp)[:10]),
                                       int(str(grab_result.TimeStamp)[10:]))
                try:
                    image_msg = self.cv_bridge.cv2_to_imgmsg(np.asarray(self.raw_image), "bgr8")
                    image_msg.header.frame_id = self.frame_id
                    image_msg.header.stamp = timestamp
                    self.image_pub.publish(image_msg)
                    self.camera_info.header.stamp = timestamp
                    self.info_pub.publish(self.camera_info)
                    if self.publish_raw:
                        raw_msg = self.cv_bridge.cv2_to_imgmsg(np.asarray(self.raw_image), "bgr8")
                        self.raw_image_pub.publish(raw_msg)
                except CvBridgeError as e:
                    print(e)
            else:
                while not self.camera.GetGrabResultWaitObject().Wait(0):
                    time.sleep(0.01)


def main():
    print("Auf gehts!")
    # Define workspace
    os.chdir("/home/ameise/catkin_ws/src/pano_pkg/scripts/")
    camera_name = [camera_param_name]
    config_file = '../cam_config.pfs'
    rospy.init_node("cam_node", anonymous=True)
    camera_instances = []
    for i in pylon.TlFactory.GetInstance().EnumerateDevices():
        if i.GetUserDefinedName() in camera_name:
            camera_instances.append(i)
    cameras = []
    for camera_pointer in camera_instances:
        cameras.append(pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(camera_pointer)))

    topics = []
    threads = []
    for camera in cameras:
        camera_name = camera.GetDeviceInfo().GetUserDefinedName()
        topics.append(CameraNode(camera,
                                 config_file,
                                 calib_path + camera_name + ".yaml",
                                 camera_name,
                                 publish_raw=publish_raw))
        threads.append(threading.Thread(target=topics[-1].publish_images))

    try:
        for publishing_task in threads:
            # TODO: use software trigger to sync or implement PTP
            publishing_task.start()
            print(publishing_task.name + " has started publishing!")
    except RuntimeError:
        for data_stream in topics:
            print("closed ...")
            data_stream.camera.Close()


if __name__ == '__main__':
    main()
