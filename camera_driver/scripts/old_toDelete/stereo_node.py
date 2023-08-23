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
    def __init__(self, cameras, config_files, calib_files, frame_ids):
        self.frame_ids = frame_ids
        self.cameras = cameras
        self.camera_names = frame_ids
        self.config_file = config_files
        self.raw_image = None
        self.camera_infos = []
        for i in range(cameras.GetSize()):
            self.camera_infos.append(load_camera_information(frame_ids[i], calib_files[i]))
            self.__camera_startup(i)
            #self.image_pubs = rospy.Publisher(f'/camera/{self.camera_names[i]}/image_raw', Images, queue_size=1)
            #self.info_pubs = rospy.Publisher(f'/camera/{self.camera_names[i]}/camera_info', CameraInfos, queue_size=1)

    def __camera_startup(self, cam_num):
        self.cameras[cam_num].Open()
        # camera configuration
        pylon.FeaturePersistence.Load(self.config_file, self.cameras[cam_num].GetNodeMap(), True)
        if "stereo" in self.camera_names[cam_num]:
            self.cameras[cam_num].ReverseX.SetValue(True)
            self.cameras[cam_num].ReverseY.SetValue(True)
        # use PTP
        self.__init_ptp(cam_num)

    def __init_ptp(self, cam_num):
        # PTP setup
        self.cameras[cam_num].PtpEnable.SetValue(False)
        self.cameras[cam_num].BslPtpPriority1.SetValue(128)
        self.cameras[cam_num].BslPtpProfile.SetValue("DelayRequestResponseDefaultProfile")
        self.cameras[cam_num].BslPtpNetworkMode.SetValue("Multicast")
        self.cameras[cam_num].BslPtpManagementEnable.SetValue(False)
        self.cameras[cam_num].BslPtpTwoStep.SetValue(True)
        self.cameras[cam_num].PtpEnable.SetValue(True)
        self.cameras[cam_num].PtpDataSetLatch.Execute()

    def __check_ptp(self, cam_num):
        while self.cameras[cam_num].PtpStatus.GetValue() != 'Slave':
            self.cameras[cam_num].PtpDataSetLatch.Execute()
            time.sleep(0.5)
        # print(self.camera_name + " PTP sync: " + str(self.camera.PtpStatus.GetValue()))
        # print(self.camera_name + str(self.camera.PtpOffsetFromMaster.GetValue()))
        # print(self.camera_name + str(self.camera.PtpParentClockID.GetValue()))

    def publish_images(self):
        self.__check_ptp(0)
        self.__check_ptp(1)
        self.cameras.StartGrabbing(pylon.GrabStrategy_OneByOne)
        while not rospy.is_shutdown():
            time.sleep(0.09)
            self.cameras[0].executeSoftwareTrigger()
            grab_results = self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            print(grab_results)
            if grab_result.IsValid():
                self.raw_image = grab_result.Array
                seconds = int(grab_result.TimeStamp // 1e9)
                nanoseconds = int(grab_result.TimeStamp % 1e9)
                timestamp = rospy.Time(seconds, nanoseconds)
                try:
                    image_msg = CvBridge().cv2_to_imgmsg(np.asarray(self.raw_image), "bgr8")
                    image_msg.header.frame_id = self.frame_id
                    image_msg.header.stamp = timestamp
                    self.image_pub.publish(image_msg)
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
    camera_names = ["stereo_left", "stereo_right"]
    #camera_name = 'mono_left'
    config_file = 'test.pfs'
    calib_files = [f'{camera_names[0]}.yaml', f'{camera_names[1]}.yaml']
    rospy.init_node(f'{"stereo"}_node')

    #camera_left = setCameraObject(camera_name_left)
    #camera_right = setCameraObject(camera_name_right)

    #camera_topic_left = CameraNode(camera_left,
    #                         config_file,
    #                         calib_file_left,
    #                         camera_name_left)


    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(len(devices))

    l = cameras.GetSize()

    # Create and attach all Pylon Devices.
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))

        # Print the model name of the camera.
        print("Using device ", cam.GetDeviceInfo().GetModelName())

    camera_topic = CameraNode(cameras,
                                config_file,
                                calib_files,
                                camera_names)

    camera_topic.publish_images()

if __name__ == '__main__':
    main()
