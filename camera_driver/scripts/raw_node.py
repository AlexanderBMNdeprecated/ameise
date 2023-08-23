#!/usr/bin/env python3

from pypylon import pylon
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import yaml
import rospy
import time
import os
import cv2 as cv
import datetime
import sys
from decimal import Decimal
import rospkg

class CamCalib:
    def __init__(self, cam_data, with_extrinsic=False):
        self.height = cam_data["resolution"][1]
        self.width = cam_data["resolution"][0]

        if "radtan" in cam_data["distortion_model"]:
            self.dist_model = "plumb_bob"
        else:
            self.dist_model = "unknown"

        self.cam_matrix = np.array([[cam_data["intrinsics"][0], 0.0, cam_data["intrinsics"][2]],
                         [0.0, cam_data["intrinsics"][1], cam_data["intrinsics"][3]],
                         [0.0, 0.0, 1.0]])
        self.dist_coeff = np.array(cam_data["distortion_coeffs"] + [0.0])

        if with_extrinsic == True:
            transformation = cam_data["T_cn_cnm1"]
            self.rotation = np.array(transformation[:3])[:, :3]
            self.translation = np.array(transformation[:3])[:, 3]

        self.rect_matrix = np.eye(3)
        self.proj_matrix = None
        self.roi = None


class CameraNode:
    def __init__(self, camera, config_file, calib_file, frame_id,
                 publish_info=True, compression=False, set_pixel_format_to_bayer=False):
        self.frame_id = frame_id
        self.camera = camera
        self.camera_name = frame_id
        self.config_file = config_file
        self.calib_file = calib_file
        self.enable_compression = compression
        self.encoding = 'bgr8' if compression else 'passthrough'
        self.bayer = set_pixel_format_to_bayer
        self.publish_info = publish_info
        self.image_raw_pub = rospy.Publisher(f'/camera/{self.camera_name}/image_raw', Image, queue_size=10)
        if publish_info:
            self.info_pub = rospy.Publisher(f'/camera/{self.camera_name}/camera_info', CameraInfo, queue_size=10)
        self.raw_image = None
        self.timestamp = None
        self.rect_image = None
        self.camera_info = self.__create_camera_info_msg()
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
            time.sleep(1)
            print("PTP not synced yet")
        # print(self.camera_name + " PTP sync: " + str(self.camera.PtpStatus.GetValue()))
        # print(self.camera_name + str(self.camera.PtpOffsetFromMaster.GetValue()))
        # print(self.camera_name + str(self.camera.PtpParentClockID.GetValue()))

    def __load_camera_information(self):
        with open(self.calib_file, "r") as stream:
            calib = yaml.safe_load(stream)
            cam_data = calib["cam0"]
            cam = CamCalib(cam_data)
            if "stereo" in self.camera_name:
                cam_2_data = calib["cam1"]
                cam_2 = CamCalib(cam_2_data, True)

                cam.rect_matrix, cam_2.rect_matrix, \
                cam.proj_matrix, cam_2.proj_matrix, \
                disp_depth_matrix, \
                cam.roi, cam_2.roi = \
                cv.stereoRectify(cam.cam_matrix, cam.dist_coeff,
                                 cam_2.cam_matrix, cam_2.dist_coeff,
                                 (cam.width, cam.height), cam_2.rotation,
                                 cam_2.translation, alpha=1)

                if "right" in self.camera_name:
                    return cam_2
            else:
                new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(cam.cam_matrix, cam.dist_coeff, (cam.width, cam.height), 1)
                cam.proj_matrix = np.hstack((new_camera_matrix, np.zeros((3, 1))))
                cam.roi = roi
            return cam

    def __create_camera_info_msg(self):
        cam = self.__load_camera_information()
        camera_info = CameraInfo()
        camera_info.header.frame_id = self.frame_id
        camera_info.height = cam.height
        camera_info.width = cam.width
        camera_info.distortion_model = cam.dist_model
        camera_info.K = cam.cam_matrix.flatten().tolist()
        camera_info.D = cam.dist_coeff.flatten().tolist()
        camera_info.R = cam.rect_matrix.flatten().tolist()
        camera_info.P = cam.proj_matrix.flatten().tolist()
        camera_info.roi.x_offset = cam.roi[0]
        camera_info.roi.y_offset = cam.roi[1]
        camera_info.roi.width = cam.roi[2]
        camera_info.roi.height = cam.roi[3]
        camera_info.roi.do_rectify = True
        return camera_info

    def publish_images(self):
        self.__check_ptp()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        decompressor = None

        if self.enable_compression:
            decompressor = pylon.ImageDecompressor()
            descriptor = self.camera.BslImageCompressionBCBDescriptor.GetAll()
            decompressor.SetCompressionDescriptor(descriptor)
            print("compression ready!")

        print("Ready for receiving images ...")

        while not rospy.is_shutdown():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            if grab_result.IsValid(): #and grab_result.Height != 0 and grab_result.Width != 0:
                try:
                    if self.enable_compression:
                        image_grab = decompressor.DecompressImage(grab_result)
                    else:
                        image_grab = grab_result

                    # Create image and timestamp
                    if image_grab.Height != 0 and image_grab.Width != 0:
                        self.raw_image = image_grab.Array
                        # minimal workaround for bayer conversion
                        if self.bayer:
                            pass
                            '''
                            NOT FINISHED
                            bayer_image = cv.cvtColor(self.raw_image, cv.COLOR_BAYER_BG2RGB)
                            img_BGR8bit = cv.normalize(bayer_image, None, 0, 255, cv.NORM_MINMAX)
                            self.raw_image = img_BGR8bit.astype(np.uint8)
                            '''
                    else:
                        raise Exception("Fail to retrieve image!")

                    seconds = int(grab_result.TimeStamp // Decimal('1e9'))
                    nanoseconds = int(grab_result.TimeStamp % Decimal('1e9'))
                    self.timestamp = rospy.Time(seconds, nanoseconds)
                except Exception as e:
                    print(e)
                    continue
                grab_result.Release()

                try:
                    image_raw_msg = CvBridge().cv2_to_imgmsg(np.asarray(self.raw_image), encoding=self.encoding)
                    image_raw_msg.header.frame_id = self.frame_id
                    image_raw_msg.header.stamp = self.timestamp
                    self.image_raw_pub.publish(image_raw_msg)

                    if self.publish_info:
                        self.camera_info.header.stamp = self.timestamp
                        self.info_pub.publish(self.camera_info)

                except CvBridgeError as e:
                    print(e)
            else:
                while not self.camera.GetGrabResultWaitObject().Wait(0):
                    print("Invalid grab ...")
                    time.sleep(0.01)


def set_camera_object(cam_name):
    cam_object = None
    for i in pylon.TlFactory.GetInstance().EnumerateDevices():
        if i.GetUserDefinedName() == cam_name:
            cam_object = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(i))
            break
    return cam_object

def main():
    # Define workspace
    package_path = rospkg.RosPack().get_path('camera_driver')
    camera_name = sys.argv[1]
    mode = rospy.get_param(f'/{camera_name}/raw_data/mode')
    publish_info = rospy.get_param(f'/{camera_name}/raw_data/publish_camera_info')
    set_pixel_format_to_bayer = rospy.get_param(f'/{camera_name}/raw_data/set_pixel_format_to_bayer')
    # manual testing
    #camera_name, mode, publish_info, set_pixel_format_to_bayer = "stereo_left", "record", True, False
    compression = False

    if "show" in mode:
        config_path = f'{package_path}/config/show/'
    elif "calib" in mode:
        config_path = f'{package_path}/config/calib/'
    else:
        config_path = f'{package_path}/config/record/'
        #compression = False if set_pixel_format_to_bayer else True

    if "stereo" in camera_name:
        calib_file = f'{config_path}stereo.yaml'
    else:
        calib_file = f'{config_path}{camera_name}.yaml'

    if set_pixel_format_to_bayer:
        config_file = f'{config_path}cam_config_bayer.pfs'
    else:
        config_file = f'{config_path}cam_config.pfs'

    rospy.init_node(f'{camera_name}_raw_data', anonymous=True)

    camera = set_camera_object(camera_name)

    assert os.path.isfile(config_file), "Check if the cam_config file exists (check if mode fits to settings)"
    assert os.path.isfile(calib_file), "Check if the calib file exists"

    camera_topic = CameraNode(camera=camera,
                              config_file=config_file,
                              calib_file=calib_file,
                              frame_id=camera_name,
                              publish_info=publish_info,
                              compression= compression,
                              set_pixel_format_to_bayer=set_pixel_format_to_bayer
                              )

    camera_topic.publish_images()

if __name__ == '__main__':
    main()
