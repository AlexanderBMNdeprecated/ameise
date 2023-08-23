#!/usr/bin/env python3
import os

from pypylon import pylon
from basler_cam.scripts.old_toDelete.image_stitcher import ImageStitcher

import numpy as np
import yaml
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


publish_pano = False
publish_image_raw = True
publish_image_rect = True

def main():
    # Setup
    serial_numbers = ['mono_left', 'stereo_left', 'mono_right']  #
    os.chdir("/home/ameise/catkin_ws/src/pano_pkg/scripts/")
    confFile = "cam_config.pfs"
    countOfImagesToGrab = 1  # Number of images to be grabbed.
    info = []
    cameras = []
    PanoImage = ImageStitcher()
    rospy.init_node('cam_node_script', anonymous=True)

    """ Find Cameras"""
    for i in pylon.TlFactory.GetInstance().EnumerateDevices():
        if i.GetUserDefinedName() in serial_numbers:
            info.append(i)
            # print(i.GetSerialNumber())
        else:
            pass

    if info is not None:
        for cam in info:
            cameras.append(pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cam)))

    #for cam in cameras:
     #   cam.Open()
        # print("Using device ", cam.GetDeviceInfo().GetModelName())

    """ Setup Cameras """
    for camera in cameras:
        camera.Open()
        print(os.getcwd())
        cam_name = camera.DeviceUserID.GetValue()
        print(cam_name)
        pylon.FeaturePersistence.Load(confFile, camera.GetNodeMap(), True)
        if "stereo" in cam_name:
            camera.ReverseX.SetValue(True)
            camera.ReverseY.SetValue(True)
        # print(camera.GevSCPSPacketSize.GetValue())
        # demonstrate some feature access
        new_width = 1920
        if new_width >= camera.Width.GetMin():
            camera.Width.SetValue(new_width)

        camera.StartGrabbing(pylon.GrabStrategy_LatestImages)
        # The parameter MaxNumBuffer can be used to control the count of buffers
        # allocated for grabbing. The default value of this parameter is 10.
        # camera.MaxNumBuffer = 5

    """ Grab Images """
    # TODO enable continuous shot
    # TODO: make parallel
    # TODO: PTP sync

    while True:
        for camera in cameras:
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                # Access the image data.
                # print("Time: ", grabResult.TimeStamp)
                cam_name = camera.DeviceUserID.GetValue()
                # print(cam_name)
                img = grabResult.Array
                if publish_image_raw:
                    pub_image(img, cam_name, "raw")
                if publish_image_rect:
                    rect_image = undistort(img, cam_name)
                    pub_image(rect_image, cam_name, "rect")
                PanoImage.images.append(img)
                # cv2.imwrite(cam_name + "_raw.png", img)
                # cv2.imwrite(cam_name + "_rect.png", rect_image)
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()
        if publish_pano:
            pano_image = PanoImage.get_panorama()
            # cv2.imwrite("pano.png", pano_image)
            pub_panorama(pano_image)

        PanoImage = ImageStitcher()
    camera.Close()

def pub_image(img, name, type):
    panorama_pub = rospy.Publisher(name + "_" + type, Image, queue_size=1)
    bridge = CvBridge()
    try:
        panorama_pub.publish(bridge.cv2_to_imgmsg(np.asarray(img), "bgr8"))
    except CvBridgeError as e:
        print(e)

def pub_panorama(panorama):
    panorama_pub = rospy.Publisher("panorama", Image, queue_size=1)
    bridge = CvBridge()
    try:
        panorama_pub.publish(bridge.cv2_to_imgmsg(np.asarray(panorama), "bgr8"))
    except CvBridgeError as e:
        print(e)

def undistort(image, name):
    with open("config/" + name + ".yaml", "r") as stream:
        try:
            calib = yaml.safe_load(stream)
            mtx = np.array(calib["camera_matrix"]["data"]).reshape((3, 3))
            dist_mtx = calib["distortion_coefficients"]["data"]
            distortionCoefficient = np.zeros((1, 14))
            nDSTYaml = len(dist_mtx)
            distortionCoefficient[0, :nDSTYaml] = dist_mtx
            dist = distortionCoefficient
        except yaml.YAMLError as exc:
            print(exc)
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst
    #cv2.imwrite(name + ".png", dst)


if __name__ == '__main__':
    main()




