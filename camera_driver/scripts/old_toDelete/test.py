import time
import numpy as np
import warnings
from pypylon import pylon
import cv2

framerate = 1.
exposure = 20.
gain = 1.
compression_ratio = 70.
pixel_scaling = 2.79


def init_ptp(camera, index):
    # PTP setup
    camera.PtpEnable.SetValue(False)
    camera.BslPtpPriority1.SetValue(128)
    camera.BslPtpProfile.SetValue("DelayRequestResponseDefaultProfile")
    camera.BslPtpNetworkMode.SetValue("Multicast")
    camera.BslPtpManagementEnable.SetValue(False)
    camera.BslPtpTwoStep.SetValue(False)
    camera.PtpEnable.SetValue(True)

def check_ptp(cameras):
    initialized = False
    locked = False

    # Wait until correctly initialized or timeout
    time1 = time.time()
    while not initialized:
        status_arr = np.zeros(cameras.GetSize(), dtype=np.bool8)
        for i, camera in enumerate(cameras):
            camera.PtpDataSetLatch.Execute()
            status_arr[i] = (camera.PtpStatus.GetValue() == 'Master' \
                            or camera.PtpStatus.GetValue() != 'Initializing')
        initialized = np.all(status_arr)
        if (time.time() - time1) > 3:
            if not initialized:
                warnings.warn('PTP not initialized -> Timeout')
            break

    # If correctly initialized, wait until settled or timeout
    if initialized:
        time2 = time.time()
        while not locked:
            status_arr = np.zeros(cameras.GetSize(), dtype=np.bool8)
            status_string = ''
            for i, camera in enumerate(cameras):
                camera.PtpDataSetLatch.Execute()
                status_arr[i] = (camera.PtpStatus.GetValue() == 'Master' \
                                or camera.PtpServoStatus.GetValue() == 'Locked')
                status_string += 'Camera {:d} locked: {} | '.format(i, status_arr[i])
            print(status_string)
            locked = np.all(status_arr)
            if (time.time() - time2) > 30:
                if not locked:
                    warnings.warn('PTP not locked -> Timeout')
                break

    return initialized and locked

def init_camera(camera, index):
    camera.GainAuto.SetValue("Off")
    camera.Gain.SetValue(gain)

    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(int(exposure*1000))

    # Beyond Pixel setup
    camera.PixelFormat.SetValue("RGB8")
    camera.BslScalingFactor.SetValue(pixel_scaling)

    # Beyond Compression setup
    camera.ImageCompressionMode.SetValue("BaslerCompressionBeyond")
    camera.ImageCompressionRateOption.SetValue("FixRatio")
    camera.BslImageCompressionRatio.SetValue(compression_ratio)

    # Periodic Signal setup
    if camera.BslPeriodicSignalSource.GetValue() != 'PtpClock':
        warnings.warn('Clock source of periodic signal is not `PtpClock`')
    camera.BslPeriodicSignalPeriod.SetValue(1 / framerate * 1e6)
    camera.BslPeriodicSignalDelay.SetValue(0)
    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("PeriodicSignal1")

    # Transport Layer Control
    camera.GevSCPD.SetValue(222768)
    camera.GevSCFTD.SetValue(8018*index)
    camera.GevSCPSPacketSize.SetValue(8000)


# Get the transport layer factory
tlFactory = pylon.TlFactory.GetInstance()

# Get all attached devices and exit application if no device is found
devices = tlFactory.EnumerateDevices()
cam_count = len(devices)
if not cam_count:
    raise EnvironmentError('No camera device found')

# Create and attach all Pylon Devices
cameras = pylon.InstantCameraArray(cam_count)
for camera, device in zip(cameras, devices):
    print('Using {:s} @ {:s}'.format(device.GetModelName(), device.GetIpAddress()))
    camera.Attach(tlFactory.CreateDevice(device))
    camera.Open()

# Initialize PTP and check initialization
for i, camera in enumerate(cameras):
    init_ptp(camera, i)

success = check_ptp(cameras)
if not success:
    raise EnvironmentError('PTP initialization was not successful')

# Initialize general camera parameters
for i, camera in enumerate(cameras):
    init_camera(camera, i)

# Image decompression
decompressor = pylon.ImageDecompressor()
descriptor = cameras[0].BslImageCompressionBCBDescriptor.GetAll()
decompressor.SetCompressionDescriptor(descriptor)

# Prepare image grabbing
imgs = [None] * cam_count
ids = [None] * cam_count

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

cameras.StartGrabbing(pylon.GrabStrategy_LatestImages)

while True:
    with cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as grabResult:
        idx = grabResult.GetCameraContext()

        if grabResult.GrabSucceeded():
            payload = grabResult.GetBuffer()
            decompressed_image = decompressor.DecompressImage(payload)

            imgs[idx] = decompressed_image.Array
            ids[idx] = grabResult.ID

            # Display image
            if ids.count(ids[0]) == len(ids):
                cv2.imshow('image', np.concatenate(imgs, axis=1)[...,::-1]/255.0)
                if cv2.waitKey(5)==27:
                    break

        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)

cameras.StopGrabbing()
cameras.Close()