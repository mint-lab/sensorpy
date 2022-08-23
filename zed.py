import numpy as np
import cv2 as cv
import pyzed.sl as sl
import sys

class ZED():
    '''StereoLabs ZED image grabber'''

    def __init__(self):
        '''Instantiate the camera and others'''
        self.camera = sl.Camera()
        self.is_opened = False
        self.runtime_params = sl.RuntimeParameters()
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        self.depth_image = sl.Mat()
        self.depth_float = sl.Mat()
        self.MAP_RESOLUTION = {'2k': sl.RESOLUTION.HD2K, '1080p': sl.RESOLUTION.HD1080, '720p': sl.RESOLUTION.HD720, 'vga': sl.RESOLUTION.VGA}
        self.MAP_DEPTH_MODE = {'neural': sl.DEPTH_MODE.NEURAL, 'ultra': sl.DEPTH_MODE.ULTRA, 'quality': sl.DEPTH_MODE.QUALITY, 'performance': sl.DEPTH_MODE.PERFORMANCE}
        self.MAP_COORD_UNIT = {'mm': sl.UNIT.MILLIMETER, 'cm': sl.UNIT.CENTIMETER, 'm': sl.UNIT.METER}

    def open(self, resolution='720p', fps=30, depth_mode='neural', coord_unit='m', config=sl.InitParameters()):
        '''Start the camera'''
        if resolution is not None:
            config.camera_resolution = self.MAP_RESOLUTION[resolution.lower()]
        if fps is not None:
            config.camera_fps = fps
        if depth_mode is not None:
            config.depth_mode = self.MAP_DEPTH_MODE[depth_mode.lower()]
        status = self.camera.open(config)
        self.is_opened = (status == sl.ERROR_CODE.SUCCESS)
        return self.is_opened

    def close(self):
        '''Stop the camera'''
        if self.is_opened:
            self.is_opened = False
            return self.camera.close()
        return False

    def is_open(self):
        '''Check whether the camera is openend or not'''
        return self.is_opened

    def grab(self):
        '''Grab a set of sensor data'''
        status = self.camera.grab(self.runtime_params)
        return status == sl.ERROR_CODE.SUCCESS

    def get_images(self):
        '''Retrieve all images'''
        self.camera.retrieve_image(self.left_image, sl.VIEW.LEFT)
        self.camera.retrieve_image(self.right_image, sl.VIEW.RIGHT)
        self.camera.retrieve_image(self.depth_image, sl.VIEW.DEPTH)
        return self.left_image.get_data()[:,:,:3], self.right_image.get_data()[:,:,:3], self.depth_image.get_data()[:,:,:3]

    def get_depth(self):
        '''Retrieve the depth data'''
        self.camera.retrieve_measure(self.depth_float, sl.MEASURE.DEPTH)
        return self.depth_float.get_data()



def print_zed_info(zed:ZED):
    '''Print camera information of the ZED camera'''
    cam_info = zed.camera.get_camera_information()

    print('* Device information')
    print(f'  * Model     : {cam_info.camera_model}')
    print(f'  * Serial    : {cam_info.serial_number}')
    print(f'  * Firmware  : {cam_info.camera_firmware_version}')
    print(f'  * Connection: {cam_info.input_type}')
    print(f'  * Resolution: {cam_info.camera_resolution.width}x{cam_info.camera_resolution.height} (FPS: {cam_info.camera_fps})')
    print('')
    print('* Intrinsic parameters')
    param = cam_info.calibration_parameters.left_cam
    print(f'  * Left : fx={param.fx}, fy={param.fy}, cx={param.cx}, cy={param.cy}, dist={param.disto}')
    param = cam_info.calibration_parameters.right_cam
    print(f'  * Right: fx={param.fx}, fy={param.fy}, cx={param.cx}, cy={param.cy}, dist={param.disto}')
    print('')
    print('* Extrinsic parameters')
    print(f'  * R: {cam_info.calibration_parameters.R}')
    print(f'  * T: {cam_info.calibration_parameters.T}')
    print('')

def test_zed(resolution='720p', fps=30, depth_mode='neural', zoom=0.5):
    '''Show live images of the ZED camera'''

    zed = ZED()
    zed.open(resolution, fps, depth_mode)
    print_zed_info(zed)
    sys.stdout.flush()

    cv.namedWindow('ZED Live: Color and Depth')
    while True:
        if zed.grab():
            # Get all images
            color, _, depth = zed.get_images()

            # Show the images
            depth_color = cv.applyColorMap(depth, cv.COLORMAP_JET)
            merge = cv.resize(np.vstack((color, depth_color)), (0, 0), fx=zoom, fy=zoom)
            cv.imshow('ZED Live: Color and Depth', merge)

        key = cv.waitKey(1)
        if key == ord(' '): # Space
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break

    cv.destroyAllWindows()
    zed.close()



if __name__ == '__main__':
    test_zed()