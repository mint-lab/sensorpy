import numpy as np
import cv2 as cv
import pyrealsense2 as rs
import sys

class RealSense():
    '''Intel RealSense D435 image grabber'''

    def __init__(self):
        '''Instantiate the pipeline'''
        self.pipeline = rs.pipeline()
        self.profile = None
        self.depth_scale = 1.
        self.is_opened = False
        self.frame = None
        self.enable_align, self.align, self.colorizer = False, None, None

    def open(self, width=1280, height=720, fps=30, use_bgr=True, enable_color=True, enable_color_align=True, enable_depth=True, enable_infrared=True, config=rs.config()):
        '''Start the camera'''
        config.disable_all_streams()
        if enable_color:
            if use_bgr:
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            else:
                config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if enable_color_align:
            self.align = rs.align(rs.stream.color)
            self.colorizer = rs.colorizer()
            self.enable_align = True
        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        if enable_infrared:
            config.enable_stream(rs.stream.infrared, width, height, rs.format.y8, fps)

        try:
            self.profile = self.pipeline.start(config)
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            self.is_opened = True
            return True
        except RuntimeError:
            return False

    def close(self):
        '''Stop the camera'''
        if self.is_opened:
            self.is_opened = False
            return self.pipeline.stop()
        return False

    def is_open(self):
        '''Check whether the camera is openend or not'''
        return self.is_opened

    def grab(self, timeout_sec=5):
        '''Grab a set of sensor data'''
        success, self.frame = self.pipeline.try_wait_for_frames(timeout_sec*1000)
        return success

    def get_images(self, aligning=True):
        '''Retrieve all images'''
        # aligning color and depth image
        if aligning and self.enable_align:
            frameset = self.align.process(self.frame)
            color = np.asanyarray(frameset.get_color_frame().get_data())
            infra = np.asanyarray(frameset.get_infrared_frame().get_data())
            depth = np.asanyarray(frameset.get_depth_frame().get_data())
        # Not aligning
        else:
            color = np.asanyarray(self.frame.get_color_frame().get_data())
            infra = np.asanyarray(self.frame.get_infrared_frame().get_data())
            depth = np.asanyarray(self.frame.get_depth_frame().get_data())

        return color, depth, infra

def print_realsense_info(realsense:RealSense):
    '''Print camera information of the RealSense camera'''

    active_device = realsense.profile.get_device()
    color_profile = realsense.profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = realsense.profile.get_stream(rs.stream.depth).as_video_stream_profile()
    infra_profile = realsense.profile.get_stream(rs.stream.infrared).as_video_stream_profile()

    print('* Device information')
    print(f'  * Model   : {active_device.get_info(rs.camera_info.name)}')
    print(f'  * Serial  : {active_device.get_info(rs.camera_info.serial_number)}')
    print(f'  * Firmware: {active_device.get_info(rs.camera_info.firmware_version)}')
    print('')
    print('* Intrinsic parameters')
    print(f'  * Color   : {color_profile.get_intrinsics()}')
    print(f'  * Depth   : {depth_profile.get_intrinsics()}')
    print(f'  * Infrared: {infra_profile.get_intrinsics()}')
    print('')
    print('* Extrinsic parameters')
    print(f'  * Depth-to-Color   : {depth_profile.get_extrinsics_to(color_profile)}')
    print(f'  * Depth-to-Infrared: {depth_profile.get_extrinsics_to(infra_profile)}')
    print('')

def test_realsense(width=1280, height=720, fps=30, max_depth = 5000, zoom=0.5):
    '''Show live images of the RealSense camera'''

    realsense = RealSense()
    realsense.open(width, height, fps)
    print_realsense_info(realsense)
    sys.stdout.flush()

    cv.namedWindow('RealSense Live: Color, Depth, and Infrared')
    while True:
        if realsense.grab():
            # Get all images
            color, depth, infra = realsense.get_images()

            # Show the images
            _, depth_color = cv.threshold(depth / max_depth * 255., 255, 255, cv.THRESH_TRUNC)
            depth_color = cv.applyColorMap(depth_color.astype('uint8'), cv.COLORMAP_JET)
            infra_color = cv.applyColorMap(infra, cv.COLORMAP_JET)
            merge = cv.resize(np.vstack((color, depth_color, infra_color)), (0, 0), fx=zoom, fy=zoom)
            cv.imshow('RealSense Live: Color, Depth, and Infrared', merge)

        key = cv.waitKey(1)
        if key == ord(' '): # Space
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break

    cv.destroyAllWindows()
    realsense.close()



if __name__ == '__main__':
    test_realsense()