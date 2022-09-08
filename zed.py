import numpy as np
import cv2 as cv
import pyzed.sl as sl
import sys
import pickle

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
        self.pose = sl.Pose()
        self.transform = sl.Transform()
        self.sensors_data = sl.SensorsData()
        self.py_translation = sl.Translation()
        self.py_orientation = sl.Orientation()
        self.zed_imu_pose = sl.Transform()
        self.input_type = sl.InputType()
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
        if coord_unit is not None:
            config.coordinate_units = self.MAP_COORD_UNIT[coord_unit.lower()]
        config.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        status = self.camera.open(config)
        self.is_opened = (status == sl.ERROR_CODE.SUCCESS)
        return self.is_opened

    def close(self):
        '''Stop the camera'''
        if self.is_opened:
            self.is_opened = False
            return self.camera.close()
        return False
    
    def get_depth(self):
        '''Retrieve the depth data'''
        self.camera.retrieve_measure(self.depth_float, sl.MEASURE.DEPTH)
        return self.depth_float.get_data()
    
    def get_sensor(self):
        self.camera.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
        self.camera.get_sensors_data(self.sensors_data, sl.TIME_REFERENCE.IMAGE)
        cam_imu = self.sensors_data.get_imu_data()
        timestamp = self.camera.get_timestamp(sl.TIME_REFERENCE.CURRENT)
        time = timestamp.get_milliseconds()
        
        baro_value = self.sensors_data.get_barometer_data().pressure
        
        tx = round(self.pose.get_translation(self.py_translation).get()[0], 3)
        ty = round(self.pose.get_translation(self.py_translation).get()[1], 3)
        tz = round(self.pose.get_translation(self.py_translation).get()[2], 3)
        
        # Display the orientation quaternion
        
        ox = round(self.pose.get_orientation(self.py_orientation).get()[0], 3)
        oy = round(self.pose.get_orientation(self.py_orientation).get()[1], 3)
        oz = round(self.pose.get_orientation(self.py_orientation).get()[2], 3)
        ow = round(self.pose.get_orientation(self.py_orientation).get()[3], 3)
        t_poses = [tx, ty, tz, ox, oy, oz, ow]
        
        #Display the IMU acceleratoin
        acceleration = [0,0,0]
        cam_imu.get_linear_acceleration(acceleration)
        ax = round(acceleration[0], 3)
        ay = round(acceleration[1], 3)
        az = round(acceleration[2], 3)
        
        #Display the IMU angular velocity
        a_velocity = [0,0,0]
        cam_imu.get_angular_velocity(a_velocity)
        vx = round(a_velocity[0], 3)
        vy = round(a_velocity[1], 3)
        vz = round(a_velocity[2], 3)

        # Display the IMU orientation quaternion
        
        ox = round(cam_imu.get_pose(self.zed_imu_pose).get_orientation().get()[0], 3)
        oy = round(cam_imu.get_pose(self.zed_imu_pose).get_orientation().get()[1], 3)
        oz = round(cam_imu.get_pose(self.zed_imu_pose).get_orientation().get()[2], 3)
        ow = round(cam_imu.get_pose(self.zed_imu_pose).get_orientation().get()[3], 3)
        imu_poses = [ax, ay, az, vx, vy, vz, ox, oy, oz, ow]
        
        return time, baro_value, t_poses, imu_poses
    
    def get_sensors_data(self):
        status = self.camera.get_sensors_data(self.sensors_data, sl.TIME_REFERENCE.CURRENT)
        return status == sl.ERROR_CODE.SUCCESS

    def get_images(self):
        '''Retrieve all images'''
        self.camera.retrieve_image(self.left_image, sl.VIEW.LEFT)
        self.camera.retrieve_image(self.right_image, sl.VIEW.RIGHT)
        self.camera.retrieve_image(self.depth_image, sl.VIEW.DEPTH)
        return self.left_image.get_data()[:,:,:3], self.right_image.get_data()[:,:,:3], self.depth_image.get_data()[:,:,:3]
    
    def grab(self):
        '''Grab a set of sensor data'''
        status = self.camera.grab(self.runtime_params)
        return status == sl.ERROR_CODE.SUCCESS
    
    def is_open(self):
        '''Check whether the camera is openend or not'''
        return self.is_opened
    
    def load_svo(self, path, realtime=False, depth_mode='neural', coord_unit='m'):
        self.input_type.set_from_svo_file(path)
        config=sl.InitParameters(input_t = self.input_type, svo_real_time_mode=realtime)
        if depth_mode is not None:
            config.depth_mode = self.MAP_DEPTH_MODE[depth_mode.lower()]
        if coord_unit is not None:
            config.coordinate_units = self.MAP_COORD_UNIT[coord_unit.lower()]
        status = self.camera.open(config)
        self.is_opened = (status == sl.ERROR_CODE.SUCCESS)
        return self.is_opened
    
    def start_recording(self, path):
        recording_param = sl.RecordingParameters(path+'.svo', sl.SVO_COMPRESSION_MODE.H264)
        err = self.camera.enable_recording(recording_param)
        return err
    
    def tracking_pose(self):
        tracking_parameters = sl.PositionalTrackingParameters(_init_pos=self.transform)
        err = self.camera.enable_positional_tracking(tracking_parameters)
        return err
        
    
def save_pickle(dic, path):
    with open(path+'.pickle', "wb") as f:
        pickle.dump(dic, f)

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
