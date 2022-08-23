import numpy as np
import serial, threading, time, pickle
from scipy.spatial.transform import Rotation
import pyvista as pv

class myAHRSPlus:
    def __init__(self, port='', baudrate=115200, timeout=1, verbose=False):
        self.dev = serial.Serial()
        self.seq = 0
        self.xyzw = [0., 0., 0., 0.]
        self.verbose = verbose
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.recv_data)
        self.thread_exit = False

        if port:
            self.open(port, baudrate, timeout)

    def __del__(self):
        self.close()

    def open(self, port, baudrate=115200, timeout=1):
        if not self.dev.is_open:
            self.dev.port = port
            self.dev.baudrate = baudrate
            self.dev.timeout = timeout
            self.dev.open()
            if self.dev.is_open:
                self.dev.reset_input_buffer()
                self.send_cmd(b'@mode,AT')
                self.send_cmd(b'@asc_out,QUAT')
                self.send_cmd(b'@divider,1')
                self.send_cmd(b'@mode,AC')
                self.thread.start()
                return True
            else:
                if self.verbose:
                    print('[myAHRSPlus] Cannot open the port, {port}')
        else:
            if self.verbose:
                print('[myAHRSPlus] The port, {port}, is already open.')
        return False

    def is_open(self):
        return self.dev.is_open

    def close(self, timeout=10):
        self.thread_exit = True
        for trial in range(1, 100):
            if not self.thread.is_alive():
                break
            time.sleep(timeout / 100.)
        self.dev.close()

    def get_xyzw(self):
        with self.lock:
            if np.linalg.norm(self.xyzw) < 0.1: # Less than 1 (but use smaller number for tolerance)
                if self.verbose:
                    print(f'[myAHRSPlus] The quaternion is not unit magnitude, {self.xyzw}.')
                return None
            return list(self.xyzw) # Return the copy

    def send_cmd(self, command):
        if command:
            crc = 0
            for b in command:
                crc = crc ^ b
            packet = command + b'*' + ('%02X' % crc).encode() + b'\r\n'
            self.dev.write(packet)
            self.dev.read_until()
        return False

    def recv_data(self):
        self.thread_exit = False
        if self.verbose:
            print("[myAHRSPlus] Start 'recv_data' thread.")
        while not self.thread_exit:
            packet = self.dev.read_until()
            token = packet.split(b',')
            if len(token) == 6:
                try:
                    with self.lock:
                        self.seq = int(token[1])
                        self.xyzw = [float(token[2]), float(token[3]), float(token[4]), float(token[5].split(b'*')[0])]
                except ValueError:
                    if self.verbose:
                        print(f'[myAHRSPlus] Cannot parse the wrong packet, {str(packet)}')
            else:
                if self.verbose:
                    print(f'[myAHRSPlus] Receive the wrong packet, {str(packet)}')
            self.dev.reset_input_buffer()
        if self.verbose:
            print("[myAHRSPlus] Terminate 'recv_data' thread.")

class AHRSCube:
    def __init__(self, plotter, scale=1., x_length=0.027, y_length=0.022, z_length=0.0005, a_length=0.011, color='orange'):
        self.ahrs_mesh = pv.Cube(x_length=scale*x_length, y_length=scale*y_length, z_length=scale*z_length)
        self.vecx_mesh = pv.Arrow(scale=scale*a_length)
        self.vecy_mesh = pv.Arrow(scale=scale*a_length)
        self.vecz_mesh = pv.Arrow(scale=scale*a_length)
        self.ahrs_actor = plotter.add_mesh(self.ahrs_mesh, color=color)
        self.vecx_actor = plotter.add_mesh(self.vecx_mesh, color='r')
        self.vecy_actor = plotter.add_mesh(self.vecy_mesh, color='g')
        self.vecz_actor = plotter.add_mesh(self.vecz_mesh, color='b')
        self.Ry = Rotation.from_euler("xyz", [0, 0, np.pi / 2])
        self.Rz = Rotation.from_euler("xyz", [0, -np.pi / 2, 0])

    def set_position(self, xyz):
        self.ahrs_actor.SetPosition(xyz)
        self.vecx_actor.SetPosition(xyz)
        self.vecy_actor.SetPosition(xyz)
        self.vecz_actor.SetPosition(xyz)

    def set_orientation(self, q_xyzw):
        R = Rotation.from_quat(q_xyzw)
        y, x, z = R.as_euler('yxz', degrees=True)
        self.ahrs_actor.SetOrientation(x, y, z)
        self.vecx_actor.SetOrientation(x, y, z)
        y, x, z = (R * self.Ry).as_euler('yxz', degrees=True)
        self.vecy_actor.SetOrientation(x, y, z)
        y, x, z = (R * self.Rz).as_euler('yxz', degrees=True)
        self.vecz_actor.SetOrientation(x, y, z)



if __name__ == '__main__':
    # Configuration
    ahrs_port = 'COM4'
    ahrs_save = ''

    # Open the device
    ahrs_dev = myAHRSPlus(ahrs_port)
    if ahrs_dev.is_open():

        # Prepare visualization
        plotter = pv.Plotter()
        ahrs_viz = AHRSCube(plotter, scale=30)
        plotter.add_axes_at_origin('r', 'g', 'b')
        plotter.show(title='SeoulTech AHRS Visualization', interactive_update=True)

        # Get data and show them
        ahrs_data = []
        try:
            start = time.time()
            while True:
                q_xyzw = ahrs_dev.get_xyzw()
                if q_xyzw:
                    if ahrs_save:
                        ahrs_data.append([time.time() - start] + q_xyzw)
                    ahrs_viz.set_orientation(q_xyzw)
                    plotter.update()
        except KeyboardInterrupt:
            pass

        # Terminate
        plotter.close()
        ahrs_dev.close()

        # Save data if necessary
        if ahrs_save and len(ahrs_data) > 0:
            with open(ahrs_save, 'wb') as f:
                pickle.dump(ahrs_data, f)