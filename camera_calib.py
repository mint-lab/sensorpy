import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys, json, glob, math

def get_default_config():
    config = {
        'cam_file'          : '',
        'cam_K'             : None,
        'cam_dist'          : None,
        'chess_pattern_cols': 9,
        'chess_pattern_rows': 6,
        'chess_cellsize'    : 0.03, # [m]
        'subpx_winsize'     : 11,
        'subpx_max_iter'    : 100,
        'subpx_eps'         : 0.001,
        'calib_option'      : [],
        'calib_output'      : '',
        'save_cam_pose'     : False,
        'img_plot_rows'     : 3,
    }
    return config

def get_calib_flags(calib_option):
    calib_flags = 0
    if 'CALIB_USE_INTRINSIC_GUESS'  in calib_option: calib_flags += cv.CALIB_USE_INTRINSIC_GUESS
    if 'CALIB_FIX_PRINCIPAL_POINT'  in calib_option: calib_flags += cv.CALIB_FIX_PRINCIPAL_POINT
    if 'CALIB_FIX_ASPECT_RATIO'     in calib_option: calib_flags += cv.CALIB_FIX_ASPECT_RATIO
    if 'CALIB_ZERO_TANGENT_DIST'    in calib_option: calib_flags += cv.CALIB_ZERO_TANGENT_DIST
    if 'CALIB_FIX_FOCAL_LENGTH'     in calib_option: calib_flags += cv.CALIB_FIX_FOCAL_LENGTH
    if 'CALIB_FIX_K1'               in calib_option: calib_flags += cv.CALIB_FIX_K1
    if 'CALIB_FIX_K2'               in calib_option: calib_flags += cv.CALIB_FIX_K2
    if 'CALIB_FIX_K3'               in calib_option: calib_flags += cv.CALIB_FIX_K3
    if 'CALIB_FIX_K4'               in calib_option: calib_flags += cv.CALIB_FIX_K4
    if 'CALIB_FIX_K5'               in calib_option: calib_flags += cv.CALIB_FIX_K5
    if 'CALIB_FIX_K6'               in calib_option: calib_flags += cv.CALIB_FIX_K6
    if 'CALIB_RATIONAL_MODEL'       in calib_option: calib_flags += cv.CALIB_RATIONAL_MODEL
    if 'CALIB_THIN_PRISM_MODEL'     in calib_option: calib_flags += cv.CALIB_THIN_PRISM_MODEL
    if 'CALIB_FIX_S1_S2_S3_S4'      in calib_option: calib_flags += cv.CALIB_FIX_S1_S2_S3_S4
    if 'CALIB_TILTED_MODEL'         in calib_option: calib_flags += cv.CALIB_TILTED_MODEL
    if 'CALIB_FIX_TAUX_TAUY'        in calib_option: calib_flags += cv.CALIB_FIX_TAUX_TAUY
    return calib_flags

def load_config(config_file):
    config = get_default_config()
    with open(config_file, 'rt') as f:
        config_new = json.load(f)
        config.update(config_new)

    if type(config['cam_K']) is list:
        config['cam_K'] = np.array(config['cam_K'])
    if type(config['cam_dist']) is list:
        config['cam_dist'] = np.array(config['cam_dist'])
    if (type(config['subpx_winsize']) is not tuple) or (type(config['subpx_winsize']) is not list):
        config['subpx_winsize'] = (config['subpx_winsize'], config['subpx_winsize'])
    config['subpx_criteria'] = [cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, config['subpx_max_iter'], config['subpx_eps']]
    config['calib_flags'] = get_calib_flags(config['calib_option'])
    return config

def get_2d_points(given_files, chess_pattern_rows, chess_pattern_cols, subpx_winsize, subpx_criteria):
    cam_pts, cam_img, cam_files = [], [], []
    for file in given_files:
        # Load an image
        img = cv.imread(file)
        if len(img.shape) >= 3 and img.shape[2] == 3:
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # Extract corner points from the images
        ret, pts = cv.findChessboardCorners(img, (chess_pattern_cols, chess_pattern_rows))
        if ret:
            pts = cv.cornerSubPix(img, pts, subpx_winsize, (-1, -1), subpx_criteria)
            cam_pts.append(pts)
            cam_img.append(img)
            cam_files.append(file)
    return cam_pts, cam_img, cam_files

def get_3d_points(chess_pattern_rows, chess_pattern_cols, chess_cellsize):
    x, y = np.meshgrid(range(chess_pattern_cols), range(chess_pattern_rows))
    z = np.zeros_like(x)
    pts = chess_cellsize * np.dstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    return pts

def show_reproject_pts(cam_img, cam_K, cam_dist, cam_rvec, cam_tvec, cam_pts, obj_pts, n_rows):
    if n_rows > 0:
        plt.figure()
        for i in range(len(cam_img)):
            ax = plt.subplot(n_rows, math.ceil(len(cam_img)/n_rows), i + 1)
            ax.imshow(cam_img[i], cmap='gray')
            ax.plot(cam_pts[i][:,:,0], cam_pts[i][:,:,1], 'r+', label='extract')
            proj, _ = cv.projectPoints(obj_pts[i], cam_rvec[i], cam_tvec[i], cam_K, cam_dist)
            ax.plot(proj[:,:,0], proj[:,:,1], 'b+', label='project')
            ax.axis('off')
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)



if __name__ == '__main__':
    config_file = 'cv_calib_mono.json'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    # Load configuration
    config = load_config(config_file)

    # Extract 2D points on the given images
    if not config['cam_file']:
        raise('An image sequence for the camera is not given')
    all_files = sorted(glob.glob(config['cam_file']))
    cam_pts, cam_img, cam_files = get_2d_points(all_files, config['chess_pattern_rows'], config['chess_pattern_cols'], config['subpx_winsize'], config['subpx_criteria'])

    # Prepare 3D points
    chessboard = get_3d_points(config['chess_pattern_rows'], config['chess_pattern_cols'], config['chess_cellsize'])
    obj_pts = [chessboard.astype(np.float32)] * len(cam_img)

    # Calibrate the camera
    cam_rms, cam_K, cam_dist, cam_rvec, cam_tvec = cv.calibrateCamera(obj_pts, cam_pts, cam_img[0].shape[::-1], config['cam_K'], config['cam_dist'], flags=config['calib_flags'])

    # Write the calibration result
    if config['calib_output']:
        with open(config['calib_output'], 'wt') as f:
            calib_result = {'cam_K': cam_K.tolist(), 'cam_dist': cam_dist.tolist()}
            if config['save_cam_pose']:
                calib_result['cam_pose'] = []
                for idx, file in enumerate(cam_files):
                    calib_result['cam_pose'].append({'file': file, 'rvec': cam_rvec[idx].tolist(), 'tvec': cam_tvec[idx].tolist()})
            json.dump(calib_result, f, indent=4)

    # Print the calibration result briefly
    print('### Brief Calibration Report')
    print(f'* Calibration flags: {bin(config["calib_flags"])}')
    print(f'* Camera files: {config["cam_file"]}')
    print(f'* The number of used images: {len(cam_files)} / {len(all_files)}')
    print(f'* RMS error: {cam_rms:.6f} [pixel]')

    # Visualize reprojected points
    show_reproject_pts(cam_img, cam_K, cam_dist, cam_rvec, cam_tvec, cam_pts, obj_pts, config['img_plot_rows'])
    plt.show()