import numpy as np
import cv2 as cv
from zed import ZED, print_zed_info
import sys, argparse

def play_svo(svo_file, svo_realtime=False, depth_mode='neural', zoom=0.5):
    '''Play the given SVO file'''

    zed = ZED()
    zed.open(svo_file=svo_file, depth_mode=depth_mode)
    print_zed_info(zed)
    sys.stdout.flush()

    cv.namedWindow('ZED SVO Player: Color and Depth')
    while zed.is_open():
        if zed.grab():
            # Get all images
            color, _, depth = zed.get_images()

            # Show the images
            depth_color = cv.applyColorMap(depth, cv.COLORMAP_JET)
            merge = cv.resize(np.vstack((color, depth_color)), (0, 0), fx=zoom, fy=zoom)
            cv.imshow('ZED SVO Player: Color and Depth', merge)

        key = cv.waitKey(1)
        if key == ord(' '): # Space
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break

    cv.destroyAllWindows()
    zed.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZED SVO Player')
    parser.add_argument('svo_file',             nargs=1, type=str,                       help='the name of the SVO file (e.g. test.svo)')
    parser.add_argument('--svo_realtime', '-r', nargs=1, type=bool,  default=[False],    help='the flag to enable realtime SVO play (default: False)')
    parser.add_argument('--depth_mode',   '-d', nargs=1, type=str,   default=['neural'], help='the depth mode (default: "neural")')
    parser.add_argument('--zoom',         '-z', nargs=1, type=float, default=[0.5],      help='the zoom ratio of images (default: 0.5)')
    args = parser.parse_args()

    play_svo(args.svo_file[0], svo_realtime=args.svo_realtime[0], depth_mode=args.depth_mode[0], zoom=args.zoom[0])