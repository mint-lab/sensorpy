import numpy as np
import cv2
import depthai as dai
import blobconverter

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shave=6))
detection_nn.setConfidenceThreshold(0.5)

cam_rgb.preview.link(detection_nn.input)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
    frame = None
    detections = []
    
    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None: frame = in_rgb.getCvFrame()
        if in_nn  is not None: detections = in_nn.detections
        
        if frame is not None:
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            cv2.imshow("depthai", frame)

        if cv2.waitKey(1) == ord('q') : break