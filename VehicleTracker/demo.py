from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from PIL import Image
from yolo4implementation import YOLOV4
from yolo3implementation import YOLOV3
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

warnings.filterwarnings('ignore')


flags.DEFINE_string('video', 'Input/test1.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', 'Output/Output.avi', 'path to output video')
flags.DEFINE_string('choice', 'yolov4', 'Choose between Yolov3 and Yolov4')
flags.DEFINE_boolean('show_detections', True, 'Flag for showing detections')
flags.DEFINE_boolean('write_video', True, 'Flag for Writing detections')

def main(_argv):

    if FLAGS.choice == "yolov3":
        yolo = YOLOV3()
    else:
        yolo = YOLOV4()


    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = FLAGS.show_detections
    writeVideo_flag = FLAGS.write_video

    file_path = FLAGS.video

    
    video_capture = cv2.VideoCapture(file_path)

    if writeVideo_flag:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(FLAGS.output, fourcc, 15, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
             break

        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        
        boxes, confidence, classes = yolo.detect_image(image)

        features = encoder(frame, boxes)
        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                      zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.cls for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for det in detections:
            bbox = det.to_tlbr()
            if show_detections and len(classes) > 0:
                det_cls = det.cls
                score = "%.2f" % (det.confidence * 100) + "%"
                cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                        1e-3 * frame.shape[0], (0, 255, 0), 2)
            if not show_detections:
                track_cls = track.cls
                cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0], (0, 255, 0), 2)
                cv2.putText(frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 2)

        if frame.shape[0] > 1000:
            cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Demo', 1000, 600)
        cv2.imshow('Demo', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        fps = (fps + (1./(time.time()-t1))) / 2
        print("FPS = %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    
    video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
