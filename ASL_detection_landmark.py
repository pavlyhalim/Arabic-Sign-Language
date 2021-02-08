from utils import detector_utils as detector_utils
from cv2 import cv2
import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import os
import keras
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper

# install: pip install python-bidi
from bidi.algorithm import get_display
import dlib
import imutils
from imutils import face_utils
# install: pip install Pillow
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
frame_processed = 0
score_thresh = 0.18

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue

res, score = '', 0.0
sequence = ''
fontFile = "fonts/Sahel.ttf"
font = ImageFont.truetype(fontFile, 70)
categories=[
["ain",'ع'],
["al","ال"],
["aleff",'أ'],
["bb",'ب'],
["dal",'د'],
["dha",'ط'],
["dhad","ض"],
["fa","ف"],
["gaaf",'ج'],
["ghain",'غ'],
["ha",'ه'],
["haa",'ه'],
["jeem",'ج'],
["kaaf",'ك'],
["la",'لا'],
["laam",'ل'],
["meem",'م'],
["nun","ن"],
["ra",'ر'],
["saad",'ص'],
["seen",'س'],
["sheen","ش"],
["ta",'ت'],
["taa",'ط'],
["thaa","ث"],
["thal","ذ"],
["toot",'ت'],
["waw",'و'],
["ya","ى"],
["yaa","ي"],
["zay",'ز']]
def process_image(img):
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (64, 64))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, 64 , 64 , 3))
    img = img.astype('float32') / 255.
    return img

def worker(input_q, output_q, cropped_output_q, inferences_q, landmark_ouput_q,cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("landmarks/shape_predictor_68_face_landmarks.dat")

    print(">> loading keras model for worker")
    try:
        model = tf.keras.models.load_model('models/asl_char_model.h5', compile=False)
    except Exception as e:
        print(e)

    while True:
        frame = input_q.get()
        if (frame is not None):
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)


            # get region of interest
            res = detector_utils.get_box_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
            
            # draw bounding boxes
            detector_utils.draw_box_on_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
               scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
            
            # classify hand 
            if res is not None:
                class_res = ""
                try:
                    proba = model.predict(process_image(res))[0]
                    mx = np.argmax(proba)

                    score = proba[mx] * 100
                    sequence = categories[mx][1]
                    class_res = str(score) + "/" + sequence
                except:
                    score = 0.0
                    sequence = ""
                    class_res = "empty"

                inferences_q.put(class_res)    

            image_np1 = imutils.resize(frame, width=400)
            gray = cv2.cvtColor(image_np1, cv2.COLOR_BGR2GRAY)

            #lanmarking

            # detect faces in the grayscale frame
            rects = detector(gray, 0)
                # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(image_np1, (x, y), 1, (0, 0, 255), -1)

            # add frame annotated with bounding box to queue
            landmark_ouput_q.put(image_np1)
            cropped_output_q.put(res)
            output_q.put(frame)
            
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=800,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=600,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    input_q             = Queue(maxsize=args.queue_size)
    output_q            = Queue(maxsize=args.queue_size)
    cropped_output_q    = Queue(maxsize=args.queue_size)
    inferences_q        = Queue(maxsize=args.queue_size)
    landmark_ouput_q     = Queue(maxsize=args.queue_size)

    video_capture = WebcamVideoStream(
        src=args.video_source, width=args.width, height=args.height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    print(cap_params['im_width'], cap_params['im_height'])
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # spin up workers to paralleize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cropped_output_q,inferences_q, landmark_ouput_q,cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    cv2.namedWindow('ASL', cv2.WINDOW_NORMAL)

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        output_frame = output_q.get()
        cropped_output = cropped_output_q.get()
        landmark_ouput = landmark_ouput_q.get()

        inferences      = None

        try:
            inferences = inferences_q.get_nowait()
        except Exception as e:
            pass      

        if(inferences is not None):
            try:
                score = inferences.split('/')[0]
                sequence = inferences.split('/')[1]
            except:
                score = 0.0
                sequence = ""

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time

        reshaped_text = arabic_reshaper.reshape(sequence)   
        bidi_text = get_display(reshaped_text)    


        if (cropped_output is not None):
            cropped_output = cv2.cvtColor(cropped_output, cv2.COLOR_RGB2BGR)
            img_pil = Image.fromarray(cropped_output)
            draw = ImageDraw.Draw(img_pil)
            draw.text((30, 100), bidi_text, (255,0,0), font=font)
            cropped_output = np.array(img_pil)
            if (args.display > 0):
                cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Cropped', 450, 300)
                cv2.putText(cropped_output, '(score= %.2f)' % (float(score)), (10,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
                cv2.imshow('Cropped', cropped_output)
                #cv2.imwrite('image_' + str(num_frames) + '.png', cropped_output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))

    
        if (landmark_ouput is not None):
            landmark_ouput = cv2.cvtColor(landmark_ouput, cv2.COLOR_RGB2BGR)
            if (args.display > 0):
                cv2.namedWindow('LandMark', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('LandMark', 450, 300)
                cv2.imshow('LandMark', landmark_ouput)
                #cv2.imwrite('image_' + str(num_frames) + '.png', cropped_output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))

        if (output_frame is not None):
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            if (args.display > 0):
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)
                cv2.imshow('ASL', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))
        else:
            print("video end")
            break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()

