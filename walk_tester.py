import logging
import sys
import multiprocessing as mp
from signal import signal
from signal import SIGTERM
import time
import random

import torch
import torchvision
import cv2
import numpy as np
import yaml

# Stuff to stream RTSP
from rtsp_stream import StreamCapture, StreamCommands

# Import yolov7 stuff
from models.experimental import attempt_load


class WalkTester:
    COLORS = {
        '0' : (255, 0, 0),
        '1' : (0, 255, 0),
        '2' : (0, 0, 255),
    }
    CLASSES = {
        '0' : 'person',
        '1' : 'animal',
        '2' : 'vehicle',    
    }

    def __init__(self, model_name='weights/rf_intrusion_yolov7_tiny_v2_4.pt'):
        # Get parameters for test - model name, confidence threshold, iou threshold, 
        # and classes to save
        parms = self.get_parms('config/parms.yaml')
        if parms and len(parms) == 3:
            self.conf_thres = parms['conf_thres']
            self.iou_thres = parms['iou_thres']
            self.classes_to_save = [int(x) for x in parms['classes_to_save']]
        else:
            self.conf_thres = 0.35
            self.iou_thres = 0.65

        self.model_name = model_name

        # Load model
        print("Loading model...")
        self.MODEL = self.load_yolov7x_model(self.model_name)
        print("Model loaded")

        # Get device model is being run on
        self.DEVICE = next(self.MODEL.parameters()).device


    def load_yolov7x_model(self, weights_path):
        # Attempt to load the model
        model = attempt_load(
                    weights_path, 
                    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                )
        return model

    def letterbox(self, img, new_shape=[640, 640], color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [height, width]
        r1 = new_shape[0]
        r1 /= shape[0]
        r2 = new_shape[1]
        r2 /= shape[1]
        r = min(r1, r2)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = dw / 2, dh / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(
            self, 
            prediction, 
            conf_thres=0.25, 
            iou_thres=0.45, 
            classes=None, 
            agnostic=False, 
            multi_label=False,
            labels=()
        ):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            if nc == 1:
                x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                    # so there is no need to multiplicate.
            else:
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output

    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
        return boxes

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        coords = self.clip_coords(coords, img0_shape)
        return coords

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        img = cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            img = cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            img = cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img



    # Get parameters
    def get_parms(self, parms_yaml):
        with open(parms_yaml, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        return data



    # Method for deciding whether to save an image or not if it has a detection
    def contains_save_class(self, pred, classes_to_save):
        for det in pred:
            if int(det[5]) in classes_to_save:
                return True
        return False


    # Handle signal
    def handler(self, sig, frame):
        # get all active child processes
        active = mp.active_children()
        # terminate all active children
        for child in active:
            child.terminate()

        for child in active:
            child.join()

        logging.info("Terminating child processes from start_process .....")

        # terminate the process
        sys.exit(0)

    def run_processor(self, rtsp_url, camera_name):

        signal(SIGTERM, self.handler)

        stopbit = mp.Event()

        try:

            logging.info(f"Starting stream processor for camera {camera_name} .....")

            # Dict with camera info used to instantiate StreamCapture object
            camera = {
                'rtsp_url' : rtsp_url,
                'camera_id' : camera_name
            }

            # Queue to store the frames from the video feed
            cam_queue = mp.Queue(maxsize=100)

            # Frame rate of the stream to fetch images from
            framerate = 6

            # Start stream capture process
            logging.info(f"Starting stream capture for camera {camera_name} ({rtsp_url})...")
            camProcess = StreamCapture(camera, stopbit, cam_queue, framerate)
            camProcess.start()

            frame_count = 0

            while True:
                if not cam_queue.empty():
                    # Check if stopbit is set in case it tries to get frame while queue is emptying
                    if stopbit.is_set():
                        break

                    # Get latest frame from stream
                    try:
                        cmd, val = cam_queue.get()
                    except cam_queue.Empty:
                        cmd, val = None, None
                        
                    if cmd == StreamCommands.FRAME:
                        # val is OpenCV image (I think) - CHECK THAT IT IS
                        if val is not None:
                            # Copy image
                            img = cv2.cvtColor(val, cv2.COLOR_RGB2BGR)
                            img0 = img.copy()

                            # Convert to format to use for inference
                            img = self.letterbox(img, (640, 640))[0]
                            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
                            img = np.ascontiguousarray(img)

                            # Normalize image
                            img = img.astype(np.float32)
                            img /= 255.0
                            img = torch.from_numpy(img).unsqueeze(0).to(self.DEVICE)

                            # Perform inference
                            with torch.no_grad():
                                pred = self.MODEL(img.float())[0]

                            # Perform NMS on prediction
                            pred = self.non_max_suppression(
                                        pred, 
                                        self.conf_thres, 
                                        self.iou_thres, 
                                        classes=[0, 1, 2],
                                        agnostic=False
                                    )

                            # Scale coords
                            for det in pred:
                                if len(det):
                                    # Rescale boxes from img_sz to img0 size
                                    det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                            # Draw inference results on frame to display
                            for det in pred[0]:
                                # Get the color of the bbox and the label for it
                                color = self.COLORS[f'{int(det[5])}']
                                label = "{} {:.2f}".format(self.CLASSES[f'{int(det[5])}'], det[4])
                                # Draw on image
                                img0 = self.plot_one_box(
                                    det[:4], 
                                    img0, 
                                    color=color, 
                                    label=label, 
                                    line_thickness=1
                                )

                            # Show frame
                            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
                            cv2.imshow('Stream', img0)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                logging.info(f"Terminating streaming for camera {camera_name} ({rtsp_url})")
                                break

                            # Write to disk if there are detections from classes to save
                            if len(pred[0]) > 0:
                                # Check if any of the detections are from the classes to save
                                if self.contains_save_class(pred[0], self.classes_to_save):
                                    # Save original image
                                    cv2.imwrite(
                                        f'alerts/detections/cam_{camera_name}_frame_{frame_count}.png', 
                                        val
                                    )
                                    # Save image with bboxes plotted
                                    cv2.imwrite(
                                        f'alerts/plotted_detections/cam_{camera_name}_frame_{frame_count}.png', 
                                        img0
                                    )
                                else:
                                    # Save image that doesn't have any detections from selected classes
                                    cv2.imwrite(
                                        f'alerts/detectionless/cam_{camera_name}_frame_{frame_count}.png', 
                                        val
                                    )
                            else:
                                # Save image that doesn't have any detections
                                cv2.imwrite(
                                    f'alerts/detectionless/cam_{camera_name}_frame_{frame_count}.png', 
                                    val
                                )

                            # Increment frame count
                            frame_count += 1

                if stopbit.is_set():
                    print("Stopbit has been set")
                    break

            cv2.destroyAllWindows()

            stopbit.set()

            # print("Emptying and closing queue...")
            # while not cam_queue.empty():
            #     print("Emptying...")
            #     cam_queue.get()
            # cam_queue.close()
            # print("Closed queue")

            print("Waiting for stream process to terminate...")
            camProcess.join()
            print("Stream process has terminated - done waiting")

        except Exception as error:

            logging.error("Error occured in stream processor for stream{}: {}".format(0, error))
            
            # Set stop bit so that camera process terminates
            stopbit.set()
            # Attempt to close the camera queue in case it exists
            try:
                cam_queue.close()
            except NameError:
                pass
            # Attempt to wait for the cam process to terminate if it exists
            try:
                print("Waiting for stream process to terminate...")
                camProcess.join()
                print("Stream process has terminated - done waiting")
            except NameError:
                print("No stream process exists - done waiting")

            # sys.exit(0) # commented out because I don't want the program to terminate