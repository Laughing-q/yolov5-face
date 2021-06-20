from utils.torch_utils import select_device, load_classifier, time_synchronized, time_synchronized
from models.experimental import attempt_load
import random
import os
import cv2
from utils.datasets import LoadStreams, LoadImages, letterbox
import numpy as np
import torch
from utils.plots import plot_one_box
from utils.face_align import align_img
from utils.general import non_max_suppression_face, scale_coords


class Yolov5Face:
    def __init__(self, weight_path, device, img_hw=(384, 640)):
        self.weights = weight_path
        self.device = select_device(device)
        self.half = True
        self.model = attempt_load(self.weights, map_location=self.device)
        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(len(self.names))]
        self.show = False
        self.img_hw = img_hw
        self.pause = False

    def preprocess(self, image, auto=True):  # (h, w)
        if type(image) == str and os.path.isfile(image):
            img0 = cv2.imread(image)
        else:
            img0 = image
        # img, _, _ = letterbox(img0, new_shape=new_shape)
        img, _, _ = letterbox(img0, new_shape=self.img_hw, auto=auto)
        # cv2.imshow('x', img)
        # cv2.waitKey(0)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img, img0

    def dynamic_detect(self, image, img0s, areas=None, classes=None, conf_threshold=0.6, iou_threshold=0.4):
        output = {}
        if classes is not None:
            for c in classes:
                output[self.names[int(c)]] = 0
        else:
            for n in self.names:
                output[n] = 0
        # image, img0 = self.preprocess(image)

        # if len(image.shape) == 4:
        #     _, img0 = self.preprocess(image[0])
        #     image = np.stack([self.preprocess(i)[0] for i in image], axis=0)
        # else:
        #     image, img0 = self.preprocess(image)
        # image = np.repeat(np.expand_dims(image, axis=0), 4, axis=0)
        # print(image.shape)
        # ttx = time.time()
        img = torch.from_numpy(image).to(self.device)
        # print('xxxxxxxxxxxxxx:', time.time() - ttx)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # print(img.shape)
        torch.cuda.synchronize()
        pred = self.model(img)[0] 
        pred = non_max_suppression_face(
            pred, conf_threshold, iou_threshold, classes=classes, agnostic=False)

        torch.cuda.synchronize()
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0s[i].shape).round()
                det[:, 6:] = scale_coords(img.shape[2:], det[:, 6:], 
                                          img0s[i].shape, landmark=True).round()
                # if areas is not None and len(areas[i]):
                #     _, warn = polygon_ROIarea(
                #         det[:, :4], areas[i], img0s[i])
                #     det = det[warn]
                #     pred[i] = det
                for di, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    landmarks = det[di, 6:]   # xy * 5
                    # alignImg = align_img(img0s[i], landmarks, 224)
                    # cv2.imshow('x', alignImg)
                    # cv2.waitKey(1)
                    output[self.names[int(cls)]] += 1
                    # label = '%s %.2f' % (self.names[int(cls)], conf)
                    label = '%s' % (self.names[int(cls)])
                    # print(conf)
                    # xyxy = [int(i) for i in xyxy]
                    # im0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :] = 114
                    # if not self.names[int(cls)] in ['uniform', 'no-uniform']:
                    if self.show:
                        plot_one_box(xyxy, img0s[i], label=None,
                                     color=self.colors[int(cls)], 
                                     line_thickness=2, landmarks=landmarks)

        if self.show:
            for i in range(len(img0s)):
                cv2.namedWindow(f'p{i}', cv2.WINDOW_NORMAL)
                cv2.imshow(f'p{i}', img0s[i])
            key = cv2.waitKey(0 if self.pause else 1)
            self.pause = True if key == ord(' ') else False
            if key == ord('q') or key == ord('e') or key == 27:
                exit()
        return pred, output


if __name__ == "__main__":
    detector = Yolov5Face(weight_path='weights/yolov5s-face.pt', device='0', img_hw=(640, 640))

    detector.show = True
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img, img_raw = detector.preprocess(frame, auto=True)
        preds, _ = detector.dynamic_detect(img, [img_raw])
