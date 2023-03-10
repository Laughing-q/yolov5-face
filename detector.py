from utils.torch_utils import select_device
from models.experimental import attempt_load
import random
import os
import cv2
from utils.datasets import letterbox
import numpy as np
import torch
from utils.plots import plot_one_box

# from utils.face_align import align_img
from utils.general import non_max_suppression_face, scale_coords, polygon_ROIarea, distance, vLineAngle, point_line_distance
from pfld import PFLD


def plot_text(x, img, text, color=None, thickness=2):
    tf = max(thickness - 1, 1)  # font thickness
    t_size = cv2.getTextSize(text, 0, fontScale=thickness / 3, thickness=tf)[0]
    c1 = int(x[0]), int(x[1])
    c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(
        img,
        text,
        (c1[0], c1[1] + t_size[1]),
        0,
        thickness / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )


class Yolov5Face:
    def __init__(self, weight_path, device, img_hw=(384, 640)):
        self.weights = weight_path
        self.device = select_device(device)
        self.half = True
        self.model = attempt_load(self.weights, map_location=self.device)
        # torch.save(self.model.state_dict(), weight_path.replace('pt', 'pth'))
        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.show = False
        self.img_hw = img_hw
        self.pause = False
        # self.save = True

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
        pred = non_max_suppression_face(pred, conf_threshold, iou_threshold, classes=classes, agnostic=False)

        torch.cuda.synchronize()
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0s[i].shape).round()
                det[:, 6:] = scale_coords(img.shape[2:], det[:, 6:], img0s[i].shape, landmark=True).round()
                if areas is not None and len(areas[i]):
                    _, warn = polygon_ROIarea(det[:, :4], areas[i], frame=None)
                    det = det[warn]
                    pred[i] = det
                for di, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                    landmarks = det[di, 6:].view(-1, 2)  # xy * 5
                    # alignImg = align_img(img0s[i], landmarks, 224)
                    # cv2.imshow('x', alignImg)
                    # cv2.waitKey(1)
                    output[self.names[int(cls)]] += 1
                    # label = '%s' % (self.names[int(cls)])
                    positive = self.is_positive(xyxy, landmarks.cpu().numpy())
                    label = "%s %.2f %s" % (self.names[int(cls)], conf, "positive" if positive else 'crooked')
                    # label = '%s %.2f' % (self.names[int(cls)], conf)
                    # if not positive:
                        # self.pause = True
                    if self.show:
                        plot_one_box(
                            xyxy,
                            img0s[i],
                            label=label,
                            color=self.colors[int(cls)] if positive else (0, 0, 255),
                            line_thickness=2,
                            landmarks=landmarks,
                        )

        if self.show:
            for i in range(len(img0s)):
                cv2.namedWindow(f"p{i}", cv2.WINDOW_NORMAL)
                cv2.imshow(f"p{i}", img0s[i])
            key = cv2.waitKey(0 if self.pause else 1)
            self.pause = True if key == ord(" ") else False
            if key == ord("q") or key == ord("e") or key == 27:
                exit()
        return pred, output, img0s

    def is_positive(self, xyxy, landmark):
        x1, y1, x2, y2 = xyxy
        leye = landmark[0]
        reye = landmark[1]
        nose = landmark[2]
        lmouth = landmark[3]
        rmouth = landmark[4]
        bh, bw = y2 - y1, x2 - x1
        angle = vLineAngle([leye, reye], [(x1, y1), (x2, y1)])
        line1 = [leye, reye]
        line2 = [lmouth, rmouth]
        line3 = [lmouth, leye]
        line4 = [rmouth, reye]
        base_line1 = [leye, rmouth]
        base_line2 = [reye, lmouth]
        distance_x = np.asarray([point_line_distance(nose, line) for line in [line1, line2, line3, line4]])
        distance_base = np.asarray([point_line_distance(nose, line) / 2 for line in [base_line1, base_line2]])
        condition = (distance_base[0] < distance_x).all() or (distance_base[1] < distance_x).all()
        # if not condition:
        #     return False
        if (
            distance(leye, reye) < bw / 4
            or distance((leye + reye) / 2, (lmouth + rmouth) / 2) < bh / 4
            or (not condition)
            # or angle % 90
        ):
            return False
        return True

def is_positive(angle, pitch_thres=0.5, yaw_thres=0.5, roll_thres=0.3):
    assert len(angle) == 3
    pitch, yaw, roll = angle
    return (abs(pitch) < pitch_thres and abs(yaw) < yaw_thres and abs(roll) < roll_thres)


if __name__ == "__main__":
    detector = Yolov5Face(weight_path="./weights/yolov5l-face.pt", device="0", img_hw=(640, 640))
    pfld = PFLD(weight="weights/checkpoint.pth.tar")

    detector.show = False

    save = False
    save_path = "./test"

    cap = cv2.VideoCapture("/d/dataset/sleep/videos/172.16.11.133_01_2022102412014046.mp4")
    # cap = cv2.VideoCapture("/home/laughing/Videos/test.mp4")
    frame_num = 0
    areas = []
    # areas = np.array([[973, 287],
    #                  [1239, 281],
    #                  [1265, 635],
    #                  [ 943, 810],
    #                  [ 973, 287]])

    fourcc = "XVID"  # output video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = (
        cv2.VideoWriter(
            # './face.mp4', cv2.VideoWriter_fourcc(*fourcc), fps,
            save_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (w, h),
        )
        if save
        else None
    )
    pause = False
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame_num == 0:
        # for _ in range(4):
        #     areas.append(cv2.selectROI('p', frame)[:2])
        # areas.append(areas[0])
        # areas = np.array(areas)
        # print(areas)
        if not ret:
            break
        img, img_raw = detector.preprocess(frame, auto=True)
        preds, _, img_raws = detector.dynamic_detect(img, [img_raw], areas=[areas], conf_threshold=0.5)
        frame_num += 1
        bboxes = preds[0][:, :4].cpu().numpy()
        if len(bboxes):
            # landmarks = pfld.inference(img_raw, bboxes)
            landmarks, poses = pfld.inference(img_raw, bboxes, return_pose=True)
            # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
            for i, landmark in enumerate(landmarks):
                box = bboxes[i]
                for j, (x, y) in enumerate(landmark.astype(np.int32)):
                    # if j in TRACKED_POINTS:
                    cv2.circle(img_raw, (x, y), 1, (255, 0, 0), 2)
                # cv2.rectangle(img_raw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                angle = poses[i]
                positive = is_positive(angle)
                label = "positive" if positive else "crooked"
                plot_one_box(
                    [int(b) for b in box],
                    img_raw,
                    label=label,
                    color=(0, 255, 0) if positive else (0, 0, 255),
                    line_thickness=2,
                )
                # pitch = "pitch: " + str(round(angle[0], 2))
                # yaw = "yaw: " + str(round(angle[1], 2))
                # roll = "roll: " + str(round(angle[2], 2))
                # plot_text(
                #     (box[0], box[1] - (18 * 3)),
                #     img_raw,
                #     pitch,
                #     color=(255, 150, 125),
                #     thickness=2,
                # )
                # plot_text(
                #     (box[0], box[1] - (18 * 2)),
                #     img_raw,
                #     yaw,
                #     color=(255, 150, 125),
                #     thickness=2,
                # )
                # plot_text(
                #     (box[0], box[1] - (18 * 1)),
                #     img_raw,
                #     roll,
                #     color=(255, 150, 125),
                #     thickness=2,
                # )
        cv2.imshow('cv2', img_raw)
        key = cv2.waitKey(0 if pause else 1)
        pause = True if key == ord(' ') else False
        if key == ord('q') or key == ord('e') or key == 27:
            break

        # if vid_writer is not None:
        # vid_writer.write(img_raws[0])
