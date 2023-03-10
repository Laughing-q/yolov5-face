import torch
import torch.nn as nn
import cv2
import numpy as np
from .utils import calculate_pitch_yaw_roll


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False
            ),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)

        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        # return out1, landmarks # for training
        return landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(64, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class PFLD:
    def __init__(self, weight, imgsz=112):
        self.imgsz = imgsz
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weight, map_location=self.device)
        self.model = PFLDInference().to(self.device)
        self.model.load_state_dict(checkpoint["pfld_backbone"])
        self.model.eval()
        self.pose_points = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]  # for 98 points

        # dlib (68 landmark) trached points
        # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
        # wflw(98 landmark) trached points
        # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        # X-Y-Z with X pointing forward and Y on the left and Z up.
        # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
        # OpenCV uses the reference usually used in computer vision:
        # X points to the right, Y down, Z to the front
        self.landmarks_3D = np.float32([
            [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT, 
            [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT, 
            [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
            [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
            [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
            [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
            [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
            [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
            [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
            [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
            [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
            [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
            [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
            [0.000000, -7.415691, 4.070434],  # CHIN
        ])

    def preprocess(self, ori_img, bboxes):
        height, width = ori_img.shape[:2]
        faces = []
        sizes = []
        edx1s = []
        edy1s = []
        x1s = []
        y1s = []
        for box in bboxes:
            x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)

            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cx = x1 + w // 2
            cy = y1 + h // 2

            size = int(max([w, h]) * 1.1)
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            edx1 = max(0, -x1)
            edy1 = max(0, -y1)
            edx2 = max(0, x2 - width)
            edy2 = max(0, y2 - height)

            cropped = ori_img[y1:y2, x1:x2]
            if edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0:
                cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2, cv2.BORDER_CONSTANT, 0)
            faces.append(cv2.resize(cropped, (self.imgsz, self.imgsz)))
            sizes.append(size)
            edx1s.append(edx1)
            edy1s.append(edy1)
            x1s.append(x1)
            y1s.append(y1)
            # cv2.imshow('face', cropped)
            # cv2.waitKey(0)
        return faces, sizes, edx1s, edy1s, x1s, y1s

    def inference(self, ori_img, face_bboxes, return_pose=False):
        landmarks = []
        angles = []
        for face, size, edx1, edy1, x1, y1 in zip(*self.preprocess(ori_img, face_bboxes)):
            face = face.transpose(2, 0, 1)#[:, :, ::-1]
            face = torch.from_numpy(np.ascontiguousarray(face)).to(self.device).float()
            face /= 255.
            landmark = self.model(face[None])
            pre_landmark = landmark[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size] - [edx1, edy1] + [x1, y1]
            landmarks.append(pre_landmark)

        if return_pose:
            h, w = ori_img.shape[:2]
            for landmark in landmarks:
                angle_landmark = []
                for index in self.pose_points:
                    angle_landmark.append(landmark[index])
                angle_landmark = np.asarray(angle_landmark).reshape((-1, 28))
                pitch, yaw, roll = calculate_pitch_yaw_roll(angle_landmark[0], self.landmarks_3D, cam_w=w, cam_h=h)
                angles.append([pitch, yaw, roll])
            return landmarks, angles
        return landmarks
