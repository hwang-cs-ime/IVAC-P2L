import imp
import os

import numpy as np
import csv
import cv2
from torch.utils.data import Dataset, DataLoader
import kornia
import torch
from scipy.io import loadmat
import torchvision.transforms as transforms


class TestData(Dataset):
    def __init__(self, root_path, video_path, label_path, num_frame):
        """
        Args:
            root_path:
            video_path:
            label_path:
            num_frame:
        """
        self.root_path = root_path
        self.video_path = video_path
        self.label_dir = os.path.join(root_path, label_path)
        self.video_dir, self.label_dict = get_video_anno(root_path, label_path)
        # self.video_dir = os.listdir(os.path.join(self.root_path, self.video_path))
        # self.label_dict = get_labels_dict(self.label_dir)  # get all labels
        self.num_frame = num_frame

    def __getitem__(self, inx):
        """获取数据集中的item  """

        video_name = self.video_dir[inx]
        file_path = os.path.join(self.root_path, self.video_path, video_name + '.avi')
        # file_path = os.path.join(self.root_path, self.video_path, video_name + '.mp4')
        video_rd = VideoRead(file_path, num_frames=self.num_frame)
        video_tensor = video_rd.crop_frame()
        video_tensor = video_tensor.transpose(0, 1)  # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
        count = self.label_dict[video_name]
        count = torch.tensor(count)

        return video_tensor, count

    def __len__(self):
        """返回数据集的大小"""
        return len(self.video_dir)


class VideoRead:
    def __init__(self, video_path, num_frames):
        self.video_path = video_path
        self.frame_length = 0
        self.num_frames = num_frames

    def get_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened(), "video name = {}".format(self.video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        self.frame_length = len(frames)
        return frames

    def crop_frame(self):
        """to crop frames to 64 frames"""
        frames = self.get_frame()
        frames_tensor = []
        if self.num_frames <= self.frame_length:
            for i in range(1, self.num_frames + 1):
                frame = frames[i * len(frames) // self.num_frames - 1]  # Proportional extraction (64 frames) 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
                # frame = transform(frame).unsqueeze(0)
                frames_tensor.append(frame)

        else:  # if raw frames number lower than 64, padding it. 
            for i in range(self.frame_length):
                frame = frames[i]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
                frames_tensor.append(frame)
            for i in range(self.num_frames - self.frame_length):
                frame = frames[self.frame_length - 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
                frames_tensor.append(frame)
        # frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
        # frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)

        # frames_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(frames_tensor)

        Frame_Tensor = np.asarray(frames_tensor)
        # Frame_Tensor = Frame_Tensor.transpose(0, 3, 1, 2)  # [f, h, w, c] -> [f, c, h, w]
        Frame_Tensor = Frame_Tensor.transpose(0, 3, 2, 1)  # [f,w,h,c] -> [f, c, h, w]，和训练的RepCount A保持一致
        Frame_Tensor = torch.FloatTensor(Frame_Tensor)

        Frame_Tensor = transforms.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))(Frame_Tensor)
        Frame_Tensor = Frame_Tensor.type(torch.FloatTensor)

        return Frame_Tensor


def get_video_anno(root_path, label_path):
    video_dir_path = os.path.join(root_path, label_path)
    lists = os.listdir(video_dir_path)
    lists.sort()
    video_names = []
    labels_dict = {}
    for i in range(len(lists)):
        video_name = lists[i][0:-4]

        if video_name=='v_HandStandPushups_g22_c02' or video_name=='v_HandStandPushups_g23_c04' or video_name=='v_HandStandPushups_g21_c04' or video_name=='v_HandStandPushups_g24_c02' or video_name=='v_HandStandPushups_g25_c06':
            continue

        video_names.append(video_name)
        anno = loadmat(os.path.join(video_dir_path, lists[i]))
        count = len(anno['label'][0, 0]['temporal_bound'][:, 0]) - 1
        if count == 0:
            print("The count of {}.avi video is 0".format(video_name))
        else:
            assert count > 0, "count<=0"
            labels_dict[video_name] = int(count)

    return video_names, labels_dict




