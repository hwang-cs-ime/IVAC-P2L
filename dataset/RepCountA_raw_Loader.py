"""
If you don't want to pre-process the RepCount dataset, 
you can use this script to load data from raw video(.mp4).
The trainning speed is slower and the memory cost more.
"""

import os
import os.path as osp
import numpy as np
import math
import cv2
from torch.utils.data import Dataset
import torch
import csv
import kornia
from .label_norm import normalize_label
import torchvision.transforms as transforms
import random

class MyData(Dataset):

    def __init__(self, root_path, video_path, label_path, num_frame=0, aug=False):
        self.root_path = root_path
        self.video_path = video_path
        self.label_dir = os.path.join(root_path, label_path)
        self.video_dir = os.listdir(os.path.join(self.root_path, self.video_path))
        self.label_dict = get_labels_dict(self.label_dir)  # get all labels
        self.num_frame = num_frame
        self.aug = aug

    def __getitem__(self, inx):
        video_file_name= self.video_dir[inx]
        time_points = self.label_dict[video_file_name]
        file_path = os.path.join(self.root_path, self.video_path, video_file_name)
        video_rd = VideoRead(file_path, time_points, num_frames=self.num_frame)
        # video_frame_length = video_rd.frame_length

        video_tensor, label, index_pos, index_neg = video_rd.Normalize_frame_label(annotation=time_points, Num_Frames_Norm=self.num_frame, aug=self.aug)

        return [video_tensor, label, index_pos, index_neg]

    def __len__(self):
        """返回数据集的大小"""
        return len(self.video_dir)

class VideoRead:
    def __init__(self, video_path, time_points, num_frames):
        self.video_path = video_path
        self.frame_length = 0
        self.time_points = time_points
        self.num_frames = num_frames

    def get_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened()
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        self.frame_length = len(frames)
        return frames

    def Normalize_frame_label(self, annotation=[], Num_Frames_Norm=64, aug=False):
        """
        1. crop 到64帧
        2. 尺寸crop 到[224, 224]
        3. label也进行相应的crop
        4. 数据增强：（1）count_gt > 15的，随机取其中一部分作为增强后的，起始时间是不确定的，持续时间也是不确定的
                    (2) 随机crop到[224, 224], 水平翻转，垂直翻转
        to crop frames to tensor
        return: tensor [64, 3, 224, 224]
        """
        frames = self.get_frame()

        count_gt = int(len(self.time_points) / 2)
        if aug==True and random.random() > 0.5:
            if count_gt > 15:
                # 对视频在时序上做了增强
                # start = random.randint(0, int(self.frame_length/2))
                # # end = random.randint(int(self.frame_length/2)+1, self.frame_length-1)
                # end = random.randint(start+int(self.frame_length/4), self.frame_length - 1)
                # frames_new = frames[start:end+1]

                # # 相应地对label也进行crop
                # annotation_new = []
                # for index_label in range(len(annotation)):
                #     if start <= annotation[index_label] <= end:
                #         annotation_new.append(annotation[index_label])
                #
                # if len(annotation_new) % 2 != 0:
                #     del annotation_new[-1]
                #     if annotation_new[-1]+1 > len(frames):
                #         print(1)
                #     frames_new = frames[start:annotation_new[-1]+1]
                #
                # annotation_new = [annotation_new[i]-start for i in range(len(annotation_new))]

                index_start = random.randrange(0, int(len(annotation)/2), 2)
                start = annotation[index_start]
                if int(len(annotation)/2)%2==0:
                    index_end = random.randrange(int(len(annotation)/2)+1, int(len(annotation)), 2)
                elif int(len(annotation)/2)%2 != 0:
                    index_end = random.randrange(int(len(annotation)/2), int(len(annotation)), 2)
                end = annotation[index_end]
                annotation_new = annotation[index_start:index_end+1]
                annotation_new = [anno-start for anno in annotation_new]

                # if len(annotation_new) % 2 != 0:
                #     print(1)

                frames_new = frames[start:end+1]
                frames = frames_new
                annotation = annotation_new
                # 更新self.frame_length
                self.frame_length = len(frames)

        frames_tensor = []
        if self.num_frames <= self.frame_length:
            for i in range(1, self.num_frames + 1):
                frame = frames[i * self.frame_length // self.num_frames - 1]
                frame = cv2.resize(frame, (224, 224))
                frames_tensor.append(frame)
        else:
            for i in range(self.frame_length):
                frame = frames[i]
                frame = cv2.resize(frame, (224, 224))
                frames_tensor.append(frame)
            for i in range(self.num_frames - self.frame_length):
                frame = frames[self.frame_length - 1]
                frame = cv2.resize(frame, (224, 224))
                frames_tensor.append(frame)

        Frame_Tensor = np.asarray(frames_tensor)
        Frame_Tensor = Frame_Tensor.transpose(0, 3, 2, 1)  # [f,w,h,c] -> [f, c, h, w]
        Frame_Tensor = torch.FloatTensor(Frame_Tensor)

        if aug == True:
            Frame_Tensor = transforms.RandomVerticalFlip()(Frame_Tensor)
            Frame_Tensor = transforms.RandomHorizontalFlip()(Frame_Tensor)

        # Frame_Tensor -= 127.5
        # Frame_Tensor /= 127.5

        Frame_Tensor = transforms.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))(Frame_Tensor)

        label_new = []
        for i in range(len(annotation)):  # frame_length -> 64
            item = min(math.ceil((float((annotation[i])) / float(self.frame_length)) * Num_Frames_Norm), Num_Frames_Norm - 1)
            label_new.append(item)
        label_new = np.sort(label_new)
        label, index_pos = normalize_label(label_new, self.num_frames)

        index_neg = []
        if label_new[0] > 0:
            index_neg.append(0)
            index_neg.append(label_new[0])
        for i in range(1, len(label_new), 2):
            if i==len(label_new)-1:
                if label_new[i] < self.num_frames:
                    index_neg.append(label_new[i])
                    index_neg.append(self.num_frames)
            else:
                x_a = label_new[i]
                x_b = label_new[i + 1]
                num = x_b - x_a
                if num > 0:
                    index_neg.append(x_a)
                    index_neg.append(x_b)
        # assert len(index_neg) <= self.num_frames, "len(index_neg) > num_frames"
        index_neg.extend([-1 for i in range(self.num_frames * 5 - len(index_neg))])

        label = torch.tensor(label)
        index_pos = torch.tensor(index_pos)
        index_neg = torch.tensor(index_neg)

        Frame_Tensor = Frame_Tensor.type(torch.FloatTensor)
        Frame_Tensor = Frame_Tensor.transpose(0, 1)
        label = label.type(torch.FloatTensor)

        return Frame_Tensor, label, index_pos, index_neg


def get_labels_dict(path):
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(row[key]) for key in row.keys() if 'L' in key and row[key] != '']
            if not row['count']:
                print('{} error, and it has no count annotation.'.format(row['name']))
            elif row['count'] == '0':
                print("The count of {} is {}.".format(row['name'], row['count']))
            else:
                labels_dict[row['name']] = cycle

    return labels_dict

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


