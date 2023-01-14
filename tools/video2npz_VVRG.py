"""
根据VVRG的采样方式生成npz
fids_sample_gt = np.round(np.linspace(0, len(video_anno['trajectories']) - 1, int(num_frames_sample)))
fids_sample_gt = list(fids_sample_gt)
fids_index_gt = [int(fid_gt) for fid_gt in fids_sample_gt]
fids_index = sorted(fids_index_gt)
"""

import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset


def isdata(self, file, label):
    """
    :param file: original video
    :param label: original label
    :return: the number of load error
    """
    video_path = os.path.join(self.video_dir, file)
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    if label.size == 0:
        self.error += 1
        return False
    elif frames_num >= max(label):
        return True
    else:
        self.error += 1
        return False


class MyDataset(Dataset):

    def __init__(self, root_dir, label_dir, frames, method):
        """
        :param root_dir: dataset root dir
        :param label_dir: dataset child dir
        """
        super().__init__()
        self.root_dir = root_dir
        if method == 'train':
            self.video_dir = os.path.join(self.root_dir, r'video/train')
        elif method == 'valid':
            self.video_dir = os.path.join(self.root_dir, r'video/valid')
        elif method == 'test':
            self.video_dir = os.path.join(self.root_dir, r'video/test')
        else:
            raise ValueError('module is wrong.')
        self.video_filename = os.listdir(self.video_dir)
        self.label_filename = os.path.join(self.root_dir, label_dir)
        self.file_list = []
        self.label_list = []
        self.num_idx = 4
        self.num_frames = frames  # model frames
        self.error = 0
        df = pd.read_csv(self.label_filename)
        for i in range(0, len(df)):
            filename = df.loc[i, 'name']
            label_tmp = df.values[i][self.num_idx:].astype(np.float64)
            label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
            self.file_list.append(filename)
            self.label_list.append(label_tmp)

    def __getitem__(self, index):
        """
        Save the preprocess frames and original video length in NPZ.
        npz[img = frames, fps = original_frames_length]
        """
        filename = self.file_list[index]
        video_path = os.path.join(self.video_dir, filename)
        label = self.label_list[index]
        npz_pth = npz_dir + os.path.splitext(filename)[0] + '.npz'
        # if not os.path.exists(npz_pth):
        print('video :', filename, ' does not been saved to npz.')
        a = self.read_video(video_path, filename, label)
        if a == 1:
            print(index, filename, 'have been saved to npz.')
        else:
            print('error:', index, filename, 'can not be saved to npz.')
        # else:
        #     try:
        #         print('good:', index, filename, 'loading success')
        #     except:
        #         print('error:', index, filename, 'is wrong npz which cant opened.')
        return 1

    def __len__(self):
        return len(self.file_list)

    def read_video(self, video_filename, filename, label, width=224, height=224):
        """
        Read video from file
        """
        try:
            if filename=='stu3_48.mp4':
                print(1)
            cap = cv2.VideoCapture(video_filename)
            frames = []
            if cap.isOpened():
                while True:
                    success, frame_bgr = cap.read()
                    if success is False:
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.resize(frame_rgb, (width, height))
                    frames.append(frame_rgb)
            cap.release()
            original_frames_length = len(frames)
            frames, label = self.adjust_frames(frames, label)  # uncomment to adjust frames
            frames = np.asarray(frames)  # [f,h,w,c]
            if (frames.size != 0):
                frames = frames.transpose(0, 3, 2, 1)  # [f,c,h,w]
            else:
                print(filename, ' is wrong video. size = 0')
                return -1
            npz_pth = npz_dir + os.path.splitext(filename)[0]
            # np.savez(npz_pth, imgs=frames, fps=original_frames_length, label=label)  # [f,c,h,w]
        except:
            print('error: ', video_filename, ' cannot open')
        return 1

    def adjust_frames(self, frames, label):
        """
        # adjust the number of total video frames to the target frame num.
        :param frames: original frames
        :return: target number of frames
        """
        frames_adjust = []
        frame_length = len(frames)
        if self.num_frames <= len(frames):
            # for i in range(1, self.num_frames + 1):
            #     frame = frames[i * frame_length // self.num_frames - 1]  # // 整除的意思
            #     frames_adjust.append(frame)
            index = sorted([int(i) for i in (np.round(np.linspace(0, len(frames) - 1, int(self.num_frames))).tolist())])
            frames_adjust = [frames[k] for k in index]
            label_new = []
            for i in range(0, len(label), 2):
                for j in range(len(index)):
                    if index[j] <= label[i] <= index[j+1]:
                        label_new.append(j+1)
                        break
                for k in range(j+1, len(index)):
                    if index[k] <= label[i+1] <= index[k+1]:
                        label_new.append(k)
                        break
            label = label_new
        else:
            for i in range(frame_length):
                frame = frames[i]
                frames_adjust.append(frame)
            for _ in range(self.num_frames - frame_length):
                if len(frames) > 0:
                    frame = frames[-1]
                    frames_adjust.append(frame)
        return frames_adjust, label  # [f,h,w,3]


if __name__ == '__main__':
    data_root = r'/data0/wanghang/VRAC_2/dataset_REPCountA_resume'
    tag = ['train', 'valid', 'test']
    for split in range(len(tag)):
        npz_dir = r'/data0/wanghang/VRAC_2/dataset_REPCountA_resume/video_npz_64_VVRG/'
        mod = tag[split]
        label_file = 'annotation/' + mod + '.csv'
        test = MyDataset(data_root, label_file, 64, mod)
        npz_dir = npz_dir + mod + '/'
        print('=========================================')
        print(mod, ' : ', npz_dir)
        if not os.path.exists(npz_dir):
            os.mkdir(npz_dir)
        for i in range(len(test)):
            a = test[i]
