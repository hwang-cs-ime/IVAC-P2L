"""
对于cycles数量>15的视频，进行切分
training set:   1170 videos
validation set: 208 videos
testing set:    246 videos
"""

import numpy as np
import os
import cv2
import pandas as pd
# from label2rep import rep_label
from torch.utils.data import Dataset


count_gt =[]

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
        # print('empty data:', file)
        self.error += 1
        return False
    elif frames_num >= max(label):
        return True
    else:
        # print("error data:", file, frames_num)
        # print('error data:', label)
        # print("error data:", len(label))
        self.error += 1
        return False


class MyDataset(Dataset):

    def __init__(self, root_dir, label_dir, frames, method):
        """
        :param root_dir: dataset root dir
        :param label_dir: dataset child dir
         # data_root = r'/data/train'
         # data_lable = data_root + 'label.xlsx'
         # data_video = data_root + 'video/'
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
        # self.path = os.path.join(self.root_dir, self.child_dir)
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
            # if self.isdata(filename, label_tmp):  # data is right format ,save in list
            self.file_list.append(filename)
            self.label_list.append(label_tmp)
            count_gt.append(label_tmp)

        np.save('count_gt_' + method + '.npy', count_gt)
        print('Finished!')

    def __getitem__(self, index):
        """
        Save the preprocess frames and original video length in NPZ.
        npz[img = frames, fps = original_frames_length]
        """

        filename = self.file_list[index]
        labels = self.label_list[index]
        video_path = os.path.join(self.video_dir, filename)
        npz_pth = npz_dir + os.path.splitext(filename)[0] + '.npz'
        if not os.path.exists(npz_pth):
            print('video :', filename, ' does not been saved to npz.')
            a = self.read_video(video_path, filename, labels)
            if a == 1:
                print(index, filename, 'have been saved to npz.')
            else:
                print('error:', index, filename, 'can not be saved to npz.')
        else:
            try:
                data = np.load(npz_pth)
                print('good:', index, filename, 'loading success')
            except:
                print('error:', index, filename, 'is wrong npz which can not be opened.')

        # uncomment to check data
        # y1, y2, num_period = self.adjust_label(self.label_list[index], original_frames_length, self.num_frames)
        # count = len(self.label_list[index]) / 2
        # if count == 0:
        #     print('file:', filename)
        # if count - num_period > 0.1:
        #     print('file:', filename)
        #     print('y1:', y1)
        #     print('y2:', y2)
        #     print('count:', count)
        #     print('sum_y:', num_period)
        return 1

    def __len__(self):
        return len(self.file_list)

    def read_video(self, video_filename, filename, labels, width=224, height=224):
        """Read video from file."""
        try:
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
            frames_save = self.adjust_frames(frames, filename, labels)  # uncomment to adjust frames
        except:
            print('error: ', video_filename, ' cannot open')
        return 1

    def adjust_frames(self, frames, filename, labels):
        """
        # adjust the number of total video frames to the target frame num.
        : param frames: original frames
        :return: target number of frames
        """
        count_target = int(len(labels) / 2)
        num_split = int(count_target // 15)
        for i in range(num_split+1):
            label_temp = labels[i * 30 : min((i+1) * 30, len(labels))]

            if i == 0 and label_temp[0] > 0:
                frames_temp = frames[ : label_temp[0]]
            else:
                frames_temp = []

            frames_temp = frames_temp + frames[label_temp[0] : label_temp[-1]+1]

            if i == num_split:
                if (i+1) * 30 < len(labels):
                    label_temp = label_temp + labels[(i+1) * 30 : len(labels)]
                if label_temp[-1] + 1 < len(frames):
                    frames_temp = frames_temp + frames[label_temp[-1] + 1 : ]

            frames_adjust = []
            frame_length = len(frames_temp)
            if self.num_frames <= frame_length:
                for k1 in range(1, self.num_frames + 1):
                    frame = frames_temp[k1 * frame_length // self.num_frames - 1]
                    frames_adjust.append(frame)
            else:
                for k2 in range(frame_length):
                    frame = frames_temp[k2]
                    frames_adjust.append(frame)
                for _ in range(self.num_frames - frame_length):
                    if len(frames_temp) > 0:
                        frame = frames_temp[-1]
                        frames_adjust.append(frame)

            frames_adjust = np.asarray(frames_adjust)
            if (frames_adjust.size != 0):
                frames_adjust = frames_adjust.transpose(0, 3, 2, 1)  # [f,c,h,w]
            else:
                print(filename, ' is wrong video. size = 0')

            npz_pth = npz_dir + os.path.splitext(filename)[0] + '_part_' + str(i)
            np.savez(npz_pth, imgs=frames_adjust, annotation=label_temp, fps=frame_length)

if __name__ == '__main__':
    data_root = r'/data0/wanghang/VRAC_2/dataset_REPCountA_resume'
    tag = ['train', 'valid', 'test']
    for split in range(3):
        mod = tag[split]
        npz_dir = r'/data0/wanghang/VRAC_2/dataset_REPCountA_resume/video_npz_64_aug_step1/'
        npz_dir = npz_dir + mod + '/'
        label_file = 'annotation/' + mod + '.csv'
        test = MyDataset(data_root, label_file, 64, mod)
        print('=========================================')
        print(mod, ' : ', npz_dir)
        if not os.path.exists(npz_dir):
            os.mkdir(npz_dir)
        for i in range(len(test)):
            a = test[i]