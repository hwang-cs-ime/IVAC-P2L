"""
test TransRAC model
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging

# if your data is .mp4 form, please use RepCountA_raw_Loader.py
# from dataset.RepCountA_raw_Loader import MyData
# if your data is .npz form, please use RepCountA_Loader.py. It can speed up the training
from dataset.RepCountA_Loader import MyData
from models.IVAC import IVAC_P2L
from testing.test_looping import test_loop

N_GPU = 1
device_ids = [i for i in range(N_GPU)]

# we pick out the fixed frames from raw video file, and we store them as .npz file
# we currently support 64 or 128 frames
# data root path
root_path = '/data0/wanghang/VRAC_2/dataset_REPCountA_resume/'

test_video_dir = 'video_npz_64_resume/test'
test_label_dir = 'annotation/test.csv'

# please make sure the pretrained model path is correct
checkpoint = '/data0/wanghang/VRAC_2/RAC_136_AE_20_1_V100/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

log_path = f'./log_test/RepCount_A'
if not os.path.exists(log_path):
    os.makedirs(log_path)
logging.basicConfig(level=logging.DEBUG, filename=f"{log_path}/VRAC_P2L_22_A", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

NUM_FRAME = 64
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [1, 4, 8]
NUM_EPOCHS = 1
test_dataset = MyData(root_path, test_video_dir, test_label_dir, num_frame=NUM_FRAME)
my_model = IVAC_P2L(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)

lastckpt_path = '/data0/wanghang/VRAC_2/VRAC_P2L_22/checkpoint/VRAC_P2L_22/'
pt_file = os.listdir(lastckpt_path)
pt_file.sort()

results_test = []
for i in range(len(pt_file)):
    if int(pt_file[i].split('_')[0]) > 68:
        lastckpt = lastckpt_path + pt_file[i]
        MAE, OBO = test_loop(NUM_EPOCHS, my_model, test_dataset, lastckpt=lastckpt)
        results_test.append({'pt_file': pt_file[i],
                             'testMAE': '%.4f' % MAE,
                             'testOBO': '%.4f' % OBO})
        print_info = "pt_file:{}, testMAE={}, testOBO={}".format(pt_file[i], '%.4f' % MAE, '%.4f' % OBO)
        print(print_info)
        logging.info(print_info)

print(results_test)
