"""
test IVAC_P2L model
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
root_path = '../VRAC_2/dataset_REPCountA_resume/'

test_video_dir = 'video_npz_64_resume/test'
test_label_dir = 'annotation/test.csv'

# please make sure the pretrained model path is correct
checkpoint = '../VRAC_2/RAC_136_AE_20_1_V100/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

log_path = f'./log_test/RepCount_A'
if not os.path.exists(log_path):
    os.makedirs(log_path)
logging.basicConfig(level=logging.DEBUG, filename=f"{log_path}/VRAC_P2L_22_A", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

NUM_FRAME = 64
# multi scales(list). we currently support 1, 4, 8 scale.
SCALES = [1, 4, 8]
test_dataset = MyData(root_path, test_video_dir, test_label_dir, num_frame=NUM_FRAME)
my_model = IVAC_P2L(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)

lastckpt = './checkpoints/Epoch-67_MAE_0.4022_OBO_0.3444.pt'

MAE, OBO = test_loop(my_model, test_dataset, lastckpt=lastckpt)
print_info = "testMAE={}, testOBO={}".format('%.4f' % MAE, '%.4f' % OBO)
print(print_info)
logging.info(print_info)
