"""
testing on UCFRep dataset
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import logging

# load UCFRep dataset
from dataset.UCFRep_loader import TestData

from models.TransRAC import TransferModel
from testing.ucf_test import test_loop

N_GPU = 1
device_ids = [i for i in range(N_GPU)]

# we pick out the fixed frames from raw video file, and we store them as .npz file
# we currently support 64 or 128 frames
# data root path
root_path = '/data0/wanghang/VRAC_1/dataset/UCFRep/'

test_video_dir = 'UCF101'
# test_video_dir = 'videos/val'
test_label_dir = 'annotations/val'

# video swin transformer pretrained model and config
checkpoint = '/data0/wanghang/VRAC_2/RAC_136_AE_20_1_V100/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

# TransRAC trained model checkpoint, we will upload soon.
# lastckpt = None

NUM_FRAME = 64
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [1, 4, 8]
test_dataset = TestData(root_path, test_video_dir, test_label_dir, num_frame=NUM_FRAME)
my_model = TransferModel(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)

log_path = f'./log_test'
if not os.path.exists(log_path):
    os.makedirs(log_path)
logging.basicConfig(level=logging.DEBUG, filename=f"{log_path}/VRAC_P2L_26_seed4_8_1_2_3_aug3_resume_UCFRep", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

lastckpt_path = '/data0/wanghang/VRAC_2/VRAC_P2L_26_seed4_8_1_2_3_aug3_resume/checkpoint/VRAC_P2L_26_seed4_8_1_2_3_aug3_resume/'
pt_file = os.listdir(lastckpt_path)
pt_file.sort()

bs = 128
results_test = []
for i in range(len(pt_file)):
    # if int(pt_file[i].split('_')[0]) > 106:
        lastckpt = lastckpt_path + pt_file[i]
        testMAE, testOBO = test_loop(my_model, test_dataset, batch_size=bs, lastckpt=lastckpt)
        results_test.append({'pt_file': pt_file[i],
                             'testMAE': '%.4f' % testMAE,
                             'testOBO': '%.4f' % testOBO})
        print_info = "pt_file:{}, testMAE={}, testOBO={}".format(pt_file[i], '%.4f' % testMAE, '%.4f' % testOBO)
        print(print_info)
        logging.info(print_info)

print(results_test)

