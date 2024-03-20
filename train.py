# from platform import node
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# if your data is .mp4 form, please use RepCountA_raw_Loader.py (slowly)
from dataset.RepCountA_raw_Loader import MyData

# you can use 'tools.video2npz.py' to transform .mp4 to .npz
from models.IVAC import IVAC_P2L
from training.train_looping import train_loop

# CUDA environment
N_GPU = 1
device_ids = [i for i in range(N_GPU)]

root_path = '/data0/wanghang/VRAC_2/dataset_REPCountA_resume/'

train_video_dir = 'video/train'
train_label_dir = 'annotation/train.csv'
test_video_dir = 'video/test'
test_label_dir = 'annotation/test.csv'

# please make sure the pretrained model path is correct
checkpoint = '/data0/wanghang/VRAC_2/RAC_136_AE_20_1_V100/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

# TransRAC trained model checkpoint, we will upload soon.
lastckpt = None

NUM_FRAME = 64
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [1, 4, 8]

train_dataset = MyData(root_path, train_video_dir, train_label_dir, num_frame=NUM_FRAME, aug=True)
test_dataset = MyData(root_path, test_video_dir, test_label_dir, num_frame=NUM_FRAME, aug=False)
my_model = IVAC_P2L(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)
NUM_EPOCHS = 200
BATCH_SIZE = 64
LR = 8e-5

train_loop(NUM_EPOCHS, 
           my_model, 
           train_dataset, 
           test_dataset, 
           train=True, 
           inference=True,
           batch_size=BATCH_SIZE, 
           lr=LR, 
           saveckpt=True, 
           ckpt_name='VRAC_P2L_26_seed4_8_1_2_3_aug3_resume',
           log_dir='VRAC_P2L_26_seed4_8_1_2_3_aug3_resume',
           device_ids=device_ids,
           lastckpt=lastckpt, 
           mae_error=False)
