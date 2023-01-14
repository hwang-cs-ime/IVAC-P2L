"""
train or valid looping
"""
import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.associative_loss import associative_loss
import logging
import torch.backends.cudnn as cudnn
import random


# seed = 13
# cudnn.benchmark = False
# # cudnn.benchmark = True
# cudnn.deterministic = True
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


torch.manual_seed(1)


def train_loop(n_epochs,
               model,
               train_set,
               test_set,
               train=True,
               inference=True,
               batch_size=64,
               lr=8e-5,
               ckpt_name='VRAC_P2L_26_seed4_8_1_2_3_aug3_resume',
               lastckpt=None,
               saveckpt=False,
               log_dir='VRAC_P2L_26_seed4_8_1_2_3_aug3_resume',
               device_ids=[0],
               mae_error=False):
    
    log_path = f'./log_test/RepCount_A'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.DEBUG, filename=f"{log_path}/VRAC_P2L_26_seed4_8_1_2_3_aug3_resume_A", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0

    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=1)
    testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=1)

    model = nn.DataParallel(model.to(device), device_ids=device_ids)
    
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # milestones = [30, 80]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)  # three step decay

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    milestones = [i for i in range(0, n_epochs, 40)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.8)  # three step decay

    writer = SummaryWriter(log_dir=os.path.join('log/', log_dir))
    scaler = GradScaler()

    if lastckpt is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch'] + 1
        # load hyperparameters by pytorch
        # if change model
        # net_dict=model.state_dict()
        # state_dict={k: v for k, v in checkpoint.items() if k in net_dict.keys()}
        # net_dict.update(state_dict)
        # model.load_state_dict(net_dict, strict=False)

        # or don't change model
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    lossMSE = nn.MSELoss()
    lossSL1 = nn.SmoothL1Loss()
    AE_loss = associative_loss()

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        trainLosses = []
        trainLoss1 = []
        trainLoss3 = []
        trainLoss4 = []
        trainOBO = 0
        trainMAE = []
        if train:
            pbar = tqdm(trainloader, total=len(trainloader))
            for input, target, index_pos, index_neg in pbar:
                with autocast():
                    model.train()
                    optimizer.zero_grad()
                    acc = 0
                    input = input.type(torch.FloatTensor).to(device)
                    density = target.type(torch.FloatTensor).to(device)
                    count = torch.sum(target, dim=1).type(torch.FloatTensor).round().to(device)
                    output, _, feat_x = model(input)
                    predict_count = torch.sum(output, dim=1).type(torch.FloatTensor).to(device)
                    predict_density = output
                    loss1 = lossMSE(predict_density, density)
                    # loss2 = lossSL1(predict_count, count)
                    loss2 = lossMSE(predict_count, count)
                    loss3 = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / predict_count.flatten().shape[0]  # mae
                    loss4 = AE_loss(feat_x, index_pos, index_neg)
                    loss = 10 * loss1 + 10 * loss4

                    if mae_error:
                        loss += loss3

                    # calculate MAE or OBO
                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    trainOBO += acc
                    MAE = torch.div(torch.abs(predict_count - count), count + 1e-1).tolist()  # mae
                    trainMAE += MAE

                    train_loss = loss.item()
                    train_loss1 = loss1.item()
                    train_loss3 = loss3.item()
                    train_loss4 = loss4.item()
                    trainLosses.append(train_loss)
                    trainLoss1.append(train_loss1)
                    trainLoss3.append(train_loss3)
                    trainLoss4.append(train_loss4)
                    pbar.set_postfix({'Epoch': epoch})
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            writer.add_scalars('learning_rate', {"learning_rate": optimizer.state_dict()['param_groups'][0]['lr']}, epoch)
            writer.add_scalars('train/trainMAE', {"trainMAE": torch.sum(torch.tensor(trainMAE)) / len(trainMAE)}, epoch)
            writer.add_scalars('train/trainOBO', {"trainOBO": trainOBO / len(trainMAE)}, epoch)
            writer.add_scalars('train/trainloss', {"trainloss": np.mean(trainLosses)}, epoch)
            writer.add_scalars('train/trainloss_1', {"trainloss_1": np.mean(trainLoss1)}, epoch)
            writer.add_scalars('train/trainloss_3', {"trainloss_3": np.mean(trainLoss3)}, epoch)
            writer.add_scalars('train/trainloss_4', {"trainloss_4": np.mean(trainLoss4)}, epoch)

        testLosses = []
        testLoss1 = []
        testLoss3 = []
        testLoss4 = []
        testMAE = []
        testOBO = 0
        testOBO_2 = 0
        testOBO_3 = 0
        if inference:
            with torch.no_grad():
                pbar_test = tqdm(testloader, total=len(testloader))
                for input_test, target_test, index_pos_test, index_neg_test in pbar_test:
                    model.eval()
                    acc_test = 0
                    acc_test_2 = 0
                    acc_test_3 = 0
                    input_test = input_test.to(device)
                    density_test = target_test.type(torch.FloatTensor).to(device)
                    count_test = torch.sum(target_test, dim=1).type(torch.FloatTensor).clone().detach().round().to(device)

                    output_test, _, feat_x_test = model(input_test)
                    predict_count_test = torch.sum(output_test, dim=1).round()
                    predict_density_test = output_test

                    loss1_test = lossMSE(predict_density_test, density_test)
                    # loss2_test = lossSL1(predict_count_test, count_test)
                    loss2_test = lossMSE(predict_count_test, count_test)
                    loss3_test = torch.sum(torch.div(torch.abs(predict_count_test - count_test), count_test + 1e-1)) / predict_count_test.flatten().shape[0]  # mae
                    loss4_test = AE_loss(feat_x_test, index_pos_test, index_neg_test)
                    loss_test = 10 * loss1_test + 10 * loss4_test

                    if mae_error:
                        loss_test += loss3_test
                    gaps_test = torch.sub(predict_count_test, count_test).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item_test in gaps_test:
                        if abs(item_test) <= 1:
                            acc_test += 1
                        if abs(item_test) <= 2:
                            acc_test_2 += 1
                        if abs(item_test) <= 3:
                            acc_test_3 += 1

                    testOBO += acc_test
                    testOBO_2 += acc_test_2
                    testOBO_3 += acc_test_3

                    MAE_test = torch.div(torch.abs(predict_count_test - count_test), count_test + 1e-1).tolist()  # mae
                    testMAE += MAE_test

                    test_loss = loss_test.item()
                    test_loss1 = loss1_test.item()
                    test_loss3 = loss3_test.item()
                    test_loss4 = loss4_test.item()
                    testLosses.append(test_loss)
                    testLoss1.append(test_loss1)
                    testLoss3.append(test_loss3)
                    testLoss4.append(test_loss4)
                    pbar_test.set_postfix({'Epoch': epoch})
                writer.add_scalars('test/MAE', {"MAE": torch.sum(torch.tensor(testMAE))/len(testMAE)}, epoch)
                writer.add_scalars('test/OBO', {"OBO": testOBO / len(testMAE)}, epoch)
                writer.add_scalars('test/OBO_2', {"OBO_2": testOBO_2 / len(testMAE)}, epoch)
                writer.add_scalars('test/OBO_3', {"OBO_3": testOBO_3 / len(testMAE)}, epoch)
                writer.add_scalars('test/loss', {"loss": np.mean(testLosses)}, epoch)
                writer.add_scalars('test/loss_1', {"loss_1": np.mean(testLoss1)}, epoch)
                writer.add_scalars('test/loss_3', {"loss_3": np.mean(testLoss3)}, epoch)
                writer.add_scalars('test/loss_4', {"loss_4": np.mean(testLoss4)}, epoch)

        scheduler.step()

        if not os.path.exists('checkpoint/{0}/'.format(ckpt_name)):
            os.mkdir('checkpoint/{0}/'.format(ckpt_name))
        if saveckpt:
            checkpoint = {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'trainLosses': trainLosses,
                          'testLosses': testLosses}
            torch.save(checkpoint, 'checkpoint/{0}/'.format(ckpt_name) + str(epoch) + '_' + str(round((torch.sum(torch.tensor(testMAE))/len(testMAE)).item(), 4)) + '_' + str(round(testOBO/len(testMAE), 4)) + '.pt')
        
        print_info = "Epoch:{}, testMAE={}, testOBO={}, testOBO_2={}, testOBO_3={}".format(
                     epoch, 
                     round((torch.sum(torch.tensor(testMAE))/len(testMAE)).item(), 4), 
                     round(testOBO/len(testMAE), 4),
                     round(testOBO_2/len(testMAE), 4),
                     round(testOBO_3/len(testMAE), 4))
        logging.info(print_info)

