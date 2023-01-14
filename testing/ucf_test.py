import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from tools.my_tools import paint_smi_matrixs, density_map

torch.manual_seed(1)


def test_loop(model, test_set, batch_size=128, lastckpt=None, paint=False, device_ids=[0]):
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=20)
    model = nn.DataParallel(model.to(device), device_ids=device_ids)

    if lastckpt != None:
        print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint

    testOBO = 0
    testMAE = []
    with torch.no_grad():
        pbar = tqdm(testloader, total=len(testloader))
        for input, target in pbar:
            model.eval()
            acc_test = 0
            input = input.to(device)
            count = target.to(device)
            output, _, _ = model(input)
            predict_count = torch.sum(output, dim=1).round()

            gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
            for item in gaps:
                if abs(item) <= 1:
                    acc_test += 1
                
            testOBO += acc_test
            MAE_test = torch.div(torch.abs(predict_count - count), count + 1e-1).tolist()  # mae
            testMAE += MAE_test

    print("testMAE={0},testOBO={1}".format(round((torch.sum(torch.tensor(testMAE))/len(testMAE)).item(), 4), round(testOBO/len(testMAE), 4)))
    return round((torch.sum(torch.tensor(testMAE))/len(testMAE)).item(), 4), round(testOBO/len(testMAE), 4)

