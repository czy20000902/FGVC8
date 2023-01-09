# coding:utf-8

from __future__ import print_function
import os
from PIL import Image
import time
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys

from utils import *



def test(test_dir, model_path=None, backbone="resnet101", outputdir=None):
    
    # 根据模型判断输入尺寸
    input_size = 224
    if backbone == "efficientnet_b1":
        input_size = 240
    if backbone == "efficientnet_b3b":
        input_size = 300
    if backbone == "efficientnet_b4b":
        input_size = 380

    # 获取模型参数
    net = load_model(model_name=backbone, pretrain=True, require_grad=True)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0")

    # 构建数据集
    transform_test = transforms.Compose([
        transforms.Scale((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    class_res = open(outputdir + "/class_result.txt", "w+")
    feature_res = open(outputdir + "/feature_result.txt", "w+")

    # 遍历数据集
    for batch_idx, (inputs, targets) in enumerate(testloader):
        
        if use_cuda:
            inputs = inputs.to(device)
        inputs = Variable(inputs, volatile=True)

        outputs_com, features = net(inputs)
        outputs_com = outputs_com.data.cpu().numpy()
        outputs_com = outputs_com.tolist()
        features = features.data.cpu().numpy()
        features = features.tolist()


        imgname = testset.imgs[batch_idx]

        class_res_line = "%s "%(imgname)
        feature_res_line = "%s "%(imgname)

        for out in outputs_com:
            class_res_line = "%s %.6f"%(class_res_line, out)

        for fea in features:
            feature_res_line =  "%s %.6f"%(feature_res_line, out)

        class_res.write("%s\n"%class_res_line)
        feature_res.write("%s\n"%feature_res_line)


    
if __name__ == "__main__":

    testdir = ""
    model_path = ""
    backbone = ""
    outputdir = ""

    test(testdir,model_path,backbone,outputdir)
