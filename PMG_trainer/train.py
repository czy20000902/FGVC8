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



def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None, backbone="resnet50"):
    
    print("#"*20)
    print(backbone)
    print("#"*20)
    
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    
    input_size = 224
    if backbone == "efficientnet_b1":
        input_size = 240
    if backbone == "efficientnet_b3b":
        input_size = 300
    if backbone == "efficientnet_b4b":
        input_size = 380
    
    transform_train = transforms.Compose([
        transforms.Scale((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root='/userhome/21inat/l_train/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model
    # if resume:
    #    net = torch.load(model_path)
    # else:
    net = load_model(model_name=backbone, pretrain=True, require_grad=True)
    netp = net

    # GPU
    device = torch.device("cuda:0")
    net.to(device)
    # cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier.parameters(), 'lr': 0.002},
        {'params': net.features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)
    save_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Step 1
            optimizer.zero_grad()
            output_1, _ = netp(inputs)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            train_loss += loss1.item()
            
             #  training log
            _, predicted = torch.max(output_1.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 10 == 0:
                print(
                    '%s Step: %d | Loss1: %.3f |  Acc: %.3f%% (%d/%d)' % (
                    backbone,
                    batch_idx, train_loss / (batch_idx + 1), 
                    100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)

        if epoch < 5 or epoch >= 80 or True:
            val_acc, val_acc_com, val_loss = test(net, CELoss, 3, input_size)
            if val_acc_com > max_val_acc:
                max_val_acc = val_acc_com
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc, val_acc_com, val_loss))
            print("Max Acc: %.4f"%(max_val_acc))
            
            bs = "fgcv_" + backbone + "_" + save_time
            if not os.path.exists("checkpoints/%s"%(bs)):
                os.mkdir("checkpoints/%s"%(bs))
            if epoch % 9 == 0:
                torch.save(net.state_dict(), "checkpoints/%s/epoch_%d.pt"%(bs, epoch))



train(nb_epoch=200,             # number of epoch
         batch_size=32,         # batch size
         store_name='bird',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='',
         backbone=sys.argv[1]
     )
