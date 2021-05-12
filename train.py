import torch
from torch.utils.data import DataLoader as Data
from torch import nn
from torch import optim
import dataloader
from net import VGG
import time
from tqdm import tqdm
# import sys
from config import *
import os
from utils import evaluate_accuracy


# cat_dog_dataset = dataloader.CatVsDogDataset(TRAIN_PATH)
# dataloader = Data(cat_dog_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


if torch.cuda.is_available():
    net = VGG(NET).cuda()
else:
    net = VGG(NET)

params = net.parameters()

if ONE_HOT:
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR)
else:
    loss_func = nn.BCELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=LR, alpha=0.9)

def train():
    global TRAIN_PATH
    global EPOCH
    global BATCH_SIZE
    global DATASET_PATH
    global ONE_HOT

    if ONE_HOT:
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LR)
    else:
        loss_func = nn.BCELoss()
        optimizer = optim.RMSprop(net.parameters(), lr=LR, alpha=0.9)

    if DATASET_PATH is not None:
        if sys.platform.startswith('win'):
            TRAIN_PATH = DATASET_PATH + '\\train'
            TEST_PATH = DATASET_PATH + '\\test'
        elif sys.platform.startswith('linux'):
            TRAIN_PATH = DATASET_PATH + '/train'
            TEST_PATH = DATASET_PATH + '/test'
    else:
        raise ValueError("Dataset can not be None")

    cat_dog_dataset = dataloader.CatVsDogDataset(TRAIN_PATH, mode="train", one_hot=ONE_HOT)
    train_loader = Data(cat_dog_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    cat_dog_dataset_test = dataloader.CatVsDogDataset(TRAIN_PATH, mode="test")
    test_loader = Data(cat_dog_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # valid_dataset = dataloader.ValidDataset(VALID_PATH)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    start_time = time.time()
    print("Net: VGG%s, Total epoch: %d, BATCH_SIZE: %d, LR: %f"%(NET, EPOCH, BATCH_SIZE, LR))
    time.sleep(0.1)

    for epoch in range(EPOCH):
        print("Epoch:%d"%(epoch + 1))
        time.sleep(0.1)

        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0

        for batch, (x, y) in enumerate(tqdm(train_loader)):
            y_hat = net(x)
            # if batch_size > 1, use sum() to calculate per batch loss
            loss = loss_func(y_hat, y)

            # print("\t\tBatch #{0}/{1}".format(batch+1, len(train_loader)) + "Loss = %.6f"%float(loss))

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            loss.backward()
            if optimizer is None:
                optimizer = optim.SGD(net.parameters(), lr=globals(LR))
                optimizer.step()
            else:
                optimizer.step()

            # convert tensor data type to float data type
            # train_loss_sum += loss.item()
            # train_acc_sum += (y_hat == y).sum().item()

            if ONE_HOT:
                train_loss_sum += loss_func(y_hat, y).sum().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            else:
                train_loss_sum += loss.item()
                train_acc_sum += (torch.round(y_hat) == y).float().mean().item()
            
            # print(train_loss_sum)
            # print(train_acc_sum)
            # train_loss_sum += float(loss_func(y_hat, y))
            
        print('Epoch: {epoch}, Loss:{loss}, Accuracy:{accuracy}, Average_loss:{average_loss}, Average_accuracy:{average_accuracy}%'.\
            format(epoch=epoch+1, loss=float('%.6f' % train_loss_sum), accuracy=float('%.6f' % train_acc_sum), \
                average_loss=float('%.6f' %(train_loss_sum/(batch+1))), \
                    average_accuracy=float('%.6f' % (train_acc_sum/(batch+1)*100))))
    end_time = time.time()
    print("Total trainning time:{:.2f}s".format(end_time-start_time))
    start_time = time.time()
    valid_acc = evaluate_accuracy(test_loader, net)
    end_time = time.time()
    print("Valid accuracy {:.6f}%, Eval time {:.2f}s".format(valid_acc*100, end_time-start_time))

train()
