import torch
from torch.utils.data import DataLoader as Data
from torch import nn
from torch import optim
import dataloader
from net import VGG
import time
from tqdm import tqdm
from config import *
import os
from utils import evaluate_accuracy, second2clock, show_result
import argparse


def str2bool(arg):
    return arg.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Training config')

parser.add_argument('--device', default=DEVICE, type=str, help='Use my device or colab or kaggle to train')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Batch size for trainning')
parser.add_argument('--epoch', default=EPOCH, type=int, help='Trainning epoch')
parser.add_argument('--lr', default=LR, type=float, help='Learning rate')
parser.add_argument('--record_epoch', default=RECORD_EPOCH, type=int, help='Record pth files how many epochs')
parser.add_argument('--vgg_net', default=NET, type=str, help='Choose net to train the model, you can choose 11, 11-LRN, 13, 16, 16-1, 19')
parser.add_argument('--one_hot', default=ONE_HOT, type=str2bool, help='Use one hot type to train or not')
parser.add_argument('--show_picture_num', default=SHOW_PIC_NUM, type=int, help='During test period show how many picture')
parser.add_argument('--dataset_path', default=DATASET_PATH, type=str, help='Dataset path, not include train or test path')

# 设置默认参数不改变，否则修改为输入参数
parser.set_defaults(keep_latest=False)

# 命令行解析
args = parser.parse_args()

DEVICE = args.device
BATCH_SIZE = args.batch_size
EPOCH = args.epoch
LR = args.lr
RECORD_EPOCH = args.record_epoch
NET = args.vgg_net
ONE_HOT = args.one_hot
SHOW_PIC_NUM = args.show_picture_num
DATASET_PATH = args.dataset_path

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

set_path(DEVICE)

def train(epoch=10, batch_size=10, dataset_path=None, one_hot=False):

    if one_hot:
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LR)
    else:
        loss_func = nn.BCELoss()
        optimizer = optim.RMSprop(net.parameters(), lr=LR, alpha=0.9)

    if dataset_path is not None and DEVICE is not "kaggle":
        if sys.platform.startswith('win'):
            TRAIN_PATH = dataset_path + '\\train'
            VALID_PATH = dataset_path + '\\test'
        elif sys.platform.startswith('linux'):
            TRAIN_PATH = dataset_path + '/train'
            VALID_PATH = dataset_path + '/test'
    elif DEVICE is "kaggle":
        TRAIN_PATH = '../input/dogs-vs-cats/train/train'
        VALID_PATH = '../input/dogs-vs-cats/test/test'
        # DATASET_PATH = '../input/dogs-vs-cats'
    else:
        raise ValueError("Dataset can not be None")

    cat_dog_dataset = dataloader.CatVsDogDataset(TRAIN_PATH, mode="train", one_hot=one_hot)
    # train_loader = Data(cat_dog_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = Data(cat_dog_dataset, batch_size=batch_size, shuffle=True)
    cat_dog_dataset_test = dataloader.CatVsDogDataset(TRAIN_PATH, mode="test", one_hot=one_hot)
    # test_loader = Data(cat_dog_dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = Data(cat_dog_dataset_test, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    print("Net: VGG%s, Total epoch: %d, batch_size: %d, LR: %f"%(NET, epoch, batch_size, LR))
    time.sleep(0.1)

    for epoch in range(epoch):
        print("\nEpoch: %d"%(epoch + 1))
        time.sleep(0.1)

        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0

        for batch, (x, y) in enumerate(tqdm(train_loader)):
            y_hat = net(x)
            # if batch_size > 1, use sum() to calculate per batch loss
            if one_hot:
                loss = loss_func(y_hat, y).sum()
            else:
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

            if one_hot:
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

        if (epoch+1) % RECORD_EPOCH == 0:
            valid_acc = evaluate_accuracy(test_loader, net)
            print('Epoch: {epoch}, Valid accuracy: {valid:.6f}%'.format(epoch=epoch+1, valid=valid_acc*100))

    end_time = time.time()
    h, m, s = second2clock(end_time - start_time)
    print("Total trainning time: " + "%d hours %02d mins %.2f seconds" % (h, m, s))
    start_time = time.time()
    valid_acc = evaluate_accuracy(test_loader, net)
    end_time = time.time()
    h, m, s = second2clock(end_time - start_time)
    print("Valid accuracy: {:.6f}".format(valid_acc*100) + "%, Eval time: " + "%d hours %02d mins %.2f seconds" % (h, m, s))

    test_img, test_label = iter(test_loader).__next__()
    show_result(net, test_img[0:SHOW_PIC_NUM], test_label[0:SHOW_PIC_NUM])
train(epoch=EPOCH, batch_size=BATCH_SIZE, dataset_path=DATASET_PATH, one_hot=ONE_HOT)
