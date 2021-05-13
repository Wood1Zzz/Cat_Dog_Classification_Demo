import os
import sys
import cv2
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from utils import DatasetModeError
from config import ONE_HOT
from dataprocess import Transform_train, Transform_test
from torch.utils.data import DataLoader as Data
from config import *

class CatVsDogDataset(Dataset):
    def __init__(self, file_path, mode="train", one_hot=ONE_HOT):
        self.file_path = file_path
        self.mode = mode
        self.one_hot = one_hot

        if self.mode == "train":
            self.file_name = os.listdir(file_path)[2500: 22500]
        elif self.mode == "test":
            self.file_name = os.listdir(file_path)[:2500] + os.listdir(file_path)[22500:]
            # print(type(self.file_name))
            # .extend(os.listdir(file_path)[22500:])
            # print(self.file_name) 
        else:
            raise DatasetModeError(self.mode)

        labels = []
        for i in self.file_name:
            if i.split('.')[0] == 'cat':
                if one_hot:
                    labels.append([1, 0])
                else:
                    labels.append(0)
            elif i.split('.')[0] == 'dog':
                if one_hot:
                    labels.append([0, 1])
                else:
                    labels.append(1)
        self.labels = labels

    def __getitem__(self, index):
        if sys.platform.startswith('win'):
            img = Image.open(self.file_path + '\\' + self.file_name[index])
        elif sys.platform.startswith('linux'):
            img = Image.open(self.file_path + '/' + self.file_name[index])
        if self.mode == "train":
            data = Transform_train(img)
            
        elif self.mode == "test":
            data = Transform_test(img)
            
        img.close()
        
        if torch.cuda.is_available():
            if self.one_hot:
                return data.cuda(), torch.LongTensor([self.labels[index]]).cuda()
            else:
                return data.cuda(), torch.FloatTensor([self.labels[index]]).cuda()
        else:
            if self.one_hot:
                return data, torch.LongTensor([self.labels[index]])
            else:
                return data, torch.FloatTensor([self.labels[index]])

    def __len__(self):
        return len(self.file_name)

# cat_dog_dataset_test = CatVsDogDataset('G:\dogs-vs-cats-redux-kernels-edition\\train', mode="test", one_hot=False)
# test_loader = Data(cat_dog_dataset_test, batch_size=8)
# print(len(test_loader))