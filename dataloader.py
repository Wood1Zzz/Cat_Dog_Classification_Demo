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

# file_path = 'G:\dogs-vs-cats-redux-kernels-edition\\train'

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

Transform_train = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.RandomCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                      ])

Transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
                                     ])




class CatVsDogDataset(Dataset):
    def __init__(self, file_path, mode="train", one_hot=ONE_HOT):
        self.file_path = file_path
        self.mode = mode
        self.one_hot = one_hot

        if self.mode == "train":
            self.file_name = os.listdir(file_path)[0: 22500]
        elif self.mode == "test":
            self.file_name = os.listdir(file_path)[22500:]
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
        global Transform_train
        global Transform_test

        if sys.platform.startswith('win'):
            img = Image.open(self.file_path + '\\' + self.file_name[index])
        elif sys.platform.startswith('linux'):
            img = Image.open(self.file_path + '/' + self.file_name[index])
        if self.mode == "train":
            data = Transform_train(img)
        elif self.mode == "test":
            data = Transform_test(img)
        img.close()
        if self.one_hot:
            return data.cuda(), torch.LongTensor([self.labels[index]]).cuda()
        else:
            return data.cuda(), torch.FloatTensor([self.labels[index]]).cuda()

    def __len__(self):
        return len(self.file_name)


# class ValidDataset(Dataset):
#     def __init__(self, file_path):
#         super(ValidDataset, self).__init__()
#         self.file_path = file_path
#         self.file_name = os.listdir(file_path)
#
#     def __getitem__(self, index):
#         global Transform_test
#         img = Image.open(self.file_path + '\\' + self.file_name[index])
#         data = Transform_test(img)
#         img.close()
#         return data.cuda()
#
#     def __len__(self):
#         return len(self.file_name)

# cat_dog_dataset = CatVsDogDataset(file_path)
# dataloader = DataLoader(cat_dog_dataset, batch_size=2, shuffle=True, num_workers=0)
# for x, y in dataloader:
#     print(y.shape)

# img = cv2.imread(file_path + '\\' + file_name[0])
# cv2.imshow(file_name[0], img)
# cv2.waitKey(0)
