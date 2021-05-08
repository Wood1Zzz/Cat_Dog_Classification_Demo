import sys

# Config
BATCH_SIZE = 50
TRAIN_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\train'
VALID_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\test'
DATASET_PATH = 'G:\dogs-vs-cats-redux-kernels-edition'
EPOCH = 20
RECORD_EPOCH = 5
LR = 0.0001
NET = '11'
ONE_HOT = False

# Dataset path for myself
if sys.platform.startswith('win'):
    TRAIN_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\train'
    VALID_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\test'
    DATASET_PATH = 'G:\dogs-vs-cats-redux-kernels-edition'
elif sys.platform.startswith('linux'):
    TRAIN_PATH = '/home/danzer/PycharmProject/Dataset/dogs-vs-cats-redux-kernels-edition/train'
    VALID_PATH = '/home/danzer/PycharmProject/Dataset/dogs-vs-cats-redux-kernels-edition/test'
    DATASET_PATH = '/home/danzer/PycharmProject/Dataset/dogs-vs-cats-redux-kernels-edition'
