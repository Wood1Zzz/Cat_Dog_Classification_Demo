import sys

# Config
BATCH_SIZE = 30
TRAIN_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\train'
VALID_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\test'
DATASET_PATH = 'G:\dogs-vs-cats-redux-kernels-edition'
EPOCH = 20
RECORD_EPOCH = 5
LR = 0.0001
NET = '16'
ONE_HOT = False
DEVICE = "my"

# Dataset path for myself
if DEVICE == "my":
    if sys.platform.startswith('win'):
        TRAIN_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\train'
        VALID_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\test'
        DATASET_PATH = 'G:\dogs-vs-cats-redux-kernels-edition'
    elif sys.platform.startswith('linux'):
        TRAIN_PATH = '/home/danzer/PycharmProject/Dataset/dogs-vs-cats-redux-kernels-edition/train'
        VALID_PATH = '/home/danzer/PycharmProject/Dataset/dogs-vs-cats-redux-kernels-edition/test'
        DATASET_PATH = '/home/danzer/PycharmProject/Dataset/dogs-vs-cats-redux-kernels-edition'
elif DEVICE == "kaggle":
    TRAIN_PATH = '../input/dogs-vs-cats/train/train'
    VALID_PATH = '../input/dogs-vs-cats/test/test'
    DATASET_PATH = '../input/dogs-vs-cats'
elif DEVICE == "colab":
    pass