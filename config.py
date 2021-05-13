import sys
import os

# Config
BATCH_SIZE = 10
TRAIN_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\train'
VALID_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\test'
DATASET_PATH = 'G:\dogs-vs-cats-redux-kernels-edition'
EPOCH = 1
RECORD_EPOCH = 5
LR = 0.0001
NET = '11'
ONE_HOT = False
SHOW_PIC_NUM = 10

# DEVICE CONFIG "my_device", 'colab', "kaggle", no recommand use "kaggle"
DEVICE = "colab"

# Special config or dataset path for different device
if DEVICE == "my_device":
    if sys.platform.startswith('win'):
        # TRAIN_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\train'
        # VALID_PATH = 'G:\dogs-vs-cats-redux-kernels-edition\\test'
        DATASET_PATH = 'G:\dogs-vs-cats-redux-kernels-edition'
        BATCH_SIZE = 50
    elif sys.platform.startswith('linux'):
        # TRAIN_PATH = '/home/danzer/PycharmProject/Dataset/dogs-vs-cats-redux-kernels-edition/train'
        # VALID_PATH = '/home/danzer/PycharmProject/Dataset/dogs-vs-cats-redux-kernels-edition/test'
        DATASET_PATH = '/home/danzer/PycharmProject/Dataset/dogs-vs-cats-redux-kernels-edition'
        BATCH_SIZE = 50
elif DEVICE == "kaggle":
    TRAIN_PATH = '../input/dogs-vs-cats/train/train'
    VALID_PATH = '../input/dogs-vs-cats/test/test'
    DATASET_PATH = '../input/dogs-vs-cats'
    BATCH_SIZE = 100
elif DEVICE == "colab":
    os.system(r'pip install -U -q kaggle')
    os.system(r'mkdir -p ~/.kaggle')
    os.system(r'echo \{\"username\":\"woodzzz\",\"key\":\"0ebbc1f4862b43cd57cf939fec48aac3\"}\ > ~/.kaggle/kaggle.json')
    os.system(r'chmod 600 ~/.kaggle/kaggle.json')

    if not os.path.exists('/content/cat_dog_dataset'):
        os.system(r'kaggle datasets download -d biaiscience/dogs-vs-cats -p /content')
        os.system(r'unzip /content/dogs-vs-cats.zip -d /content/cat_dog_dataset')
        os.system(r'mv /content/cat_dog_dataset/train/train/* /content/cat_dog_dataset/train')
        os.system(r'rm -rf /content/cat_dog_dataset/train/train')
        os.system(r'mv /content/cat_dog_dataset/test/test/* /content/cat_dog_dataset/test')
        os.system(r'rm -rf /content/cat_dog_dataset/test/test')
        print("Download and unzip finish!\n")
    else:
        print('do nothing finished\n')
    TRAIN_PATH = '/content/cat_dog_dataset/train'
    VALID_PATH = '/content/cat_dog_dataset/test'
    DATASET_PATH = '/content/cat_dog_dataset'
    BATCH_SIZE = 80