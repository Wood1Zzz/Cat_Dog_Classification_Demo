import torch
import torchvision
from torchvision import transforms

class Data_process(object):
    def __init__(self):
        pass
    def data_process_train(self):
        TRANSFORM = transforms.Compose([
            transforms.Resize((256, 256)),
        ])