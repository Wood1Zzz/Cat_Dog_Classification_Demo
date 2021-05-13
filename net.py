# import torch
from torch import nn
from config import ONE_HOT


class VGG(nn.Module):
    def __init__(self, name="11", one_hot=ONE_HOT):
        super(VGG, self).__init__()
        self.name = "VGG" + name
        self.conv = nn.Sequential()
        if one_hot:
            self.output = 2
        else:
            self.output = 1

        i = 1
        p = 1
        # construct conv layer based on input module name
        # different VGG Net Config
        # input img size is casual, and the input size of fc depend on maxpooling layer
        self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
        i += 1

        if name in ["13", "16-1", "16", "19"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1
        elif name in ["11-LRN"]:
            self.conv.add_module('LRN', nn.LocalResponseNorm(size=2))
        self.conv.add_module('MaxPooling-{0}'.format(p), nn.MaxPool2d(kernel_size=2, stride=2))
        p += 1

        self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
        i += 1
        if name in ["13", "16-1", "16", "19"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1
        self.conv.add_module('MaxPooling-{0}'.format(p), nn.MaxPool2d(kernel_size=2, stride=2))
        p += 1

        self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
        i += 1
        self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
        i += 1
        if name in ["16", "19"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1
        elif name in ["16-1"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1

        if name in ["19"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1

        self.conv.add_module('MaxPooling-{0}'.format(p), nn.MaxPool2d(kernel_size=2, stride=2))
        p += 1

        self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('ReLU-{0}', nn.ReLU())
        i += 1
        self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
        i += 1
        if name in ["16", "19"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1
        elif name in ["16-1"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1
        if name in ["19"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1
        self.conv.add_module('MaxPooling-{0}'.format(p), nn.MaxPool2d(kernel_size=2, stride=2))
        p += 1

        self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('ReLU-{0}', nn.ReLU())
        i += 1
        self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
        i += 1
        if name in ["16", "19"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1
        elif name in ["16-1"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1
        if name in ["19"]:
            self.conv.add_module('conv-{0}'.format(i), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
            self.conv.add_module('ReLU-{0}'.format(i), nn.ReLU())
            i += 1
        self.conv.add_module('MaxPooling-{0}'.format(p), nn.MaxPool2d(kernel_size=2, stride=2))
        p += 1

        # define fc layer
        if one_hot:
            self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.output),
            nn.Softmax(),
            nn.Sigmoid()
        )
        else:
            self.fc = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 1000),
                nn.ReLU(),
                nn.Linear(1000, self.output),
                nn.Sigmoid()
            )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(x.shape[0], -1)
        output = self.fc(output)

        return output


