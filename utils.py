import torch
import matplotlib.pyplot as plt
from torch._C import Size
from dataprocess import crop_size
# import matplotlib.image as

class DatasetModeError(Exception):
    def __init__(self, mode):
        super(DatasetModeError, self).__init__()
        self.mode = mode

    def __str__(self):
        print("Dataset mode" + self.mode + "error and it must be train or test")


def get_labels(label_digital):
    label_map = ['cat', 'dog']
    return [label_map[int(i)] for i in label_digital]

def evaluate_accuracy(test_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in test_iter:
        acc_sum += (torch.round(net(x)) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def second2clock(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return h, m, s

def rgb2gray(img):
    # img = img.convert("RGB")
    r, g, b = img[0, :, :], img[1, :, :], img[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray

def show_result(net, x, y, num=10):
    true_labels = get_labels(y.cpu().detach().numpy())
    predict_labels = get_labels(torch.round(net(x)).cpu().detach().numpy())
    titles = ["label: " + true + '\n' + "predict: " + pred for true, pred in zip(true_labels, predict_labels)]
    
    _, axs = plt.subplots(1, len(x), figsize=crop_size)

    for ax, img, lbl in zip(axs, x, titles):
        ax.imshow(rgb2gray(img.cpu().detach().numpy()))
        # ax.imshow(img.cpu().detach().resize((224, 224, 3)).numpy())
        ax.set_title(lbl)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    # plt.imshow(x[0: num-1], titles[0: num-1])
    plt.show()
