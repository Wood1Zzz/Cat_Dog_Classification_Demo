import torch

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