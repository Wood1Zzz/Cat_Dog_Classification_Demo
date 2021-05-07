class DatasetModeError(Exception):
    def __init__(self, mode):
        super(DatasetModeError, self).__init__()
        self.mode = mode

    def __str__(self):
        print("Dataset mode" + self.mode + "error and it must be train or test")


def get_labels(label_digital):
    label_map = ['cat', 'dog']
    return [label_map[int(i)] for i in label_digital]
