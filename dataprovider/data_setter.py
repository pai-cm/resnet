from base.base_data_setter import BaseDataSetter
from torchvision.datasets import CIFAR10


class CIFAR10DataSetter(BaseDataSetter):
    def __init__(self, *args, **kwargs):
        self.cifar10 = CIFAR10(*args, **kwargs)

    def __getitem__(self, item):
        return self.cifar10.__getitem__(item)

    def __len__(self):
        return len(self.cifar10)
