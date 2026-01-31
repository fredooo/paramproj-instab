from torchvision import datasets

from .utils import load_torchvision_split


def load_mnist_split(seed=777):
    return load_torchvision_split(datasets.MNIST, seed)
