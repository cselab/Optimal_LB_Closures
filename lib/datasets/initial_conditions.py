import torchvision.datasets

class MNIST(torchvision.datasets.MNIST):
    """
    Adapted MNIST dataset to drop label of image
    """
    def __getitem__(self, item):
        img, target = super().__getitem__(item)
        return img


class FashionMNIST(torchvision.datasets.FashionMNIST):
    """
    Adapted FashionMNIST dataset to drop label of image
    """
    def __getitem__(self, item):
        img, target = super().__getitem__(item)
        return img
