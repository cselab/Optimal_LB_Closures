import torch
import torchvision
import os
import torch.utils.data
import copy

from lib.datasets.initial_conditions import MNIST, FashionMNIST


def get_train_val_test_initial_conditions_dataset(dataset_name: str,
                                                  img_size: int):
    _train_val_split = [0.9, 0.1]

    if dataset_name == 'mnist':
        mnist_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))}/data"
        _transforms_list = [torchvision.transforms.ToTensor(),
                            torchvision.transforms.RandomRotation(degrees=90, fill=0),
                            torchvision.transforms.Resize(size=(img_size, img_size), antialias=True)]
        _transforms = torchvision.transforms.Compose(_transforms_list)
        _full_dataset = MNIST(train=True,
                              root=mnist_dir,
                              download=True,
                              transform=_transforms)
        train_data, val_data = torch.utils.data.random_split(_full_dataset, _train_val_split)
        _test_transforms_list = copy.deepcopy(_transforms_list)
        #_test_transforms_list.append(torchvision.transforms.RandomInvert())
        _transforms = torchvision.transforms.Compose(_test_transforms_list)
        test_data = MNIST(train=False,
                          root=mnist_dir,
                          download=True,
                          transform=_transforms)
        return train_data, val_data, test_data

    if dataset_name == 'fashion':
        mnist_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))}/data"
        _transforms = [torchvision.transforms.ToTensor(),
                       torchvision.transforms.Resize(size=(img_size, img_size), antialias=True)]
        _transforms = torchvision.transforms.Compose(_transforms)
        _full_dataset = FashionMNIST(train=True,
                                     root=mnist_dir,
                                     download=True,
                                     transform=_transforms)
        train_data, val_data = torch.utils.data.random_split(_full_dataset, _train_val_split)
        test_data = FashionMNIST(train=False,
                                 root=mnist_dir,
                                 download=True,
                                 transform=_transforms)
        return train_data, val_data, test_data

    else:
        raise ValueError(f"Dataset {dataset_name} not implemented!")
