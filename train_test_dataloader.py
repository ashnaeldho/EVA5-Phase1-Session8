"""Train and Test data set downloader and allows access through PyTorch's Dataloader"""

import torch
import CONSTANTS as const
from torchtoolbox.transform import Cutout
from torchvision import datasets, transforms
from utility import get_config_details, check_gpu_availability

config = get_config_details()


def define_train_test_transformers(dataset_name=None, *, session):
    if dataset_name and "mnist" in dataset_name.lower():

        # Train data transformation

        train_transforms = transforms.Compose([transforms.RandomRotation((-9.0, 9.0), fill=(1,)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                                               ])

        # Test transform

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    elif "s8" in session.lower() or "cifar10" in dataset_name.lower():
        mean = tuple([125.30691805 / 255, 122.95039414 / 255, 113.86538318 / 255])
        standard_deviation = tuple([62.99321928 / 255, 62.08870764 / 255, 66.70489964 / 255])
        # Train Phase transformations
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=1),
            Cutout(p=0.25, scale=(0.02, 0.10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=standard_deviation)

        ])

        # Test Phase transformations
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=standard_deviation)
        ])


    else:
        train_transforms = transforms.Compose([transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.ToTensor()])

    return train_transforms, test_transforms


def download_data(*, dataset_name, train_transforms, test_transforms):
    """Downloads and returns train test dataset after the mandatory train and test transforms respectively"""

    if not (isinstance(train_transforms, transforms.Compose) and isinstance(test_transforms, transforms.Compose)):
        raise Exception("\n The train and test transformers passed are invalid.")

    if dataset_name:
        if "mnist" in dataset_name.lower():

            train = datasets.MNIST(root='../data', train=True, download=True, transform=train_transforms)  # Train data

            test = datasets.MNIST(root='../data', train=False, download=True, transform=test_transforms)  # Test data

        elif "cifar10" in dataset_name.lower():
            train = datasets.CIFAR10(root='../data', train=True,
                                     download=True, transform=train_transforms)

            test = datasets.CIFAR10(root='../data', train=False,
                                    download=True, transform=test_transforms)

    return train, test


dataloader_args = dict(shuffle=bool(config[const.MODEL_CONFIG][const.SHUFFLE]),
                       batch_size=int(config[const.MODEL_CONFIG][const.BATCH_SIZE]),
                       num_workers=int(config[const.MODEL_CONFIG][const.WORKERS]),
                       pin_memory=bool(config[const.MODEL_CONFIG][const.PIN_MEMORY])) if check_gpu_availability() \
    else dict(shuffle=bool(config[const.MODEL_CONFIG][const.SHUFFLE]),
              batch_size=int(config[const.MODEL_CONFIG][const.BATCH_SIZE]))


def get_train_test_dataloaders(*, train_data, test_data, data_loader_args):
    """Generates and returns data loaders for train and test data sets"""

    '''train dataloader'''
    train_loader = torch.utils.data.DataLoader(dataset=train_data, **data_loader_args)

    '''test dataloader'''
    test_loader = torch.utils.data.DataLoader(dataset=test_data, **data_loader_args)

    return train_loader, test_loader
