from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18
import config


def get_dataset_model():
    """
        Get the training and testing data loaders and initialize the model.
        Returns:
            tuple: Train DataLoader, Test DataLoader, and the model.
    """
    # Define data transformations for training and testing datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Initialize the ResNet-18 model with modified first convolution layer for CIFAR10
    model = resnet18(num_classes=config.NUM_CLASS)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader, model
