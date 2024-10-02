import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_train_valid_loader(data_dir,
                           batch_size,
                           download=True):
    cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
    cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.Resize((32,32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar_norm_mean, cifar_norm_std)])
    transform_valid = transforms.Compose([transforms.Resize((32,32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar_norm_mean, cifar_norm_std)])
    train_set = datasets.CIFAR10(root=data_dir, train=True, transform=transform_train, download=download)
    valid_set = datasets.CIFAR10(root=data_dir, train=False, transform=transform_valid, download=False)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def get_grey_valid_loader(data_dir, batch_size):
    cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
    cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
    transform_valid = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                            transforms.Resize((32,32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar_norm_mean, cifar_norm_std)])
    valid_set = datasets.CIFAR10(root=data_dir, train=False, transform=transform_valid, download=False)
    testloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    return testloader

def get_mask_valid_loader(data_dir, batch_size):
    cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
    cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
    transform_valid = transforms.Compose([
                                            transforms.Resize((32,32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(cifar_norm_mean, cifar_norm_std),
                                          transforms.RandomErasing(p=1, scale=(0.1, 0.1), ratio=(1, 1), value=0, inplace=False),])
    valid_set = datasets.CIFAR10(root=data_dir, train=False, transform=transform_valid, download=False)
    testloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    return testloader