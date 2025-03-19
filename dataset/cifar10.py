import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
def get_cifar10_dataloader(batch_size=128, num_workers=8):
    # Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
    transforms_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Loading the CIFAR-10 dataset:
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

    """
    创建训练数据加载器，用于批量加载并预处理训练数据
    参数说明：
        train_dataset (Dataset): 训练数据集对象
        batch_size (int): 每个批次的样本数量，控制内存占用和梯度更新频率
        shuffle (bool): 是否打乱数据顺序，训练时建议启用以避免样本顺序偏差
        num_workers (int): 数据加载并行进程数，提升IO密集型操作效率
    """
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader