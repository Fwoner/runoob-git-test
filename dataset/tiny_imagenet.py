import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.train = train
        
        if self.train:
            self.data_dir = os.path.join(root, 'train')
        else:
            self.data_dir = os.path.join(root, 'val')
            
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        if self.train:
            for idx, class_dir in enumerate(sorted(glob.glob(os.path.join(self.data_dir, '*')))):
                class_name = os.path.basename(class_dir)
                self.class_to_idx[class_name] = idx
                image_paths = glob.glob(os.path.join(class_dir, 'images', '*.JPEG'))
                self.image_paths.extend(image_paths)
                self.labels.extend([idx] * len(image_paths))
        else:
            with open(os.path.join(root, 'val', 'val_annotations.txt'), 'r') as f:
                for line in f:
                    image_name, class_name, _, _, _, _ = line.strip().split('\t')
                    image_path = os.path.join(self.data_dir, 'images', image_name)
                    if os.path.exists(image_path):
                        self.image_paths.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_tiny_imagenet_dataloader(batch_size=128, num_workers=8):
    # ����Ԥ����
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transforms_val = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # �������ݼ�
    train_dataset = TinyImageNet(root='./data/tiny-imagenet-200', 
                                train=True, 
                                transform=transforms_train)
    test_dataset = TinyImageNet(root='./data/tiny-imagenet-200', 
                               train=False, 
                               transform=transforms_val)

    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, 
                           batch_size=batch_size,
                           shuffle=False, 
                           num_workers=num_workers)
    
    return train_loader, test_loader