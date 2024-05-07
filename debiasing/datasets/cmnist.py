from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from glob import glob
import os
import torch
    
    
class CMNISTDataset(Dataset):
    def __init__(self, root, split, conflict_ratio, transform=None, selected_label=None):
        super().__init__()
        self.transform = transform
        self.root = root

        if conflict_ratio != 'unbiased': conflict_ratio = str(conflict_ratio)+'pct'

        if split=='train':
            self.align = glob(os.path.join(root, 'cmnist', f'{conflict_ratio}', 'align', '*', '*'))
            self.conflict = glob(os.path.join(root, 'cmnist', f'{conflict_ratio}', 'conflict', '*', '*'))
            self.data = self.align + self.conflict
        elif split=='valid':
            if selected_label is None:
                self.data = glob(os.path.join(root, 'cmnist', f'{conflict_ratio}', 'valid', '*'))
            else:
                self.data = glob(os.path.join(root, 'cmnist', f'{conflict_ratio}', 'valid', f'*_{selected_label}_*'))
        elif split=='test':
            if selected_label is None:
                self.data = glob(os.path.join(root, 'cmnist', 'test', '*', '*'))
            else:
                self.data = glob(os.path.join(root, 'cmnist', 'test', f'{selected_label}', '*'))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, bias = torch.LongTensor([int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, self.data[index], bias