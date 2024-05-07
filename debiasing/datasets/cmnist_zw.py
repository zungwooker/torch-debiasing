from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from glob import glob
import os
import torch
    
    
class CMNIST_ZWDataset(Dataset):
    def __init__(self, split, conflict_ratio):
        super().__init__()
        root = '/mnt/sdc/zungwooker/workspace/debiasing/data/dataset/cmnist-zw'
        self.transform = T.ToTensor()

        if conflict_ratio != 'unbiased': 
            conflict_ratio = str(conflict_ratio)+'pct'

        if split=='train':
            self.data = glob(os.path.join(root, f'{conflict_ratio}', 'train', '*', '*'))
        elif split=='valid':
            self.data = glob(os.path.join(root, f'{conflict_ratio}', 'valid', '*', '*'))
        elif split=='test':
            self.data = glob(os.path.join(root, 'cmnist-zw', 'test', '*', '*'))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = torch.tensor(int(self.data[index].split('_')[-2]))
        bias = torch.tensor(int(self.data[index].split('_')[-1].split('.')[0]))
        image = Image.open(self.data[index]).convert('RGB')
        image = self.transform(image)
        
        return image, label, bias, self.data[index]