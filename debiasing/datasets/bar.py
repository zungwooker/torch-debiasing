from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from glob import glob
import os
import torch


class BARDataset(Dataset):
    def __init__(self, root, split, transform=None, percent=None):
        super(BARDataset, self).__init__()
        self.transform = transform
        self.percent = percent # 1pct, 5pct
        self.split = split

        self.train_align = glob(os.path.join(root,'train/align',"*/*"))
        self.train_conflict = glob(os.path.join(root,'train/conflict',f"{self.percent}/*/*"))
        self.valid = glob(os.path.join(root,'valid',"*/*"))
        self.test = glob(os.path.join(root,'test',"*/*"))

        if self.split=='train':
            self.data = self.train_align + self.train_conflict
        elif self.split=='valid':
            self.data = self.valid
        elif self.split=='test':
            self.data = self.test

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')
        image_path = self.data[index]

        
        if 'bar/train/conflict' in image_path:
            attr[1] = (attr[0] + 1) % 6 # assign non-related attribute to bias attribute
        elif 'bar/train/align' in image_path:
            attr[1] = attr[0]

        if self.transform is not None:
            image = self.transform(image)  
        return image, attr, (image_path, index)