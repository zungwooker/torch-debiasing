from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from glob import glob
import os
import torch


class DogCatDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(DogCatDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image_path_list = image_path_list

        if split == "train":
            self.align = glob(os.path.join(root, "align", "*", "*"))
            self.conflict = glob(os.path.join(root, "conflict", "*", "*"))
            self.data = self.align + self.conflict
        elif split == "valid":
            self.data = glob(os.path.join(root, split, "*"))
        elif split == "test":
            self.data = glob(os.path.join(root, "../test", "*", "*"))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)  
        return image, attr, self.data[index]