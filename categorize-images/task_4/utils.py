import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """ A torch Dataset class for hw4 """
    def __init__(self, root_dir, mode, transform=None):
        """
        Args:
            root_dir - root dir for all images
            transform - optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.out_list = []

    def __getitem__(self, idx):
        train_path = os.path.join(self.root_dir, "train")
        test_path = os.path.join(self.root_dir, "test")
        img_dirs = [d for d in os.listdir(train_path) if not d.startswith('.')]
        labels = None

        if self.mode == 'train':
            # train_img_list = []
            train_img_names = [os.listdir(os.path.join(train_path, d)) for d in img_dirs if not d.startswith('.')]
            for i in range(len(img_dirs)):
                for name in train_img_names[i]:
                    if not name.startswith('.'):
                        self.out_list.append(os.path.join(train_path, img_dirs[i], name))
            labels = np.repeat(np.arange(15), 100)
        elif self.mode == 'test':        
            # test_img_list = []
            test_img_names = [os.listdir(os.path.join(test_path, d)) for d in img_dirs if not d.startswith('.')]
            for i in range(len(img_dirs)):
                for name in test_img_names[i]:
                    if not name.startswith('.'):
                        self.out_list.append(os.path.join(test_path, img_dirs[i], name))
            labels = np.repeat(np.arange(15), 10)
        else:
            raise ValueError("mode argument has to be 'train' or 'test'.")
        
    
        img = Image.open(self.out_list[idx])
        label = labels[idx]
        # sample = {'image': img, 'label':label}

        if self.transform:
            # sample['image'] = self.transform(sample['image'])
            img = self.transform(img)

        return (img, label)
    
    def __len__(self):
        return len(self.out_list)
