import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

class galDataset(TorchvisionDataset):

    def __init__(self, train_test_root: str, apply_model_root: str, normal_class=3):
        super().__init__(train_test_root, apply_model_root)

        self.n_classes = 4  # 3: normal, 0,1,2: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = [0,1,2]
        self.outlier_classes.remove(normal_class)

        # Preprocessing: GCN (with L2 norm)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l2'))])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MygalDat(root=self.train_test_root, transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.target_ten.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MygalDat(root=self.train_test_root, transform=transform, target_transform=target_transform)

        # Unlabelled data
        self.apply_set = MygalDat(root=self.apply_model_root, transform=transform,  target_transform=None) # labels are NaN


class MygalDat(Dataset): #images have labels

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        
        # Read the csv file
        self.data_info = pd.read_csv(root)
        
        # First column contains the image paths
        self.image_arr = np.array(self.data_info.iloc[:, 0])
        
        # Second column is the targets (labels)
        self.target_arr = np.array(self.data_info.iloc[:, 1])
        self.target_ten = torch.from_numpy(self.target_arr)
        
        # Calculate len        
        self.data_len = len(self.data_info.index)
        
        #initializing transforms        
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        # Get image name
        image_name = self.image_arr[index]
        # Open image and convert to greyscale
        img = Image.open(image_name).convert('L')

        # Get target (label) of the image
        target = torch.from_numpy(np.array(self.target_arr[index]))

        # Transform image
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)        

        return img, target, index

    def __len__(self):
        return self.data_len

def load_dataset(train_test_data_path, apply_model_data_path, normal_class):
    """Loads the dataset."""
    dataset = galDataset(train_test_root=train_test_data_path, apply_model_root=apply_model_data_path, normal_class=normal_class)

    return dataset

