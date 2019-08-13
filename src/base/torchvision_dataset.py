from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, train_test_root: str, apply_model_root: str):
        super().__init__(train_test_root, apply_model_root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        apply_loader = DataLoader(dataset=self.apply_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader, apply_loader
