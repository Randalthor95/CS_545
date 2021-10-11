import torch
import os
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class FinalProjectEEGDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the EEG data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.eeg_data = pd.read_csv(os.path.join(root_dir, csv_file), delimiter=',', usecols=range(15))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        open_or_closed = self.eeg_data.iloc[idx, 14]
        channel_data = self.eeg_data.iloc[0, 0:14]
        channel_data = np.array([channel_data])
        channel_data = channel_data.astype('float').reshape(-1, 14)
        sample = {'open_or_closed': open_or_closed, 'channel_data': channel_data}

        if self.transform:
            sample = self.transform(sample)

        return sample
