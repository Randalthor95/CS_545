from collections import OrderedDict

import torch
import os
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
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
        print(self.eeg_data.dtypes)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        open_or_closed = self.eeg_data.iloc[idx, 14].astype('float')
        channel_data = self.eeg_data.iloc[idx, 0:14]
        channel_data = np.array([channel_data])
        channel_data = channel_data.astype('float').reshape(-1, 14)
        sample = {'open_or_closed': open_or_closed, 'channel_data': channel_data}

        if self.transform:
            sample = self.transform(sample)

        return channel_data, open_or_closed


class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_hidden_units_by_layers, num_outputs):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        od = OrderedDict([])
        if num_hidden_units_by_layers:
            for i in range(len(num_hidden_units_by_layers)):
                if i == 0:
                    print('i == 0')
                    od['Linear0'] = nn.Linear(num_inputs, num_hidden_units_by_layers[0])
                    od['ReLU0'] = nn.ReLU()
                else:
                    od['Linear' + str(i)] = nn.Linear(num_hidden_units_by_layers[i - 1], num_hidden_units_by_layers[i])
                    od['ReLU' + str(i)] = nn.ReLU()
            od['Linear' + str(len(num_hidden_units_by_layers))] = \
                nn.Linear(num_hidden_units_by_layers[len(num_hidden_units_by_layers) - 1], num_outputs)
        else:
            od['Linear0'] = nn.Linear(num_inputs, num_outputs)

        self.linear_relu_stack = nn.Sequential(od)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
