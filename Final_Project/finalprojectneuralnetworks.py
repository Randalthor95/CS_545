from collections import OrderedDict

import torch
import os
from torch import nn
import torch.nn.functional as F
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
        self.labels = pd.read_csv(os.path.join(root_dir, csv_file), delimiter=',', nrows=1)
        self.eeg_data = pd.read_csv(os.path.join(root_dir, csv_file), delimiter=',', usecols=range(self.labels.size))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        open_or_closed = self.eeg_data.iloc[idx, self.labels.size - 1].astype('float')
        channel_data = self.eeg_data.iloc[idx, 0:self.labels.size - 1]
        channel_data = np.array([channel_data])
        channel_data = channel_data.astype('float').reshape(-1, self.labels.size - 1)
        sample = {'open_or_closed': open_or_closed, 'channel_data': channel_data}

        if self.transform:
            sample = self.transform(sample)

        return channel_data, open_or_closed


class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_hidden_units_by_layers, num_outputs, activation_function='relu'):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        od = OrderedDict([])
        if num_hidden_units_by_layers:
            for i in range(len(num_hidden_units_by_layers)):
                if i == 0:
                    od['Linear0'] = nn.Linear(num_inputs, num_hidden_units_by_layers[0])
                    if activation_function == 'sigmoid':
                        od['Sigmoid0'] = nn.Sigmoid()
                    elif activation_function == 'tanh':
                        od['Tanh0'] = nn.Tanh()
                    else:
                        od['ReLU0'] = nn.ReLU()

                else:
                    od['Linear' + str(i)] = nn.Linear(num_hidden_units_by_layers[i - 1], num_hidden_units_by_layers[i])

                    if activation_function == 'sigmoid':
                        od['Sigmoid' + str(i)] = nn.Sigmoid()
                    elif activation_function == 'tanh':
                        od['Tanh' + str(i)] = nn.Tanh()
                    elif activation_function == 'softplus':
                        od['Softplus' + str(i)] = nn.Softplus()
                    else:
                        od['ReLU' + str(i)] = nn.ReLU()
            od['Linear' + str(len(num_hidden_units_by_layers))] = \
                nn.Linear(num_hidden_units_by_layers[len(num_hidden_units_by_layers) - 1], num_outputs)

        else:
            od['Linear0'] = nn.Linear(num_inputs, num_outputs)

        self.linear_stack = nn.Sequential(od)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


class CNN(nn.Module):
    def __init__(self, num_channels, num_inputs, num_hiddens_per_conv_layer, num_hiddens_per_fc_layer, num_outputs,
                 kernel_size_per_conv_layer, stride_per_conv_layer, activation_function="tanh"):
        super(CNN, self).__init__()
        n_conv_layers = len(num_hiddens_per_conv_layer)
        self.num_inputs = num_inputs
        if (
                len(kernel_size_per_conv_layer) != n_conv_layers
                or len(stride_per_conv_layer) != n_conv_layers
        ):
            raise Exception(
                "The lengths of num_hiddens_per_conv_layer, patch_size_per_conv_layer, and stride_per_conv_layer must "
                "be equal. "
            )

        if activation_function == "tanh":
            self.activation_function = torch.tanh
        else:
            self.activation_function = torch.relu

        # Create all convolutional layers
        # First argument to first Conv2d is number of channels for each pixel.
        # Just 1 for our grayscale images.
        n_in = 1
        input_w = num_inputs  # original input image height (=width because image assumed square)
        self.conv_layers = torch.nn.ModuleList()

        for nh, patch_size, stride in zip(
                num_hiddens_per_conv_layer, kernel_size_per_conv_layer, stride_per_conv_layer
        ):
            self.conv_layers.append(
                torch.nn.Conv1d(n_in, nh, kernel_size=patch_size, stride=stride)
            )
            conv_layer_output_hw = (input_w - patch_size) // stride + 1
            input_w = conv_layer_output_hw  # for next trip through this loop
            n_in = nh

        # Create all fully connected layers.  First must determine number of inputs to first
        # fully-connected layer that results from flattening the images coming out of the last
        # convolutional layer.
        n_in = input_w * n_in  # n_hiddens_per_fc_layer[0]
        self.fc_layers = torch.nn.ModuleList()
        for nh in num_hiddens_per_fc_layer:
            self.fc_layers.append(torch.nn.Linear(n_in, nh))
            n_in = nh
        self.fc_layers.append(torch.nn.Linear(n_in, num_outputs))

        # self.conv_layers = torch.nn.ModuleList()
        # self.fc_layers = torch.nn.ModuleList()
        # self.cod = OrderedDict([])
        # self.fcod = OrderedDict([])
        # if num_hiddens_per_conv_layer:
        #     for i in range(len(num_hiddens_per_conv_layer)):
        #         print("i: ", i)
        #         if i == 0:
        #             self.conv_layers.append(
        #                 nn.Conv1d(num_channels, num_hiddens_per_conv_layer[0],
        #                           kernel_size=kernel_size_per_conv_layer[0],
        #                           stride=stride_per_conv_layer[0])
        #             )
        #             # self.cod['Conv1d0'] = nn.Sequential(
        #             #     nn.Conv1d(num_channels, num_hiddens_per_conv_layer[0],
        #             #               kernel_size=kernel_size_per_conv_layer[0],
        #             #               stride=stride_per_conv_layer[0]),
        #             #     nn.ReLU())
        #             liner_input = ((num_inputs - (kernel_size_per_conv_layer[len(kernel_size_per_conv_layer) - 1] - 1)
        #                             * len(num_hiddens_per_conv_layer)) // (stride_per_conv_layer[i])) * \
        #                           num_hiddens_per_conv_layer[
        #                               len(num_hiddens_per_conv_layer) - 1]
        #             sum = 0
        #             for i in range(len(kernel_size_per_conv_layer) - 1):
        #                 print("i: ", i)
        #                 print("stride_per_conv_layer:", stride_per_conv_layer[i])
        #                 print(((kernel_size_per_conv_layer[i] - kernel_size_per_conv_layer[
        #                     len(kernel_size_per_conv_layer) - 1]) // (stride_per_conv_layer[i])) \
        #                       * num_hiddens_per_conv_layer[len(num_hiddens_per_conv_layer) - 1])
        #                 sum += ((kernel_size_per_conv_layer[i] - kernel_size_per_conv_layer[
        #                     len(kernel_size_per_conv_layer) - 1]) // (stride_per_conv_layer[i])) \
        #                        * num_hiddens_per_conv_layer[len(num_hiddens_per_conv_layer) - 1]
        #             liner_input -= sum
        #             print("num_inputs", num_inputs)
        #             print(kernel_size_per_conv_layer[len(kernel_size_per_conv_layer) - 1] - 1)
        #             print(len(num_hiddens_per_conv_layer))
        #             print(num_hiddens_per_conv_layer[
        #                       len(num_hiddens_per_conv_layer) - 1])
        #             print(sum)
        #             print("liner_input: ", liner_input)
        #
        #             self.fc_layers.append(nn.Linear(liner_input, num_hiddens_per_fc_layer[0]))
        #
        #             # self.fcod['Linear0'] = nn.Sequential(
        #             #     nn.Linear(liner_input, num_hiddens_per_fc_layer[0]),
        #             #     nn.ReLU())
        #             # self.cod['ReLU0'] = nn.ReLU()
        #         else:
        #             self.conv_layers.append(nn.Conv1d(num_hiddens_per_conv_layer[i - 1],
        #                           num_hiddens_per_conv_layer[i],
        #                           kernel_size=kernel_size_per_conv_layer[i],
        #                           stride=stride_per_conv_layer[i]))
        #             self.fc_layers.append(nn.Linear(num_hiddens_per_fc_layer[i - 1],
        #                                             num_hiddens_per_fc_layer[i]))
        #
        #             # self.conv_layers.append(nn.Conv1d(num_hiddens_per_conv_layer[i - 1],
        #             #               num_hiddens_per_conv_layer[i],
        #             #               kernel_size=kernel_size_per_conv_layer[i],
        #             #               stride=stride_per_conv_layer[i]))
        #             # self.cod['Conv1d' + str(i)] = nn.Sequential(
        #             #     nn.Conv1d(num_hiddens_per_conv_layer[i - 1],
        #             #               num_hiddens_per_conv_layer[i],
        #             #               kernel_size=kernel_size_per_conv_layer[i],
        #             #               stride=stride_per_conv_layer[i]),
        #             #     nn.ReLU())
        #
        #
        #             # self.fcod['Linear' + str(i)] = nn.Sequential(
        #             #     nn.Linear(num_hiddens_per_fc_layer[i - 1],
        #             #               num_hiddens_per_fc_layer[i]),
        #             #     nn.ReLU())
        #
        #     self.last_layer = nn.Linear(num_hiddens_per_fc_layer[len(num_hiddens_per_fc_layer) - 1],
        #                                 num_outputs)
        #     # self.cod['ReLU' + str(i)] = nn.ReLU()
        #
        # # self.od = OrderedDict([])
        # # if num_hiddens_per_fc_layer:
        # #     for i in range(len(num_hiddens_per_fc_layer)):
        # #         if i == 0:
        # #             print('i == 0')
        # #             self.od['Linear0'] = nn.Linear(num_hiddens_per_conv_layer[len(num_hiddens_per_conv_layer)-1], num_hiddens_per_fc_layer[0])
        # #             # self.od['ReLU0'] = nn.ReLU()
        # #         else:
        # #             self.od['Linear' + str(i)] = nn.Linear(num_hiddens_per_fc_layer[i - 1], num_hiddens_per_fc_layer[i])
        # #             # self.od['ReLU' + str(i)] = nn.ReLU()
        # #     self.od['Linear' + str(len(num_hiddens_per_fc_layer))] = \
        # #         nn.Linear(num_hiddens_per_fc_layer[len(num_hiddens_per_fc_layer) - 1], num_outputs)
        # # else:
        # #     self.od['Linear0'] = nn.Linear(num_inputs, num_outputs)
        #
        # # self.conv1d_linear_stack = nn.Sequential(od)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.activation_function(conv_layer(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        for i, fc_layer in enumerate(self.fc_layers):
            if i == len(self.fc_layers) - 1:
                break
            x = self.activation_function(fc_layer(x))

        x = self.fc_layers[-1](x)
        return x
