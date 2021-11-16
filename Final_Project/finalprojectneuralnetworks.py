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
                 kernel_size_per_conv_layer, stride_per_conv_layer):
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

        # self.flatten = nn.Flatten()
        self.odc = OrderedDict([])
        self.fcod = OrderedDict([])
        if num_hiddens_per_conv_layer:
            for i in range(len(num_hiddens_per_conv_layer)):
                print("i: ", i)
                if i == 0:
                    self.odc['Conv1d0'] = nn.Sequential(
                        nn.Conv1d(num_channels, num_hiddens_per_conv_layer[0],
                                  kernel_size=kernel_size_per_conv_layer[0],
                                  stride=stride_per_conv_layer[0]),
                        nn.ReLU())
                    liner_input = ((num_inputs - (kernel_size_per_conv_layer[len(kernel_size_per_conv_layer) - 1] - 1)
                                    * len(num_hiddens_per_conv_layer)) // (stride_per_conv_layer[i])) * \
                                  num_hiddens_per_conv_layer[
                                      len(num_hiddens_per_conv_layer) - 1]
                    sum = 0
                    for i in range(len(kernel_size_per_conv_layer) - 1):
                        print("i: ", i)
                        print("stride_per_conv_layer:", stride_per_conv_layer[i])
                        print(((kernel_size_per_conv_layer[i] - kernel_size_per_conv_layer[
                            len(kernel_size_per_conv_layer) - 1]) // (stride_per_conv_layer[i])) \
                              * num_hiddens_per_conv_layer[len(num_hiddens_per_conv_layer) - 1])
                        sum += ((kernel_size_per_conv_layer[i] - kernel_size_per_conv_layer[
                            len(kernel_size_per_conv_layer) - 1]) // (stride_per_conv_layer[i])) \
                               * num_hiddens_per_conv_layer[len(num_hiddens_per_conv_layer) - 1]
                    liner_input -= sum
                    print("num_inputs", num_inputs)
                    print(kernel_size_per_conv_layer[len(kernel_size_per_conv_layer) - 1] - 1)
                    print(len(num_hiddens_per_conv_layer))
                    print(num_hiddens_per_conv_layer[
                              len(num_hiddens_per_conv_layer) - 1])
                    print(sum)
                    print("liner_input: ", liner_input)
                    self.fcod['Linear0'] = nn.Sequential(
                        nn.Linear(liner_input, num_hiddens_per_fc_layer[0]),
                        nn.ReLU())
                    # self.odc['ReLU0'] = nn.ReLU()
                else:
                    self.odc['Conv1d' + str(i)] = nn.Sequential(
                        nn.Conv1d(num_hiddens_per_conv_layer[i - 1],
                                  num_hiddens_per_conv_layer[i],
                                  kernel_size=kernel_size_per_conv_layer[i],
                                  stride=stride_per_conv_layer[i]),
                        nn.ReLU())
                    self.fcod['Linear' + str(i)] = nn.Sequential(
                        nn.Linear(num_hiddens_per_fc_layer[i - 1],
                                  num_hiddens_per_fc_layer[i]),
                        nn.ReLU())
            self.last_layer = nn.Linear(num_hiddens_per_fc_layer[len(num_hiddens_per_fc_layer) - 1],
                                        num_outputs)
            # self.odc['ReLU' + str(i)] = nn.ReLU()

        # self.od = OrderedDict([])
        # if num_hiddens_per_fc_layer:
        #     for i in range(len(num_hiddens_per_fc_layer)):
        #         if i == 0:
        #             print('i == 0')
        #             self.od['Linear0'] = nn.Linear(num_hiddens_per_conv_layer[len(num_hiddens_per_conv_layer)-1], num_hiddens_per_fc_layer[0])
        #             # self.od['ReLU0'] = nn.ReLU()
        #         else:
        #             self.od['Linear' + str(i)] = nn.Linear(num_hiddens_per_fc_layer[i - 1], num_hiddens_per_fc_layer[i])
        #             # self.od['ReLU' + str(i)] = nn.ReLU()
        #     self.od['Linear' + str(len(num_hiddens_per_fc_layer))] = \
        #         nn.Linear(num_hiddens_per_fc_layer[len(num_hiddens_per_fc_layer) - 1], num_outputs)
        # else:
        #     self.od['Linear0'] = nn.Linear(num_inputs, num_outputs)

        # self.conv1d_linear_stack = nn.Sequential(od)

    def forward(self, x):

        for conv_layer in self.odc:
            print("x.shape1: ", x.shape)
            print("self.odc[conv_layer]: ", self.odc[conv_layer])
            x = self.odc[conv_layer](x)
        print("x.shape2: ", x.shape)
        # out = x.view(x_shape_zero, x.size(1))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        for linear_layer in self.fcod:
            print("x.shape1: ", x.shape)
            print("self.fcod[conv_layer]: ", self.fcod[linear_layer])
            x = self.fcod[linear_layer](x)
        logit = self.last_layer(x)
        return logit
        # x = x.unsqueeze(dim=0)
        # for conv_layer in self.odc:
        #     print(self.odc[conv_layer])
        #     x = F.relu(self.odc[conv_layer](x))
        #     print(x.size())
        #
        # # x = torch.flatten(x, 0)  # flatten all dimensions except batch
        # x = x.squeeze()
        # for i, layer in enumerate(self.od):
        #     print(self.od[layer])
        #     if i == len(self.od) - 1:
        #         x = self.od[layer](x)
        #     else:
        #         x = F.relu(self.od[layer](x))
        #     print(print(x.size()))
        # return x
