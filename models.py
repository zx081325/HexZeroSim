"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn


class HexSize11Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(363, 360)
        self.dense2 = nn.Linear(360, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 32)
        self.dense6 = nn.Linear(32, 1)

    def forward(self, x, return_value=False, flags=None):
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


class HexSize7Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(147, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, 16)
        self.dense5 = nn.Linear(16, 8)
        self.dense6 = nn.Linear(8, 1)

    def forward(self, x, return_value=False, flags=None):
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


class HexSize5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(75, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 128)
        self.dense4 = nn.Linear(128, 128)
        self.dense5 = nn.Linear(128, 128)
        self.dense6 = nn.Linear(128, 1)

    def forward(self, x, return_value=False, flags=None):
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


# Model dict is only used in evaluation but not training
def get_model_dict(board_size):
    model_dict = {}
    model_dict['white'] = eval("HexSize" + str(board_size) + "Model")
    model_dict['black'] = eval("HexSize" + str(board_size) + "Model")
    return model_dict


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """

    def __init__(self, device=0, board_size=5):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['white'] = eval("HexSize" + str(board_size) + "Model")().to(torch.device(device))
        self.models['black'] = eval("HexSize" + str(board_size) + "Model")().to(torch.device(device))

    def forward(self, position, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(x, training, flags)

    def share_memory(self):
        self.models['white'].share_memory()
        self.models['black'].share_memory()

    def eval(self):
        self.models['white'].eval()
        self.models['black'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
