import numpy as np
import torch
import torch.nn as nn


class CannonEnviroment:
    def __init__(self, max_dist, angle_bottom, angle_top):
        self.g = 9.81
        self.max_dist = max_dist
        self.angle_bottom = angle_bottom
        self.angle_top = angle_top
        self.epsilon = 1
        self.angle = None
        self.distance = None

    def step(self, speed):
        distance_hit = (speed ** 2 * np.sin(2 * self.angle)) / self.g
        distance_diff = np.abs(self.distance - distance_hit)
        return [self.angle, distance_diff], -distance_diff, distance_diff < self.epsilon

    def reset(self):
        self.angle = np.radians(np.random.uniform(self.angle_bottom, self.angle_top))
        self.distance = np.random.uniform(0, self.max_dist)
        return [self.angle, self.distance]


class Actor(nn.Module):
    def __init__(self, hid_size, hid_layers_num):
        super().__init__()
        self.fc_i = nn.Linear(2, hid_size)
        seq = []
        for _ in range(hid_layers_num):
            seq.append(nn.Linear(hid_size, hid_size))
            seq.append(nn.ReLU())
        self.seq = nn.Sequential(*seq)
        self.fc_o = nn.Linear(hid_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc_i(x))
        x = self.seq(x)
        x = self.fc_o(x)
        return x


class Critic(nn.Module):
    def __init__(self, hid_size, hid_layers_num):
        super().__init__()
        self.fc_i = nn.Linear(3, hid_size)
        seq = []
        for _ in range(hid_layers_num):
            seq.append(nn.Linear(hid_size, hid_size))
            seq.append(nn.ReLU())
        self.seq = nn.Sequential(*seq)
        self.fc_o = nn.Linear(hid_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc_i(x))
        x = self.seq(x)
        x = self.fc_o(x)
        return x
