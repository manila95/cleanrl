import torch
import torch.nn as nn



class RiskNetwork(nn.Module):
    def __init__(self, obs_size, risk_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, risk_size),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.network(x)

