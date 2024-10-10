import torch

import torch.nn as nn
import torch.nn.functional as F

# Define the simplified model
class FootballPredictor(nn.Module):
    def __init__(self, input_dim):
        dropout_rate = 0.7

        super(FootballPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, 3)  # Assuming 3 possible outcomes: win, lose, draw

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x