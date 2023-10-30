import torch
import torch.nn as nn

class NNOfPerson(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super(NNOfPerson, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)