from constants import EDUCATION_EARNINGS,EXPENSE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


class Person:
    id_generator = id_generator_function()

    def __init__(self,NNOfPerson,edu_level):
        self.model= NNOfPerson
        self.net_worth = 0
        self.education_level = edu_level
        self._idx = next(self.id_generator)


    def earn(self):
        return EDUCATION_EARNINGS[self.education_level]

    def spend(self):
        return EXPENSE

    def update_net_worth(self):
        self.net_worth += self.earn() - self.spend()

    @property
    def idx(self):
        """Index used to identify this agent. Must be unique within the environment."""
        return self._idx


class NNOfPerson(nn.module):
    def __init__(self, input_dim, output_dim):
        super(NNOfPerson, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

