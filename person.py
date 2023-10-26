from constants_person import EDUCATION_EARNINGS,EXPENSE,ALPHA, GAMMA, BATCH_SIZE, MEMORY_SIZE, EDUCATION_INCREASE
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

    def __init__(self,NNOfPerson,edu_level, epsilon=0.1):
        self.model= NNOfPerson
        self.net_worth = 0
        self.education_level = edu_level
        self.potential_income = 400 * self.education_level
        self.income_for_the_round = 0
        self._idx = next(self.id_generator)
        self.memory = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)
        self.loss_fn = nn.Loss()

        self.state = [self.net_worth, self.potential_income]
        self.action_space = ["Earn", "Learn"]

        self.epsilon = epsilon

    def update_net_worth(self):
        self.net_worth += self.earn()

    @property
    def idx(self):
        """Index used to identify this agent. Must be unique within the environment."""
        return self._idx

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.get_state())
                q_values = self.model(state_tensor)
                max_values, max_indices = torch.max(q_values, dim=0)

                if max_indices.item() == 0:
                    return "Earn"
                else:
                    return "Learn"
                

    # Can include number of hours worked at later stages
    def get_reward(self):
        return self.income_for_the_round
    
    def get_state(self):
        return [self.net_worth, self.potential_income]
    
    def remember(self):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch_indices = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
        batch = [self.memory[i] for i in batch_indices]

        for state, action, reward, next_state in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            with torch.no_grad():
                target = reward + GAMMA * torch.max(self.model(next_state_tensor))

            q_values = self.model(state_tensor)
            loss = nn.MSELoss()(q_values[0][action], target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # returns the next state
    def take_action(self, action):
        if action == "Earn":
            self.earn()

        else:
            self.learn()

    def earn(self):
        self.income_for_the_round = self.potential_income
        
    def learn(self):
        self.income_for_the_round = 0
        self.education_level += EDUCATION_INCREASE

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


