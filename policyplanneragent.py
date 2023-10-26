import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from constants import GAMMA, ALPHA,EPSILON, BATCH_SIZE, MEMORY_SIZE, ACTIONS, SALARIES

# Constants for the agent and learning process.
# You can also move these to a separate configuration file or constants module.



class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 21)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 3, 7) 

class PolicyPlannerAgent:
    def __init__(self, input_dim, num_actions):
        self.model = QNetwork(input_dim, num_actions)
        self.current_tax_rate = [10,12,22,24,32,35,37]
        self.memory = []  # For experience replay
        self.history_of_auctions = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)

    def select_action(self, state):
        if np.random.uniform(0, 1) < EPSILON:
            return np.random.choice(ACTIONS)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor)
                max_values, max_indices = torch.max(q_values, dim=1) 
                return max_indices[0] 
                # EX Output [1, 0, 1, 2, 0, 1, 0]
                # 0 = remain same tax rate     1= +0.1 tax rate for the bracket    2 = -0.1 tax rate for the bracket  
                

    def remember(self, state, action, reward, next_state):
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

    def apply_action(self, action, persons):
        total_cost = 0
        current_tax_rate = self.current_tax_rate
        action_modifiers = [0 if act == 0 else 0.1 if act == 1 else -0.1 for act in action]
        new_tax_rate = [rate + modifier for rate, modifier in zip(current_tax_rate, action_modifiers)]
        self.current_tax_rate = new_tax_rate
        for person in persons:
            
        # this is the old back bone so we will probably change it later
        return total_cost
    

    #need to change this one
    def get_reward(self, total_cost, persons):  
        net_worth_sum = sum([person.net_worth for person in persons])
        reward = net_worth_sum - total_cost
        return reward
    
    # that the agent will use to interact with the environment.
