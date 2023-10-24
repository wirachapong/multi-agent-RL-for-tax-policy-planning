import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from constants import GAMMA, ALPHA,EPSILON, BATCH_SIZE, MEMORY_SIZE, ACTIONS, SALARIES

# Constants for the agent and learning process.
# You can also move these to a separate configuration file or constants module.



class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PolicyPlannerAgent:
    def __init__(self, input_dim, num_actions):
        self.model = QNetwork(input_dim, num_actions)
        self.memory = []  # For experience replay
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)

    def select_action(self, state):
        if np.random.uniform(0, 1) < EPSILON:
            return np.random.choice(ACTIONS)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()

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

        if action == 0:  # Assuming 0 is the action for direct money granting
            grant_amount = 500  # Define this as per requirement
            for person in persons:
                person.net_worth += grant_amount
            total_cost = grant_amount * len(persons)
            
        elif action == 1:  # Upgrade education level of random 5 people
            # Select 5 random people who have education level less than 4
            eligible_persons = [p for p in persons if p.education_level < 4]
            selected_persons = random.sample(eligible_persons, min(5, len(eligible_persons)))

            for person in selected_persons:
                # Increase the educational level
                person.education_level += 1

                # Calculate the salary difference due to the upgrade
                salary_difference = SALARIES[person.education_level] - SALARIES[person.education_level - 1]

                # Calculate the cost of the upgrade and add to the total_cost
                upgrade_cost = 5 * salary_difference
                total_cost += upgrade_cost


        return total_cost
    
    def get_reward(self, total_cost, persons):  # need to check this one too
        net_worth_sum = sum([person.net_worth for person in persons])
        reward = net_worth_sum - total_cost
        return reward
    
    # that the agent will use to interact with the environment.
