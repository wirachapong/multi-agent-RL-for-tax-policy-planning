import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from constants import GAMMA, ALPHA,EPSILON, BATCH_SIZE, MEMORY_SIZE, ACTIONS, SALARIES


# Constants for the agent and learning process.
# You can also move these to a separate configuration file or constants module.

class QNetwork(nn.Module):
    def __init__(self, input_dim:int, num_actions:int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7 * num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 3, 7) 

class PolicyPlannerAgent:
    def __init__(self, input_dim:int, num_actions:int):
        self.model = QNetwork(input_dim, num_actions)
        self.current_tax_rate = [10,12,22,24,32,35,37]
        self.memory = deque()  # For experience replay
        self.history_of_auctions = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)

    def select_action(self, state):
        if np.random.uniform(0, 1) < EPSILON or len(self.memory) < MEMORY_SIZE:
            return torch.tensor([np.random.choice(ACTIONS) for _ in range(len(self.current_tax_rate))])

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
            #self.memory.pop(0)
            self.memory.popleft()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch_indices = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
        batch = [self.memory[i] for i in batch_indices]

        for state, action, reward, next_state in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            with torch.no_grad():
                max_values, max_indices = torch.max(self.model(next_state_tensor), dim=1)
                target = reward + GAMMA * max_values

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
        
        
        # accumulated_tax = self.apply_tax(persons, self.current_tax_rate)

        # for person in persons:
        #     person.net_worth+=accumulated_tax/len(persons)
        
        # this is the old back bone so we will probably change it later
        return total_cost
    
    def tax_rate_for_income(self, income, bracket_gap:int=1000):
        
        brackets = self.current_tax_rate
        income_bracket_index = min(int(income / bracket_gap), len(brackets)-1)
        
        income_over_last_index = income - income_bracket_index * bracket_gap
        tax_over_last_index = income_over_last_index * (brackets[income_bracket_index]/100)

        tax_income = tax_over_last_index
        for i in range(income_bracket_index):
            tax_income += (brackets[i]/100) * bracket_gap

        person_income = income - tax_income 

        return person_income, tax_income
    
    #need to change this one
    def get_reward(self, total_cost, persons):  
        net_worth_sum = sum([person.net_worth for person in persons])
        reward = net_worth_sum - total_cost
        return reward
    
    # def apply_tax(self, persons, brackets, bracket_gap:int=5000):
    #     accumulated_tax=0
    #     for person in persons:
    #         # Calculate the person's income bracket based on their income.
    #         income_bracket_index = int(person.income_for_the_round / bracket_gap)

    #         # Make sure we don't exceed the number of defined brackets.
    #         if income_bracket_index > len(brackets) - 1:
    #             income_bracket_index = len(brackets) - 1

    #         # Get the tax rate for the person's bracket.
    #         tax_rate = brackets[income_bracket_index]

    #         # Calculate the tax amount.
    #         tax_amount = (tax_rate / 100.0) * person.income_for_the_round

    #         accumulated_tax += tax_amount

    #         # Deduct the tax from the person's income.
    #         person.income_for_the_round -= tax_amount

    #         return accumulated_tax

    
    
    # that the agent will use to interact with the environment.

