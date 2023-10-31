from constants_person import EDUCATION_EARNINGS,EXPENSE,ALPHA, GAMMA, BATCH_SIZE, MEMORY_SIZE, EDUCATION_INCREASE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from NNOfPerson import NNOfPerson

# def id_generator_function():
#     """Generate integers 0, 1, 2, and so on."""
#     task_id = 0
#     while True:
#         yield task_id
#         task_id += 1

class Person:
    # id_generator = id_generator_function()

    def __init__(self, idx:int, education_level:float, net_worth:float, base_salary:float = 400.0, epsilon:float = 0.1, category:str):
        # self.model= NNOfPerson --- Dont think this is needed because each person are independent objects
        
        # QNetwork definition
        self.model = NNOfPerson(2, 2) # QNetwork[net_worth, potential_income] -> [earn, learn]
        # Needed for all definitions of a person

        self._idx = idx
        self.memory = deque()
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)
        self.epsilon = epsilon

        # Value trackings
        self.net_worth = net_worth
        self.education_level = education_level
        self.base_salary = base_salary
        self.potential_income = self.base_salary * self.education_level
        self.income_for_the_round = 0
        self.tax_for_the_round = 0
        self.category = category
        self.category_token_value =0

        
        self.state = [self.net_worth, self.potential_income]
        self.action_space = [0, 1] # ["earn", "learn"]
    
    # def update_net_worth(self):
    #     self.net_worth += self.earn()

    def earn(self, tax_function):
        self.income_for_the_round = self.potential_income
    
        self.income_for_the_round, self.tax_for_the_round = tax_function(self.income_for_the_round)
        # self.net_worth += self.income_for_the_round
        
    def learn(self):
        self.income_for_the_round, self.tax_for_the_round = 0, 0
        self.education_level += EDUCATION_INCREASE
        self.potential_income = self.base_salary * self.education_level

    # Can include number of hours worked at later stages
    def get_reward(self):
        return self.income_for_the_round
    
    def get_state(self):
        return [self.net_worth, self.potential_income]

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
                max_indices = torch.argmax(q_values)
                return int(max_indices) # 0-> earn, 1-> learn
    
    # returns the next state
    def take_action(self, action:int, tax_function):
        if action == 0: # Earn
            self.earn(tax_function)

        else:
            self.learn()

    # def step(self):
    #     action = self.select_action()
    #     self.take_action(action)

    def remember(self, state, action:int, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > MEMORY_SIZE:
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
                max_value, max_index = torch.max(self.model(next_state_tensor), dim=1)
                target = reward + GAMMA * max_value

            q_values = self.model(state_tensor)
            loss = nn.MSELoss()(q_values[0][action], target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



