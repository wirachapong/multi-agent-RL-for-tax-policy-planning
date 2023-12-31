import configuration
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from NNOfPerson import NNOfPerson

class Person:
    """
    A Person is an agent that takes part in the system. has the ability to make actions and earn.
    """

    def __init__(self, idx:int, education_level:int, net_worth:float, epsilon:float = 0.1, category:str='A'):

        
        # QNetwork definition
        self.model = NNOfPerson(2, 2) # QNetwork[net_worth, potential_income] -> [earn, learn]

        self._idx = idx
        self.memory = deque()
        self.optimizer = optim.Adam(self.model.parameters(), lr=configuration.config.get_constant("ALPHA_PERSON"))
        self.epsilon = epsilon

        self.turns_left_in_current_education = 0
        self.education_level = education_level
        self.education_levels = configuration.config.get_constant("EDUCATION_LEVELS")
        self.education_earnings = configuration.config.get_constant("EDUCATION_EARNINGS")
        self.education_turns_required = configuration.config.get_constant("EDUCATION_TURNS_REQUIRED")

        # Value tracking
        self.cost_of_living = configuration.config.get_constant("COST_OF_LIVING")
        self.net_worth = net_worth
        self.base_salary = configuration.config.get_constant("BASE_SALARY")
        self.potential_income = self.base_salary * self.education_level
        self.income_for_the_round = 0
        self.tax_for_the_round = 0
        self.last_tax_income = 0

        # Informational 
        self.state = [self.net_worth, self.potential_income]
        self.action_space = [0, 1] # ["earn", "learn"]


        self.category = category
        self.category_token_value = {'A':0,'B':0,'C':0}
        self.bid_amount_A=np.random.choice([1,2,3,4,5])
        self.bid_amount_B=np.random.choice([1,2,3,4,5])
        self.bid_amount_C=np.random.choice([1,2,3,4,5])
        self.sell_amount_A=np.random.choice([1,2,3,4,5])
        self.sell_amount_B=np.random.choice([1,2,3,4,5])
        self.sell_amount_C=np.random.choice([1,2,3,4,5])

        self.bid_history_A = deque(maxlen=100)
        self.bid_counter_A = 0
        self.sell_history_A = deque(maxlen=100)
        self.sell_counter_A = 0

        self.bid_history_B = deque(maxlen=100)
        self.bid_counter_B = 0
        self.sell_history_B = deque(maxlen=100)
        self.sell_counter_B = 0

        self.bid_history_C = deque(maxlen=100)
        self.bid_counter_C = 0
        self.sell_history_C = deque(maxlen=100)
        self.sell_counter_C = 0

        self.reward_from_token = deque(maxlen=100)
    
    def earn_category_token(self):
        self.category_token_value[self.category] += int(self.education_level)

    def earn(self, tax_function):
        self.potential_income, self.tax_for_the_round = tax_function(self.education_earnings[self.education_level])
        self.income_for_the_round = self.potential_income
        self.net_worth += self.income_for_the_round
        self.net_worth -= self.cost_of_living
        
    def learn(self, tax_function=None):
        if not self.turns_left_in_current_education:
            self.turns_left_in_current_education=self.education_turns_required[self.education_level+1]

        if self.turns_left_in_current_education:
            self.turns_left_in_current_education -= 1
            if not self.turns_left_in_current_education:
                self.education_level += 1

            
        self.income_for_the_round, self.tax_for_the_round = 0, 0
        self.potential_income, _ = tax_function(self.education_earnings[self.education_level])
        self.net_worth -= self.cost_of_living


    # Can include number of hours worked at later stages
    def get_reward(self, is_terminal_state=False):
        if is_terminal_state:
            return self.income_for_the_round + self.net_worth
        
        return self.income_for_the_round
    
    def get_state(self):
        return [self.net_worth, self.potential_income]

    def select_action(self, time_step: int = 0, horizon: int = 100, tax_function = None):
        if np.random.random() < self.epsilon or len(self.memory) < configuration.config.get_constant("MEMORY_SIZE_PERSON"):
            return np.random.choice(self.action_space)
        
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.get_state())
                q_values = self.model(state_tensor)
                max_indices = torch.argmax(q_values)
                return int(max_indices) # 0-> earn, 1-> learn
    
    def take_action(self, action:int, tax_function):
        """
        Returns: the next state
        """

        if action == 0: # Earn
            self.earn(tax_function)

        else:
            self.learn(tax_function)

    def get_tax_dist(self, amount: float):
        """
        performs necessary actions for each tax distribution
        """

        self.last_tax_income = amount
        self.net_worth += amount

    def remember(self, state, action:int, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > configuration.config.get_constant("MEMORY_SIZE"):
            self.memory.popleft()

    def replay(self):
        if len(self.memory) < configuration.config.get_constant("BATCH_SIZE_PERSON"):
            return

        batch_indices = np.random.choice(len(self.memory), configuration.config.get_constant("BATCH_SIZE_PERSON"), replace=False)
        batch = [self.memory[i] for i in batch_indices]

        for state, action, reward, next_state in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            with torch.no_grad():
                max_value, max_index = torch.max(self.model(next_state_tensor), dim=1)
                target = reward + configuration.config.get_constant("GAMMA_PERSON") * max_value

            q_values = self.model(state_tensor)
            loss = nn.MSELoss()(q_values[0][action], target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def check_full_combination(self):
        """
        checks for possible sets of Resources.
        """

        if self.category_token_value['A']>=5 and self.category_token_value['B']>=5 and self.category_token_value['C']>=5:
            self.net_worth+=100
            self.category_token_value['A']-=5
            self.category_token_value['B']-=5
            self.category_token_value['C']-=5
            return 1
        else:
            self.reward_from_token.append(0)
            return 0

    def learn_bid_A(self):
        # count_bid=sum(1 for elem in self.bid_history_A if elem != 0)
        sum_bid=sum(self.bid_history_A) # ไปเติมเคสที่บิดเป็น0ด้วยจ้า
        sum_reward=sum(self.reward_from_token)
        result= sum_reward-sum_bid
        if result>0:
            self.bid_amount_A=int(10*self.bid_amount_A+1)/10
            self.sell_amount_A=int(10*self.bid_amount_A-1)/10
        else: 
            self.bid_amount_A=int(10*self.bid_amount_A-1)/10
            self.sell_amount_A=int(10*self.bid_amount_A+1)/10

    def update_bid_token_transaction_history_A(self,amount_money):
        self.bid_history_A.append(amount_money)
        self.bid_counter_A+=1

    def update_sell_token_transaction_history_A(self,amount_money):
        self.sell_history_A.append(amount_money)    
        self.sell_counter_A+=1

    def learn_bid_B(self):
        # - Input =  how many things they buy
        # - Reward = how much money they got- sum of cost they spent
        # count_bid=sum(1 for elem in self.bid_history_B if elem != 0)
        sum_bid=sum(self.bid_history_B) # ไปเติมเคสที่บิดเป็น0ด้วยจ้า
        sum_reward=sum(self.reward_from_token)
        result= sum_reward-sum_bid
        if result>0:
            self.bid_amount_B=int(10*self.bid_amount_B+1)/10
            self.sell_amount_B=int(10*self.bid_amount_B-1)/10
        else: 
            self.bid_amount_B=int(10*self.bid_amount_B-1)/10
            self.sell_amount_B=int(10*self.bid_amount_B+1)/10


    def update_bid_token_transaction_history_B(self,amount_money):
        self.bid_history_B.append(amount_money)
        self.bid_counter_B+=1

    def update_sell_token_transaction_history_B(self,amount_money):
        self.sell_history_B.append(amount_money)    
        self.sell_counter_B+=1

    def learn_bid_C(self):
        # - Input =  how many things they buy
        # - Reward = how much money they got- sum of cost they spent
        # count_bid=sum(1 for elem in self.bid_history_C if elem != 0)
        sum_bid=sum(self.bid_history_C) # ไปเติมเคสที่บิดเป็น0ด้วยจ้า
        sum_reward=sum(self.reward_from_token)
        result= sum_reward-sum_bid
        if result>0:
            self.bid_amount_C=int(10*self.bid_amount_C+1)/10
            self.sell_amount_C=int(10*self.bid_amount_C-1)/10
        else: 
            self.bid_amount_C=int(10*self.bid_amount_C-1)/10
            self.sell_amount_C=int(10*self.bid_amount_C+1)/10

    def update_bid_token_transaction_history_C(self,amount_money):
        self.bid_history_C.append(amount_money)
        self.bid_counter_C+=1

    def update_sell_token_transaction_history_C(self,amount_money):
        self.sell_history_C.append(amount_money)    
        self.sell_counter_C+=1

    def can_sell_A(self):
        if self.category_token_value['A']>2:
            return True
        else:
            return False

    def can_sell_B(self):
        if self.category_token_value['B']>2:
            return True
        else:
            return False
        
    def can_sell_C(self):
        if self.category_token_value['C']>2:
            return True
        else:
            return False

    def set_tax_income_for_round(self, tax_income):
        self.last_tax_income = tax_income
        self.net_worth+=tax_income

    
    