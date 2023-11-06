from person import Person
from constants_person import EDUCATION_EARNINGS,EXPENSE,ALPHA, GAMMA, BATCH_SIZE, MEMORY_SIZE, EDUCATION_INCREASE

class Person_0nn(Person):

    def __init__(self, idx:int, education_level:float, net_worth:float, base_salary:float = 400.0, epsilon:float=0.1, category:str='A'):
        super().__init__(idx, education_level, net_worth, base_salary, epsilon, category)

        # No neural network for decision making
        self.model = None
        self.memory = None
        self.optimizer = None
        self.epsilon = None
    
    # Closed form optimal choice ---> BEST RESPONSE 
    def select_action(self, time_step: int, horizon: int):
        time_steps_left = horizon - 1 - time_step
        income_if_earn = self.potential_income * time_steps_left
        income_if_learn = (self.base_salary * (self.education_level + EDUCATION_INCREASE)) * (time_steps_left - 1)

        if income_if_earn > income_if_learn:
            return 0
        else:
            return 1

    # Non existing functions without neural networks
    def remember(self, state, action: int, reward, next_state):
        raise Exception("No neural network in this object")
    
    def replay(self):
        raise Exception("No neural network in this object")

        