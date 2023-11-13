import configuration
import utils
from person import Person




class Person_0nn(Person):

    def __init__(self, idx:int, education_level:int, net_worth:float, epsilon:float=0.1, category:str='A',
                 discount_rate: float = 0):
        super().__init__(idx, education_level, net_worth, epsilon, category)

        # No neural network for decision making
        self.model = None
        self.memory = None
        self.optimizer = None
        self.epsilon = None
        
        self.learning_years_remaining = 0
        self.discount_rate = discount_rate
    
    # Closed form optimal choice ---> BEST RESPONSE 
    def select_action(self, time_step: int = 0, horizon: int = 100, tax_function = None):
        if self.education_level == max(self.education_levels):
            return 0
        
        if self.net_worth < (self.cost_of_living * self.education_turns_required[self.education_level + 1]):
            return 0

        if self.turns_left_in_current_education:
            return 1 # Continues ongoing education

        time_steps_left = horizon - 1 - time_step
        income_if_earn, _  = tax_function(self.education_earnings[self.education_level])
        income_if_earn += self.last_tax_income
        income_if_earn = utils.discounted_sum_constant_reward_vectorized(income_if_earn, self.discount_rate, time_steps_left)

        turns_needed = 0
        for education_level in range(self.education_level + 1, max(self.education_levels) + 1):
            turns_needed += self.education_turns_required[education_level]
            if time_steps_left - turns_needed<0:
                continue

            income_if_learn, _ = tax_function(self.education_earnings[education_level])
            income_if_learn += self.last_tax_income
            income_if_learn = utils.discounted_sum_constant_reward_vectorized(income_if_learn, self.discount_rate, time_steps_left - turns_needed) * self.discount_rate**turns_needed
            income_if_learn +=  utils.discounted_sum_constant_reward_vectorized(self.last_tax_income, self.discount_rate, turns_needed)
            # income_if_learn *=  (time_steps_left - turns_needed)  # todo Apply time discounting

            if income_if_learn > income_if_earn:
                return 1 # Learn
        
        return 0 # Earn 


        #TODO Complete this
        # if self.learning_years_remaining:
        #     self.learning_years_remaining -= 1
        #     if self.learning_years_remaining == 0:
        #         education_level += 1 
        #     return 2

        # time_steps_left = horizon - 1 - time_step
        # income_if_earn = self.potential_income * time_steps_left
        # income_if_learn = (self.base_salary * (self.education_level + configuration.config.get_constant("EDUCATION_INCREASE"))) * (time_steps_left - 1)

        # if income_if_earn > income_if_learn:
        #     return 0
        # else:
        #     return 1

    # Non existing functions without neural networks
    def remember(self, state, action: int, reward, next_state):
        raise Exception("No neural network in this object")
    
    def replay(self):
        raise Exception("No neural network in this object")

        