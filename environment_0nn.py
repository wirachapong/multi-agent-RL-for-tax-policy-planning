from environment import Environment
from person_0nn import Person_0nn
from policyplanneragent import PolicyPlannerAgent
from constants import ACTIONS
import random
from double_auction import *
from bid_sell import *

class Environment_0nn(Environment):
    def __init__(self, n_persons:int):
        available_category_of_person = ["A","B","C"]
        education_level_turn0 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0]
        net_worth_turn0 = 0.0
        base_salary = 400.0
        n_brackets = 7
        self.time_step = 0

        self.persons = [Person_0nn(idx,  random.choice(education_level_turn0), net_worth_turn0, base_salary) for idx in range(n_persons)] 

        self.PolicyPlannerAgent = PolicyPlannerAgent(2 * n_persons + n_brackets, len(ACTIONS))
        

        self.bid_sell_system = BidSellSystem(commodities=["A","B","C"],agents=self.persons)

    def persons_step(self, is_terminal_state=False):
        # Approach with individual comprehensions
        # current_states = [person.get_state() for person in self.persons]
        person_actions = [person.select_action(self.time_step, self.horizon) for person in self.persons]

        for action, person in zip(person_actions, self.persons):
            person.take_action(action, self.PolicyPlannerAgent.tax_rate_for_income)

        accumulated_tax = self.get_tax_for_round_for_all()
        self.distribute_tax(accumulated_tax)
        
        # person_rewards = [person.get_reward(is_terminal_state) for person in self.persons]
        # person_next_states = [person.get_state() for person in self.persons]

        # for i, person in enumerate(self.persons):
        #     person.remember(current_states[i], person_actions[i], person_rewards[i], person_next_states[i])
        
        # for person in self.persons:
        #     person.replay()
        
        next_state = self.get_state()

        self.time_step += 1
        return next_state
