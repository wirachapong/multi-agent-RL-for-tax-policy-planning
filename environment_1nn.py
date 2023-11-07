from environment import Environment
from person_1nn import Person_1nn
from NNOfPerson import NNOfPerson
from policyplanneragent import PolicyPlannerAgent
from constants import ACTIONS
import random
from double_auction import *
from bid_sell import *

class Environment_1nn(Environment):
    def __init__(self, n_persons:int):
        super().__init__(n_persons)
        available_category_of_person = ["A","B","C"]
        education_level_turn0 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0]
        net_worth_turn0 = 0.0
        base_salary = 400.0
        n_brackets = 7
        self.person_model = NNOfPerson(2, 2) # QNetwork[net_worth, potential_income] -> [earn, learn]

        self.persons = [Person_1nn(self.person_model, idx,  random.choice(education_level_turn0), net_worth_turn0, base_salary, category=random.choice(available_category_of_person)) for idx in range(n_persons)] 

        self.PolicyPlannerAgent = PolicyPlannerAgent(2 * n_persons + n_brackets, len(ACTIONS))

        self.bid_sell_system = BidSellSystem(commodities=["A","B","C"],agents=self.persons)