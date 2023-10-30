from environment import Environment
from person_1nn import Person_1nn
from NNOfPerson import NNOfPerson
from policyplanneragent import PolicyPlannerAgent
from constants import ACTIONS


class Environment_1nn(Environment):
    def __init__(self, n_persons:int):
        
        education_level_turn0 = 1.0
        net_worth_turn0 = 0
        base_salary = 400
        n_brackets = 7
        self.person_model = NNOfPerson(2, 2) # QNetwork[net_worth, potential_income] -> [earn, learn]

        self.persons = [Person_1nn(self.person_model, idx,  education_level_turn0, net_worth_turn0, base_salary) for idx in range(n_persons)] 

        self.PolicyPlannerAgent = PolicyPlannerAgent(2 * n_persons + n_brackets, len(ACTIONS))