import configuration
from NNOfPerson import NNOfPerson
from person import Person
import random

class Person_1nn(Person):
    """
    A person that uses a nn in order to decide his actions.
    """

    def __init__(self, model, idx:int, education_level:int, net_worth:float, epsilon:float=0.1, category:str='A'):
        super().__init__(idx, education_level, net_worth, epsilon, category)

        # This is the thing that is different from person.py
        self.model = model
        self.available_category_of_person = ["A","B","C"]
        
        self.education_level_turn0 = configuration.config.get_constant("EDUCATION_LEVELS")
        self.net_worth_turn0 = configuration.config.get_constant("NETWORTH_TURN0")

    
    def reset_persons(self):
        self.person_model = NNOfPerson(2, 2) # QNetwork[net_worth, potential_income] -> [earn, learn]

        self.persons = [Person_1nn(self.person_model, idx,  random.choice(self.education_level_turn0), self.net_worth_turn0, base_salary, category=random.choice(available_category_of_person)) for idx in range(n_persons)] 