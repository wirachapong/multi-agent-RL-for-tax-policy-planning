# environment.py
from person import Person,NNOfPerson
from policyplanneragent import PolicyPlannerAgent
import numpy as np
from constants import EDUCATION_EARNINGS,EDUCATION_LEVELS,EXPENSE,NUM_PERSONS,ACTIONS



class Environment:
    def __init__(self):
        self.NNOfPerson= NNOfPerson
        self.persons = [Person(self.NNOfPerson,np.random.choice(EDUCATION_LEVELS)) for _ in range(NUM_PERSONS)]
        self.PolicyPlannerAgent= PolicyPlannerAgent(2*len(self.persons),len(ACTIONS))

    def get_state(self):
        # This method compiles the net worths and education levels of all persons
        # into a single list representing the current state.
        net_worths = [person.net_worth for person in self.persons]
        educations = [person.education_level for person in self.persons]
        
        state = net_worths + educations
        return state

    def step(self, action):
        # This method updates the net worth of all persons and gets the new state.
        # The 'action' parameter is included because it might affect how the environment changes.
        for person in self.persons:
            person.update_net_worth()
        
        # 'action' is not used in the current method, but it's here for future use
        # if you want the environment to react based on the actions taken.
        
        next_state = self.get_state()
        return next_state