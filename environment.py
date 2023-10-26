# environment.py
from person import Person,NNOfPerson
from policyplanneragent import PolicyPlannerAgent
import numpy as np
from constants import EDUCATION_EARNINGS,EDUCATION_LEVELS,EXPENSE,NUM_PERSONS,ACTIONS



class Environment:
    def __init__(self):
        self.NNOfPerson= NNOfPerson
        self.persons = [Person(self.NNOfPerson,np.random.choice(EDUCATION_LEVELS)) for _ in range(NUM_PERSONS)]
        self.PolicyPlannerAgent= PolicyPlannerAgent(2*len(self.persons)+7,len(ACTIONS))
        # len 2*len(self.persons)+7 = from net_worths+educations+tax_rate

    # class PolicyPlannerAgent:
    #     def __init__(self, input_dim, num_actions):
    #         self.model = QNetwork(input_dim, num_actions)
    #         self.current_tax_rate = [10,12,22,24,32,35,37]
    #         self.memory = []  # For experience replay
    #         self.history_of_auctions = []
    #         self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)

    def get_state(self):
        # This method compiles the net worths and education levels of all persons
        # into a single list representing the current state.
        net_worths = [person.net_worth for person in self.persons]
        educations = [person.education_level for person in self.persons]
        tax_rate= PolicyPlannerAgent.current_tax_rate

        # it should also be history_of_auctions here but I'm not sure how to include it.
        


        state = net_worths + educations + tax_rate
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