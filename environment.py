# environment.py
from person import Person
from policyplanneragent import PolicyPlannerAgent
import numpy as np
from constants import EDUCATION_EARNINGS,EDUCATION_LEVELS,EXPENSE,NUM_PERSONS,ACTIONS

class Environment:
    def __init__(self, n_persons:int):
        #self.persons = [Person(self.NNOfPerson,np.random.choice(EDUCATION_LEVELS)) for _ in range(n_persons)]
        
        # # Starts with uniformly distributed educations levels:
        # lower_bound, higher_bound = 1,4
        # net_worth_turn0 = 0
        # base_salary = 400
        # education_levels = np.random.uniform(lower_bound, higher_bound, self.NNOfPerson)
        # self.persons = [Person(i, education_levels[i], net_worth_turn0, base_salary) for i in range(self.n_persons)]

        # Starts with same education level
        education_level_turn0 = 1.0
        net_worth_turn0 = 0.0
        base_salary = 400.0
        n_brackets = 7
        self.persons = [Person(i,  education_level_turn0, net_worth_turn0, base_salary) for i in range(n_persons)] 

        self.PolicyPlannerAgent = PolicyPlannerAgent(2 * n_persons + n_brackets, len(ACTIONS))
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
        
        #? is this with brackets?
        tax_rate = self.PolicyPlannerAgent.current_tax_rate

        # it should also be history_of_auctions here but I'm not sure how to include it.
        
        state = net_worths + educations + tax_rate
        return state

    #! Either this in main.py or in Environment.py
    def persons_step(self):
        # This method updates the net worth of all persons and gets the new state.
        # The 'action' parameter is included because it might affect how the environment changes.
        # Here should be the space that each individual start doing actions
        for person in self.persons:
            current_state = person.get_state()
            person_action = person.select_action()

            person.take_action(person_action, self.PolicyPlannerAgent.tax_rate_for_income)
            
            person_reward = person.get_reward()
            person_next_state = person.get_state()

            person.remember(current_state, person_action, person_reward, person_next_state)
            
            #! Maybe not do this to batch training later instead
            person.replay()
        # 'action' is not used in the current method, but it's here for future use
        # if you want the environment to react based on the actions taken.
        
        next_state = self.get_state()
        return next_state