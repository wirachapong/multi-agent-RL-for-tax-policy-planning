# environment.py
import configuration
from person import Person
from policyplanneragent import PolicyPlannerAgent
import numpy as np
from double_auction import *
from bid_sell import *
from collections import deque
import random
from configuration import config
import torch

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
        education_level_turn0 = configuration.config.get_constant("EDUCATION_LEVELS")
        net_worth_turn0 = configuration.config.get_constant("NETWORTH_TURN0")
        n_brackets = configuration.config.get_constant("N_BRACKETS")
        commodities = configuration.config.get_constant("AVAILABLE_COMMODITIES")

        self.persons = [Person(i,  np.random.choice(education_level_turn0), net_worth_turn0) for i in range(n_persons)] 

        self.PolicyPlannerAgent = PolicyPlannerAgent(2 * n_persons + n_brackets, len(configuration.config.get_constant("ACTIONS")))
        # len 2*len(self.persons)+7 = from net_worths+educations+tax_rate
        self.bid_sell_system = BidSellSystem(commodities=commodities,agents=self.persons)
        self.EPSILON = configuration.config.get_constant("EPSILON")

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
    
    def get_tax_for_round_for_all(self):
        return sum([person.tax_for_the_round for person in self.persons])
    
    def get_income_for_round_for_all(self):
        return sum([person.income_for_the_round for person in self.persons])
    
    def distribute_tax(self, accumulated_tax):
        for person in self.persons:
            person.income_for_the_round += accumulated_tax/len(self.persons)

    #! Either this in main.py or in Environment.py
    def persons_step(self, is_terminal_state=False):
        # Approach with individual comprehensions
        current_states = [person.get_state() for person in self.persons]
        person_actions = [person.select_action() for person in self.persons]

        for action, person in zip(person_actions,self.persons):
            person.take_action(action, self.PolicyPlannerAgent.tax_rate_for_income)

        accumulated_tax = self.get_tax_for_round_for_all()
        self.distribute_tax(accumulated_tax)
        
        person_rewards = [person.get_reward(is_terminal_state) for person in self.persons]
        person_next_states = [person.get_state() for person in self.persons]

        for i, person in enumerate(self.persons):
            person.remember(current_states[i], person_actions[i], person_rewards[i], person_next_states[i])
        
        for person in self.persons:
            person.replay()

        # 'action' is not used in the current method, but it's here for future use
        # if you want the environment to react based on the actions taken.
        
        next_state = self.get_state()
        return next_state
    
    def save_policy_planner(self, lifecycle):
        path = f"saved_models/lifecycle_{lifecycle}_"
    
        tax_rate = np.array(self.PolicyPlannerAgent.current_tax_rate)
        tax_rate = np.save(tax_rate, path + "tax_rate")

        model = self.PolicyPlannerAgent.model
        torch.save(model.state_dict(), path + "model")
        

    def simulate_episode(self, is_terminal_state=False, verbose = False):

        current_state = self.get_state()

        next_state0= self.persons_gain_category_token()

        next_state1= self.fill_random_action_history()
        
        next_state2= self.persons_do_bid_sell() # learn of buying and selling is already included in here

        next_state3= self.bid_sell_system.clear_previous_round()

        action = self.PolicyPlannerAgent.select_action(next_state3)
        
        if verbose:
            print(["%.2f" % tax_rate for tax_rate in self.PolicyPlannerAgent.current_tax_rate])
        
        total_cost = self.PolicyPlannerAgent.apply_action(action, self.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
        next_state2 = self.persons_step(is_terminal_state) # all persons learn or earn and tax is collected.
        reward_policy_planner = self.PolicyPlannerAgent.get_reward(0, self.persons)  # Assumes you've added this method to DQNAgent, similar to PolicyMaker
        
        # we used 0 for now in the (a,b) for previously used get_reward function due to how there's a change in how the policy changed from our first structure
        self.PolicyPlannerAgent.remember(current_state, action, reward_policy_planner, next_state2)
        self.PolicyPlannerAgent.replay()  # Experience replay
        self.bid_sell_system.end_round()
        total_reward_individual = sum([person.get_reward() for person in self.persons])
        return [reward_policy_planner, total_reward_individual]

    def simulate_lifecycle(self, NUM_EPISODES):
        total_reward_policy_planner = 0
        total_reward_individual = 0
        is_terminal_state = False
        verbose = False

        for episode in range(NUM_EPISODES):
            if episode == NUM_EPISODES - 1:
                is_terminal_state = True
            
            if episode % 10:
                print('Episode', episode)
                verbose = True

            reward_policy_planner, reward_individual = self.simulate_episode(is_terminal_state, verbose)
            verbose = False
            total_reward_policy_planner += reward_policy_planner
            total_reward_individual += reward_individual
            # Optionally decrease epsilon over time to reduce exploration
            if EPSILON > 0.01:
                EPSILON *= 0.995

        print(f"Total reward after {NUM_EPISODES} episodes: {[total_reward_policy_planner,total_reward_individual]}")



    def fill_random_action_history(self):    #! Think maybe there is an error in this function??? - person loop doesn't use the person object
        for person in self.persons:
            person.bid_history_A=self.create_random_deque(100,0,4,5,60,20)
            person.bid_history_B=self.create_random_deque(100,0,4,5,60,20)
            person.bid_history_C=self.create_random_deque(100,0,4,5,60,20)
            person.sell_history_A=self.create_random_deque(100,0,4,5,60,20)
            person.sell_history_B=self.create_random_deque(100,0,4,5,60,20)
            person.sell_history_C=self.create_random_deque(100,0,4,5,60,20)
            person.reward_from_token=self.create_random_deque(100,0,0,100,60,35)
        next_state = self.get_state()
        # self.bid_sell_system.bid_dictionary_A
        # self.bid_sell_system.bid_dictionary_B
        # self.bid_sell_system.bid_dictionary_C
        # self.bid_sell_system.
        return next_state
    
    def create_random_deque(self,size, a, b,c, percentage_a,percentage_b):
        count_a = int(size * (percentage_a / 100))  # Number of elements for value a
        count_b = int(size * (percentage_b / 100)) 
        count_c = size - count_a -count_b                    # Number of elements for value b

        # Generate the deque with the given percentage of a and b
        elements = [a] * count_a + [b] * count_b + [c] * count_c
        random.shuffle(elements)  # Shuffle to distribute a and b randomly
        return deque(elements, maxlen=size)

    def persons_do_bid_sell(self):
        # new version of double_auction_system

        full_combination_added=0

        for person in self.persons: 
            if person.bid_amount_A>=self.bid_sell_system.current_sell_price_A:
                smallest_key_less_than_amount = None
                for key in sorted(self.bid_sell_system.bid_dictionary_A):
                    if key < person.bid_amount_A and (smallest_key_less_than_amount is None or key > smallest_key_less_than_amount):
                        smallest_key_less_than_amount = key
                if smallest_key_less_than_amount is not None and self.bid_sell_system.bid_dictionary_A[smallest_key_less_than_amount]:
                    idx_of_the_one_selling = self.bid_sell_system.bid_dictionary_A[smallest_key_less_than_amount].popleft()
                    person.bid_history_A.append(person.bid_amount_A)
                    self.bid_sell_system.update_bid_sell_price()
                    person.category_token_value['A']+=2
                    self.persons[idx_of_the_one_selling].category_token_value['A']-=2
                    person.net_worth-=person.bid_amount_A
                    # not sure if I should use net_worth or income_for_the_round or other things instead
                    self.persons[idx_of_the_one_selling].net_worth+=person.bid_amount_A
                    full_combination_added+=person.check_full_combination()
            else:
                if person.bid_amount_A in self.bid_sell_system.bid_dictionary_A:
                    person.bid_history_A.append(0)
                    self.bid_sell_system.bid_dictionary_A[person.bid_amount_A].append(person.idx)
                else:
                    person.bid_history_A.append(0)
                    self.bid_sell_system.bid_dictionary_A[person.bid_amount_A]=deque([person.idx])
                self.bid_sell_system.update_bid_sell_price()

            if person.bid_counter_A//10==0:
                person.learn_bid_A()
            if person.can_sell_A():
                if person.sell_amount_A>=self.bid_sell_system.current_bid_price_A:
                    if person.sell_amount_A in self.bid_sell_system.sell_dictionary_A:
                        person.sell_history_A.append(0)
                        self.bid_sell_system.sell_dictionary_A[person.sell_amount_A].append(person.idx)
                    else:
                        person.sell_history_A.append(0)
                        self.bid_sell_system.sell_dictionary_A[person.sell_amount_A]=deque([person.idx])
                    self.bid_sell_system.update_bid_sell_price()
                elif person.sell_amount_A<self.bid_sell_system.current_bid_price_A:
                    smallest_key_greater_than_sell_amount = None
                    for key in self.bid_sell_system.sell_dictionary_A.keys():
                        if key > person.sell_amount_A and (smallest_key_greater_than_sell_amount is None or key < smallest_key_greater_than_sell_amount):
                            smallest_key_greater_than_sell_amount = key
                    if smallest_key_greater_than_sell_amount is not None and self.bid_sell_system.sell_dictionary_A[smallest_key_greater_than_sell_amount]:
                        person.sell_history_A.append(person.sell_amount_A)
                        idx_of_the_person_bidding = self.bid_sell_system.sell_dictionary_A[smallest_key_greater_than_sell_amount].popleft()
                        self.bid_sell_system.update_bid_sell_price()
                        person.category_token_value['A']-=2
                        self.persons[idx_of_the_person_bidding].category_token_value['A']+=2
                        person.net_worth-=person.sell_amount_A
                        # not sure if I should use net_worth or income_for_the_round or other things instead
                        self.persons[idx_of_the_person_bidding].net_worth+=person.sell_amount_A
            else:
                person.sell_history_A.append(0)

            if person.bid_amount_B>=self.bid_sell_system.current_sell_price_B:
                smallest_key_less_than_amount = None
                for key in sorted(self.bid_sell_system.bid_dictionary_B):
                    if key < person.bid_amount_B and (smallest_key_less_than_amount is None or key > smallest_key_less_than_amount):
                        smallest_key_less_than_amount = key
                if smallest_key_less_than_amount is not None and self.bid_sell_system.bid_dictionary_B[smallest_key_less_than_amount]:
                    idx_of_the_one_selling = self.bid_sell_system.bid_dictionary_B[smallest_key_less_than_amount].popleft()
                    person.bid_history_B.append(person.bid_amount_B)
                    self.bid_sell_system.update_bid_sell_price()
                    person.category_token_value['B']+=2
                    self.persons[idx_of_the_one_selling].category_token_value['B']-=2
                    person.net_worth-=person.bid_amount_B
                    # not sure if I should use net_worth or income_for_the_round or other things instead
                    self.persons[idx_of_the_one_selling].net_worth+=person.bid_amount_B
                    full_combination_added+=person.check_full_combination()
            else:
                if person.bid_amount_B in self.bid_sell_system.bid_dictionary_B:
                    person.bid_history_B.append(0)
                    self.bid_sell_system.bid_dictionary_B[person.bid_amount_B].append(person.idx)
                else:
                    person.bid_history_B.append(0)
                    self.bid_sell_system.bid_dictionary_B[person.bid_amount_B]=deque([person.idx])
                self.bid_sell_system.update_bid_sell_price()
            
            if person.bid_counter_B//10==0:
                person.learn_bid_B()
            if person.can_sell_B():
                if person.sell_amount_B>=self.bid_sell_system.current_bid_price_B:
                    if person.sell_amount_B in self.bid_sell_system.sell_dictionary_B:
                        person.sell_history_B.append(0)
                        self.bid_sell_system.sell_dictionary_B[person.sell_amount_B].append(person.idx)
                    else:
                        person.sell_history_B.append(0)
                        self.bid_sell_system.sell_dictionary_B[person.sell_amount_B]=deque([person.idx])
                    self.bid_sell_system.update_bid_sell_price()
                elif person.sell_amount_B<self.bid_sell_system.current_bid_price_B:
                    smallest_key_greater_than_sell_amount = None
                    for key in self.bid_sell_system.sell_dictionary_B.keys():
                        if key > person.sell_amount_B and (smallest_key_greater_than_sell_amount is None or key < smallest_key_greater_than_sell_amount):
                            smallest_key_greater_than_sell_amount = key
                    if smallest_key_greater_than_sell_amount is not None and self.bid_sell_system.sell_dictionary_B[smallest_key_greater_than_sell_amount]:
                        person.sell_history_B.append(person.sell_amount_B)
                        idx_of_the_person_bidding = self.bid_sell_system.sell_dictionary_B[smallest_key_greater_than_sell_amount].popleft()
                        self.bid_sell_system.update_bid_sell_price()
                        person.category_token_value['B']-=2
                        self.persons[idx_of_the_person_bidding].category_token_value['B']+=2
                        person.net_worth-=person.sell_amount_B
                        # not sure if I should use net_worth or income_for_the_round or other things instead
                        self.persons[idx_of_the_person_bidding].net_worth+=person.sell_amount_B
            else:
                person.sell_history_B.append(0)


            if person.bid_amount_C>=self.bid_sell_system.current_sell_price_C:
                smallest_key_less_than_amount = None
                for key in sorted(self.bid_sell_system.bid_dictionary_C):
                    if key < person.bid_amount_C and (smallest_key_less_than_amount is None or key > smallest_key_less_than_amount):
                        smallest_key_less_than_amount = key
                if smallest_key_less_than_amount is not None and self.bid_sell_system.bid_dictionary_C[smallest_key_less_than_amount]:
                    idx_of_the_one_selling = self.bid_sell_system.bid_dictionary_C[smallest_key_less_than_amount].popleft()
                    person.bid_history_C.append(person.bid_amount_C)
                    self.bid_sell_system.update_bid_sell_price()
                    person.category_token_value['C']+=2
                    self.persons[idx_of_the_one_selling].category_token_value['C']-=2
                    person.net_worth-=person.bid_amount_C
                    # not sure if I should use net_worth or income_for_the_round or other things instead
                    self.persons[idx_of_the_one_selling].net_worth+=person.bid_amount_C
                    full_combination_added+=person.check_full_combination()
            else:
                if person.bid_amount_C in self.bid_sell_system.bid_dictionary_C:
                    person.bid_history_C.append(0)
                    self.bid_sell_system.bid_dictionary_C[person.bid_amount_C].append(person.idx)
                else:
                    person.bid_history_C.append(0)
                    self.bid_sell_system.bid_dictionary_C[person.bid_amount_C]=deque([person.idx])
                self.bid_sell_system.update_bid_sell_price()
            
            
            if person.bid_counter_C//10==0:
                person.learn_bid_C()
            if person.can_sell_C():
                if person.sell_amount_C>=self.bid_sell_system.current_bid_price_C:
                    if person.sell_amount_C in self.bid_sell_system.sell_dictionary_C:
                        person.sell_history_C.append(0)
                        self.bid_sell_system.sell_dictionary_C[person.sell_amount_C].append(person.idx)
                    else:
                        person.sell_history_C.append(0)
                        self.bid_sell_system.sell_dictionary_C[person.sell_amount_C]=deque([person.idx])
                    self.bid_sell_system.update_bid_sell_price()
                elif person.sell_amount_C<self.bid_sell_system.current_bid_price_C:
                    smallest_key_greater_than_sell_amount = None
                    for key in self.bid_sell_system.sell_dictionary_C.keys():
                        if key > person.sell_amount_C and (smallest_key_greater_than_sell_amount is None or key < smallest_key_greater_than_sell_amount):
                            smallest_key_greater_than_sell_amount = key
                    if smallest_key_greater_than_sell_amount is not None and self.bid_sell_system.sell_dictionary_C[smallest_key_greater_than_sell_amount]:
                        person.sell_history_C.append(person.sell_amount_C)
                        idx_of_the_person_bidding = self.bid_sell_system.sell_dictionary_C[smallest_key_greater_than_sell_amount].popleft()
                        self.bid_sell_system.update_bid_sell_price()
                        person.category_token_value['C']-=2
                        self.persons[idx_of_the_person_bidding].category_token_value['C']+=2
                        person.net_worth-=person.sell_amount_C
                        # not sure if I should use net_worth or income_for_the_round or other things instead
                        self.persons[idx_of_the_person_bidding].net_worth+=person.sell_amount_C

            else:
                person.sell_history_C.append(0)
                
            if full_combination_added==0:
                person.reward_from_token.append(0)
            else:
                person.reward_from_token.append(100)
        next_state= self.get_state()
        return next_state
    
    def persons_gain_category_token(self):
        for person in self.persons:
            person.earn_category_token()
        next_state= self.get_state()
        return next_state