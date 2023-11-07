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
        education_level_turn0 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0]
        net_worth_turn0 = 0.0
        n_brackets = 7
        self.persons = [Person(i,  np.random.choice(education_level_turn0), net_worth_turn0) for i in range(n_persons)] 

        self.PolicyPlannerAgent = PolicyPlannerAgent(2 * n_persons + n_brackets, len(configuration.config.get_constant("ACTIONS")))
        # len 2*len(self.persons)+7 = from net_worths+educations+tax_rate
        self.bid_sell_system = BidSellSystem(commodities=["A","B","C"],agents=self.persons)

        self.current_round_bid_dictA = self.bid_sell_system.current_round_bid_dict_A()
        self.current_round_sell_dictA = self.bid_sell_system.current_round_sell_dict_A()
        self.current_round_bid_dictB = self.bid_sell_system.current_round_bid_dict_B()
        self.current_round_sell_dictB = self.bid_sell_system.current_round_sell_dict_B()
        self.current_round_bid_dictC = self.bid_sell_system.current_round_bid_dict_C()
        self.current_round_sell_dictC = self.bid_sell_system.current_round_sell_dict_C()

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
        # This method updates the net worth of all persons and gets the new state.
        # The 'action' parameter is included because it might affect how the environment changes.
        # Here should be the space that each individual start doing actions

        # # Approach with one for loop
        # for person in self.persons:
        #     current_state = person.get_state()
        #     person_action = person.select_action()

        #     person.take_action(person_action, self.PolicyPlannerAgent.tax_rate_for_income)
            
        #     person_reward = person.get_reward()
        #     person_next_state = person.get_state()

        #     person.remember(current_state, person_action, person_reward, person_next_state)
            
        #     #! Maybe not do this to batch training later instead
        #     person.replay()


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
            # print(self.current_round_sell_dictA)
            # print(self.bid_sell_system.sell_dictionary_A)
            # for key, small_values in self.current_round_sell_dictA.items():
            #     if key in self.bid_sell_system.sell_dictionary_A:
            #         to_remove = list(small_values)
            #         for item in to_remove:
            #             try:
            #                 self.bid_sell_system.sell_dictionary_A[key].remove(item)
            #             except ValueError:
            #                 pass
            # self.current_round_sell_dictA = self.bid_sell_system.current_round_sell_dict_A()
            # print(self.current_round_sell_dictA)
            # print(self.bid_sell_system.sell_dictionary_A)
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

            # print("watching bit update")

            # print(self.current_round_bid_dictA)
            # print(self.bid_sell_system.bid_dictionary_A)

            # # it's name is current round but it hasn't been updated yet so it's actually previous round here
            # for key, small_values in self.current_round_bid_dictA.items():
            #     if key in self.bid_sell_system.bid_dictionary_A:
            #         to_remove = list(small_values)
            #         for item in to_remove:
            #             try:
            #                 self.bid_sell_system.bid_dictionary_A[key].remove(item)
            #             except ValueError:
            #                 pass
            # self.current_round_bid_dictA = self.bid_sell_system.bid_dictionary_A

            # print(self.current_round_bid_dictA)
            # print(self.bid_sell_system.bid_dictionary_A)

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
    
    def remove_redundant_current_dict(self):
        for key,value in self.bid_sell_system.bid_dictionary_A.items():
            leng=int(len(self.bid_sell_system.bid_dictionary_A[key])/2)
            for i in range(leng):
                self.bid_sell_system.bid_dictionary_A[key].popleft()
        for key,value in self.bid_sell_system.bid_dictionary_B.items():
            leng=int(len(self.bid_sell_system.bid_dictionary_B[key])/2)
            for i in range(leng):
                self.bid_sell_system.bid_dictionary_B[key].popleft()
        for key,value in self.bid_sell_system.bid_dictionary_C.items():
            leng=int(len(self.bid_sell_system.bid_dictionary_C[key])/2)
            for i in range(leng):
                self.bid_sell_system.bid_dictionary_C[key].popleft()
        for key,value in self.bid_sell_system.sell_dictionary_A.items():
            leng=int(len(self.bid_sell_system.sell_dictionary_A[key])/2)
            for i in range(leng):
                self.bid_sell_system.sell_dictionary_A[key].popleft()
        for key,value in self.bid_sell_system.sell_dictionary_B.items():
            leng=int(len(self.bid_sell_system.sell_dictionary_B[key])/2)
            for i in range(leng):
                self.bid_sell_system.sell_dictionary_B[key].popleft()
        for key,value in self.bid_sell_system.sell_dictionary_C.items():
            leng=int(len(self.bid_sell_system.sell_dictionary_C[key])/2)
            for i in range(leng):
                self.bid_sell_system.sell_dictionary_C[key].popleft()