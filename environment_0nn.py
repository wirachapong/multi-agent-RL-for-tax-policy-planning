import configuration
import utils
from environment import Environment
from person_0nn import Person_0nn
from policyplanneragent import PolicyPlannerAgent
import random
from double_auction import *
from bid_sell import *

class Environment_0nn(Environment):
    def __init__(self, n_persons:int, horizon: int, random_seed = 1):
        super().__init__(n_persons, random_seed = random_seed)
        self.available_category_of_person = ["A","B","C"]
        
        # For person decisions
        self.horizon = horizon
        self.time_step = 0
        education_level_turn0 = configuration.config.get_constant("EDUCATION_LEVELS")
        net_worth_turn0 = configuration.config.get_constant("NETWORTH_TURN0")
        n_brackets = configuration.config.get_constant("N_BRACKETS")
        self.discount_rate_func = utils.get_discount_rate_heuristic(configuration.config.get_constant("DISCOUNT_RATE_HEURISTIC"))

        random.seed(self.random_seed)
        self.persons = [Person_0nn(idx, random.choice(self.education_level_turn0), self.net_worth_turn0, category=random.choice(self.available_category_of_person), discount_rate=self.discount_rate_func()) for idx in range(n_persons)]

        self.PolicyPlannerAgent = PolicyPlannerAgent(2 * n_persons + n_brackets, len(configuration.config.get_constant("ACTIONS")))
        

        self.bid_sell_system = BidSellSystem(commodities=self.available_category_of_person ,agents=self.persons)

    def persons_step(self, is_terminal_state=False):
        # Approach with individual comprehensions
        # current_states = [person.get_state() for person in self.persons]
        person_actions = [person.select_action(self.time_step, self.horizon, self.PolicyPlannerAgent.tax_rate_for_income) for person in self.persons]

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

    def reset_persons(self, reset_seed: bool = True):
        if reset_seed:
            random.seed(self.random_seed)
        self.persons = [Person_0nn(idx, random.choice(self.education_level_turn0), self.net_worth_turn0, category=random.choice(["A","B","C"]), discount_rate=self.discount_rate_func()) for idx in range(len(self.persons))]

    def reset(self):
        self.time_step = 0

    # override functions
    def persons_gain_category_token(self):
        pass
    def persons_do_bid_sell(self):
        pass
    def update_history_of_auctions(self):
        pass
    def remove_redundant_current_dict(self):
        pass


