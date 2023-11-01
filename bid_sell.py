import numpy as np


class BidSellSystem:
    def __init__(
            self,
            *args,
            commodities,
            agents,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bid_dictionary = {}
        self.sell_dictionary = {}
        self.bid_current_round = {}
        self.sell_current_round = {}
        self.bid_previous_round ={}
        self.sell_previous_round = {}
    
    def clear_previous_round(self):
        # Remove bids from the previous round from the bid dictionary
        for bid_key in self.bid_previous_round:
            if bid_key in self.bid_dictionary:
                del self.bid_dictionary[bid_key]

        # Remove sells from the previous round from the sell dictionary
        for sell_key in self.sell_previous_round:
            if sell_key in self.sell_dictionary:
                del self.sell_dictionary[sell_key]