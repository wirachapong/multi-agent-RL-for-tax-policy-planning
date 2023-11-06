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
        self.bid_dictionary_A = {}
        self.bid_dictionary_B = {}
        self.bid_dictionary_C = {}

        self.sell_dictionary_A = {}
        self.sell_dictionary_B = {}
        self.sell_dictionary_C = {}

        self.bid_current_round_A = {}
        self.bid_current_round_B = {}
        self.bid_current_round_C = {}

        self.sell_current_round_A = {}
        self.sell_current_round_B = {}
        self.sell_current_round_C = {}

        self.bid_previous_round_A = {}
        self.bid_previous_round_B = {}
        self.bid_previous_round_C = {}

        self.sell_previous_round_A = {}
        self.sell_previous_round_B = {}
        self.sell_previous_round_C = {}

        self.current_bid_price_A=-1000
        self.current_bid_price_B=-1000
        self.current_bid_price_C=-1000

        self.current_sell_price_A=1000        
        self.current_sell_price_B=1000
        self.current_sell_price_C=1000

    def clear_previous_round(self):
        # Clear previous round's bids for each token type
        for bid_key in self.bid_previous_round_A:
            if bid_key in self.bid_dictionary_A:
                del self.bid_dictionary_A[bid_key]
        
        for bid_key in self.bid_previous_round_B:
            if bid_key in self.bid_dictionary_B:
                del self.bid_dictionary_B[bid_key]
        
        for bid_key in self.bid_previous_round_C:
            if bid_key in self.bid_dictionary_C:
                del self.bid_dictionary_C[bid_key]

        # Clear previous round's sells for each token type
        for sell_key in self.sell_previous_round_A:
            if sell_key in self.sell_dictionary_A:
                del self.sell_dictionary_A[sell_key]
        
        for sell_key in self.sell_previous_round_B:
            if sell_key in self.sell_dictionary_B:
                del self.sell_dictionary_B[sell_key]
        
        for sell_key in self.sell_previous_round_C:
            if sell_key in self.sell_dictionary_C:
                del self.sell_dictionary_C[sell_key]

    def end_round(self):
        pass

    def update_bid_sell_price(self):
        # For A
        if self.bid_dictionary_A:
            self.current_bid_price_A = max(self.bid_dictionary_A.keys())
        if self.sell_dictionary_A:
            self.current_sell_price_A = min(self.sell_dictionary_A.keys())

        # For B
        if self.bid_dictionary_B:
            self.current_bid_price_B = max(self.bid_dictionary_B.keys())
        if self.sell_dictionary_B:
            self.current_sell_price_B = min(self.sell_dictionary_B.keys())

        # For C
        if self.bid_dictionary_C:
            self.current_bid_price_C = max(self.bid_dictionary_C.keys())
        if self.sell_dictionary_C:
            self.current_sell_price_C = min(self.sell_dictionary_C.keys())