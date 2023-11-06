import json
from typing import Optional


class Configuration:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as config_file:
            return json.load(config_file)
    def get_constant(self, const_name):
        return self.config_data.get(const_name)

config: Optional[Configuration] = None