import json
from typing import Optional


class Configuration:
    """
    A simple class for loading and accessing configuration data from a JSON file.
    """

    def __init__(self, config_file):
        """
        Initializes the Configuration object with the specified configuration file path.

        Args:
            config_file: The path to the JSON configuration file.
        """
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self) -> dict:
        """
        Reads the configuration data from the JSON file and returns it as a dictionary.

        Returns: The loaded configuration data.
        """
        with open(self.config_file, 'r') as config_file:
            return json.load(config_file)

    def get_constant(self, const_name):
        """
        Retrieves the value of a constant from the loaded configuration data.

        Args:
            const_name: the name of the constant to retrieve.

        Returns: The value of the constant if it exists, otherwise None.
        """
        return self.config_data.get(const_name)


config: Optional[Configuration] = None
