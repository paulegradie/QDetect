from abc import ABC, abstractmethod


class BaseContainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def show_random_sample_of_data(self):
        pass

    @abstractmethod
    def display_metadata(self):
        pass

    @abstractmethod
    def return_config(self):
        pass