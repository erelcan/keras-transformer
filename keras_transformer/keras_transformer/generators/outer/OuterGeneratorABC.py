from abc import ABC, abstractmethod


class OuterGeneratorABC(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def refresh(self):
        pass
