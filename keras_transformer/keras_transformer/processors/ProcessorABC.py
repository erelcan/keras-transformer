from abc import ABC, abstractmethod


class ProcessorABC(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, data, **kwargs):
        pass
