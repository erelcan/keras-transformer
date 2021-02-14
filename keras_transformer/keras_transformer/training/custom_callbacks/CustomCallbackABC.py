from abc import ABC, abstractmethod


class CustomCallbackABC(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_artifacts(self):
        pass

    @abstractmethod
    def prepare_from_artifacts(self, artifacts):
        pass
