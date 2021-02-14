from abc import abstractmethod

from keras_transformer.processors.ProcessorABC import ProcessorABC


class DecodingCompatibleProcessorABC(ProcessorABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, data, usage="encoder", **kwargs):
        pass

    @abstractmethod
    def decode(self, data, usage="decoder", **kwargs):
        pass

    @abstractmethod
    def get_tag_ids(self, usage="decoder", **kwargs):
        # Specifically; start, end and pad tag ids of decoder
        # Re-consider unknown~
        pass

    @abstractmethod
    def get_max_seq_length(self, usage="decoder", **kwargs):
        pass
