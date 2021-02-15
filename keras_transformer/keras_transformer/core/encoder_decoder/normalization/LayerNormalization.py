from keras import backend as K
from keras.layers import Layer
from keras.initializers import Ones, Zeros


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self._eps = eps
        self._gamma = None
        self._beta = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self._gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self._beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # In layer normalization, instances are handled independent of each other.
        # Normalize on the feature (last) dimension for each instance.
        # Not handling masks for now..
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        return self._gamma * (inputs - mean) / (std + self._eps) + self._beta

    def compute_output_shape(self, input_shape):
        return input_shape
