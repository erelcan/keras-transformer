from keras.layers import Layer
from keras import backend as K
from keras import activations
from keras.initializers import Constant

from keras_transformer.training.factories import custom_weight_factory


class ProjectionLayer(Layer):
    def __init__(self, input_dim, output_dim, trainable=True, activation="linear", weight_info=None, **kwargs):
        super().__init__(**kwargs)
        # This is for projection layer, however we could also use a dense layer directly..
        # Assume that embedding weights is already transposed when passed here..
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._trainable = trainable
        self._activation = activations.get(activation)
        self._weight_info = weight_info

        self._kernel = None

    def build(self, input_shape=None):
        weight_initializer = None if self._weight_info is None else Constant(
            custom_weight_factory.get_weight(self._weight_info))
        self._kernel = self.add_weight(shape=(self._input_dim, self._output_dim), initializer=weight_initializer, trainable=self._trainable)

    def call(self, inputs, **kwargs):
        output = K.dot(inputs, self._kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + self._output_dim

    def get_config(self):
        configs = {
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "trainable": self._trainable,
            "activation": self._activation,
            "weight_info": self._weight_info
        }
        configs.update(super().get_config())
        return configs
