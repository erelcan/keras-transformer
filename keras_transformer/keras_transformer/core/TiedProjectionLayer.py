from keras.layers.wrappers import Wrapper
from keras import backend as K
from keras import activations


class TiedProjectionLayer(Wrapper):
    def __init__(self, layer, activation="linear", **kwargs):
        super().__init__(layer, **kwargs)

        # Assuming that the tied layer is an embedding layer.
        # Hence, using its weights named "embeddings".
        # Otherwise, ensure accessing to the right weights with appropriate naming.
        self.tied_to = layer
        self.activation = activations.get(activation)

    def build(self, input_shape=None):
        # Consider adding bias later on...
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = K.dot(inputs, K.transpose(self.tied_to.embeddings))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        configs = {
            "activation": self.activation
        }
        configs.update(super().get_config())
        return configs

    def get_weights(self):
        # May re-consider in case...
        return [K.get_value(K.transpose(self.tied_to.embeddings))]

    def set_wrapped_layer(self, layer):
        self.tied_to = layer
