from keras.layers import Layer, Embedding
from keras import backend as K
from keras import activations
from keras.initializers import Constant

from keras_transformer.training.factories import custom_weight_factory


class TiedEmbedderProjector(Layer):
    def __init__(self, input_dim, output_dim, trainable, activation="linear",  weight_info=None, **kwargs):
        super().__init__(**kwargs)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._trainable = trainable
        self._activation = activations.get(activation)
        self._weight_info = weight_info

        self._projection_mode = False

        weight_initializer = None if self._weight_info is None else Constant(
            custom_weight_factory.get_weight(self._weight_info))
        self._embedder = Embedding(input_dim, output_dim, trainable=trainable, embeddings_initializer=weight_initializer)

    def call(self, inputs, projection_mode=False, **kwargs):
        if projection_mode:
            return self._handle_projection(inputs)
        else:
            return self._handle_embedding(inputs)

        # return K.switch(K.equal(K.constant(embed_mode), True), self._handle_embedding(inputs), self._handle_projection(inputs))

    def _handle_embedding(self, inputs):
        self._projection_mode = False
        return self._embedder(inputs)

    def _handle_projection(self, inputs):
        self._projection_mode = True
        output = K.dot(inputs, K.transpose(self._embedder.embeddings))
        if self._activation is not None:
            output = self._activation(output)
        return output

    def compute_output_shape(self, input_shape):
        embed_output_shape = self._embedder.compute_output_shape(input_shape)
        if self._embed_mode:
            return embed_output_shape
        else:
            # last one is vocabulary..
            return embed_output_shape[:-1] + self._embedder.input_dim

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
