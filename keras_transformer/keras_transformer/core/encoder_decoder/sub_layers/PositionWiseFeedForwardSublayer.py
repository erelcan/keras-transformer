from keras.layers import Layer, Dense, Add, Dropout
from keras_transformer.core.encoder_decoder.misc.LayerNormalization import LayerNormalization


class PositionWiseFeedForwardSublayer(Layer):
    def __init__(self, d_model, inner_length, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

        self._d_model = d_model
        self._inner_length = inner_length
        self._dropout_rate = dropout_rate

        # We might use conv1d as stated in paper rather than dense layers; but dense layers seems to perform faster~
        # self._layer1 = Conv1D(self._inner_length, 1, activation="relu")
        # self._layer2 = Conv1D(self._d_model, 1)

        # Using ReLu due to truncated activation.
        self._dense1 = Dense(self._inner_length, activation="relu")
        # Activation should be None for the second layer.
        self._dense2 = Dense(self._d_model)

        self._layer_normalization = LayerNormalization()
        self._adder = Add()
        self._dropout = Dropout(dropout_rate)

    def call(self, inputs, mask=None, **kwargs):
        # Input shape will be equal to the output shape.
        # Input mask and output mask will have the same shape.
        # input: (bs, tq, fv)
        #
        # We may apply the following masking, but it seems redundant since the linear transformations on the last
        # dimension. Hence, masked time-steps do not effect each other!
        # if mask is not None:
        #     inputs *= K.cast(K.expand_dims(mask, axis=-1), K.floatx())

        output = self._dense1(inputs)
        output = self._dense2(output)
        output = self._dropout(output)
        output = self._adder([inputs, output])
        output = self._layer_normalization(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = {
            'd_model': self._d_model,
            'inner_length': self._inner_length,
            'dropout_rate': self._dropout_rate
        }
        base_config = super().get_config()
        config.update(base_config)

        return config
