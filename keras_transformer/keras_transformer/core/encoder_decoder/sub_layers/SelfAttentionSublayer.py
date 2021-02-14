from keras.layers import Layer, Add, Dropout
from keras_transformer.core.attention.MultiHeadAttention import MultiHeadAttention
from keras_transformer.core.encoder_decoder.misc.LayerNormalization import LayerNormalization


class SelfAttentionSublayer(Layer):
    def __init__(self, head_num, context_mask_type, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

        self._head_num = head_num
        self._context_mask_type = context_mask_type
        self._dropout_rate = dropout_rate

        # For transformer in the paper, only "narrow" head_type should be allowed since the adder would require same
        # feature length with the query.
        self._attention_layer = MultiHeadAttention(self._head_num, head_type="narrow", context_mask_type=self._context_mask_type)
        self._layer_normalization = LayerNormalization()
        self._adder = Add()
        self._dropout = Dropout(self._dropout_rate)

    def call(self, inputs, mask=None, **kwargs):
        output = self._attention_layer(inputs, mask=mask)
        output = self._dropout(output)
        # If inputs is a list, add Q in the adder. Assume that the inputs is [Q, K, V] if it is an array.
        # We should assert that fv equals to fq; otherwise adder will fail.
        if isinstance(inputs, list):
            output = self._adder([inputs[0], output])
        else:
            output = self._adder([inputs, output])
        output = self._layer_normalization(output)
        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        else:
            return input_shape

    def compute_mask(self, inputs, mask=None):
        # If mask is single tensor:
        # Mask on input will be used for queries, keys and values. Multi-head attention will return mask on queries.
        # Therefore, at the end we will produce the mask on queries, which is indeed the input mask!
        # If a list, we will return the first element which is the mask on queries.
        if isinstance(mask, list):
            return mask[0]
        else:
            return mask

    def get_config(self):
        config = {
            "head_num": self._head_num,
            "context_mask_type": self._context_mask_type,
            "dropout_rate": self._dropout_rate
        }
        base_config = super().get_config()
        config.update(base_config)

        return config
