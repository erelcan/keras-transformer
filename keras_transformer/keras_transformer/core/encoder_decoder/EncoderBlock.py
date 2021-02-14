from keras.layers import Layer

from keras_transformer.core.encoder_decoder.sub_layers.SelfAttentionSublayer import SelfAttentionSublayer
from keras_transformer.core.encoder_decoder.sub_layers.PositionWiseFeedForwardSublayer import PositionWiseFeedForwardSublayer


class EncoderBlock(Layer):
    def __init__(self, d_model, attention_info, pff_info, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

        self._d_model = d_model
        self._attention_info = attention_info
        self._pff_info = pff_info

        # No context mask is needed for the encoder.
        self._self_attention = SelfAttentionSublayer(self._attention_info["head_num"], None, self._attention_info["dropout_rate"])
        self._position_wise_feed_forward = PositionWiseFeedForwardSublayer(self._d_model, self._pff_info["inner_length"], self._pff_info["dropout_rate"])

    def call(self, inputs, mask=None, **kwargs):
        # Assuming that inputs is a tensor which will be copied to Q, K and V in multi-head attention.
        # We may customize this for taking a list of tensors representing Q, K and V which will be directly mapped
        # in multi-head attention.
        output = self._self_attention(inputs, mask=mask)
        output = self._position_wise_feed_forward(output, mask=mask)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self._d_model

    def compute_mask(self, inputs, mask=None):
        # Mask on input will be used for queries, keys and values. Multi-head attention will return mask on queries.
        # Feed forward sublayer will also carry the same masking as well. Therefore, at the end we will produce the mask
        # on queries, which is indeed the input mask!
        return mask

    def get_config(self):
        config = {
            "d_model": self._d_model,
            "attention_info": self._attention_info,
            "pff_info": self._pff_info
        }
        base_config = super().get_config()
        config.update(base_config)

        return config
