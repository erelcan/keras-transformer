from keras.layers import Layer

from keras_transformer.core.encoder_decoder.sub_layers.SelfAttentionSublayer import SelfAttentionSublayer
from keras_transformer.core.encoder_decoder.sub_layers.PositionWiseFeedForwardSublayer import PositionWiseFeedForwardSublayer


class DecoderBlock(Layer):
    def __init__(self, d_model, attention_info, pff_info, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

        self._d_model = d_model
        self._attention_info = attention_info
        self._pff_info = pff_info

        # masked_self_attention masks the future for each query~. Hence, using left_context_mask.
        self._masked_self_attention = SelfAttentionSublayer(self._attention_info["head_num"], "left_context_mask", self._attention_info["dropout_rate"])
        self._self_attention = SelfAttentionSublayer(self._attention_info["head_num"], None, self._attention_info["dropout_rate"])
        self._position_wise_feed_forward = PositionWiseFeedForwardSublayer(self._d_model, self._pff_info["inner_length"], self._pff_info["dropout_rate"])

    def call(self, inputs, mask=None, **kwargs):
        # Assuming that inputs is a tensor which will be copied to Q, K and V in multi-head attention.
        # We may customize this for taking a list of tensors representing Q, K and V which will be directly mapped
        # in multi-head attention.

        # input: [encoder_output, decoder_input]
        # mask: [encoder_output_mask, decoder_mask]

        encoder_output, decoder_input = inputs
        encoder_output_mask, decoder_mask = mask

        # In sublayer2, encoder_output is used for keys and queries.
        sublayer1_output = self._masked_self_attention(decoder_input, mask=decoder_mask)
        sublayer2_output = self._self_attention([sublayer1_output, encoder_output, encoder_output], [decoder_mask, encoder_output_mask, encoder_output_mask])
        output = self._position_wise_feed_forward(sublayer2_output, mask=decoder_mask)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][1], self._d_model

    def compute_mask(self, inputs, mask=None):
        # The input mask is [encoder_output_mask, decoder_mask]
        # This part is tricky:
        # If we want to explicitly manage the encoder and decoder masks outside; this method should have no effect.
        # Depending on the usage, we may need to return both masks or just the one on the decoder output/sequence.
        # For now, we return the  decoder_mask only; for next decoder in decoder layer, mask will be prepared by
        # the DecoderStack.
        return mask[1]

    def get_config(self):
        config = {
            "d_model": self._d_model,
            "attention_info": self._attention_info,
            "pff_info": self._pff_info
        }
        base_config = super().get_config()
        config.update(base_config)

        return config
