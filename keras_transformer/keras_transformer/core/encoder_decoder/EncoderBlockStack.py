from keras.layers import Layer, Dropout

from keras_transformer.core.encoder_decoder.EncoderBlock import EncoderBlock


class EncoderBlockStack(Layer):
    def __init__(self, d_model, num_of_blocks, attention_info, pff_info, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        if num_of_blocks < 1:
            raise Exception("There should be at least 1 block in the EncoderBlockStack")

        self.supports_masking = True

        self._d_model = d_model
        self._num_of_blocks = num_of_blocks
        self._attention_info = attention_info
        self._pff_info = pff_info
        self._dropout_rate = dropout_rate

        self._dropout = Dropout(self._dropout_rate)
        self._encoder_blocks = [EncoderBlock(self._d_model, self._attention_info, self._pff_info) for _ in range(self._num_of_blocks)]

    def call(self, inputs, mask=None, **kwargs):
        # Assuming that inputs is a tensor which will be copied to Q, K and V in multi-head attention.
        # We may customize this for taking a list of tensors representing Q, K and V which will be directly mapped
        # in multi-head attention.

        # Keras should handle auto-mask passing, but we want to make it explicit to be sure.

        cur_mask = mask
        cur_input = self._dropout(inputs)

        for i in range(self._num_of_blocks):
            output = self._encoder_blocks[i](cur_input, mask=cur_mask)
            cur_mask = self._encoder_blocks[i].compute_mask(cur_input, mask=cur_mask)
            cur_input = output

        return cur_input

    def compute_output_shape(self, input_shape):
        # Rather than manually setting the output_shape, let's compute it implicitly.
        output_shape = input_shape
        for i in range(self._num_of_blocks):
            output_shape = self._encoder_blocks[i].compute_output_shape(output_shape)
        return output_shape

    def compute_mask(self, inputs, mask=None):
        # It may not be appropriate to compute the mask as in the computation of output_shape since computing inputs
        # for each layer may be costly and prone to error.
        # Hence, use the prior knowledge:
        # Mask on input will be used for queries, keys and values. Multi-head attention will return mask on queries.
        # Feed forward sublayer will also carry the same masking as well. Therefore, at the end we will produce the mask
        # on queries, which is indeed the input mask!
        return mask

    def get_config(self):
        config = {
            "d_model": self._d_model,
            "num_of_blocks": self._num_of_blocks,
            "attention_info": self._attention_info,
            "pff_info": self._pff_info,
            "dropout_rate": self._dropout_rate
        }
        base_config = super().get_config()
        config.update(base_config)

        return config
