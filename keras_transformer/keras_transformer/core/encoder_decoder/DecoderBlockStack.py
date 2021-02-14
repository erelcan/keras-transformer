from keras.layers import Layer, Dropout

from keras_transformer.core.encoder_decoder.DecoderBlock import DecoderBlock


class DecoderBlockStack(Layer):
    def __init__(self, d_model, num_of_blocks, attention_info, pff_info, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        if num_of_blocks < 1:
            raise Exception("There should be at least 1 block in the DecoderBlockStack")

        self.supports_masking = True

        self._d_model = d_model
        self._num_of_blocks = num_of_blocks
        self._attention_info = attention_info
        self._pff_info = pff_info
        self._dropout_rate = dropout_rate

        self._dropout = Dropout(self._dropout_rate)
        self._decoder_blocks = [DecoderBlock(self._d_model, self._attention_info, self._pff_info) for _ in range(self._num_of_blocks)]

    def call(self, inputs, mask=None, **kwargs):
        # Assuming that inputs is a tensor which will be copied to Q, K and V in multi-head attention.
        # We may customize this for taking a list of tensors representing Q, K and V which will be directly mapped
        # in multi-head attention.

        encoder_output, decoder_input = inputs
        encoder_output_mask, decoder_mask = mask

        cur_decoder_mask = decoder_mask
        cur_input = self._dropout(decoder_input)

        for i in range(self._num_of_blocks):
            output = self._decoder_blocks[i]([encoder_output, cur_input], [encoder_output_mask, cur_decoder_mask])
            cur_decoder_mask = self._decoder_blocks[i].compute_mask([encoder_output, cur_input], mask=[encoder_output_mask, cur_decoder_mask])
            cur_input = output

        return cur_input

    def compute_output_shape(self, input_shape):
        # Rather than manually setting the output_shape, let's compute it implicitly.

        encoder_output_shape, decoder_output_shape = input_shape
        for i in range(self._num_of_blocks):
            decoder_output_shape = self._decoder_blocks[i].compute_output_shape([encoder_output_shape, decoder_output_shape])
        return decoder_output_shape

    def compute_mask(self, inputs, mask=None):
        # We want this layer not to propagate the mask
        return None

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
