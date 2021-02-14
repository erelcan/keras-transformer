from keras.layers import Layer

from keras_transformer.core.encoder_decoder.EncoderBlockStack import EncoderBlockStack
from keras_transformer.core.encoder_decoder.DecoderBlockStack import DecoderBlockStack


class EncoderDecoder(Layer):
    def __init__(self, d_model, num_of_blocks, attention_info, pff_info, input_dropout_rates, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

        self._d_model = d_model
        self._num_of_blocks = num_of_blocks
        self._attention_info = attention_info
        self._pff_info = pff_info
        self._input_dropout_rates = input_dropout_rates

        self._encoder_stack = EncoderBlockStack(self._d_model, self._num_of_blocks, self._attention_info, self._pff_info, self._input_dropout_rates["encoder"])
        self._decoder_stack = DecoderBlockStack(self._d_model, self._num_of_blocks, self._attention_info, self._pff_info, self._input_dropout_rates["decoder"])

    def call(self, inputs, mask=None, **kwargs):
        # For now, allowing only single encoder_input and decoder input rather than q/v/k for each.
        # First implementation has that feature which takes a list of 6 tensors (as an alternative)
        # and arranges the inputs and masks. However, for the sake of simplicity, we decided to remove that. In case,
        # you need such a feature, implement a new encoder-decoder or add input/mask arranger.
        #
        # input: [(b, tE, embE), (b, tD, embD)]
        # output: (b, tD, d_model)
        # where embE = embD = d_model

        encoder_input, decoder_input = inputs
        encoder_mask, decoder_mask = mask

        encoder_output = self._encoder_stack(encoder_input, mask=encoder_mask)
        encoder_output_mask = self._encoder_stack.compute_mask(encoder_input, encoder_mask)
        decoder_output = self._decoder_stack([encoder_output, decoder_input], mask=[encoder_output_mask, decoder_mask])

        return decoder_output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            if len(input_shape) == 2:
                batch_size = input_shape[0][0]
                decoder_sequence_length = input_shape[1][1]
                return batch_size, decoder_sequence_length, self._d_model
            else:
                raise Exception("Input to the transformer should be a list of 2.")
        else:
            raise Exception("Input to the transformer should be a list: [source_input, target_input]")

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            "d_model": self._d_model,
            "num_of_blocks": self._num_of_blocks,
            "attention_info": self._attention_info,
            "pff_info": self._pff_info
        }
        base_config = super().get_config()
        config.update(base_config)

        return config
