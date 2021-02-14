from keras import backend as K
from keras.layers import Layer


class PositionalEncodingLayer(Layer):
    def __init__(self, encoding_type=0, min_timescale=1.0, max_timescale=1.0e4, **kwargs):
        super().__init__(**kwargs)
        self._encoding_type = encoding_type
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale
        self._signal = None

    def build(self, input_shape=None):
        seq_length = input_shape[-2]
        encoding_length = input_shape[-1]
        if self._encoding_type == 0:
            self._signal = self._get_positional_encoding(seq_length, encoding_length)
        else:
            self._signal = self._get_positional_encoding2(seq_length, encoding_length, self._min_timescale, self._max_timescale)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self._signal

    def get_config(self):
        configs = {
            "encoding_type": self._encoding_type,
            "min_timescale": self._min_timescale,
            "max_timescale": self._max_timescale
        }
        configs.update(super().get_config())
        return configs

    @staticmethod
    def _get_positional_encoding(max_seq_length, encoding_length):
        seq_positions = K.expand_dims(K.cast_to_floatx(K.arange(0, max_seq_length)), 1)
        sin_locations = K.cast_to_floatx(K.arange(0, encoding_length, 2))
        cos_locations = K.cast_to_floatx(K.arange(1, encoding_length, 2))

        div_term_const = K.log(K.constant(10000)) * K.constant(-2 / encoding_length)

        sin_encodings = K.sin(seq_positions * K.expand_dims(K.exp(sin_locations * div_term_const), 0))
        cos_encodings = K.cos(seq_positions * K.expand_dims(K.exp(cos_locations * div_term_const), 0))

        expanded_sin_encodings = K.expand_dims(sin_encodings, 2)
        expanded_cos_encodings = K.expand_dims(cos_encodings, 2)

        # Trick to alternate sines and cosines~
        concatenated_encodings = K.concatenate([expanded_sin_encodings, expanded_cos_encodings])
        new_shape = concatenated_encodings.shape[:-1].as_list()
        new_shape[-1] *= 2
        final_encoding = K.reshape(concatenated_encodings, new_shape)

        return final_encoding

    @staticmethod
    def _get_positional_encoding2(max_seq_length, encoding_length, min_timescale, max_timescale):
        # Similar to https://github.com/tensorflow/tensor2tensor/blob/5f9dd2db6d7797162e53adf152310ed13e9fc711/tensor2tensor/layers/common_attention.py

        seq_positions = K.expand_dims(K.cast_to_floatx(K.arange(0, max_seq_length)), 1)
        num_of_scales = encoding_length / 2
        scales_locations = K.cast_to_floatx(K.arange(0, num_of_scales))

        div_term_const = K.constant(-1) * (K.log(K.constant(max_timescale / min_timescale)) / K.maximum(K.constant(num_of_scales), 1))

        sin_encodings = K.sin(seq_positions * K.expand_dims(K.constant(min_timescale) * K.exp(scales_locations * div_term_const), 0))
        cos_encodings = K.cos(seq_positions * K.expand_dims(K.constant(min_timescale) * K.exp(scales_locations * div_term_const), 0))

        expanded_sin_encodings = K.expand_dims(sin_encodings, 2)
        expanded_cos_encodings = K.expand_dims(cos_encodings, 2)

        # Trick to alternate sines and cosines~
        concatenated_encodings = K.concatenate([expanded_sin_encodings, expanded_cos_encodings])
        new_shape = concatenated_encodings.shape[:-1].as_list()
        new_shape[-1] *= 2
        final_encoding = K.reshape(concatenated_encodings, new_shape)

        return final_encoding

    def get_signal(self):
        return self._signal
