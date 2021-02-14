from keras import backend as K
from keras_transformer.utils.context_utils import create_context_mask
import tensorflow as tf


def test_context_mask():
    batch_size = 3
    query_seq_length = 4
    key_sequence_length = 4

    inside_softmax_shape = (batch_size, query_seq_length, key_sequence_length)
    inside_softmax = K.reshape(K.cast_to_floatx(K.arange(1, batch_size * query_seq_length * key_sequence_length + 1)), inside_softmax_shape)
    tf.print("inside_softmax:")
    tf.print(inside_softmax)

    mask = create_context_mask("left_context_mask", query_sequence_length=query_seq_length, key_sequence_length=key_sequence_length)
    tf.print("mask:")
    tf.print(mask)

    inside_softmax *= mask

    tf.print("inside_softmax after masking:")
    tf.print(inside_softmax)


# test_context_mask()
