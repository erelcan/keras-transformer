from keras import backend as K
import tensorflow as tf

from keras_transformer.utils import context_utils


def test_qk_mask():
    context_mask = context_utils.create_context_mask("left_context_mask", query_sequence_length=4, key_sequence_length=4)
    tf.print("context_mask:")
    tf.print(context_mask)

    Mq = K.constant([[True, True, False, False], [True, False, False, False]])
    Mk = K.constant([[True, False, True, False], [True, True, True, False]])

    tf.print("Mq:")
    tf.print(Mq)
    tf.print("Mk:")
    tf.print(Mk)

    padding_mask = K.batch_dot(K.expand_dims(Mq, axis=-1), K.expand_dims(Mk, axis=1))
    tf.print("padding_mask:")
    tf.print(padding_mask)

    tf.print("qk_mask:")
    qk_mask = (1 - (padding_mask * context_mask)) * -1e9
    tf.print(qk_mask)


# test_qk_mask()
