from keras import backend as K
import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical


def custom_crossentropy(y_true, y_pred, num_classes, mask_value=0, from_logits=False, label_smoothing=0.0):
    mask = K.cast_to_floatx(K.not_equal(y_true, mask_value))
    tf.print("mask:")
    tf.print(mask)

    y_true_one_hot = K.one_hot(y_true, num_classes)
    tf.print("y_true_one_hot:")
    tf.print(y_true_one_hot)

    loss = categorical_crossentropy(y_true_one_hot, y_pred, from_logits=from_logits, label_smoothing=label_smoothing)
    tf.print("loss:")
    tf.print(loss)

    masked_loss = loss * mask
    tf.print("masked_loss:")
    tf.print(masked_loss)

    tf.print("sum_loss:")
    tf.print(K.sum(masked_loss))

    tf.print("sum_mask:")
    tf.print(K.sum(mask))

    reduced_loss = K.sum(masked_loss) / K.sum(mask)
    tf.print("reduced_loss:")
    tf.print(reduced_loss)

    return reduced_loss


def test_masked_loss():
    y_true = [[1, 2, 2], [1, 0, 0]]
    y_pred = to_categorical([[1, 2, 3], [1, 2, 1]], num_classes=4)
    custom_crossentropy(y_true, y_pred, 4)

    y_true = [[1, 0, 2], [1, 2, 0]]
    y_pred = to_categorical([[1, 2, 3], [1, 2, 1]], num_classes=4)
    custom_crossentropy(y_true, y_pred, 4)


def test_label_smoothing():
    y_true = [[1, 2, 2], [1, 0, 0]]
    y_pred = to_categorical([[1, 2, 2], [1, 2, 1]], num_classes=4)

    custom_crossentropy(y_true, y_pred, 4, label_smoothing=0.0)
    custom_crossentropy(y_true, y_pred, 4, label_smoothing=0.1)
    custom_crossentropy(y_true, y_pred, 4, label_smoothing=0.8)


# test_masked_loss()
# test_label_smoothing()
