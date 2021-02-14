from keras import backend as K
from keras.losses import categorical_crossentropy


def create_loss_function(loss_info):
    return _loss_functions[loss_info["type"]](loss_info["parameters"])


def _get_custom_crossentropy_loss(num_classes, mask_value=0, from_logits=False, label_smoothing=0.0):
    def inner_loss(y_true, y_pred):
        mask = K.cast_to_floatx(K.not_equal(y_true, mask_value))
        y_true_one_hot = K.one_hot(y_true, num_classes)

        loss = categorical_crossentropy(y_true_one_hot, y_pred, from_logits=from_logits, label_smoothing=label_smoothing)
        masked_loss = loss * mask
        reduced_loss = K.sum(masked_loss) / K.sum(mask)

        return reduced_loss

    return inner_loss


_loss_functions = {
    "custom_crossentropy": lambda parameters: _get_custom_crossentropy_loss(**parameters)
}
