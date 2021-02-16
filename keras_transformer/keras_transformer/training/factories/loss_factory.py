from keras_transformer.utils.common_utils import get_function_by_name


def create_loss(loss_info):
    if loss_info["is_custom"]:
        return get_function_by_name("keras_transformer.training.custom_losses.loss_functions", loss_info["type"])(**loss_info["parameters"])
    else:
        # May consider returning instantiations rather than string..
        return loss_info["type"]
