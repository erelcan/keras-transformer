from keras_transformer.utils.common_utils import get_class_instance_by_name


def create_optimizer(optimizer_info):
    instantiation_info = _optimizers[optimizer_info["type"]]
    return get_class_instance_by_name(instantiation_info["module"], instantiation_info["class_name"], optimizer_info["parameters"])


_optimizers = {
    "Adam": {"module": "keras.optimizers", "class_name": "Adam"}
}
