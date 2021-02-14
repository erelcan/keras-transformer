from keras_transformer.utils.common_utils import get_class_instance_by_name


def create_processor(processor_info):
    instantiation_info = _processors[processor_info["type"]]
    return get_class_instance_by_name(instantiation_info["module"], instantiation_info["class_name"], processor_info["parameters"])


_processors = {
    "SubWordProcessor": {"module": "keras_transformer.processors.SubWordProcessor", "class_name": "SubWordProcessor"}
}
