from keras_transformer.utils.common_utils import get_class_instance_by_name


# If you need to use a stateful keras callback, wrap it and save artifacts.
# Then, prepare its state after creation.
# Serialization is not straightforward for callbacks!
def create_callback(callback_info):
    callback = get_class_instance_by_name("keras.callbacks", callback_info["type"], callback_info["parameters"])
    return callback


def create_custom_callback(callback_info, callback_artifacts):
    # All custom callbacks must implement CustomCallbackABC!!!
    instantiation_info = _custom_callbacks[callback_info["type"]]
    callback = get_class_instance_by_name(instantiation_info["module"], instantiation_info["class_name"], callback_info["parameters"])
    if callback.get_name() in callback_artifacts:
        callback.prepare_from_artifacts(callback_artifacts[callback.get_name()])
    return callback


def create_callbacks(callbacks_info, callback_artifacts, return_custom_callbacks=True):
    # Assuming that all built-in callbacks are stateless.
    # Otherwise, wrap them into custom callbacks and implement necessary methods for serialization!
    callbacks = []
    custom_callbacks = []
    for info in callbacks_info:
        is_custom = info.get("is_custom", False)
        if is_custom:
            custom_callbacks.append(create_custom_callback(info, callback_artifacts))
        else:
            callbacks.append(create_callback(info))

    callbacks += custom_callbacks
    if return_custom_callbacks:
        return callbacks, custom_callbacks
    else:
        return callbacks


_custom_callbacks = {
    "CustomLearningRateScheduler": {"module": "keras_transformer.training.CustomLearningRateScheduler", "class_name": "CustomLearningRateScheduler"}
}
