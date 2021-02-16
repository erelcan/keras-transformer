import os
from abc import ABC, abstractmethod
from keras.models import load_model

from keras_transformer.generators.InnerGenerator import InnerGenerator
from keras_transformer.training.factories.loss_factory import create_loss
from keras_transformer.training.factories.optimizer_factory import create_optimizer
from keras_transformer.training.factories.callback_factory import create_callbacks
from keras_transformer.training.custom_callbacks.CustomCheckpointer import CustomCheckpointer
from keras_transformer.utils.io_utils import load_from_pickle
from keras_transformer.training.custom_serialization.custom_object_handler import prepare_custom_objects


class TrainerABC(ABC):
    def __init__(self, preprocessor, outer_generator, generator_info, model_info, checkpointing_info):
        self._preprocessor = preprocessor
        self._data_generator = InnerGenerator(outer_generator, self._preprocessor, generator_info.get("pass_count", None), generator_info.get("use_remaining", None))
        self._model_info = model_info
        self._checkpointing_info = checkpointing_info
        self._workspace_path = self._checkpointing_info.pop("workspace_path")

        self._artifacts = None
        self._checkpointer = None

        if "continue_from" in self._checkpointing_info:
            self._artifacts = load_from_pickle(os.path.join(self._workspace_path, "artifacts-" + str(self._checkpointing_info["continue_from"]) + ".pkl"))
        else:
            self._artifacts = {"preprocessor": self._preprocessor, "custom_objects_info": {"loss_info": self._model_info["loss_info"]}, "callbacks": {}}

    def train(self):
        if "continue_from" in self._checkpointing_info:
            model = self._load_model(os.path.join(self._workspace_path, "model-" + str(self._checkpointing_info["continue_from"]) + ".h5"))
        else:
            model = self._get_model()
            model.compile(optimizer=create_optimizer(self._model_info["optimizer_info"]), loss=create_loss(self._model_info["loss_info"]), metrics=self._model_info.get("metrics", None))

        model.summary()

        model.fit(self._data_generator.get_generator(), initial_epoch=self._checkpointing_info.get("continue_from", 0), steps_per_epoch=self._data_generator.get_num_of_steps(), callbacks=self._prepare_callbacks(), **self._model_info["fit_parameters"])

        return model

    def _prepare_callbacks(self):
        callbacks, custom_callbacks = create_callbacks(self._model_info["callback_info"], self._artifacts["callbacks"])

        self._checkpointer = CustomCheckpointer(self._workspace_path, self._artifacts, custom_callbacks, **self._checkpointing_info)
        if "continue_from" in self._checkpointing_info:
            self._checkpointer.prepare_from_artifacts(self._artifacts["callbacks"][self._checkpointer.get_name()])
        self._checkpointer.update_artifacts()

        return callbacks + [self._checkpointer]

    def _load_model(self, model_path):
        return load_model(model_path, custom_objects=prepare_custom_objects(self._artifacts["custom_objects_info"]))

    @abstractmethod
    def _get_model(self):
        pass
