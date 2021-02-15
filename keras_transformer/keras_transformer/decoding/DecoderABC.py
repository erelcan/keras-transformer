from abc import ABC, abstractmethod
from keras.models import load_model

from keras_transformer.utils.io_utils import load_from_pickle
from keras_transformer.training.custom_serialization.custom_object_handler import prepare_custom_objects


class DecoderABC(ABC):
    def __init__(self, model_path, artifacts_path):
        self._artifacts = self._load_artifacts(artifacts_path)
        self._model = self._load_model(model_path)

        # Re-consider direct load vs. re-creating from parameters~
        self._preprocessor = self._artifacts["preprocessor"]
        self._tag_ids = self._preprocessor.get_tag_ids()

    @abstractmethod
    def decode(self, data):
        pass

    def _load_model(self, model_path):
        model = load_model(model_path, custom_objects=prepare_custom_objects(self._artifacts["custom_objects_info"]), compile=False)
        model.compile()
        return model

    @staticmethod
    def _load_artifacts(artifacts_path):
        return load_from_pickle(artifacts_path)



