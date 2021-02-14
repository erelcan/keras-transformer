import os
from keras.callbacks import ModelCheckpoint
from keras_transformer.training.custom_callbacks.CustomCallbackABC import CustomCallbackABC

from keras_transformer.utils.io_utils import save_to_pickle


class CustomCheckpointer(ModelCheckpoint, CustomCallbackABC):
    def __init__(self, workspace_path, artifacts, callbacks, **kwargs):
        super().__init__(os.path.join(workspace_path, "model-{epoch:01d}.h5"), **kwargs)
        self._workspace_path = workspace_path
        self._artifacts = artifacts
        self._completed_epoch = 0
        self._callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        self._completed_epoch += 1

        self.update_artifacts()

        should_save = False
        if self.epochs_since_last_save == 0:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current == self.best:
                    should_save = True
            else:
                should_save = True

        if should_save:
            save_to_pickle(self._artifacts, os.path.join(self._workspace_path, "artifacts-" + str(epoch+1) + ".pkl"))

    def update_artifacts(self):
        for callback in self._callbacks:
            self._artifacts["callbacks"][callback.get_name()] = callback.get_artifacts()

        self._artifacts["callbacks"][self.get_name()] = self.get_artifacts()

    def get_name(self):
        return self.__class__.__name__

    def get_artifacts(self):
        return {"best_score": self.best, "completed_epoch": self._completed_epoch}

    def prepare_from_artifacts(self, artifacts):
        self.best = artifacts["best_score"]
        self._completed_epoch = artifacts["completed_epoch"]
