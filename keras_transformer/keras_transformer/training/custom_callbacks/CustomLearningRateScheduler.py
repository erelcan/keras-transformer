from keras.callbacks import Callback
from keras_transformer.training.custom_callbacks.CustomCallbackABC import CustomCallbackABC
from keras import backend as K


class CustomLearningRateScheduler(Callback, CustomCallbackABC):
    def __init__(self, d_model, warmup_steps):
        super().__init__()

        self._d_model = d_model
        self._warmup_steps = warmup_steps

        self.learning_rate_history = []

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # As epoch=0 results in division error, we add 1 to it for convenience.~
        step_num = epoch + 1

        lr = self._d_model ** -0.5 + min(step_num ** -0.5, step_num * (self._warmup_steps ** -1.5))
        self.learning_rate_history.append(lr)

        K.set_value(self.model.optimizer.lr, lr)

    def get_name(self):
        return self.__class__.__name__

    def get_artifacts(self):
        return {"learning_rate_history": self.learning_rate_history}

    def prepare_from_artifacts(self, artifacts):
        self.learning_rate_history = artifacts["learning_rate_history"]
