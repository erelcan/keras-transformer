from random import shuffle

from keras_transformer.generators.outer.OuterGeneratorABC import OuterGeneratorABC


# Consider renaming~
# Not exactly like a generator; but gives more control?
# Add input validation~
class PandasGenerator(OuterGeneratorABC):
    def __init__(self, df, batch_size, target_df=None, return_targets=False, shuffle_on=True):
        super().__init__()
        self._data = df
        self._target_data = target_df
        self._batch_size = batch_size
        self._return_targets = return_targets
        self._shuffle_on = shuffle_on

        self._num_of_batches = len(self._data.index) // self._batch_size
        self._remaining_size = len(self._data.index) % self._batch_size
        self._id_list = list(range(len(self._data.index)))

        if self._shuffle_on:
            shuffle(self._id_list)

        self._cur_pointer = 0

    def __next__(self):
        # May throw exception, if end of data..
        # For now, returning empty data..
        batch_indices = []
        while len(batch_indices) < self._batch_size and self._cur_pointer < len(self._data.index):
            batch_indices.append(self._id_list[self._cur_pointer])
            self._cur_pointer += 1

        # If we can ensure, no modification on the returned value, then we can turn copy off.
        # However, this is safer for now..)
        if self._return_targets:
            return self._data.iloc[batch_indices].to_numpy(copy=True), self._target_data.iloc[batch_indices].to_numpy(copy=True)
        else:
            return self._data.iloc[batch_indices].to_numpy(copy=True)

    def __iter__(self):
        return self

    def __len__(self):
        # Returns number of batches, excluding the remaining.
        return self._num_of_batches

    def refresh(self):
        self._cur_pointer = 0
        if self._shuffle_on:
            shuffle(self._id_list)

    def get_remaining_size(self):
        return self._remaining_size
