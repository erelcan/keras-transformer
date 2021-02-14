import os
import io
import zipfile
from random import shuffle
from keras.utils import get_file

from keras_transformer.generators.outer.OuterGeneratorABC import OuterGeneratorABC
from keras_transformer.utils.common_utils import select_items


class TranslationDataGenerator(OuterGeneratorABC):
    def __init__(self, batch_size, zip_file_path, file_url, extraction_path, num_of_samples=None, shuffle_on=True, id_list=None):
        super().__init__()
        # Assuming that data fits in memory. Otherwise, change data format; and use dask etc.
        self._batch_size = batch_size
        self._shuffle_on = shuffle_on

        self._zip_file_path = zip_file_path
        self._file_url = file_url
        self._extraction_path = extraction_path
        self._num_of_samples = num_of_samples

        self._source_sentences, self._target_sentences = self._create_dataset()

        self._num_of_batches = self._num_of_samples // self._batch_size
        self._remaining_size = self._num_of_samples % self._batch_size
        if id_list is None:
            self._id_list = list(range(self._num_of_samples))
        else:
            # In case, we would like to train with a given list of samples
            # (May be handy for handling evaluation split~).
            self._id_list = id_list

        if self._shuffle_on:
            shuffle(self._id_list)

        self._cur_pointer = 0

    def __next__(self):
        # May throw exception, if end of data..
        # For now, returning empty data..
        batch_indices = []
        while len(batch_indices) < self._batch_size and self._cur_pointer < self._num_of_samples:
            batch_indices.append(self._id_list[self._cur_pointer])
            self._cur_pointer += 1

        return select_items(self._source_sentences, batch_indices), select_items(self._target_sentences, batch_indices)

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

    def _download_and_extract_data(self):
        path_to_file = get_file(fname=self._zip_file_path, origin=self._file_url, extract=True)
        file_path = os.path.dirname(path_to_file) + self._extraction_path

        # For get_file, extract option is not working when absolute path is provided. Hence manually extracting the zip.
        with zipfile.ZipFile(self._zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(path_to_file))

        return file_path

    def _create_dataset(self):
        file_path = self._download_and_extract_data()
        lines = io.open(file_path, encoding='UTF-8').read().strip().split('\n')

        if self._num_of_samples is None:
            self._num_of_samples = len(lines)
        word_pairs = [[w for w in l.split('\t')] for l in lines[:self._num_of_samples]]

        return zip(*word_pairs)
