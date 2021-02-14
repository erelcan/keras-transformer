import math


class InnerGenerator(object):
    def __init__(self, outer_generator, preprocessor, pass_count=None, use_remaining=True):
        super().__init__()
        self._outer_generator = outer_generator
        self._preprocessor = preprocessor
        self._pass_count = math.inf if pass_count is None else pass_count
        self._use_remaining = use_remaining

        self._num_of_batches = len(self._outer_generator)
        self._remaining_size = self._outer_generator.get_remaining_size()
        self._cur_pass = 0

    def get_generator(self):
        while self._cur_pass < self._pass_count:
            yield from self._one_pass()
            self._outer_generator.refresh()

    def _one_pass(self):
        for i in range(self._num_of_batches):
            yield self._preprocessor.process(next(self._outer_generator))

        if self._remaining_size > 0 and self._use_remaining:
            yield self._preprocessor.process(next(self._outer_generator))

        self._cur_pass += 1

    def get_num_of_steps(self):
        if self._use_remaining:
            return self._num_of_batches + (1 if self._remaining_size > 0 else 0)
        else:
            return self._num_of_batches
