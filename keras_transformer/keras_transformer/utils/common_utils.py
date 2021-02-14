import numpy as np
from operator import itemgetter


def get_class_instance_by_name(module_name, class_name, class_args):
    module = __import__(module_name, fromlist=[''])
    my_class = getattr(module, class_name)(**class_args)
    return my_class


def select_items(given_list, indices):
    if len(indices) == 1:
        return [given_list[indices[0]]]
    else:
        return list(itemgetter(*indices)(given_list))


def get_topk_with_indices(np_array, k):
    top_indices = np.flip(np_array.argsort()[-k:])
    top_values = np_array[top_indices]
    return top_values, top_indices
