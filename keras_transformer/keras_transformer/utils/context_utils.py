from keras import backend as K


# As a convention False/0 will represent missing/padded whereas True/1 for existing values~.
# Hence, account for 1s, handle 0s.
def create_context_mask(mask_type, **kwargs):
    return context_mask_creators[mask_type](**kwargs)


def _create_left_context_mask(query_sequence_length, key_sequence_length):
    # For each query, only the keys before it (and itself) should contribute to its attention-computation.
    # We can create a lower triangular matrix as follows.
    # E.g. first row has 1 only for its first element since no left context than itself.
    # E.g. second row has 1 for first 2 elements, 1st key and 2nd key (itself) contributes to attention computation.

    # Returns mask has a 2D shape of (query_sequence_length, key_sequence_length)
    # Masked element are zeros, others are ones!
    key_indices = K.expand_dims(K.arange(0, key_sequence_length), axis=0)
    query_indices = K.expand_dims(K.arange(0, query_sequence_length), axis=-1)
    return K.cast(key_indices <= query_indices, K.floatx())


context_mask_creators = {
    "left_context_mask": _create_left_context_mask
}
