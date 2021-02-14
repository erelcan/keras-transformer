from keras_transformer.utils import context_utils

from keras import backend as K
from keras.layers import Layer


class DotProductAttention(Layer):
    def __init__(self, return_attention=False, context_mask_type=None, should_scale=True, **kwargs):
        super().__init__(**kwargs)

        self._return_attention = return_attention
        # Use context_mask to decide which keys to mask when computing attention for a query.
        # For example, left-context-mask can be used to prevent contribution of future keys. (Which is used for
        # computing attention for the decoder: a.k.a. masked-attention)
        # For re-usability, in case there are other strategies to mask, we get it as argument rather than fixing one.
        self._context_mask_type = context_mask_type

        # To scale inside of softmax or not
        self._should_scale = should_scale

    def call(self, inputs, mask=None, **kwargs):
        # Assumes that inputs is a list/tuple, it has query (Q), keys (K) and values (V)
        # Otherwise, raises error!
        # Any custom projection is done on Q, K and V before calling this method.
        #
        # Shape of Q, K and V are (batch_size * head_num, seq_len, head_length).
        # Note that this method is agnostic of heads, therefore we should prepare Q, K and V appropriately before
        # inputting them into this method. Corresponding weights should be created outside; and reshaped according to
        # the number of heads. (This allows re-usability)
        # Choice to apply wide or narrow attention is up to the caller of this method.

        if isinstance(inputs, list):
            queries, keys, values = inputs
        else:
            raise Exception("DotProductAttention requires inputs to be a list of 3: [Q, K, V]")

        if mask is None:
            mask_q = mask_k = mask_v = mask
        else:
            if isinstance(mask, list):
                mask_q, mask_k, mask_v = mask
            else:
                raise Exception("DotProductAttention requires inputs to be a list of 3: [Q, K, V]")

        # At this point queries and keys should have a shape of (batch_size, sequence_length, feature_length).
        # batch_dot will keep the batch dimension fixed, the dimensions given by the axes will be reduced.
        # Specifying the axes removes the need for transposing the keys tensor.
        #
        # scaler is as defined in the paper to counteract the effect of large dot products.

        if self._should_scale:
            # feature_length: dk in paper...
            feature_length = K.shape(queries)[-1]
            scaler = K.sqrt(K.cast(feature_length, dtype=K.floatx()))
            inside_softmax = K.batch_dot(queries, keys, axes=2) / scaler
        else:
            inside_softmax = K.batch_dot(queries, keys, axes=2)

        # Now, we may apply any context-based masking at this stage.
        # Then, we may compute softmax.

        # Assumed shape is (batch_size * (head_num or 1), sequence_length_of_queries, sequence_length_of_keys)
        # Hence, we need to expand_dims of the mask (in batch dimension).
        # context_mask (e.g. for left_context_mask) is 2D but applied to each batch element automatically.
        qk_mask = None
        if self._context_mask_type is not None:
            qk_mask = context_utils.create_context_mask(self._context_mask_type, query_sequence_length=queries.shape[1], key_sequence_length=keys.shape[1])

        if mask_q is not None and mask_k is not None:
            padding_mask = K.batch_dot(K.expand_dims(mask_q, axis=-1), K.expand_dims(mask_k, axis=1))
            if qk_mask is None:
                qk_mask = (1 - padding_mask) * -1e9
            else:
                qk_mask = (1 - (padding_mask * qk_mask)) * -1e9

        if qk_mask is not None:
            inside_softmax += qk_mask

        attention_weights = K.softmax(inside_softmax)

        if mask_v is not None:
            # This time, we just zero out "values" which has False/0 in the mask.
            values *= K.tile(K.expand_dims(mask_v), (1, 1, K.shape(values)[-1]))

        output = K.batch_dot(attention_weights, values)

        # "attention_weights" represents the weights of each pair of queries and keys, in the sequence.

        if self._return_attention:
            return [output, attention_weights]

        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            raise Exception("DotProductAttention requires inputs to be a list of 3: [Q, K, V]")

        # Let shapeQ = (b, tq, fq), shapeK = (b, tk, fk) and shapeV = (b, tv, fv)
        # where b is batch_size or batch_size * head_num
        # t represents sequence length and f represents embedding/feature size
        #
        # QK and softmax(QK/sqrt(fq)) have shape: (b, tq, tk)
        # soft(QK/sqrt(fq))V has shape: (b, tq, fv)
        # Also note that tk must be equal to tv.
        #
        # Hence, output shape will be (b, tq, fv) whereas attention shape is (b, tq, tk)

        output_shape = query_shape[:-1] + (value_shape[-1],)
        if self._return_attention:
            attention_shape = query_shape[:-1] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        # Returns mask on queries..
        if isinstance(mask, list):
            return mask[0]
        else:
            raise Exception("DotProductAttention requires masks to be a list of 3: [Q, K, V]")

    def get_config(self):
        config = {
            'return_attention': self._return_attention,
            'context_mask_type': self._context_mask_type,
            'should_scale': self._should_scale
        }
        base_config = super().get_config()
        config.update(base_config)

        return config
