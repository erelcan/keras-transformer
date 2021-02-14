from keras import backend as K
from keras.layers import Layer
from keras import activations, initializers, regularizers, constraints

from keras_transformer.core.attention.DotProductAttention import DotProductAttention


class MultiHeadAttention(Layer):
    def __init__(self, head_num, head_type="narrow", inner_embedding_length=None, output_length=None, context_mask_type=None, should_scale=True, activation="linear", use_bias=False, kernel_initializer="glorot_normal", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

        self._head_num = head_num
        # inner_embedding_length corresponds to d_model. When all the Q, K and V has same shape, we can extract it from
        # their shapes; but in case they have different shapes (somehow for some other applications); we should know the
        # inner_embedding_length and set shapes of weights accordingly (also these shapes may be given directly,
        # in the future~). Also, output_length will be used to determine the output length of the layer,
        # in other words length of last dimension of Wo. If they are None, initialize them to fv.
        self._inner_embedding_length = inner_embedding_length
        self._output_length = output_length
        self._head_type = head_type
        self._context_mask_type = context_mask_type
        self._should_scale = should_scale

        self._attention_layer = DotProductAttention(context_mask_type=self._context_mask_type, should_scale=self._should_scale)
        self._activation = activations.get(activation)

        self._use_bias = use_bias
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)

        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None

    def build(self, input_shape):
        if isinstance(input_shape, list):
            shape_q, shape_k, shape_v = input_shape
        else:
            shape_q = shape_k = shape_v = input_shape

        if self._inner_embedding_length is None:
            self._inner_embedding_length = shape_v[-1]
        if self._output_length is None:
            self._output_length = shape_v[-1]

        # Let inputs shapes to be: shape_q = (b, tq, fq), shape_k = (b, tk, fk) and shape_v = (b, tv, fv)
        # where b is batch_size, t represents sequence length and f represents embedding/feature size
        # Let weight shapes to be: shape_Wq = (d_model, dk), shape_Wk = (d_model, dk), shape_Wv = (d_model, dv)
        # shape_Wo = (head_num * dv, d_model)
        # According to the paper, fq = fk = fv = d_model, also dk = dv = d_model // head_num
        # After linear projections, projected input shapes should be:
        # shape_Pq = (b, tq, dk), shape_Pk = (b, tk, dk) and shape_Pv = (b, tv, dv)
        # dk can be different than dv as dk's cancel out when computing QK.
        #
        # Rather than handling heads in different tensors, we will represent them in a single tensor
        # (probably will reduce computation cost and will be more maintainable).
        # Also, we will support both wide and narrow attention.
        #
        # Weights will have dk and dv multiplied by head_num since we will handle all heads in single tensor.
        if self._head_type == "narrow":
            dk, dv = self._get_narrow_attention_shapes()
        else:
            dk, dv = self._get_wide_attention_shapes()

        self.Wq = self.add_weight(shape=(shape_q[-1], dk * self._head_num), initializer=self._kernel_initializer,
                                  regularizer=self._kernel_regularizer, constraint=self._kernel_constraint,
                                  name='%s_Wq' % self.name)
        self.Wk = self.add_weight(shape=(shape_k[-1], dk * self._head_num), initializer=self._kernel_initializer,
                                  regularizer=self._kernel_regularizer, constraint=self._kernel_constraint,
                                  name='%s_Wk' % self.name)
        self.Wv = self.add_weight(shape=(shape_v[-1], dv * self._head_num), initializer=self._kernel_initializer,
                                  regularizer=self._kernel_regularizer, constraint=self._kernel_constraint,
                                  name='%s_Wv' % self.name)
        self.Wo = self.add_weight(shape=(dv * self._head_num, self._output_length), initializer=self._kernel_initializer,
                                  regularizer=self._kernel_regularizer, constraint=self._kernel_constraint,
                                  name='%s_Wo' % self.name)

        if self._use_bias:
            self.bq = self.add_weight(shape=(dk * self._head_num,), initializer=self._kernel_initializer,
                                      regularizer=self._kernel_regularizer, constraint=self._kernel_constraint,
                                      name='%s_bq' % self.name)
            self.bk = self.add_weight(shape=(dk * self._head_num,), initializer=self._kernel_initializer,
                                      regularizer=self._kernel_regularizer, constraint=self._kernel_constraint,
                                      name='%s_bk' % self.name)
            self.bv = self.add_weight(shape=(dv * self._head_num,), initializer=self._kernel_initializer,
                                      regularizer=self._kernel_regularizer, constraint=self._kernel_constraint,
                                      name='%s_bv' % self.name)
            self.bo = self.add_weight(shape=(self._output_length,), initializer=self._kernel_initializer,
                                      regularizer=self._kernel_regularizer, constraint=self._kernel_constraint,
                                      name='%s_bo' % self.name)

        super().build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            assert len(inputs) == 3, "Length of inputs must be 3."
            queries, keys, values = inputs
        else:
            queries = keys = values = inputs

        if isinstance(mask, list):
            assert len(mask) == 3, "Length of mask must be 3."
            mask_q, mask_k, mask_v = mask
        else:
            mask_q = mask_k = mask_v = mask

        projected_queries = K.dot(queries, self.Wq)
        projected_keys = K.dot(keys, self.Wk)
        projected_values = K.dot(values, self.Wv)

        if self._use_bias:
            projected_queries += self.bq
            projected_keys += self.bk
            projected_values += self.bv

        # To have linear projection, activation should be "linear".
        if self._activation is not None:
            projected_queries = self._activation(projected_queries)
            projected_keys = self._activation(projected_keys)
            projected_values = self._activation(projected_values)

        attentded_values = self._attention_layer(
            inputs=[
                self._move_heads_to_batch_dimension(projected_queries, self._head_num),
                self._move_heads_to_batch_dimension(projected_keys, self._head_num),
                self._move_heads_to_batch_dimension(projected_values, self._head_num)
            ],
            mask=[
                self._create_heads_for_mask(mask_q, self._head_num),
                self._create_heads_for_mask(mask_k, self._head_num),
                self._create_heads_for_mask(mask_v, self._head_num)
            ]
        )

        # attentded_values has shape (batch_size * head_num, tq, dv)
        # after moving heads from batch dimension, shape is (batch_size, tq, dv * head_num)
        attentded_values = self._move_heads_from_batch_dimension(attentded_values, self._head_num)

        output = K.dot(attentded_values, self.Wo)
        if self._use_bias:
            output += self.bo
        if self._activation is not None:
            output = self._activation(output)

        # output shape is (batch_size, tq, self._output_length)
        return output

    def compute_output_shape(self, input_shape):
        # Note that we keep the output size flexible and to be determined by the user. In the paper, the last dimension
        # of the output equals to the last dimension of the values.
        # output shape is (batch_size, tq, self._output_length)
        if isinstance(input_shape, list):
            shape_q, shape_k, shape_v = input_shape
            return shape_q[:-1] + (self._output_length,)
        return input_shape[:-1] + (self._output_length,)

    def compute_mask(self, inputs, input_mask=None):
        # As output of this layer has a shape of (batch_size, tq, self._output_length), we may return the mask for the
        # queries. Hence, next layers can mask on 2nd dimension of the output, which has the size of tq.
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def get_config(self):
        config = {
            'head_num': self._head_num,
            'head_type': self._head_type,
            'inner_embedding_length': self._inner_embedding_length,
            'output_length': self._output_length,
            'context_mask_type': self._context_mask_type,
            'should_scale': self._should_scale,
            'activation': activations.serialize(self._activation),
            'use_bias': self._use_bias,
            'kernel_initializer': initializers.serialize(self._kernel_initializer),
            'bias_initializer': initializers.serialize(self._bias_initializer),
            'kernel_regularizer': regularizers.serialize(self._kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self._bias_regularizer),
            'kernel_constraint': constraints.serialize(self._kernel_constraint),
            'bias_constraint': constraints.serialize(self._bias_constraint),
        }
        base_config = super().get_config()
        config.update(base_config)

        return config

    def _get_narrow_attention_shapes(self):
        if self._inner_embedding_length % self._head_num != 0:
            raise Exception("inner_embedding_length should be divisible by head_num")
        dk = self._inner_embedding_length // self._head_num
        dv = self._inner_embedding_length // self._head_num
        return dk, dv

    def _get_wide_attention_shapes(self):
        dk = self._inner_embedding_length
        dv = self._inner_embedding_length
        return dk, dv

    @staticmethod
    def _move_heads_to_batch_dimension(data, head_num):
        # Rather than computing attention separately for each head, we handle it in single tensor for
        # faster computation.
        _, sequence_length, feature_length = data.shape
        head_length = feature_length // head_num
        data = K.reshape(data, shape=(-1, sequence_length, head_num, head_length))
        data = K.permute_dimensions(data, [0, 2, 1, 3])
        return K.reshape(data, shape=(-1, sequence_length, head_length))

    @staticmethod
    def _move_heads_from_batch_dimension(data, head_num):
        # Rather than computing attention separately for each head, we handle it in single tensor for
        # faster computation.
        _, sequence_length, feature_length = data.shape
        data = K.reshape(data, shape=(-1, head_num, sequence_length, feature_length))
        data = K.permute_dimensions(data, [0, 2, 1, 3])
        return K.reshape(data, shape=(-1, sequence_length, feature_length * head_num))

    @staticmethod
    def _create_heads_for_mask(mask, head_num):
        # mask is 2D tensor with shape: (batch_size, sequence_length)
        # returns 2D tensor with shape: (batch_size * head_num, sequence_length)
        if mask is None:
            return mask
        sequence_length = mask.shape[1]
        # after expand, shape is: (batch_size, 1, sequence_length)
        mask = K.expand_dims(mask, axis=1)
        # after tile data will be copied along 2nd dimension~
        # shape will be: (batch_size, head_num, sequence_length)
        mask = K.tile(mask, [1, head_num, 1])
        # after reshape, shape is: (batch_size * head_num, sequence_length)
        return K.reshape(mask, (-1, sequence_length))
