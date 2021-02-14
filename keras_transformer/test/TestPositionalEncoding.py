import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input
from keras.models import Model

from keras_transformer.core.PositionalEncodingLayer import PositionalEncodingLayer


def plot_embeddings(signal):
    fig = plt.Figure()
    plt.pcolormesh(signal, cmap='RdBu')
    plt.xlabel('Depth')
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


def test_positional_encoding(encoding_type):

    seq_length = 20
    embedding_length = 512
    sample_size = 100

    inputs = Input(shape=(seq_length, embedding_length))
    positional_encoder = PositionalEncodingLayer(encoding_type=encoding_type)
    output = positional_encoder(inputs)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x=np.random.rand(sample_size, seq_length, embedding_length), y=np.random.randint(0, 1, size=(sample_size, seq_length, embedding_length)))

    plot_embeddings(K.eval(positional_encoder.get_signal()))


def test_interleaving_trick():

    seq_length = 3
    encoding_length = 10

    seq_positions = K.expand_dims(K.cast_to_floatx(K.arange(0, seq_length)), 1)
    sin_locations = K.cast_to_floatx(K.arange(0, encoding_length, 2))
    cos_locations = K.cast_to_floatx(K.arange(1, encoding_length, 2))

    print("seq_positions:")
    print(seq_positions)
    print("sin_locations:")
    print(sin_locations)
    print("cos_locations:")
    print(cos_locations)

    div_term_const = K.log(K.constant(10000)) * K.constant(-2 / encoding_length)

    sin_encodings = K.sin(seq_positions * K.expand_dims(K.exp(sin_locations * div_term_const), 0))
    cos_encodings = K.cos(seq_positions * K.expand_dims(K.exp(cos_locations * div_term_const), 0))

    print("sin_encodings:")
    print(sin_encodings)
    print("cos_encodings:")
    print(cos_encodings)

    expanded_sin_encodings = K.expand_dims(sin_encodings, 2)
    expanded_cos_encodings = K.expand_dims(cos_encodings, 2)

    print("expanded_sin_encodings:")
    print(expanded_sin_encodings)
    print("expanded_cos_encodings:")
    print(expanded_cos_encodings)

    # Trick to alternate sines and cosines~
    concatenated_encodings = K.concatenate([expanded_sin_encodings, expanded_cos_encodings])
    print("concatenated_encodings")
    print(concatenated_encodings)

    new_shape = concatenated_encodings.shape[:-1].as_list()
    new_shape[-1] *= 2
    final_encoding = K.reshape(concatenated_encodings, new_shape)
    print("final_encoding")
    print(final_encoding)


# test_positional_encoding(0)
# test_positional_encoding(1)
# test_interleaving_trick()
