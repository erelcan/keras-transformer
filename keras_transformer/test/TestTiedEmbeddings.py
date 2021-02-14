import numpy as np
from keras.layers import Embedding, Input, LSTM
from keras.models import Model, save_model, load_model
from keras import backend as K
# from keras.initializers import Constant
from keras_transformer.core.TiedProjectionLayer import TiedProjectionLayer
from keras_transformer.core.TiedEmbedderProjector import TiedEmbedderProjector


def test_weight_initialization():
    # 3 ways to initialize weights, but copies the original in each case.
    weights = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    embedding_info1 = {
        "input_dim": 4,
        "output_dim": 3,
        "input_length": 10,
        "trainable": True
    }

    def weight_init(shape, dtype=None):
        return weights

    inputs = Input(shape=(4,))
    embedding_layer = Embedding(weights=[np.array(weights)], **embedding_info1)
    # embedding_layer = Embedding(embeddings_initializer=weight_init, **embedding_info1)
    # embedding_layer = Embedding(embeddings_initializer=Constant(weights), **embedding_info1)
    res = embedding_layer(inputs)
    model = Model(inputs=inputs, outputs=res)
    model.compile(optimizer="adam", loss="mse")

    print(embedding_layer.get_weights())


def print_weights(encoder, decoder):
    print("embedder1_weights:")
    print(encoder.get_weights())
    print("embedder2_weights:")
    print(decoder.get_weights())


def test_embedding_weight_tying_not_working():
    # Since copies of weights are created for each embedder; they are not tied!
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    embedding_info1 = {
        "input_dim": 4,
        "output_dim": 3,
        "input_length": 10,
        "trainable": True
    }

    def weight_init(shape, dtype=None):
        return weights

    inputs = Input(shape=(4,))

    embedding_layer = Embedding(weights=[weights], **embedding_info1)
    embedding_layer2 = Embedding(weights=[weights], **embedding_info1)

    # embedding_layer = Embedding(embeddings_initializer=weight_init, **embedding_info1)
    # embedding_layer2 = Embedding(embeddings_initializer=weight_init, **embedding_info1)

    # embedding_layer = Embedding(embeddings_initializer=Constant(weights), **embedding_info1)
    # embedding_layer2 = Embedding(embeddings_initializer=Constant(weights), **embedding_info1)

    emb1 = embedding_layer(inputs)
    emb2 = embedding_layer2(inputs)
    out2 = LSTM(3, return_sequences=True)(emb2)
    res = K.concatenate([emb1, out2])

    model = Model(inputs=inputs, outputs=res)
    model.compile(optimizer="adam", loss="mse")

    print("Before fitting...")
    print_weights(embedding_layer, embedding_layer2)

    model.fit(epochs=10, x=np.random.randint(0, 4, size=(100, 4)), y=np.ones(shape=(100, 4, 6)))

    print("After fitting...")
    print_weights(embedding_layer, embedding_layer2)

    print("Init weights after fitting:")
    print(weights)


def test_embedding_weight_tying_working():
    # Only one embedder is created. Hence, there are weights only for it.
    # It is applied in two places. Hence, the embedder/weights is shared directly.
    # When we load the model, we cannot retrieve embedder1 and embedder2 directly in this example, but
    # we can observe that no duplicate weight is created!
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    embedding_info1 = {
        "input_dim": 4,
        "output_dim": 3,
        "input_length": 10,
        "trainable": True
    }

    inputs = Input(shape=(4,))

    embedder = Embedding(weights=[weights], **embedding_info1, name="embedder")
    embedding_layer = embedder
    embedding_layer2 = embedder

    emb1 = embedding_layer(inputs)
    emb2 = embedding_layer2(inputs)
    out2 = LSTM(3, return_sequences=True)(emb2)
    res = K.concatenate([emb1, out2])

    model = Model(inputs=inputs, outputs=res)
    model.compile(optimizer="adam", loss="mse")

    print("Before fitting...")
    print_weights(embedding_layer, embedding_layer2)

    model.fit(epochs=10, x=np.random.randint(0, 4, size=(100, 4)), y=np.ones(shape=(100, 4, 6)))

    print("After fitting...")
    print_weights(embedding_layer, embedding_layer2)

    print("Init weights after fitting:")
    print(weights)

    print("Model weights before save:")
    print(model.get_weights())

    save_model(model, "../model.h5")
    model = load_model("../model.h5")

    model.summary()

    print("Model weights after loading:")
    print(model.get_weights())

    print("After loading...")
    print(model.get_layer("embedder").get_weights())

    model.fit(x=np.random.randint(0, 4, size=(100, 4)), y=np.ones(shape=(100, 4, 6)))

    print("After loading and fitting..")
    print(model.get_layer("embedder").get_weights())


def print_weights2(encoder, decoder, proj):
    print("embedder1_weights:")
    print(encoder.get_weights())
    print("embedder2_weights:")
    print(decoder.get_weights())
    print("proj_weights:")
    print(proj.get_weights())


def test_tied_embedding_and_projection():
    # If we use a wrapper for handling projection, it can use the weight of the embedder (which is its inner/tied layer)
    # When we fit model, we can observe that both embedders and the projector has the same weights (actually projector
    # has the transpose, but retrieving the weights from the inner layer before transpose).
    # All good till here...
    # The problem is that when we save the model and try to load; the weigths of the inner layer is not tied to the
    # weights of the embedders. It builds another embedder inside and copies the weights.
    # Therefore, we need to discard the newly build layer and tie the "real" embedder to the projector layer.
    # This works, but might be hard to maintain or may cause unforseen problems in the computation graph (just maybe..)
    # Also, optimizer state is lost when load; due to mapping error as more weights is created for load case..
    # Hence, re-creates optimizer when handling load error.
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    embedding_info1 = {
        "input_dim": 4,
        "output_dim": 3,
        "input_length": 10,
        "trainable": True
    }

    inputs = Input(shape=(4,))
    inputs2 = Input(shape=(4,))

    embedder = Embedding(weights=[weights], name="embedder", **embedding_info1)
    embedding_layer = embedder
    embedding_layer2 = embedder
    projection_layer = TiedProjectionLayer(embedder, name="projection")

    emb1 = embedding_layer(inputs)
    emb2 = embedding_layer2(inputs2)
    out1 = LSTM(3, return_sequences=True)(emb1)
    out2 = LSTM(3, return_sequences=True)(emb2)
    proj = projection_layer(out1)
    res = K.concatenate([emb1, out2, proj])

    model = Model(inputs=[inputs, inputs2], outputs=res)
    model.compile(optimizer="adam", loss="mse")

    print("Before fitting...")
    print_weights2(embedding_layer, embedding_layer2, projection_layer)

    model.fit(epochs=10, x=[np.random.randint(0, 4, size=(100, 4)), np.random.randint(0, 4, size=(100, 4))], y=np.ones(shape=(100, 4, 10)))

    print("After fitting...")
    print_weights2(embedding_layer, embedding_layer2, projection_layer)

    print("Init weights after fitting:")
    print(weights)

    print("Model weights before save:")
    print(model.get_weights())

    save_model(model, "../model.h5")
    # model = load_model("../model.h5", compile=False, custom_objects={"TiedProjectionLayer": TiedProjectionLayer})
    model = load_model("../model.h5", custom_objects={"TiedProjectionLayer": TiedProjectionLayer})

    print("Model weights after load:")
    print(model.get_weights())

    model.summary()

    print("After loading...")
    print_weights2(model.get_layer("embedder"), model.get_layer("embedder"), model.get_layer("projection"))

    print("After re-assigning embedder to wrapped layer of projection layer:")
    model.get_layer("projection").set_wrapped_layer(model.get_layer("embedder"))
    # model.compile(optimizer="adam", loss="mse")
    print_weights2(model.get_layer("embedder"), model.get_layer("embedder"), model.get_layer("projection"))

    model.fit(epochs=10, x=[np.random.randint(0, 4, size=(100, 4)), np.random.randint(0, 4, size=(100, 4))], y=np.ones(shape=(100, 4, 10)))

    print("After loading and fitting..")
    print_weights2(model.get_layer("embedder"), model.get_layer("embedder"), model.get_layer("projection"))


def test_tied_embedder_projector():
    # To solve shared-weight problem for the save/load case as well; we may create a custom layer.
    # It will have an embedding layer; and a "conditional call" and "compute_output_shape".

    # We have to be able to pass argument to the call so that it can decide whether to embed or project.
    # K.switch did not work (maybe for some other reasons..), but we were able to modify the graph with conditional
    # statements.

    # We pass mode info and depending on the mode embed or project functionality is executed.
    # However, we needed one more trick!
    # How will other layers know the output shape?
    # Output shape depends on the mode argument passed to call.
    # We keep a parameter in the layer to track the current mode. Then, whe output shape is called, it returns according
    # to the current mode. In the way we implement, computation graph considers the condition rather than taking the
    # first value of the mode. This is crucial for us!!!

    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])

    def weight_init(shape, dtype=None):
        return weights

    embedding_info1 = {
        "input_dim": 5,
        "output_dim": 3,
        "trainable": True
    }

    inputs = Input(shape=(4,))
    inputs2 = Input(shape=(4,))

    embed_proj = TiedEmbedderProjector(embedding_info1["input_dim"], embedding_info1["output_dim"], embedding_info1["trainable"], weight_initializer=weight_init, name="embed_proj")

    emb1 = embed_proj(inputs)
    emb2 = embed_proj(inputs2)
    out1 = LSTM(3, return_sequences=True)(emb1)
    out2 = LSTM(3, return_sequences=True)(emb2)
    proj = embed_proj(out1, projection_mode=True)
    res = K.concatenate([emb1, out2, proj])

    model = Model(inputs=[inputs, inputs2], outputs=res)
    model.compile(optimizer="adam", loss="mse")

    # embed_proj.set_mode(True)
    # print(embed_proj.compute_output_shape((10, 4)))
    # embed_proj.set_mode(False)
    # print(embed_proj.compute_output_shape((10, 4)))

    print("Before fitting...")
    print(embed_proj.get_weights())

    model.fit(epochs=10, x=[np.random.randint(0, 4, size=(100, 4)), np.random.randint(0, 4, size=(100, 4))], y=np.ones(shape=(100, 4, 11)))

    print("After fitting...")
    print(embed_proj.get_weights())

    print("Init weights after fitting:")
    print(weights)

    print("Model weights before save:")
    print(model.get_weights())

    save_model(model, "../model.h5")
    model = load_model("../model.h5", custom_objects={"TiedEmbedderProjector": TiedEmbedderProjector, "weight_init": weight_init})

    print("Model weights after load:")
    print(model.get_weights())

    model.summary()

    print("After loading...")
    print(model.get_layer("embed_proj").get_weights())

    model.fit(epochs=10, x=[np.random.randint(0, 4, size=(100, 4)), np.random.randint(0, 4, size=(100, 4))], y=np.ones(shape=(100, 4, 11)))

    print("After loading and fitting..")
    print(model.get_layer("embed_proj").get_weights())


def test_tied_embedder_projector_with_different_lengths():
    # In some cases (please see the Embedding Layer for the details), Embedding layer requires input_length
    # (for reasons to be able to compute the output shape)
    # In our case, we do not need to specify input_length. In case, we wouldn't be able to use any of the solutions.
    # Since Keras copies weights, it is not possible to copy the reference given the API (at the moment at least).
    # In such a case we may create new embedding layers copying the current functions of the Embedding Layer, but adding
    # functionality to modify/manage the layer for sharing mechanisms..
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])

    def weight_init(shape, dtype=None):
        return weights

    embedding_info1 = {
        "input_dim": 5,
        "output_dim": 3,
        "trainable": True
    }

    inputs = Input(shape=(4,))
    inputs2 = Input(shape=(6,))

    embed_proj = TiedEmbedderProjector(embedding_info1["input_dim"], embedding_info1["output_dim"], embedding_info1["trainable"], weight_initializer=weight_init, name="embed_proj")

    emb1 = embed_proj(inputs)
    emb2 = embed_proj(inputs2)
    out1 = LSTM(3, return_sequences=True)(emb1)
    out2 = LSTM(8, return_sequences=True)(emb2)
    proj = embed_proj(out1, projection_mode=True)
    c1 = K.concatenate([emb1, proj])
    c2 = K.concatenate([c1, out2], axis=1)

    model = Model(inputs=[inputs, inputs2], outputs=c2)
    model.compile(optimizer="adam", loss="mse")

    print("Before fitting...")
    print(embed_proj.get_weights())

    model.fit(epochs=10, x=[np.random.randint(0, 5, size=(100, 4)), np.random.randint(0, 5, size=(100, 6))], y=np.ones(shape=(100, 10, 8)))

    print("After fitting...")
    print(embed_proj.get_weights())

    print("Init weights after fitting:")
    print(weights)

    print("Model weights before save:")
    print(model.get_weights())

    save_model(model, "../model.h5")
    model = load_model("../model.h5", custom_objects={"TiedEmbedderProjector": TiedEmbedderProjector, "weight_init": weight_init})

    print("Model weights after load:")
    print(model.get_weights())

    model.summary()

    print("After loading...")
    print(model.get_layer("embed_proj").get_weights())

    model.fit(epochs=10, x=[np.random.randint(0, 5, size=(100, 4)), np.random.randint(0, 5, size=(100, 6))], y=np.ones(shape=(100, 10, 8)))

    print("After loading and fitting..")
    print(model.get_layer("embed_proj").get_weights())


# test_weight_initialization()
# test_embedding_weight_tying_not_working()
# test_embedding_weight_tying_working()
# test_tied_embedding_and_projection()
# test_tied_embedder_projector()
# test_tied_embedder_projector_with_different_lengths()
