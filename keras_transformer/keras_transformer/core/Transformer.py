from keras import backend as K
from keras.layers import Layer
from keras.layers import Embedding
from keras_transformer.core.ProjectionLayer import ProjectionLayer
from keras_transformer.core.encoder_decoder.EncoderDecoder import EncoderDecoder
from keras_transformer.core.TiedEmbedderProjector import TiedEmbedderProjector
from keras_transformer.core.PositionalEncodingLayer import PositionalEncodingLayer
from keras.layers import Activation
from keras.initializers import Constant
from keras_transformer.training.factories import custom_weight_factory


class Transformer(Layer):
    def __init__(self, d_model, num_of_blocks, embedding_info, attention_info, pff_info, input_dropout_rates, return_logits=False, mask_value=0, **kwargs):
        super().__init__(**kwargs)

        self._d_model = d_model
        self._num_of_blocks = num_of_blocks
        self._embedding_info = embedding_info
        self._attention_info = attention_info
        self._pff_info = pff_info
        self._input_dropout_rates = input_dropout_rates
        self._return_logits = return_logits
        self._mask_value = mask_value

        self._encoder_embedder, self._decoder_embedder, self._projection_layer = self._initialize_embedding_and_projection_layers()
        self._encoder_embedding_length, self._decoder_embedding_length = self._get_embedding_lengths()

        # Check whether using the same positional encoder for both encoder and decoder causes any problems or not.
        self._positional_encoding_for_encoder = PositionalEncodingLayer()
        self._positional_encoding_for_decoder = PositionalEncodingLayer()
        self._encoder_decoder = EncoderDecoder(self._d_model, self._num_of_blocks, self._attention_info, self._pff_info, self._input_dropout_rates)
        self._softmax_layer = Activation("softmax")

    def call(self, inputs, **kwargs):
        # input: [encoder_input, decoder_input]
        # input ->  [(b, tE), (b, tD)]
        # output -> (b, tq_D, fv_E)
        # fv_E, embE, embD = d_model
        # There is no fE and fD in the inputs as built-in embedding layer requires 2D input.
        # However, for different embedding layers, we may have such 3rd dimension!

        encoder_input, decoder_input = inputs

        encoder_mask = self._create_padding_mask(encoder_input, self._mask_value)
        decoder_mask = self._create_padding_mask(decoder_input, self._mask_value)

        encoder_scaled_embeddings = self._encoder_embedder(encoder_input) * K.sqrt(K.constant(self._encoder_embedding_length))
        decoder_scaled_embeddings = self._decoder_embedder(decoder_input) * K.sqrt(K.constant(self._decoder_embedding_length))

        encoder_embeddings = self._positional_encoding_for_encoder(encoder_scaled_embeddings)
        decoder_embeddings = self._positional_encoding_for_decoder(decoder_scaled_embeddings)

        # If you would like to pass different queries/keys/values; here is the place for it. Otherwise, the input will
        # copied to queries, keys and values.
        encoder_decoder_output = self._encoder_decoder([encoder_embeddings, decoder_embeddings], mask=[encoder_mask, decoder_mask])
        if self._embedding_info["weight_sharing"] == "all" or self._embedding_info["weight_sharing"] == "decoder_projector_only":
            linear_projection = self._projection_layer(encoder_decoder_output, projection_mode=True)
        else:
            linear_projection = self._projection_layer(encoder_decoder_output)

        # We may consider applying the mask on the output of projection layer such that for a padding mask we produce
        # the one-hot representation of the padding token. As we know that next token of a padding token is a padding
        # token. For now, let's see whether it learns without imposing that constraint.
        if self._return_logits:
            output = linear_projection
        else:
            output = self._softmax_layer(linear_projection)

        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            if len(input_shape) == 2:
                batch_size = input_shape[0][0]
                decoder_sequence_length = input_shape[1][1]
                return batch_size, decoder_sequence_length, self._d_model
            else:
                raise Exception("Input to the transformer should be a list of 2.")
        else:
            raise Exception("Input to the transformer should be a list: [source_input, target_input]")

    def get_config(self):
        config = {
            "d_model": self._d_model,
            "num_of_blocks": self._num_of_blocks,
            "embedding_info": self._embedding_info,
            "attention_info": self._attention_info,
            "pff_info": self._pff_info,
            "input_dropout_rates": self._input_dropout_rates,
            "return_logits": self._return_logits,
            "mask_value": self._mask_value
        }
        base_config = super().get_config()
        config.update(base_config)

        return config

    def _initialize_embedding_and_projection_layers(self):
        # Sharing weights means that input and output is being embedded in the same space.
        # E.g. for machine translation, use sub-word vocabularies (and maybe with pre-trained weights) over both languages.
        if self._embedding_info["weight_sharing"] == "all":
            embedder_projector = TiedEmbedderProjector(input_dim=self._embedding_info["vocabulary_size"], output_dim=self._embedding_info["embedding_length"], trainable=self._embedding_info.get("is_trainable", True), activation=self._embedding_info.get("activation", "linear"))
            return embedder_projector, embedder_projector, embedder_projector
        elif self._embedding_info["weight_sharing"] == "decoder_projector_only":
            weight_initializer_enc = Constant(
                custom_weight_factory.get_weight(self._embedding_info["encoder"]["weight_info"])) if "weight_info" in self._embedding_info["encoder"] else None
            encoder_embedder = Embedding(input_dim=self._embedding_info["encoder"]["vocabulary_size"], output_dim=self._embedding_info["encoder"]["embedding_length"], embeddings_initializer=weight_initializer_enc, trainable=self._embedding_info["encoder"].get("is_trainable", True))
            embedder_projector = TiedEmbedderProjector(input_dim=self._embedding_info["decoder"]["vocabulary_size"], output_dim=self._embedding_info["decoder"]["embedding_length"], trainable=self._embedding_info["decoder"].get("is_trainable", True), activation=self._embedding_info["decoder"].get("activation", "linear"))
            return encoder_embedder, embedder_projector, embedder_projector
        else:
            weight_initializer_enc = Constant(
                custom_weight_factory.get_weight(self._embedding_info["encoder"]["weight_info"])) if "weight_info" in self._embedding_info["encoder"] else None
            weight_initializer_dec = Constant(
                custom_weight_factory.get_weight(self._embedding_info["decoder"]["weight_info"])) if "weight_info" in self._embedding_info["decoder"] else None
            encoder_embedder = Embedding(input_dim=self._embedding_info["encoder"]["vocabulary_size"], output_dim=self._embedding_info["encoder"]["embedding_length"], embeddings_initializer=weight_initializer_enc, trainable=self._embedding_info["encoder"].get("is_trainable", True))
            decoder_embedder = Embedding(input_dim=self._embedding_info["decoder"]["vocabulary_size"], output_dim=self._embedding_info["decoder"]["embedding_length"], embeddings_initializer=weight_initializer_dec, trainable=self._embedding_info["decoder"].get("is_trainable", True))
            projection_layer = ProjectionLayer(input_dim=self._embedding_info["decoder"]["embedding_length"], output_dim=self._embedding_info["decoder"]["vocabulary_size"], trainable=self._embedding_info["projection"].get("is_trainable", True), activation=self._embedding_info["projection"].get("activation", "linear"))
            return encoder_embedder, decoder_embedder, projection_layer

    def _get_embedding_lengths(self):
        if self._embedding_info["weight_sharing"] == "all":
            return self._embedding_info["embedding_length"], self._embedding_info["embedding_length"]
        else:
            return self._embedding_info["encoder"]["embedding_length"], self._embedding_info["decoder"]["embedding_length"]

    @staticmethod
    def _create_padding_mask(data, mask_value=0):
        return K.cast_to_floatx(K.not_equal(data, mask_value))
