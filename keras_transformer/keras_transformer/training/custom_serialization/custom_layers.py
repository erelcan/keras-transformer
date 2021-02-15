from keras_transformer.core.Transformer import Transformer
from keras_transformer.core.PositionalEncodingLayer import PositionalEncodingLayer
# from keras_transformer.core.TiedProjectionLayer import TiedProjectionLayer
from keras_transformer.core.attention.DotProductAttention import DotProductAttention
from keras_transformer.core.attention.MultiHeadAttention import MultiHeadAttention
from keras_transformer.core.encoder_decoder.EncoderBlock import EncoderBlock
from keras_transformer.core.encoder_decoder.EncoderBlockStack import EncoderBlockStack
from keras_transformer.core.encoder_decoder.DecoderBlock import DecoderBlock
from keras_transformer.core.encoder_decoder.DecoderBlockStack import DecoderBlockStack
from keras_transformer.core.encoder_decoder.EncoderDecoder import EncoderDecoder
from keras_transformer.core.encoder_decoder.sub_layers.PositionWiseFeedForwardSublayer import PositionWiseFeedForwardSublayer
from keras_transformer.core.encoder_decoder.sub_layers.SelfAttentionSublayer import SelfAttentionSublayer
from keras_transformer.core.encoder_decoder.normalization.LayerNormalization import LayerNormalization
from keras_transformer.core.TiedEmbedderProjector import TiedEmbedderProjector


def get_custom_layer_class(layer_name):
    return _custom_layer_mappings[layer_name]


def get_basic_custom_layer_names():
    return ["Transformer", "PositionalEncodingLayer", "TiedEmbedderProjector", "DotProductAttention", "MultiHeadAttention", "EncoderBlock", "EncoderBlockStack", "DecoderBlock", "DecoderBlockStack", "EncoderDecoder", "PositionWiseFeedForwardSublayer", "SelfAttentionSublayer", "LayerNormalization"]


_custom_layer_mappings = {
    "Transformer": Transformer,
    "PositionalEncodingLayer": PositionalEncodingLayer,
    "TiedEmbedderProjector": TiedEmbedderProjector,
    "DotProductAttention": DotProductAttention,
    "MultiHeadAttention": MultiHeadAttention,
    "EncoderBlock": EncoderBlock,
    "EncoderBlockStack": EncoderBlockStack,
    "DecoderBlock": DecoderBlock,
    "DecoderBlockStack": DecoderBlockStack,
    "EncoderDecoder": EncoderDecoder,
    "PositionWiseFeedForwardSublayer": PositionWiseFeedForwardSublayer,
    "SelfAttentionSublayer": SelfAttentionSublayer,
    "LayerNormalization": LayerNormalization,
}
