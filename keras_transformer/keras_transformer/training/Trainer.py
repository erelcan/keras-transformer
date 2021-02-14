from keras.layers import Input
from keras.models import Model

from keras_transformer.training.TrainerABC import TrainerABC
from keras_transformer.core.Transformer import Transformer
from keras_transformer.training.aux.custom_layers import get_basic_custom_layer_names
from keras_transformer.processors.processor_factory import create_processor


class Trainer(TrainerABC):
    def __init__(self, domain_info, processor_info, outer_generator, generator_info, model_info, checkpointing_info):
        super().__init__(create_processor(processor_info), outer_generator, generator_info, model_info, checkpointing_info)
        # Re-consider creating processor outside~
        self._domain_info = domain_info

    def _get_model(self):
        # Ensure that preprocessor does not change the vocabulary size!!
        # Vocabulary size must account for unknown, start, end and pad tokens!
        encoder_input = Input((self._domain_info["encoder"]["max_seq_len"]), dtype=self._domain_info["encoder"]["input_type"])
        decoder_input = Input((self._domain_info["decoder"]["max_seq_len"]), dtype=self._domain_info["decoder"]["input_type"])

        transformer = Transformer(self._model_info["d_model"], self._model_info["num_of_blocks"], self._model_info["embedding_info"], self._model_info["attention_info"], self._model_info["pff_info"], self._model_info["input_dropout_rates"], self._model_info["return_logits"], self._model_info["mask_value"])

        decoder_output = transformer([encoder_input, decoder_input])
        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

        # Add other custom layers in addition to basic ones, in case.
        self._artifacts["custom_objects_info"]["layer_info"] = get_basic_custom_layer_names()

        return model
