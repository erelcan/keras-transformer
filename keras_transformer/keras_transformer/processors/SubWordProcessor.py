from bpemb import BPEmb
from keras.preprocessing.sequence import pad_sequences

from keras_transformer.processors.DecodingCompatibleProcessorABC import DecodingCompatibleProcessorABC


class SubWordProcessor(DecodingCompatibleProcessorABC):
    def __init__(self, bpe_info, padding_info):
        super().__init__()
        self._bpe_info = bpe_info
        self._padding_info = padding_info

        self._shared_bpe = None
        self._encoder_bpe = None
        self._decoder_bpe = None
        if "shared_bpe" in self._bpe_info:
            self._shared_bpe = BPEmb(**self._bpe_info["shared_bpe"])
            self._encoder_bpe = self._shared_bpe
            self._decoder_bpe = self._shared_bpe
        else:
            self._encoder_bpe = BPEmb(**self._bpe_info["encoder_bpe"])
            self._decoder_bpe = BPEmb(**self._bpe_info["decoder_bpe"])

    def process(self, data, **kwargs):
        # data: (encoder_input, decoder_input) which are both list_of_string (may be a list of list; check again though)
        # Assuming that text-based preprocesses are done before.
        # For encoder_input and decoder_input, I decided to add start/end tokens.
        # decoder_input and target should have same length after preprocessing.
        # Hence target will have one more pad element.

        encoder_input = self._encoder_bpe.encode_ids_with_bos_eos(data[0])
        decoder_input = self._decoder_bpe.encode_ids_with_bos_eos(data[1])
        target = self._decoder_bpe.encode_ids_with_eos(data[1])

        # bpe vocab-size does not account for pad word. Hence, weight matrix has length vocab-size + 1
        # As indices start from 0; pad index will be vocab-size.
        # Notice that if bpe is shared, pad token has the same index both for encoder and decoder..
        padded_enc_input = pad_sequences(encoder_input, maxlen=self._padding_info["enc_max_seq_len"], value=self._encoder_bpe.vocab_size, padding="post")
        padded_dec_input = pad_sequences(decoder_input, maxlen=self._padding_info["dec_max_seq_len"], value=self._decoder_bpe.vocab_size, padding="post")
        padded_target = pad_sequences(target, maxlen=self._padding_info["dec_max_seq_len"], value=self._decoder_bpe.vocab_size, padding="post")

        return [padded_enc_input, padded_dec_input], padded_target

    def encode(self, data, usage="encoder", **kwargs):
        # data is a list of string (may be a list of list)
        cur_bpe = self._encoder_bpe
        max_seq_len = self._padding_info["enc_max_seq_len"]
        pad_value = self._encoder_bpe.vocab_size
        if usage != "encoder":
            cur_bpe = self._decoder_bpe
            max_seq_len = self._padding_info["dec_max_seq_len"]
            pad_value = self._decoder_bpe.vocab_size

        encoded = cur_bpe.encode_ids_with_bos_eos(data)
        padded = pad_sequences(encoded, maxlen=max_seq_len, value=pad_value, padding="post")

        return padded

    def decode(self, data, usage="decoder", **kwargs):
        # data is a list of ids (may be a list of list)
        # Designed for decoder id list to sentence mapping, but enabling for encoder as well.
        cur_bpe = self._decoder_bpe
        if usage != "decoder":
            cur_bpe = self._encoder_bpe

        # When decoding, bpe can't handle padding. Hence, we need to remove the padding first.
        pad_id = cur_bpe.vocab_size
        if any(isinstance(el, list) for el in data):
            pad_removed = []
            for elem in data:
                pad_removed.append(self.remove_padding(elem, pad_id))
            return cur_bpe.decode_ids(pad_removed)
        else:
            return cur_bpe.decode_ids(self.remove_padding(data, pad_id))

    def get_tag_ids(self, usage="decoder", **kwargs):
        # Specifically; start, end and pad tag ids of decoder
        # Re-consider unknown~
        cur_bpe = self._decoder_bpe
        if usage != "decoder":
            cur_bpe = self._encoder_bpe

        # Since pad is the last element..
        tag_ids = {"start": cur_bpe.BOS, "end": cur_bpe.EOS, "pad": cur_bpe.vocab_size}
        return tag_ids

    def get_max_seq_length(self, usage="decoder", **kwargs):
        if usage == "decoder":
            return self._padding_info["dec_max_seq_len"]
        else:
            return self._padding_info["enc_max_seq_len"]

    @staticmethod
    def remove_padding(list_of_ids, pad_value):
        return [int(i) for i in list_of_ids if i != pad_value]
