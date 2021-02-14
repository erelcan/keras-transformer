import numpy as np
from keras.preprocessing.sequence import pad_sequences

from keras_transformer.core.decoding.DecoderABC import DecoderABC


class GreedyDecoder(DecoderABC):
    def __init__(self, model_path, artifacts_path):
        super().__init__(model_path, artifacts_path)
        self._tag_ids = self._preprocessor.get_tag_ids(usage="decoder")

    def decode(self, data):
        # Let's first ensure that we are working on list of sentences~.
        return_single = False
        if isinstance(data, str):
            return_single = True
            encoder_input = [data]
        elif isinstance(data, list):
            encoder_input = data
        else:
            raise Exception("GreedyDecoder: Not supported data format..")

        result = []
        for enc_inp in encoder_input:
            result.append(self._handle_single_sequence(enc_inp))

        if return_single:
            return result[0]
        else:
            return result

    def _handle_single_sequence(self, cur_seq):
        enc_inp = np.array(self._preprocessor.encode([cur_seq]))
        dec_inp = [self._tag_ids["start"]]
        max_dec_len = self._preprocessor.get_max_seq_length(usage="decoder")
        while len(dec_inp) < max_dec_len and dec_inp[-1] != self._tag_ids["end"] and dec_inp[-1] != self._tag_ids["pad"]:
            x = pad_sequences([dec_inp], maxlen=max_dec_len, value=self._tag_ids["pad"], padding="post")
            preds = self._model.predict([enc_inp, x])
            dec_inp.append(np.argmax(preds[0][len(dec_inp) - 1]))

        best_interpretation = self._preprocessor.decode(dec_inp, usage="decoder")

        return best_interpretation
