import numpy as np
import math
from keras.preprocessing.sequence import pad_sequences

from keras_transformer.decoding.DecoderABC import DecoderABC
from keras_transformer.utils.common_utils import get_topk_with_indices


class BeamSearchDecoder(DecoderABC):
    def __init__(self, model_path, artifacts_path, num_of_candidates, beam_width):
        # Naive implementation of BeamSearchDecoder
        super().__init__(model_path, artifacts_path)
        self._num_of_candidates = num_of_candidates
        self._beam_width = beam_width
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
            raise Exception("BeamSearchDecoder: Not supported data format..")

        result = []
        for enc_inp in encoder_input:
            result.append(self._handle_single_sequence(enc_inp))

        if return_single:
            return result[0]
        else:
            return result

    def _handle_single_sequence(self, cur_seq):
        enc_max_seq_length = self._preprocessor.get_max_seq_length(usage="encoder")
        dec_max_seq_length = self._preprocessor.get_max_seq_length(usage="decoder")
        completed = []
        active = [([self._tag_ids["start"]], 0) for _ in range(self._num_of_candidates)]
        encoder_input = np.tile(self._preprocessor.encode([cur_seq]), self._num_of_candidates).reshape((-1, enc_max_seq_length))
        while len(active) > 0:
            all_candidates = completed
            preds = self._model.predict([encoder_input[:len(active)], self._obtain_sequence_array(active, dec_max_seq_length)])
            for i in range(len(active)):
                cur_t = len(active[i][0]) - 1
                topk, topk_indices = get_topk_with_indices(preds[i][cur_t], self._beam_width)
                for k in range(self._beam_width):
                    all_candidates.append((active[i][0] + [topk_indices[k].item()], active[i][1] + math.log(topk[k])))

            all_candidates.sort(key=lambda x: x[1], reverse=True)

            completed = []
            active = []
            for i in range(self._num_of_candidates):
                cur_candidate = all_candidates[i]
                if cur_candidate[0][-1] == self._tag_ids["end"] or cur_candidate[0][-1] == self._tag_ids["pad"] or len(cur_candidate[0]) == dec_max_seq_length:
                    completed.append(cur_candidate)
                else:
                    active.append(cur_candidate)

        best_seq_pair = max(completed, key=lambda x: x[1])
        best_seq = best_seq_pair[0]
        # Convert ids to tokens (Also, there may be special decoding such that
        # sub-word token ids are processed into sentences~)
        decoded_best_seq = self._preprocessor.decode(best_seq, usage="decoder")
        return decoded_best_seq

    def _obtain_sequence_array(self, seq_score_pairs, max_seq_length):
        return pad_sequences([elem[0] for elem in seq_score_pairs], maxlen=max_seq_length, value=self._tag_ids["pad"], padding="post")
