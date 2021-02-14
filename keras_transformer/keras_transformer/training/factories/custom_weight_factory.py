from bpemb import BPEmb

from keras_transformer.utils.io_utils import save_to_pickle, load_from_pickle, check_file_exists


def get_weight(weight_info):
    if weight_info is None:
        return None
    else:
        return _weight_creators[weight_info["type"]](weight_info["parameters"])


def prepare_bpe_weights(lang, vs, dim, weight_path=None):
    # Note that embedding weights and preprocessor weights should be the same.
    # If the weight computer is not deterministic, be careful..
    # May relax the requirement depending on the case..
    # E.g. weight values may be different for the embedder and the processor.
    # However, the ids should match the words in the right order..
    if weight_path is None:
        bpe = BPEmb(lang=lang, add_pad_emb=True, vs=vs, dim=dim)
        weights = bpe.vectors
    else:
        if check_file_exists(weight_path):
            weights = load_from_pickle(weight_path)
        else:
            bpe = BPEmb(lang=lang, add_pad_emb=True, vs=vs, dim=dim)
            weights = bpe.vectors
            save_to_pickle(weights, weight_path)

    return weights


_weight_creators = {
    "bpe": lambda parameters: prepare_bpe_weights(**parameters)
}