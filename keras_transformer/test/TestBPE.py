from keras.layers import Embedding
from keras import backend as K
from bpemb import BPEmb
import numpy as np


def test_tokens():
    # bpemb_en = BPEmb(lang="en")
    bpemb_en = BPEmb(lang="en", add_pad_emb=True)
    print("num_of_words:")
    print(len(bpemb_en.words))
    print("<pad> direct lookup: ")
    print(bpemb_en['<pad>'])

    print("last and first 3 words and vectors:")
    # pad-token, start-token, end-token, _t
    for i in range(-1, 4):
        print(bpemb_en.words[i])
        print(bpemb_en.vectors[i])


def test_encoding():
    text = ["This is Stratford", "<pad>"]

    bpemb_en = BPEmb(lang="en", add_pad_emb=True)

    # We can auto-add and encode start/end tokens. However, encoder can't handle <pad> directly.
    # We should pad outside with the corresponding index (index of the last word when add_pad_emb True).
    print(bpemb_en.encode(text))
    print(bpemb_en.encode_with_eos(text))
    print(bpemb_en.encode_with_bos_eos(text))
    print(bpemb_en.encode_ids(text))
    print(bpemb_en.encode_ids_with_eos(text))
    print(bpemb_en.encode_ids_with_bos_eos(text))


def test_decoding():
    # Although <pad> word is added, when decoding it can't handle. Therefore, remove padding before decoding.
    # Decoding removes start/end tokens.
    bpemb_en = BPEmb(lang="en", add_pad_emb=True)
    # ids = [1, 215, 80, 8526, 1221, 2]
    ids = [[1, 215, 80, 8526, 1221, 2], [1, 215, 80, 8526, 1221, 2]]
    # ids = [1, 215, 80, 8526, 1221, 2, 10000, 10000]
    # print(bpemb_en.vectors[10000])
    print(bpemb_en.decode_ids(ids))


def test_embedding():
    # Can pass byte-pair embedding weights to keras embedder...

    bpemb_en = BPEmb(lang="en", add_pad_emb=True)
    embedder = Embedding(bpemb_en.vectors.shape[0], bpemb_en.vectors.shape[1], weights=[bpemb_en.vectors])

    embeddings = embedder(K.constant([0, 1, 2, 3, 10000]))
    print("Embedder result:")
    print(embeddings)

    print("--------------")

    print("Vectors in bpemb:")
    for i in [0, 1, 2, 3, 10000]:
        print(bpemb_en.vectors[i])


def test_multi_language():
    text = ["This is Stratford", "Kitap okuyordu."]
    bpemb_multi = BPEmb(lang="multi", add_pad_emb=True)
    print(bpemb_multi.encode_ids_with_bos_eos(text))
    print(bpemb_multi.decode_ids([[1, 5496, 200, 23866, 3927, 2], [1, 45350, 44934, 67191, 94777, 2]]))


def test_punctuation():
    text = ["Leonidas: This's Sparta!!", "Leonidas : This ' s Sparta ! !", "Leonidas This s Sparta"]
    bpemb_multi = BPEmb(lang="multi", add_pad_emb=True)
    print(bpemb_multi.encode(text))


def test_retrieving_tags():
    bpemb_multi = BPEmb(lang="multi", add_pad_emb=True)
    print(bpemb_multi.EOS)
    print(bpemb_multi.BOS)
    # Since pad is the last element:
    print(bpemb_multi.vocab_size)


def test_retrieving_weights():
    bpemb_multi = BPEmb(lang="multi", add_pad_emb=True)
    weights = bpemb_multi.vectors
    print(weights.shape)
    print(type(weights))
    print(weights[:5])

    transposed_weights = np.transpose(weights)
    print(transposed_weights.shape)
    print(transposed_weights[:5])


# test_tokens()
# test_encoding()
# test_decoding()
# test_embedding()
# test_multi_language()
# test_punctuation()
# test_retrieving_tags()
# test_retrieving_weights()
