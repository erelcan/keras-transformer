from demo.translation.TranslationDataGenerator import TranslationDataGenerator
from keras_transformer.generators.InnerGenerator import InnerGenerator
from keras_transformer.processors.SubWordProcessor import SubWordProcessor


def test_translation_generator():
    batch_size = 3
    zip_file_path = "../data/spa-eng.zip"
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
    extraction_path = "/spa-eng/spa.txt"
    num_of_samples = 10

    bpe_info = {
        "shared_bpe": {
            "lang": "multi",
            "add_pad_emb": True
        }
    }
    # padding_info = {
    #     "enc_max_seq_len": 247,
    #     "dec_max_seq_len": 278
    # }

    padding_info = {
        "enc_max_seq_len": 20,
        "dec_max_seq_len": 20
    }

    outer_gen = TranslationDataGenerator(batch_size, zip_file_path, file_url, extraction_path, num_of_samples=num_of_samples, shuffle_on=True)
    processor = SubWordProcessor(bpe_info, padding_info)
    inner_gen = InnerGenerator(outer_gen, processor, pass_count=2, use_remaining=True).get_generator()
    # inner_gen = InnerGenerator(outer_gen, processor, pass_count=2, use_remaining=False).get_generator()
    # inner_gen = InnerGenerator(outer_gen, processor, pass_count=1, use_remaining=True).get_generator()
    # inner_gen = InnerGenerator(outer_gen, processor, pass_count=None, use_remaining=True).get_generator()

    for elem in inner_gen:
        print(elem)
        print("-------------")


# test_translation_generator()
