from demo.translation.TranslationDataGenerator import TranslationDataGenerator
from keras_transformer.training.Trainer import Trainer
from keras_transformer.decoding.BeamSearchDecoder import BeamSearchDecoder
from keras_transformer.decoding.GreedyDecoder import GreedyDecoder
from keras_transformer.utils.io_utils import read_json


def train_model(conf):
    outer_gen = TranslationDataGenerator(**conf["outer_generator_info"])
    trainer = Trainer(outer_generator=outer_gen, **conf["trainer_info"])
    trainer.train()


def decode_samples(conf, samples):
    print("BeamSearchDecoding:")
    decoder = BeamSearchDecoder(**conf)
    result = decoder.decode(samples)
    print(result)

    print("------------------------------")

    print("GreedyDecoding:")
    decoder2 = GreedyDecoder(conf["model_path"], conf["artifacts_path"])
    result = decoder2.decode(samples)
    print(result)


def execute_demo():
    training_conf = read_json("./training_conf2.json")
    decoding_conf = read_json("./decoding_conf.json")
    samples = read_json("./samples.json")

    train_model(training_conf)
    decode_samples(decoding_conf, samples)


execute_demo()
