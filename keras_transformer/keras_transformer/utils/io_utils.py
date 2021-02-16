import os
import pickle
import json


def check_file_exists(file_path):
    return os.path.exists(file_path)


def save_to_pickle(data, save_path):
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(load_path, encoding="latin1"):
    if check_file_exists(load_path):
        with open(load_path, "rb") as handle:
            data = pickle._Unpickler(handle)
            data.encoding = encoding
            return data.load()
    else:
        raise Exception("File not found: " + load_path)


def read_json(file_path):
    if check_file_exists(file_path):
        with open(file_path) as handle:
            data = json.load(handle)
        return data
    else:
        raise Exception("File not found: " + file_path)