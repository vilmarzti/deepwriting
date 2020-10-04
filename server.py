from flask import Flask, escape, request
from flask_cors import CORS

from data_scripts.json_to_numpy import fetch_sample_from_dict, scale_zero_one
from data_scripts.preprocessing import process_dataset
from classify_hw import getModel, getConfig, process_result

import numpy as np

from source import data_utils


app = Flask(__name__)
cors = CORS(app)

alphabet = list(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.,-()/"
)  # %:;&# '\x00' character, i.e., ord(0) to label concatenations.
alphabet.insert(0, chr(0))
config_dict = getConfig()


@app.route('/', methods=["POST"])
def evaluate():
    json = request.json
    parsed_json = parse_json(json)
    args = Args()
    process_dataset(args, parsed_json, 'data_preprocessed')
    model, sess, training_dataset = getModel(config_dict)
    result = model.classify_given_sample(sess, np.array([training_dataset.data_dict['samples'][0]]))
    processed_result = process_result(result[0], alphabet)
    return processed_result


class Args:
    def __init__(self):
        self.amount_validation_samples = -1
        self.data_file = ['/home/martin/Documents/code/python3/deepwriting-module/data/deepwriting_dataset/deepwriting-data.npz']
        self.eoc_labels = False
        self.fixed_length_chunks = None
        self.merge_input_dictionaries_first = False
        self.out_dir = '/home/martin/Documents/code/deepwriting-module/data/deepwriting_dataset'
        self.out_file = ['data_preprocessed']
        self.relative_representation = True
        self.scale_data_zero_one = False
        self.semantic_chunks_max_len = 0
        self.standardize_data = True
        self.translate_to_origin = True


def parse_json(json):
    data_dict_1 = create_data_dict()
    data_dict_2 = create_data_dict()
    fetch_sample_from_dict(data_dict_1, json, False, False)
    data_dict_2 = data_utils.dictionary_merge(
        [data_dict_1, data_dict_2],
        inplace_idx=0,
        keys_frozen=['alphabet'],
        verbose=0
    )
    data_dict_2 = scale_zero_one(data_dict_2)
    return data_dict_2


def create_data_dict():
    data_dict = {}
    data_dict['samples'] = []
    data_dict['char_labels'] = []
    data_dict['word_labels'] = []
    data_dict['subject_labels'] = []
    data_dict['texts'] = []
    data_dict['eow_labels'] = []
    data_dict['bow_labels'] = []
    data_dict['eoc_labels'] = []
    data_dict['boc_labels'] = []

    data_dict['alphabet'] = alphabet
    return data_dict


def main():
    app.run(port=5000)


if __name__ == "__main__":
    main()
