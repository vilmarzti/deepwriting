from flask import Flask, escape, request
from flask_cors import CORS

from data_scripts.json_to_numpy import fetch_sample_from_dict
import data_utils


app = Flask(__name__)
cors = CORS(app)

alphabet = list(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.,-()/"
)  # %:;&# '\x00' character, i.e., ord(0) to label concatenations.
alphabet.insert(0, chr(0))


@app.route('/', methods=["POST"])
def hello():
    json = request.json
    parsed_json = parse_json(json)

    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}'


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

