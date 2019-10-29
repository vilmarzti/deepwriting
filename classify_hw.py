import tensorflow as tf
import numpy as np
import sys
import config
import json

from tf_models_hw_classification import BiDirectionalRNNClassifier
from tf_dataset_hw import *
from tf_data_feeder import DataFeederTF
from dataset_hw import *


def classify(input_config):
    Model_cls = BiDirectionalRNNClassifier
    Dataset_cls = getattr(sys.modules[__name__], input_config['dataset_cls'])
    training_dataset = Dataset_cls(
        input_config['training_data'],
        use_bow_labels=input_config.get('use_bow_labels', False),
        data_augmentation=input_config.get('data_augmentation', False)
    )
    training_dataset.sample_tf_type = [tf.int32, tf.float32, tf.float32]

    data_feeder = DataFeederTF(
        training_dataset,
        input_config['num_epochs'],
        input_config['batch_size'],  # batch_size
        queue_capacity=512
    )
    sequence_length, inputs, targets = data_feeder.batch_queue(
        dynamic_pad=training_dataset.is_dynamic,
        queue_capacity=512,
        queue_threads=4
    )
    sequence_length, inputs, targets = data_feeder.batch_queue(dynamic_pad=training_dataset.is_dynamic, queue_capacity=512, queue_threads=4)
    model = Model_cls(
        input_config,
        reuse=False,
        input_op=inputs,
        target_op=targets,
        input_seq_length_op=sequence_length,
        input_dims=training_dataset.input_dims,
        target_dims=training_dataset.target_dims,
        mode='validation'
    )
    model.build_graph()

    sess = tf.Session()
    saver = tf.train.Saver()
    checkpoint_path = tf.train.latest_checkpoint(input_config['model_dir'])
    saver.restore(sess, checkpoint_path)
    test = np.array([training_dataset.data_dict['strokes'][0]])
    result = model.classify_given_sample(sess, test)
    process_result(result[0])


def process_result(result, training_dataset):
    char_prediction = result['char_prediction'][0]
    bow_positions = np.where(result['bow_prediction'] > 0.9)[1]
    eoc_position = [0] + np.where(result['eoc_prediction'] > 0.9)[1]
    argmax_char = np.argmax(char_prediction, 1)
    alphabet = training_dataset.alphabet
    alphabet = np.delete(alphabet, 0)
    np.insert(eoc_position, 0, 0)
    eoc_ranges = [(eoc_position[i], eoc_position[i+1]) for i in range(eoc_position.shape[0] - 1)]
    chars = [alphabet[(pos + 35) % 69] for pos in argmax_char]
    chars_collapsed = []
    for r in eoc_ranges:
        char_range = chars[r[0]: r[1]]
        most_frequent_char = max(set(char_range), key=char_range.count)
        chars_collapsed.append(most_frequent_char)


if __name__ == "__main__":
    config_path = './deepwriting/runs/tf-1571593867-deepwriting-classification_model/config.json'
    config_dict = json.load(open(config_path, 'r'))
    config_dict['batch_size'] = 1
    tf.set_random_seed(config_dict['seed'])
    classify(config_dict)
