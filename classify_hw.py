import tensorflow as tf
import sys
import config
import json

from tf_models_hw_classification import BiDirectionalRNNClassifier
from tf_dataset_hw import *
from tf_data_feeder import DataFeederTF
from dataset_hw import *


def classify(config):
    Model_cls = BiDirectionalRNNClassifier
    Dataset_cls = getattr(sys.modules[__name__], config['dataset_cls'])
    training_dataset = Dataset_cls(config['training_data'], use_bow_labels=config.get('use_bow_labels', False), data_augmentation=config.get('data_augmentation', False))

    data_feeder = DataFeederTF(
        training_dataset,
        config['num_epochs'],
        config['batch_size'],
        queue_capacity=512
    )
    sequence_length, inputs, targets = data_feeder.batch_queue(
        dynamic_pad=training_dataset.is_dynamic,
        queue_capacity=512,
        queue_threads=4
    )
    sequence_length, inputs, targets = data_feeder.batch_queue(dynamic_pad=training_dataset.is_dynamic, queue_capacity=512, queue_threads=4)
    model = Model_cls(
        config,
        reuse=True,
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
    checkpoint_path = tf.train.latest_checkpoint(config['model_dir'])
    #saver.restore(sess, checkpoint_path)

if __name__ == "__main__":
    config_path = './runs/tf-1571593867-deepwriting-classification_model/config.json'
    config_dict = json.load(open(config_path, 'r'))
    tf.set_random_seed(config_dict['seed'])
    classify(config_dict)
