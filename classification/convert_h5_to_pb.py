# import keras
import keras

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

from keras import backend as K


def parse_args():
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Converting Keras .h5 model to Tensorflow .pb model')

    parser.add_argument('--input',       help='Input Keras .h5 model file name', type=str, default=None)
    parser.add_argument('--output',      help='Output Tensorflow .pb model file name', type=str)

    return parser.parse_args()


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

args = parse_args()

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# This line must be executed before loading Keras model. Otherwise, the frozen model will be very messy.
K.set_learning_phase(0)

# load retinanet model
model = keras.models.load_model(args.input)

model.summary()

print(model.inputs)
print(model.outputs)

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

# Finally we serialize and dump the output graph to the filesystem
with tf.gfile.GFile(args.output, 'wb') as f:
    f.write(frozen_graph.SerializeToString())
print('%d ops in the final graph.' % len(frozen_graph.node))
