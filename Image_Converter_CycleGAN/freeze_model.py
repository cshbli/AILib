import argparse
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Convert images using pre-trained CycleGAN model.')

    parser.add_argument('--meta-filename',              help='File path for your trained model meta file .meta file.', type=str, default = './PLT_X3_to_Microscope_20x/models/PLT_X3_to_Microscope_20x.ckpt_116.meta')
    parser.add_argument('--check-point-path',           help='Directory to your latest check point', type=str, default='./PLT_X3_to_Microscope_20x/models/')
    parser.add_argument('--output-model-filename',      help='Input image DIR /image file name', type=str, default = './PLT_X3_to_Microscope_20x/models/plt_x3_to_microscope_20x.pb')

    args = parser.parse_args()

    output_node_names = ['generator_A2B/d3_conv/Tanh']    # Output nodes

    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(args.meta_filename)

        # Load weights
        saver.restore(sess, tf.train.latest_checkpoint(args.check_point_path))

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # Save the frozen graph
        with open(args.output_model_filename, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
