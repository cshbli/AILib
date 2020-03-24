import argparse
import cv2
import numpy as np
import sys
import os
import time
import glob

from keras.preprocessing import image
from keras.applications import mobilenetv2
from keras.applications import resnet50

import tensorflow as tf

sys.path.insert(0, '/data/Projects/pylibs_hb')
from utils_hb import load_graph

def parse_args():
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Classification Testing Program')   

    parser.add_argument('--model',      help='Classification Tensorflow model in .pb', type=str, default='models/model.pb')   
    parser.add_argument('--image-ext',  help='Image file extension (default JPG)', type=str, default='JPG')     
    parser.add_argument('--backbone',   help='Backbone network name', default='ResNet50', type=str)
    parser.add_argument('--image-width',  help='Input image size of width.', type=int, default=224)
    parser.add_argument('--image-height', help='Input image size of height.', type=int, default=224)

    parser.add_argument('input',        help='Input image DIR /image file name', type=str)    

    return parser.parse_args()          


def main(args):
    # load the classification model
    classification_graph = load_graph(args.model)
    classification_session = tf.Session(graph=classification_graph)

    # load image or image dir    
    if os.path.isdir(args.input):
        img_list = glob.iglob(args.input + '/*.' + args.image_ext)
    else:
        img_list = [args.input]

    img_list = sorted(img_list)  

    for i, img_name in enumerate(img_list):
        image_org = image.load_img(img_name, target_size=(args.image_height, args.image_width))
        img = image.img_to_array(image_org)
        if args.backbone == 'ResNet50':
            img = resnet50.preprocess_input(img)
        elif args.backbone == 'MobileNetV2':
            img = mobilenetv2.preprocess_input(img)            
        else:
            print('Unsupported backbone')
            continue

        # process image
        start = time.time()
        image_tensor = classification_graph.get_tensor_by_name('input_1:0')
        output_tensor= classification_graph.get_tensor_by_name('dense_2/Softmax:0')        
        scores       = classification_session.run([output_tensor], feed_dict={image_tensor: np.expand_dims(img, axis=0)})
        print(img_name)
        print("processing time: ", time.time() - start)
        print(np.round(scores[0], 3))


if __name__ == '__main__':
    args = parse_args()

    main(args)