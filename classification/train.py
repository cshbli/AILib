import time
import sys
import numpy as np
import math
import argparse
import os
import csv          # for writing class label CSV file

import keras
import tensorflow as tf

from keras.applications import mobilenetv2
from keras.applications import mobilenet
from keras.applications import resnet50

# sys.path.insert(0, '../../pylibs_hb')
from utils_hb import makedirs
from utils_hb import get_session

def parse_args():
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Keras Classification Training with different networks.')
    
    parser.add_argument('--snapshot',           help='Resume training from a snapshot.')    
    parser.add_argument('--weights',            help='Initialize the model with weights from a file.', type=str, default='imagenet')
    parser.add_argument('--backbone',           help='Backbone network name', default='ResNet50', type=str)
    parser.add_argument('--batch-size',         help='Size of the batches.', default=16, type=int)
    parser.add_argument('--gpu',                help='Id of the GPU to use (as reported by nvidia-smi).')    
    parser.add_argument('--epochs',             help='Number of epochs to train.', type=int, default=5)
    parser.add_argument('--steps',              help='Number of steps per epoch.', type=int, default=100)
    parser.add_argument('--lr',                 help='Learning rate.', type=float, default=1e-4)
    parser.add_argument('--snapshot-path',      help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir',    help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--csv-log-dir',        help='Log directory for CSVLogger output', default='./logs_csv')
    parser.add_argument('--snapshots',          help='Save the training snapshots', action='store_true', default=True)
    parser.add_argument('--no-snapshots',       help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',      help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',    help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform',   help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-width',        help='Input image size of width.', type=int, default=224)
    parser.add_argument('--image-height',       help='Input image size of height.', type=int, default=224)
    parser.add_argument('--dataset-type',       help='Dataset type purely for log purpose.', type=str, default='cat_and_dog')
    parser.add_argument('--train-data-dir',     help='Training data directory.', type=str, default='cats_and_dogs_filtered/train')
    parser.add_argument('--validation-data-dir',help='Validation data directory.', type=str, default='cats_and_dogs_filtered/validation')       
    parser.add_argument('--batch-normalization',help='Add batch normalization layer.', action='store_true')
    parser.add_argument('--fully-connected-layer-size',     help='Add one fully connected layer with size.', type=int, default=512)

    parser.add_argument('--mobilenet-alpha',    help='MobileNet width multiplier', type=float, default=1.0)

    return parser.parse_args() 


def create_callbacks(args, filename_prefix):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None
    
    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)    
   

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)

        checkpoint_filename = '{prefix}_{{epoch:02d}}.h5'.format(prefix=filename_prefix)        
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(
                    args.snapshot_path,                    
                    checkpoint_filename
            ),
            #monitor='mean_absolute_error',
            monitor='acc',
            #monitor='loss'
            # monitor="mAP",
            #monitor = 'top_1_accuracy',
            verbose=1,
            #save_best_only=True,
            #save_weights_only=True,            
            #mode='min'
            mode='max'
        )        
        callbacks.append(checkpoint)        

    if args.csv_log_dir:
        # ensure directory created first; 
        makedirs(args.csv_log_dir)
        csv_filename = filename_prefix + '.csv'        
        csv_logger = keras.callbacks.CSVLogger(os.path.join(args.csv_log_dir, csv_filename))
        callbacks.append(csv_logger)
    
    return callbacks


def top_1_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)


def create_model(weights, classes=1000, input_shape=(224, 224, 3), pooling='avg', lr=0.001):
    # import with pre-trained weights. do not include fully connected layers
    if args.backbone == "ResNet50":
        base = keras.applications.ResNet50(include_top=False, weights=args.weights, input_shape=input_shape)
    elif args.backbone == "DenseNet121":
        base = keras.applications.DenseNet121(include_top=False, weights=args.weights, input_shape=input_shape)
    elif args.backbone == "MobileNetV2":
        base = keras.applications.MobileNetV2(include_top=False, weights=args.weights, input_shape=input_shape, alpha=args.mobilenet_alpha)
    elif args.backbone == "MobileNet":
        base = keras.applications.MobileNet(include_top=False, weights=args.weights, input_shape=input_shape, alpha=args.mobilenet_alpha)
    else:
        print('Unsupported backbone network: ' + args.backbone)
        return None

    # add a global spatial average pooling layer
    x = base.output
    x = keras.layers.GlobalAveragePooling2D()(x)

    # add a fully-connected layer
    x = keras.layers.Dense(args.fully_connected_layer_size, activation='relu')(x)
    # and a fully connected output/classification layer
    predictions = keras.layers.Dense(classes, activation='softmax')(x)

    # create the full network so we can train on it
    model = keras.models.Model(inputs=base.input, outputs=predictions)

    # compile the model
    model.compile(loss='categorical_crossentropy',
              #optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              #optimizer='adam',
              optimizer=keras.optimizers.Adam(lr=lr),
              metrics=['accuracy'])
            
    return model


def main(args=None):    

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu    
    keras.backend.tensorflow_backend.set_session(get_session())

    if args.backbone == 'MobileNetV2' or args.backbone == 'MobileNet':
        filename_prefix = '{backbone}_{alpha}_{image_height}x{image_width}_{fully_connected_size}_{dataset_type}'.format(backbone=args.backbone, 
                            alpha=args.mobilenet_alpha, 
                            image_height=args.image_height,
                            image_width=args.image_width,
                            fully_connected_size=args.fully_connected_layer_size,
                            dataset_type=args.dataset_type)
    else:
        filename_prefix = '{backbone}_{image_height}x{image_width}_{fully_connected_size}_{dataset_type}'.format(backbone=args.backbone, 
                            image_height=args.image_height,
                            image_width=args.image_width,
                            fully_connected_size=args.fully_connected_layer_size,
                            dataset_type=args.dataset_type)  

    if args.backbone == 'MobileNetV2':
        preprocessing_function = mobilenetv2.preprocess_input
    elif args.backbone == 'MobileNet':
        preprocessing_function = mobilenet.preprocess_input
    elif args.backbone == 'ResNet50':
        preprocessing_function = resnet50.preprocess_input

    # Reading the data
    # default color mode is 'rgb'
    train_datagen = keras.preprocessing.image.ImageDataGenerator(        
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocessing_function)
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocessing_function)

    train_generator = train_datagen.flow_from_directory(
        args.train_data_dir,
        target_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        args.validation_data_dir,
        target_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        class_mode='categorical')

    makedirs(args.snapshot_path)        

    label_map = (train_generator.class_indices)
    #print(label_map)
    #print(len(label_map))
    label_dict = dict((v,k) for k,v in label_map.items())     
    print(label_dict) 

    # write the class label file to CSV
    classes_csv_filename = filename_prefix + '.csv'           
    f = open(os.path.join(args.snapshot_path, classes_csv_filename), 'w')
    w = csv.writer(f)  
    for key, val in label_dict.items():
        w.writerow([key, val])
    f.close() 

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')        
        model = keras.models.load_model(args.snapshot)        
    else:        
        print('Creating model, this may take a second...')    
        if (args.weights):
            print('Load pretrained weights: ' + args.weights)      
        model = create_model(args.weights, classes=len(label_map), input_shape=(args.image_height, args.image_width, 3), lr=args.lr)

    # print model summary
    print(model.summary())
    
    # create the callbacks
    callbacks = create_callbacks(args, filename_prefix)

    # train the model on the new data for a few epochs    
    step_size_train=train_generator.n//train_generator.batch_size
    model.fit_generator(train_generator,                        
                        steps_per_epoch=step_size_train,
                        epochs=args.epochs,
                        shuffle=True,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n//validation_generator.batch_size) 

            
if __name__ == '__main__':
    args = parse_args()
    main(args)
    
