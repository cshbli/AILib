# Classification with different network structure

## Setup

### Prerequisites
- python3
- opencv 3.4: image processing
- Tensorflow: 
- keras

### Dependencies
- pylibs_hb: This project uses a lot of functions from library project `pylibs_hb`. Please make sure you download `pylibs_hb` first `git clone http://192.168.114.220:8081/hongbing/pylibs_hb.git`, change library path `sys.path.insert(0, '/data/Projects/pylibs_hb')` in `train.py` and `test.py` accordingly.

### Getting Started

#### Training a model with sample cats and dogs dataset

```sh
# clone this repo
git clone http://192.168.114.220:8081/hongbing/classification.git
cd classification
# Training a MobileNetV2 classification model on sample cats and dogs dataset, it will use the default values
#    weights='imagenet', 
#    batch_size=16,
#    epochs=5,
#    lr=0.0001,
#    snapshot_path='./snapshots',
#    tensorboard_dir='./logs',
#    csv_log_dir='./logs_csv',
#    image_width=224,
#    image_height=224,
#    dataset_type='cat_and_dog',
#    train_data_dir='cats_and_dogs_filtered/train',
#    validation_data_dir='cats_and_dogs_filtered/validation',
python train.py --backbone MobileNetV2 --fully-connected-layer_size 128 --mobilenet-alpha 0.5
# --fully-connected-layer_size can be 256, 512, ..., larger size will have better accuracy, but larger model size
# --mobilenet-alpha is the MobileNet width multiplier, larger will have better accuracy, but larger model size
```

During and after training, there are multiple files generated:

    snapshots/Backbone_WidthMultiplier_imagesize_FullyConnectedLayerSize_DatasetType_Epoch.h5, which are training model snaphsots.

        snapshots/MobileNetV2_0.5_224x224_128_cat_and_dog_01.h5    
        ...
        snapshots/MobileNetV2_0.5_224x224_128_cat_and_dog_05.h5
        
    snapshots/Backbone_WidthMultiplier_imagesize_FullyConnectedLayerSize_DatasetType.csv, which is the class label map file.
        
        snapshots/MobileNetV2_0.5_224x224_128_cat_and_dog.csv
        
    logs_csv/Backbone_WidthMultiplier_imagesize_FullyConnectedLayerSize_DatasetType.csv, which records the training losses and accuracy of each epoch.
    
        logs_csv/MobileNetV2_0.5_224x224_128_cat_and_dog.csv
        
#### Converting keras .h5 model to Tensorflow .pb model

```sh
# Convert a keras .h5 model to Tensorflow .pb model, it will use the default values
#    output_dir='models', 
python /data/Projects/pylibs_hb/convert_h5_to_pb.py snapshots/MobileNetV2_0.5_224x224_128_cat_and_dog_05.h5
# the `convert_h5_to_pb.py` is in the `pylibs_hb` package, depends on where you download that project.
# The output Tensorflow .pb model file will use the same base filename of .h5.
```

After converting, there is one Tensorflow .pb model generated:

    models/MobileNetV2_0.5_224x224_128_cat_and_dog_05.pb
    
#### Inferencing images with Tensorflow .pb model

```sh
# Inferencing images with Tensorflow model, it will use the default values
#    image_width=224, 
#    image_height=224
python test.py --model models/MobileNetV2_0.5_224x224_128_cat_and_dog_05.pb --backbone MobileNetV2 cats_and_dogs_filtered/validation/dogs/dog.2000.jpg
```
which will output something like:

    processing time:  1.000859022140503
    [[0.12450953 0.8754904 ]]
    
```sh
python test.py --model models/MobileNetV2_0.5_224x224_128_cat_and_dog_05.pb --backbone MobileNetV2 cats_and_dogs_filtered/validation/cats/cat.2000.jpg
```
which will output something like:

    processing time:  1.0088074207305908
    [[0.72374606 0.27625394]]


## Notes

1) The image preprocessing function has be to consistent for training, validation, testing and inference.


## Citation

## Acknowledgments

