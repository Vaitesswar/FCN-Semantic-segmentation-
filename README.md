# Semantic-segmentation

## Models

### Fully Convolutional Networks (FCN)
![FCN image](https://user-images.githubusercontent.com/81757215/114264076-60936900-9a1b-11eb-98ce-451a92fd417a.png)

As recommended in the original paper, VGG 16 was used as the backbone for the convolution block.

### U-Net
![UNet image](https://user-images.githubusercontent.com/81757215/114264084-6c7f2b00-9a1b-11eb-82d6-095259ce1619.png)

The U-Net consists of a contracting block and an expansive block. Since the contracting block closely resembles the architecture of VGG 13, the convolution layers were initialized with the pre-trained weights of VGG 13.

## Dataset

The dataset that was used in this project is the Stanford background dataset which is an open-source dataset provided by Stanford University. This dataset consists of 715 unique images of outdoor scenes which were compiled from a variety of existing public datasets such as LabelMe, MSRC, PASCAL VOC and Geometric Context. This dataset comprises of 9 classes namely sky, tree, road, grass, water, building, mountain, foreground object and unknown.

Though the images have approximately a size of 320 pixels by 240 pixels, they were not exactly of the same size. To make batch training possible, all images were resized to a dimension size of 224 pixels by 224 pixels using nearest neighbor interpolation which is the recommended image size for pretrained image models such as AlexNet and VGG variants. Random colour jittering was applied on the images and were normalized to have zero mean and unit variance before training.

## Instructions
1. Download the stanford background dataset from http://dags.stanford.edu/projects/scenedataset.html
2. Change the model name accordingly to train the different models in Training.py.
3. 3 metrics namely pixel accuracy, mean accuracy and mean IOU can be computed from Validation.py.

## Training approach
The optimizer used is stochastic gradient descent (SGD) with momentum. A constant learning rate of 0.01 was used throughout the training process. The input images were passed in batches of 2. The loss function used is standard cross entropy loss. The dataset was split randomly in the ratio of 9:1 for training and test sets. Since the dataset is relatively small for training a deep learning model, validation set was not created.

## Results

### FCN
![FCN result](https://user-images.githubusercontent.com/81757215/114264376-26c36200-9a1d-11eb-9958-a654d671935b.JPG)

### U-Net
![UNet result](https://user-images.githubusercontent.com/81757215/114264380-29be5280-9a1d-11eb-8a98-ff240b709f4c.JPG)

### Metrics
FCN has a pixel accuracy, mean accuracy and mean IOU of 78.7 %, 68.2 % and 55.8 % respectively while U-Net has a pixel accuracy, mean accuracy and mean IOU of 67.1 %, 57 % and 44.1 % respectively


