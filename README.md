#Predict hand gestures using CNN

1. Techniques: python, numpy, tensorflow

2. Project's structure:
- prepdata.py: load and preprocess data
- train.py: for training the model using Marcel-Train dataset 
Note: when using train.py, change the line 25 in prepdata.py to path = os.path.join(train_path, fields, '*m')
- test.py: for testing the model using Marcel-Test
Note: when using test.py, change the line 25 in prepdata.py to :path = os.path.join(train_path, fields + "/uniform", '*m')
- predict.py: this file is used to predict one image class
Note: duplicate code for loading the image here, can be further optimized

3. Model's structure:
- 3 convolutional layers with filter size of (3x3)
- 1 flatten layer
- 2 fully connected layers
I have tried to add more convolutional layers but it did not improve the accuracy.

Potential further optimization: 
- implement image augmentation on the trainning set to achiver bigger set.
- mimic ResNet structure to go deeper into the network

