# Keras_AutoEncoder
A simple autoencoder written with Keras. Using 'mnist' data set to validate the efficiency of the the autoencoder.
# About mnist data set
This data set contains hand-written digit numbers of 28 by 28 pixels. It is a subset of a larger set available from NIST.
Training data contains 60000 samples and testing set contains 10000 samples.
# AutoEncoder
Autoencoders try to reduce the dimension of the input hand-written digits. From 784-d to 80-d or less.
Specifically, here autoencoders play a role just like Priminent Component Analysis (PCA) in machine learning.
# About training
For optimizer, we use 'rmsprop'; for loss, we use Mean Square Error (MSE).
# The classification model
Very simple multi-layer perceptron with only one hidden layer.
# Other
See the codes.
