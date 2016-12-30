from scipy.io import loadmat
from keras.layers import Dense,Activation
from keras.models import Sequential
import numpy as np
# loading matrix, the result should be a python dictionary
trafficSign = loadmat('mnist.mat')
# get initial input size
input_size = trafficSign['train_X'].shape[1] #should be 2209
print 'input size is {}'.format(input_size)
lowerSize = 60
Encoder = Sequential([Dense(output_dim=lowerSize, input_dim=input_size), Activation('sigmoid')])
Decoder = Sequential([Dense(output_dim=input_size, input_dim=lowerSize), Activation('linear')])
AutoEncoderModel = Sequential()
AutoEncoderModel.add(Encoder)
AutoEncoderModel.add(Decoder)
AutoEncoderModel.compile(optimizer='rmsprop',loss='mse')
print
AutoEncoderModel.fit(trafficSign['train_X']/255.0, trafficSign['train_X']/255.0, nb_epoch=25)
print 'AutoEncoder training Success, starting classification...'
# X_train = trafficSign['train_X']
# X_test = trafficSign['test_X']
X_train = Encoder.predict(trafficSign['train_X']/255.0)
# AutoEncoderModel.layers[0].output
X_test = Encoder.predict(trafficSign['test_X']/255.0)
# X_train = AutoEncoderModel.predict(trafficSign['train_X']/255.0)
# X_test = AutoEncoderModel.predict(trafficSign['test_X']/255.0)
# X_train = trafficSign['train_X']/255.0
# X_test = trafficSign['test_X']/255.0
# X_train = Decoder.predict(X_train)
# X_test = Decoder.predict(X_test)
y_train = np.zeros([trafficSign['train_Y'].shape[1], 10])
for i in range(y_train.shape[0]):
    y_train[i,trafficSign['train_Y'][0, i]] = 1
# y_test = np.zeros([trafficSign['testlabel'].shape[0], 10])
# for i in range(y_test.shape[0]):
#     y_test[i,trafficSign['testlabel']-1] = 1
y_test = trafficSign['test_Y']
classNum = 10
print 'classNum={}'.format(classNum)
ClassModel = Sequential()
ClassModel.add(Dense(30, input_dim=lowerSize))
ClassModel.add(Activation('sigmoid'))
ClassModel.add(Dense(output_dim=classNum))
ClassModel.add(Activation('softmax'))
ClassModel.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
ClassModel.fit(X_train, y_train, shuffle=True, batch_size=100, nb_epoch=50, verbose=1)
y_predict = ClassModel.predict(X_test)
y_label = np.argmax(y_predict, axis = 1)
# print y_predict[:10]
# print y_label[:10]
# print y_predict.shape
# print y_label.shape
# print y_test.shape
test_num = y_test.shape[0]
correct = (y_label==y_test.reshape(y_test.shape[1]))
print 'all testing number is {}'.format(correct.shape[0])
print 'correct number is {}'.format(np.sum(correct))
print np.sum(correct)/np.float32(correct.shape[0])
# print correct.shape
# print type(correct)
# print correct[:10]
# print y_label[:10]
# print y_test[:10]
# print y_train[:10]
# print y_train[-10:]
# print 'score={}'.format(score)
# AutoEncoderModel.evaluate(trafficSign['testdata'], trafficSign['testlabel'])