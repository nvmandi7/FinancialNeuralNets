from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from retrieve_data import design, labels, lookback_days
import numpy as np

test_days = 250

X_train = design[:-test_days]
X_test = design[-test_days:]

y_train = labels[:-test_days]
y_test_noncat = labels[-test_days:]

print("Number of BUY Training Examples", y_train.count(1))
print("Number of SELL Training Examples", y_train.count(0))
print("Number of HOLD Training Examples", y_train.count(2))
print("Number of BUY Test Examples", y_test_noncat.count(1))
print("Number of SELL Test Examples", y_test_noncat.count(0))
print("Number of HOLD Test Examples", y_test_noncat.count(2))
num_classes = 3

y_train = np_utils.to_categorical(y_train, num_classes) #Converts label data to a matrix
y_test = np_utils.to_categorical(y_test_noncat, num_classes)

#################################
# Model specification
# Start from an empty sequential model where we can stack layers
model = Sequential()

## Add a fully-connected layer.
model.add(Dense(output_dim=50, input_dim=len(design.T)))

## Add tanh activation function to each neuron
model.add(Activation("relu"))

## Add a fully-connected layer.
model.add(Dense(output_dim=50, input_dim=50))

## Add tanh activation function to each neuron
model.add(Activation("tanh"))

## Add another fully-connected layer with 2 neurons, one for each class of labels
model.add(Dense(output_dim=3, activation='softmax'))

'''
Trying Recurrent NN
'''
# model = Sequential()
# model.add(Embedding(len(X_train), 128, input_length=lookback_days, dropout=0.2))
# model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.fit(X_train, y_train, nb_epoch=20, batch_size=2,validation_split=0.1, show_accuracy=True)
# score, acc = model.evaluate(X_test, y_test,
#                             batch_size=2)
# print('Test score:', score))
# print('Test accuracy:', acc))

##################################

## Compile the model with categorical_crossentrotry as the loss, and stochastic gradient descent (learning rate=0.001, momentum=0.5,as the optimizer)
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.5), metrics=['accuracy'])
# for l in model.layers: print(l.get_weights(), '\n')

## Fit the model (10% of training data used as validation set)
model.fit(X_train, y_train, nb_epoch=2, batch_size=1,validation_split=0.1, show_accuracy=True)

score, acc = model.evaluate(X_test, y_test, batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)


test_labels = model.predict_classes(X_test, batch_size=1,verbose=1)

# test_labels = list(test_labels)
print('SELL: ', sum(test_labels==0))
print('HOLD: ', sum(test_labels==1))
print('BUY: ', sum(test_labels==2))

def accuracy_matrix(actual, predicted):
	mat = np.ndarray((3,3))
	for i in range(3):
		inds = np.where(actual==i)
		i_preds = predicted[inds]
		mat[i] = np.array([np.sum([1 for k in i_preds if k==j]) for j in range(3)])
	return mat/mat.sum(axis=1)[:,np.newaxis]

print(accuracy_matrix(np.array(y_test_noncat), test_labels))

