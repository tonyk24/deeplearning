from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras import Sequential

import matplotlib.pyplot as plt

max_features = 12000
maxlen = 600
epochs = 10

def RNNModel(input_train, y_train, input_test, y_test):
	model = Sequential()
	# Embedding layer, dictionary that maps integer indices to dense vectors
	model.add(Embedding(max_features, 32))
	model.add(SimpleRNN(32))
	# Result as negative or positive review
	model.add(Dense(1, activation='sigmoid'))

	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
	history = model.fit(input_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2)

	test_loss, test_acc = model.evaluate(input_test, y_test, verbose=2)
	return history, test_loss, test_acc

def LSTMModel(input_train, y_train, input_test, y_test):
	model = Sequential()
	# Embedding layer, dictionary that maps integer indices to dense vectors
	model.add(Embedding(max_features, 32))
	model.add(LSTM(32))
	# Result as negative or positive review
	model.add(Dense(1, activation='sigmoid'))
	
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
	history = model.fit(input_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2)

	test_loss, test_acc = model.evaluate(input_test, y_test, verbose=2)
	return history, test_loss, test_acc

def plot(rnn_history, ltsm_history):
	plt.figure(figsize=(10,10))
	
	# RNN history data
	rnn_acc = rnn_history.history['acc']
	rnn_val_acc = rnn_history.history['val_acc']
	
	rnn_loss = rnn_history.history['loss']
	rnn_val_loss = rnn_history.history['val_loss']
	
	rnn_epochs = range(1, len(rnn_acc) + 1)
	
	# LTSM history data
	ltsm_acc = ltsm_history.history['acc']
	ltsm_val_acc = ltsm_history.history['val_acc']
	
	ltsm_loss = ltsm_history.history['loss']
	ltsm_val_loss = ltsm_history.history['val_loss']
	
	ltsm_epochs = range(1, len(ltsm_acc) + 1)
	
	fig, axes = plt.subplots(2, 2)
	
	axes[0, 0].plot(rnn_epochs, rnn_acc, 'bo', label='RNN Training acc')
	axes[0, 0].plot(rnn_epochs, rnn_val_acc, 'b', label='RNN Validation acc')
	axes[0, 0].set_title('RNN Training and validation accuracy')
	
	axes[1, 0].plot(rnn_epochs, rnn_loss, 'bo', label='RNN Training loss')
	axes[1, 0].plot(rnn_epochs, rnn_val_loss, 'b', label='RNN Vailidation loss')
	axes[1, 0].set_title('RNN Training and validation loss')
	
	axes[0, 1].plot(ltsm_epochs, ltsm_acc, 'bo', label='LTSM Training acc')
	axes[0, 1].plot(ltsm_epochs, ltsm_val_acc, 'b', label='LTSM Validation acc')
	axes[0, 1].set_title('LTSM Training and validation accuracy')
	
	axes[1, 1].plot(ltsm_epochs, ltsm_loss, 'bo', label='LTSM Training loss')
	axes[1, 1].plot(ltsm_epochs, ltsm_val_loss, 'b', label='LTSM Vailidation loss')
	axes[1, 1].set_title('LTSM Training and validation loss')
	
	plt.show()

	
print('Loading data')
# Load the IMDB dataset, review words and a vector representing either positive or nagative
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print(str(y_train))

#print('first: ' + str(input_train[0]) + ', ' + str(y_train[0]))

print('Pad sequences (samples x time)')
# Pad each sequence with zero until size of each is 600
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)


print('input_train shape: ', input_train.shape)
print('input_test shape: ', input_test.shape)

#print('first: ' + str(input_train[0]) + ', ' + str(y_train[0]))

rnn_history, rnn_test_loss, rnn_test_acc = RNNModel(input_train, y_train, input_test, y_test)
ltsm_history, ltsm_test_loss, ltsm_test_acc = LSTMModel(input_train, y_train, input_test, y_test)

print('\nRNN test accuracy:', rnn_test_acc)
print('\nLTSM test accuracy:', ltsm_test_acc)

plot(rnn_history, ltsm_history)
