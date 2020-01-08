import csv
import datetime
import numpy as np
import math
import time
import sys

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

class DataPoint:
	def __init__(self, tempValue, humValue, ts):
		self.temperatureValue = tempValue
		self.humidityValue = humValue
		self.timestamp = ts

dpList = []
alldata = np.empty((0,2), float)
# Read in values from csv file
with open('expanded_outdoor_data.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		date_time_obj = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
		dp = DataPoint(float(row[1]),float(row[2]),date_time_obj)
		dpList.append(dp)
		alldata = np.append(alldata, np.array([[float(row[1]),float(row[2])]]), axis=0)
print('alldata len=' + str(len(alldata)))
data_mean = alldata.mean(axis=0)
alldata -= data_mean
#print(alldata)
data_std = alldata.std(axis=0)
alldata /= data_std

print('data_mean=' + str(data_mean))
print('data_std=' + str(data_std))
print(len(dpList))


# One for each hour of one day
data = [None] * 24
labels = [None] * 24
days_back = 5
n = 0
# Number of hours minus days_back days
m = (len(dpList)-(days_back*24))-24
while n < 24:
	# (number of hours, number of hours the last five days, 2)
	data[n] = np.zeros((m, (days_back*24), 2))
	labels[n] = np.zeros((m, 1))
	n = n + 1

# Take data from five days back
i = days_back*24

while i < len(dpList)-24:
	#print(str(dpList[i].timestamp))
	
	for j in range(24):
		
		for k in range(days_back*24):
			data[j][i-days_back*24, k, 0] = dpList[i-days_back*24+k].temperatureValue
			data[j][i-days_back*24, k, 0] -= data_mean[0]
			data[j][i-days_back*24, k, 0] /= data_std[0]
			
			data[j][i-days_back*24, k, 1] = dpList[i-days_back*24+k].humidityValue
			data[j][i-days_back*24, k, 1] -= data_mean[1]
			data[j][i-days_back*24, k, 1] /= data_std[1]
		labels[j][i-days_back*24][0] = dpList[i+j].temperatureValue
		labels[j][i-days_back*24][0] -= data_mean[0]
		labels[j][i-days_back*24][0] /= data_std[0]
	i = i + 1
	

# Verify that data is correct
for i in range(24):
	#print('a=' + str(alldata[days_back*24 + i][0]) + ', b= ' + str((labels[i][0]-data_mean[0])/data_std[0]))
	assert abs(alldata[days_back*24 + i][0] - labels[i][0]) < 0.01
	assert abs(alldata[days_back*24 + i + 1][0] - labels[i][1]) < 0.01

for i in range(days_back*24):
	for j in range(24):
		#print('[i=' + str(i) + ',j=' + str(j) + ']a=' + str(alldata[i][0]* data_std[0] + data_mean[0]) + ', b=' + str(data[j][0, i, 0] * data_std[0] + data_mean[0]))
		assert abs(alldata[i][0] - data[j][0, i, 0]) < 0.01
		assert abs(alldata[i+1][0] - data[j][1, i,0]) < 0.01
		assert abs(alldata[i][1] - data[j][0, i, 1]) < 0.01
		assert abs(alldata[i+1][1] - data[j][1, i,1]) < 0.01
	
def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    print('rand1, orginal a=' + str(a[0]) + ', b= ' + str(b[0]))
    print('rand1, random  a=' + str(a[permutation[0]]) + ', b=' + str(b[permutation[0]]))
    return shuffled_a, shuffled_b

# Shuffle the data
data_x = [None] * 24
labels_y = [None] * 24
for i in range(24):
	data_x[i], labels_y[i] = randomize(data[i], labels[i])

sys.exit()
# Split data with 20% test data
test_x = [None] * 24
test_y = [None] * 24
split_percentage = 0.2
split_percentage_int = int(len(labels[0][:])*0.2)
print('data_x shape= ' + str(data_x[0].shape))
print('labels_y shape=' + str(labels_y[0].shape))
for i in range(24):
	test_x[i] = data_x[i][0:split_percentage_int]
	data_x[i] = data_x[i][split_percentage_int:]
	
	test_y[i] = labels_y[i][0:split_percentage_int]
	labels_y[i] = labels_y[i][split_percentage_int:]
	
print('test= ' + str(test_x[0].shape) + ', data= ' + str(data_x[0].shape))

def Average(lst): 
    return sum(lst) / len(lst)

def plot_loss(histories):
	plt.figure(figsize=(10,10))
	
	fig, axes = plt.subplots(6, 4)
	
	for i in range(6):
		for j in range(4):
			index = i*4 + j
			loss = histories[index].history['loss']
			val_loss = histories[index].history['val_loss']
			epochs = range(1, len(loss)+1)
			axes[i,j].plot(epochs, loss, 'bo', 'T')
			axes[i,j].plot(epochs, val_loss, 'b', 'V')
	
	plt.show()
def plot_one_loss(history, data_std):
	for i in range(len(history.history['loss'])):
		history.history['loss'][i] = history.history['loss'][i] * data_std
	for i in range(len(history.history['val_loss'])):
		history.history['val_loss'][i] = history.history['val_loss'][i] * data_std
	plt.plot(history.history['loss'], 'bo', label='Training loss')
	plt.plot(history.history['val_loss'], 'b', label='Validation loss')
	plt.xlabel('epochs')
	plt.ylabel('MAE')
	plt.show()

def save_model_to_file(type, models):
	for i in range(24):
		model_json = models[i].to_json()
		with open('model_data\\' + type + '_model_' + str(i) + '.json', "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		models[i].save_weights('model_data\\' + type + '_model_' + str(i) + '.h5')

def model_all_to_all(data, labels, test_data, test_labels):
	result  = [None] * 24
	history = [None] * 24

	model = [None] * 24
	for i in range(24):
		model[i] = Sequential()
		model[i].add(layers.Flatten(input_shape=(data[i][-1].shape)))
		model[i].add(layers.Dense(64, activation='relu'))
		model[i].add(layers.Dense(64, activation='relu'))
		model[i].add(layers.Dense(64, activation='relu'))
		model[i].add(layers.Dense(1))
		
		model[i].summary()
		
		model[i].compile(optimizer=RMSprop(), loss='mae')
		history[i] = model[i].fit(data[i], labels[i], epochs=50, validation_split=0.2)
		result[i] = model[i].evaluate(test_data[i], test_labels[i])
	return model, history, result

def model_rnn_lstm(data, labels, test_data, test_labels):
	result  = [None] * 24
	history = [None] * 24

	model = [None] * 24
	for i in range(24):
		model[i] = Sequential()
		model[i].add(layers.LSTM(32, return_sequences=True,input_shape=(data[i][-1].shape)))
		model[i].add(layers.LSTM(32, return_sequences=True, activation='relu'))
		model[i].add(layers.LSTM(32, activation='relu'))
		model[i].add(layers.Dense(1))
		
		model[i].summary()
		
		model[i].compile(optimizer=RMSprop(), loss='mae')
		history[i] = model[i].fit(data[i], labels[i], epochs=10)
		result[i] = model[i].evaluate(test_data[i], test_labels[i])
	return model, history, result

def model_rnn_gru(data, labels, test_data, test_labels):
	result  = [None] * 24
	history = [None] * 24

	model = [None] * 24
	for i in range(24):
		model[i] = Sequential()
		model[i].add(layers.GRU(32, input_shape=(data[i][-1].shape), return_sequences=True))
		model[i].add(layers.GRU(32, activation='relu'))
		model[i].add(layers.Dense(1))
		
		model[i].summary()
		
		model[i].compile(optimizer=RMSprop(), loss='mae')
		history[i] = model[i].fit(data[i], labels[i], epochs=10, validation_split=0.2)
		result[i] = model[i].evaluate(test_data[i], test_labels[i])
	return model, history, result

data_pred = np.zeros((1, (5*24), 2))
i = 0
with open('tmp/expanded_outdoor_data.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		data_pred[0, i, 0] = float(row[1])
		data_pred[0, i, 0] -= data_mean[0]
		data_pred[0, i, 0] /= data_std[0]
		data_pred[0, i, 1] = float(row[2])
		data_pred[0, i, 1] -= data_mean[1]
		data_pred[0, i, 1] /= data_std[1]
		i = i + 1


start_time = time.time()
models, history, result = model_all_to_all(data_x, labels_y, test_x, test_y)
save_model_to_file('n_to_n', models)
elapsed_time = time.time() - start_time
print('N-to-N elapsed time: ' + str(elapsed_time) + 's')
print('MAE= ' + str(result))
print('mean MAE= ' + str(Average(result) * data_std[0]))
for i in range(24):
	result = models[i].predict(data_pred)
	result1 = result[0][0]
	result2 = result[0][0]*data_std[0]
	result3 = result[0][0] + data_mean[0]
	print('result= ' + str(result1))
	print('result= ' + str(result2))
	print('result= ' + str(result3))
#plot_one_loss(history[0], data_std[0])
#plot_one_loss(history[12], data_std[0])
#plot_one_loss(history[23], data_std[0])


#start_time = time.time()
#history_lstm, result_lstm = model_rnn_lstm(data_x, labels_y, test_x, test_y)
#elapsed_time = time.time() - start_time
#print('LSTM elapsed time: ' + str(elapsed_time))
#print(Average(result_lstm))

'''
start_time = time.time()
models_gru, history_gru, result_gru = model_rnn_gru(data_x, labels_y, test_x, test_y)

save_model_to_file('gru', models_gru)
elapsed_time = time.time() - start_time
print('GRU elapsed time: ' + str(elapsed_time) + 's')
print('MAE= ' + str(result_gru))
print('mean MAE= ' + str(Average(result_gru) * data_std[0]))
plot_one_loss(history_gru[0], data_std[0])
plot_one_loss(history_gru[12], data_std[0])
plot_one_loss(history_gru[23], data_std[0])	
'''








