import csv
import datetime
import numpy as np
import math

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

print(len(dpList))

# One for each hour of one day
data = [None] * 24
labels = [None] * 24
days_back = 5
n = 0
# Number of hours minus days_back days
m = (len(dpList)-(days_back*24))
while n < 24:
	# (number of 24 hour periods, number of hours the last five days, 2)
	data[n] = np.zeros((int(math.floor(m / 24.0)), (days_back*24), 2))
	labels[n] = np.zeros((int(math.floor(m / 24.0)), 1))
	n = n + 1

# Take data from five days back
i = days_back*24

while i < int(math.floor(m / 24.0))*24+days_back*24:
	#print(str(dpList[i].timestamp))
	
	
	
	prediction_value_temp = dpList[i].temperatureValue
	prediction_value_hum = dpList[i].humidityValue
	
	j = i-days_back*24
	k = 0
	listTemp = []
	listHum = []
	while j < i:
		# Data for the nth hour, since we will have one model for each hour of the day
		n = (j) % 24
		# Index for the current hour to insert data for
		a = int(math.floor((i-days_back*24) / 24.0))
		# Index for the current hour in last five days
		# First: 
		b = k
		#print('[n= ' + str(n) + ', a=' + str(a) + ', b=' + str(b))
		#print('i= ' + str(i) + ', j=' + str(j))
		data[n][a, b, 0] = dpList[j].temperatureValue
		data[n][a, b, 0] -= data_mean[0]
		data[n][a, b, 0] /= data_std[0]
		
		data[n][a, b, 1] = dpList[j].humidityValue
		data[n][a, b, 1] -= data_mean[1]
		data[n][a, b, 1] /= data_std[1]
		
		labels[n][a, 0] = prediction_value_temp
		
		listTemp.append(dpList[j].temperatureValue)
		listHum.append(dpList[j].humidityValue)
		j = j + 1
		k = k + 1
	i = i + 1
#print(labels[0])
#print(data[0])

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

# Shuffle the data
data_x = [None] * 24
labels_y = [None] * 24
for i in range(24):
	data_x[i], labels_y[i] = randomize(data[i], labels[i])

# Split data with 20% test data
test_x = [None] * 24
test_y = [None] * 24
split_percentage = 0.2
split_percentage_int = int(len(labels[0][:])*0.2)
print('shape= ' + str(data_x[0].shape))
for i in range(24):
	test_x[i] = data_x[i][0:split_percentage_int]
	data_x[i] = data_x[i][split_percentage_int:]
	
	test_y[i] = labels_y[i][0:split_percentage_int]
	labels_y[i] = labels_y[i][split_percentage_int:]
	
print('test= ' + str(test_x[0].shape) + ', data= ' + str(data_x[0].shape))

def model_all_to_all(data, labels, test_data, test_labels):
	result  = [None] * 24
	history = [None] * 24

	model = [None] * 24
	for i in range(24):
		model[i] = Sequential()
		model[i].add(layers.Flatten(input_shape=(data[i][-1].shape)))
		model[i].add(layers.Dense(32, activation='relu'))
		model[i].add(layers.Dense(1))
		model[i].compile(optimizer=RMSprop(), loss='mae')
		history[i] = model[i].fit(data[i], labels[i], epochs=30)
		result[i] = model[i].evaluate(test_data[i], test_labels[i])
		#test_loss[i] = tl
		#test_acc[i] = ta
	return history, result

history, result = model_all_to_all(data_x, labels_y, test_x, test_y)








