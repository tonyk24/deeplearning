import csv
import datetime

class SimpleDataPoint:
	def __init__(self, typeId, val, ts):
		self.measurementTypeId = typeId
		self.value = val
		self.timestamp = ts

class DataPoint:
	def __init__(self, tempValue, humValue, ts):
		self.temperatureValue = tempValue
		self.humidityValue = humValue
		self.timestamp = ts

sdpList = []
# Read in values from csv file
with open('output_outdoor_data.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=';')
	for row in csv_reader:
		date_time_obj = datetime.datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S')
		sdp = SimpleDataPoint(int(row[1]), float(row[3]),date_time_obj)
		sdpList.append(sdp)
		
print(len(sdpList))

dpListTemp = []
# First extract the temperature values
for item in sdpList:
	# type 1 is temperature
	if item.measurementTypeId == 1:
		dp = DataPoint(item.value, 0.0, item.timestamp)
		dpListTemp.append(dp)

print(len(dpListTemp))

# Expand the list of temperatures to one value for each hour
expandedDpListTemp = []

i=0
while True:
	if ( i < len(dpListTemp)-1):
		# Set minute and seconds to zero for current hour
		truncatedDateTime = dpListTemp[i].timestamp
		truncatedDateTime = truncatedDateTime.replace(minute=0, second=0, microsecond=0)
		# +1 hour from truncatedDateTime
		nextHour = truncatedDateTime + datetime.timedelta(seconds = 3600)
		
		# Intergrate to get the value for each hour period 
		
		# Index for next hour in vector
		j = i+1 
		sum_temp = 0.0
		currentTime = truncatedDateTime
		update = False
		while True:
			if j == (len(dpListTemp)-1):
				break
			# Next hour in vector is past next hour
			elif dpListTemp[j].timestamp >= nextHour:
				diff_period = nextHour-currentTime
				diff_period_min = (diff_period.days*24.0*60 + diff_period.seconds/60.0)
				sum_temp = sum_temp + diff_period_min*dpListTemp[j-1].temperatureValue
				expandedDpListTemp.append(DataPoint(sum_temp / 60.0, 0.0 / 60.0, truncatedDateTime))
				
				sum_temp = 0.0
				truncatedDateTimeNext = dpListTemp[j].timestamp
				truncatedDateTimeNext = truncatedDateTimeNext.replace(minute=0, second=0, microsecond=0)
				nextHour = nextHour + datetime.timedelta(seconds = 3600)
				while nextHour <= truncatedDateTimeNext:
					sum_temp = dpListTemp[j-1].temperatureValue
					expandedDpListTemp.append(DataPoint(sum_temp, 0.0, nextHour - datetime.timedelta(seconds = 3600)))
					nextHour = nextHour + datetime.timedelta(seconds = 3600)
				break
			# Next hour in vector is within the next hour
			else:
				ts = dpListTemp[j].timestamp
				diff_period = ts-currentTime
				diff_period_min = (diff_period.days*24.0*60 + diff_period.seconds/60.0)
				sum_temp = sum_temp + diff_period_min*dpListTemp[j-1].temperatureValue
			currentTime = dpListTemp[j].timestamp
			i = j
			j = j + 1
		if update:
			expandedDpListTemp.append(DataPoint(sum_temp / 60.0, 0.0 / 60.0, truncatedDateTime))
	
	if i == len(dpListTemp)-2:
		break
	i = i + 1
	print('iteration: ' + str(i))
	print(100.0*float(i)/float(len(dpListTemp)))

with open('expanded_outdoor_tempdata.csv', mode='w') as csv_file:
	fieldnames = ['period', 'temperature', 'humidity']
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
	
	writer.writeheader()
	for dp in expandedDpListTemp:
		writer.writerow({'period' : dp.timestamp,
						'temperature' : str(dp.temperatureValue),
						'humidity' : str(dp.humidityValue)})
						
dpListHum = []
# First extract the temperature values
for item in sdpList:
	# type 1 is temperature
	if item.measurementTypeId == 2:
		dp = DataPoint(0.0, item.value, item.timestamp)
		dpListHum.append(dp)

print(len(dpListHum))

# Expand the list of temperatures to one value for each hour
expandedDpListHum = []

i=0
while True:
	if ( i < len(dpListHum)-1):
		# Set minute and seconds to zero for current hour
		truncatedDateTime = dpListHum[i].timestamp
		truncatedDateTime = truncatedDateTime.replace(minute=0, second=0, microsecond=0)
		# +1 hour from truncatedDateTime
		nextHour = truncatedDateTime + datetime.timedelta(seconds = 3600)
		
		# Intergrate to get the value for each hour period 
		
		# Index for next hour in vector
		j = i+1 
		sum_hum = 0.0
		currentTime = truncatedDateTime
		update = False
		while True:
			if j == (len(dpListHum)-1):
				break
			# Next hour in vector is past next hour
			elif dpListHum[j].timestamp >= nextHour:
				diff_period = nextHour-currentTime
				diff_period_min = (diff_period.days*24.0*60 + diff_period.seconds/60.0)
				sum_hum = sum_hum + diff_period_min*dpListHum[j-1].humidityValue
				expandedDpListHum.append(DataPoint( 0.0 / 60.0, sum_hum / 60.0, truncatedDateTime))
				
				sum_hum = 0.0
				truncatedDateTimeNext = dpListHum[j].timestamp
				truncatedDateTimeNext = truncatedDateTimeNext.replace(minute=0, second=0, microsecond=0)
				nextHour = nextHour + datetime.timedelta(seconds = 3600)
				while nextHour <= truncatedDateTimeNext:
					sum_hum = dpListHum[j-1].humidityValue
					expandedDpListHum.append(DataPoint(0.0, sum_hum, nextHour - datetime.timedelta(seconds = 3600)))
					nextHour = nextHour + datetime.timedelta(seconds = 3600)
				break
			# Next hour in vector is within the next hour
			else:
				ts = dpListHum[j].timestamp
				diff_period = ts-currentTime
				diff_period_min = (diff_period.days*24.0*60 + diff_period.seconds/60.0)
				sum_hum = sum_hum + diff_period_min*dpListHum[j-1].humidityValue
			currentTime = dpListHum[j].timestamp
			i = j
			j = j + 1
		if update:
			expandedDpListHum.append(DataPoint( 0.0/ 60.0, sum_hum / 60.0, truncatedDateTime))
	
	if i == len(dpListHum)-2:
		break
	i = i + 1
	print('iteration: ' + str(i))
	print(100.0*float(i)/float(len(dpListHum)))

with open('expanded_outdoor_humdata.csv', mode='w') as csv_file:
	fieldnames = ['period', 'temperature', 'humidity']
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
	
	writer.writeheader()
	for dp in expandedDpListHum:
		writer.writerow({'period' : dp.timestamp,
						'temperature' : str(dp.temperatureValue),
						'humidity' : str(dp.humidityValue)})
						
def merge_temp_and_humidity_data(expandedDpListTemp, expandedDpListHum):
	expandedDpList = []
	
	start_timestamp = expandedDpListTemp[0].timestamp
	end_timestamp = expandedDpListTemp[-1].timestamp
	
	if expandedDpListTemp[0].timestamp.timestamp() >= expandedDpListHum[0].timestamp.timestamp():
		start_timestamp = expandedDpListTemp[0].timestamp
	else:
		start_timestamp = expandedDpListHum[0].timestamp
		
	
	i = 0
	j = 0
	while True:
		if expandedDpListTemp[i].timestamp < start_timestamp:
			i = i+1
		else:
			break
	while True: 
		if expandedDpListHum[j].timestamp < start_timestamp:
			j = j+1
		else:
			break
	
	print('i= ' + str(i) + ', j= ' + str(j))
	
	while True:
		if expandedDpListTemp[i].timestamp == expandedDpListTemp[-1].timestamp or expandedDpListHum[j].timestamp == expandedDpListHum[-1].timestamp:
			break
		if expandedDpListTemp[i].timestamp.timestamp() != expandedDpListHum[j].timestamp.timestamp():
			print('Conflict....i=' + str(i) + ', j=' + str(j))
			print(str(expandedDpListTemp[i].timestamp))
			print(str(expandedDpListHum[j].timestamp))
			break
		expandedDpList.append(DataPoint(expandedDpListTemp[i].temperatureValue, expandedDpListHum[j].humidityValue, expandedDpListTemp[i].timestamp))
		i = i + 1
		j = j + 1
	return expandedDpList

expandedDpList = merge_temp_and_humidity_data(expandedDpListTemp, expandedDpListHum)

with open('expanded_outdoor_data.csv', mode='w') as csv_file:
	fieldnames = ['period', 'temperature', 'humidity']
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
	
	writer.writeheader()
	for dp in expandedDpList:
		writer.writerow({'period' : dp.timestamp,
						'temperature' : str(dp.temperatureValue),
						'humidity' : str(dp.humidityValue)})
