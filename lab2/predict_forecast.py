import mysql.connector
import datetime
from datetime import timedelta
import csv

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json

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

# Get data from the last 10 days from the database and expand the data
# to get a temperature and humidity value for each hour in the period.
def get_last_x_days_data():
    cnx = mysql.connector.connect(user='pi', password='primax24',
                                  host='127.0.0.1',
                                  database='surveillence')
    cursor = cnx.cursor()

    query = ("select idMeasurementType, value, timestamp from device_measurement_value_last_week where timestamp >= ( CURDATE() - INTERVAL 10 DAY ) and idDevice=215 and (idMeasurementType=1 or idMeasurementType=2)  order by timestamp asc;")

    cursor.execute(query)

    sdpList = []
    for (idMeasurementType, value, timestamp) in cursor:
        date_time_obj = datetime.datetime.strptime(str(timestamp), '%Y-%m-%d %H:%M:%S')
        sdp = SimpleDataPoint(idMeasurementType, float(value), date_time_obj)
        sdpList.append(sdp)


    cursor.close()

    cnx.close()

    print(len(sdpList))
    print(str(sdpList[0].value))
    print(str(sdpList[0].timestamp))

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
    # First extract the humidity values
    for item in sdpList:
        # type 2 is humidity
        if item.measurementTypeId == 2:
            dp = DataPoint(0.0, item.value, item.timestamp)
            dpListHum.append(dp)

    print(len(dpListHum))

    # Expand the list of humidities to one value for each hour
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

    return expandedDpListTemp, expandedDpListHum

# Merge the list of temperature values with the list of humidity values
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


# Write the list of expanded data to file
def write_expanded_data(expandedDpList):
    with open('expanded_outdoor_data.csv', mode='w') as csv_file:
        fieldnames = ['period', 'temperature', 'humidity']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
	
        writer.writeheader()
        for dp in expandedDpList:
            writer.writerow({'period' : dp.timestamp,
                             'temperature' : str(dp.temperatureValue),
                             'humidity' : str(dp.humidityValue)})

# Fill out end of list to current datetime
def fill_out_end(expandedDpList):
    ret = expandedDpList
    currentDpTs = expandedDpList[-1].timestamp
    currentDpTs = currentDpTs + datetime.timedelta(hours=1)
    now = datetime.datetime.now()
    while currentDpTs < now:
        ret.append(DataPoint(expandedDpList[-1].temperatureValue,
                             expandedDpList[-1].humidityValue,
                             currentDpTs))
        currentDpTs = currentDpTs + datetime.timedelta(hours=1)
    print(str(now))
    
    return ret

# Remove items with timestamp larger than five days ago
def shave_of_past_five_days(expandedDpList):
    ret = []
    fiveDaysDeadLine = datetime.datetime.now() - datetime.timedelta(hours=120)

    for i in range(len(expandedDpList)):
        if expandedDpList[i].timestamp > fiveDaysDeadLine:
            ret.append(expandedDpList[i])
    
    return ret

# Clear the table of entries with specific location id
def clear_in_db(locationId):
    cnx = mysql.connector.connect(user='pi',
                                  password='primax24',
                                  host='127.0.0.1',
                                  database='surveillence')
    cursor = cnx.cursor()

    query = ('delete from weather_data_next_24h where weather_location_id=' + str(locationId) + ';')
             
    print('query= ' + str(query))

    cursor.execute(query)

    cnx.commit()
    print(cursor.rowcount, "Record inserted successfully")

    cursor.close()

    cnx.close()
       

# Store the predicted data in database
def store_in_db(type, data):
    if type == 'n_to_n':
        cnx = mysql.connector.connect(user='pi',
                                      password='primax24',
                                      host='127.0.0.1',
                                      database='surveillence')
        for item in data:
            cursor = cnx.cursor()

            query = ('insert into weather_data_next_24h('
                     + 'weather_location_id'
                     + ',temperature'
                     + ',air_humidity'
                     + ',timestamp) VALUES('
                     + '2,'
                     + str(item.value) + ','
                     + '0.0,\''
                     + str(item.timestamp)
                     + '\');')

            print('query= ' + str(query))

            cursor.execute(query)

            cnx.commit()
            print(cursor.rowcount, "Record inserted successfully")

            cursor.close()

        cnx.close()
    elif type == 'gru':
        cnx = mysql.connector.connect(user='pi',
                                      password='primax24',
                                      host='127.0.0.1',
                                      database='surveillence')

        for item in data:
            cursor = cnx.cursor()

            query = ('insert into weather_data_next_24h('
                     + 'weather_location_id'
                     + ',temperature'
                     + ',air_humidity'
                     + ',timestamp) VALUES('
                     + '3,'
                     + str(item.value) + ','
                     + '0.0,\''
                     + str(item.timestamp)
                     + '\');')

            cursor.execute(query)

            cnx.commit()
            cursor.close()


        cnx.close()

# Read in the models from file first
types = ['n_to_n', 'gru']

# Load models from json files and their weights
def get_model_for_hour(type, hour):
    model = None

    json_file = open('model_data/' + types[type] + '_model_' + str(hour) + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    print('Model ' + str(hour) + ' loaded model')
    # load weights into new model
    model.load_weights('model_data/' + types[type] + '_model_' + str(hour) + '.h5')
    print('Model ' + str(hour) + ' loaded model weights')
    
    return model


# Loop forever
while True:
    # Get data from database, expand it and remove data longer than five days ago
    expandedDpListTemp, expandedDpListHum = get_last_x_days_data()

    expandedDpList = merge_temp_and_humidity_data(expandedDpListTemp, expandedDpListHum)
    expandedDpList = fill_out_end(expandedDpList)

    expandedDpList = shave_of_past_five_days(expandedDpList)
    write_expanded_data(expandedDpList)
    # Data mean and std it was trained on
    data_mean=[ 9.23781749, 92.8578085 ]
    data_std=[6.54929013, 9.24427348]


    # store data from last five days in array
    data_x = np.zeros((1, (5*24), 2))

    # Read data into array, and normalize
    for i in range(24*5):
        data_x[0, i, 0] = expandedDpList[i].temperatureValue
        data_x[0, i, 0] -= data_mean[0] 
        data_x[0, i, 0] /= data_std[0]
        data_x[0, i, 1] = expandedDpList[i].humidityValue
        data_x[0, i, 1] -= data_mean[1]
        data_x[0, i, 1] /= data_std[1]


    # Predict values for coming 24 hours using data from last five days
    # using the all-to-all models
    resultSpdNtoN = []
    for i in range(24):
        model = get_model_for_hour(0, i)
        result = model.predict(data_x)
        result1 = result[0][0]
        result2 = result[0][0]*data_std[0]
        result3 = result[0][0] + data_mean[0]
        print('result= ' + str(result1))
        print('result= ' + str(result2))
        print('result= ' + str(result3))
        resultSpdNtoN.append(SimpleDataPoint(2, result3, (expandedDpList[-1].timestamp+timedelta(hours=i+1))))
    # Clear all-to-all prediction table data and insert the new prediction data
    clear_in_db(2)
    store_in_db('n_to_n', resultSpdNtoN)

    # Predict values for coming 24 hours using data from last five days
    # using the GRU models
    resultSpdNGRU = []
    for i in range(24):
        model = get_model_for_hour(1, i)
        result = model.predict(data_x)
        result1 = result[0][0]
        result2 = result[0][0]*data_std[0]
        result3 = result[0][0] + data_mean[0]
        print('result= ' + str(result1))
        print('result= ' + str(result2))
        print('result= ' + str(result3))
        resultSpdNGRU.append(SimpleDataPoint(3, result3, expandedDpList[-1].timestamp+timedelta(hours=i+1)))
    # Clear GRU prediction table data and insert the new prediction data
    clear_in_db(3)
    store_in_db('gru', resultSpdNGRU)




