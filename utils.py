import glob
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def getDataFile():
    '''Return datas if data.csv file exist/not empty else exit'''
    filenames = glob.glob('*.csv') # Get a list of all CSV files in the current file
    if "data.csv" not in filenames:
        print("Error: Missing file [data.csv]")
        sys.exit()

    datas = pd.read_csv('data.csv')
    if datas.empty:
        print("Error: data.csv has no dataframe")
        sys.exit()
    return datas

def convertNumpyArrayAndReshape(data, column):
    '''return a converted the data to numpy array and reshape it to (shape[0], 1)'''
    converted = data[column].to_numpy()
    converted = converted.reshape(converted.shape[0], 1)
    return converted


def normalizeData(data):
    '''Normalize the data using the Min-Max scaling method'''
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return {'scaled_data': scaled, 'scaler': scaler}

def denormalizeData(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)


def normalizeThisValue(value, minMax):
    normalized_value = (value - minMax[0]) / (minMax[1] - minMax[0])
    return normalized_value

def estimatePrice(mileage, minMaxMileage, minMaxPrice, thetas):
    normalized_mileage  = normalizeThisValue(mileage, minMaxMileage) * thetas[0] + thetas[1]
    estimated_price = normalized_mileage * (minMaxPrice[1] - minMaxPrice[0]) + minMaxPrice[0]
    return float(estimated_price[0])

def createThetasCSV(thetas):
    np.savetxt('thetas.csv', thetas, fmt='%.9f')