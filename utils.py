import glob
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def check_numeric(column):
    """Verify that every values in the column can be convert to numeric 
    
    Parameter:
    - column (pandas.Series) : the column to verify
    
    Return:
    - bool : true if everything can be convert . False if not
    """
    try:
        pd.to_numeric(column)
        return True
    except ValueError:
        return False
    
def check_numeric_data(data):
    """Verify that every values of every columns can be convert to numeric
    
    Parameter:
    - data (pandas.DataFrame) : the data to verify
    
    Return:
    - bool : true if everything can be convert . False if not
    """
    
    for column in data.columns:
        if not check_numeric(data[column]):
            return False
        
    return True
        
    
    
def getDataFile():
    '''Return datas if data.csv file exist/not empty/no missing value else exit'''
    filenames = glob.glob('*.csv') # Get a list of all CSV files in the current file
    if "data.csv" not in filenames:
        print("Error: Missing file [data.csv]")
        sys.exit()
    try:
        datas = pd.read_csv('data.csv')
        if datas.empty:
            print("Error: data.csv has no dataframe")
            sys.exit()
        if datas.isnull().values.any():  
            print("Error: There are some missing values or NaN in the data.")
            sys.exit()       
        if not check_numeric_data(datas):
            print("Error: data.csv has not only numerical values or missing values")
            sys.exit()
    except Exception as e:
        print(f"Error: {e}")
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
    if thetas[0] == 0:
        return 0
    normalized_mileage  = normalizeThisValue(mileage, minMaxMileage) * thetas[0] + thetas[1]
    print(normalized_mileage)
    estimated_price = normalized_mileage * (minMaxPrice[1] - minMaxPrice[0]) + minMaxPrice[0]
    return float(estimated_price[0])

def createThetasCSV(thetas):
    np.savetxt('thetas.csv', thetas, fmt='%.9f')