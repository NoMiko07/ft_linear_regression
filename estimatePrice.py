import numpy as np
import pandas as pd
from dataSetClass import DataSet
import glob
from utils import estimatePrice

def getThetas():
    thetasData = np.zeros((2, 1))
    filenames = glob.glob('*.csv')
    if "thetas.csv" not in filenames:
        print("Warning: thetas.csv is not created yet !")
        return thetasData
    
    datas = pd.read_csv('thetas.csv', header=None)
    
    thetasData[0] = float(datas.iloc[0].values)
    thetasData[1] = float(datas.iloc[1].values)
    return thetasData


def	main():
    
    newdata = DataSet()
    newdata.initialize_data()
    newdata.thetas = getThetas()
    
    try:
        user_input = int(input("Please type a mileage : "))
        if user_input < 0:
            print("Error: mileage can't be under 0!")
        else:
            estimated_price = estimatePrice(user_input, newdata.x_minMax, newdata.y_minMax, newdata.thetas)
            if estimated_price < 0:
                print("The estimated price is negative, due to an excessively high mileage.")
            else:
                print(f"Estimated price for {user_input} mileage: {estimated_price:.2f}")
    except ValueError:
            print("Error: not a number.")

if __name__ == "__main__":
	main()