import numpy as np
from utils import getDataFile, convertNumpyArrayAndReshape, normalizeData

class DataSet:
    def __init__(self):
        self.data = None
        self.mileages = None
        self.prices = None
        self.xScaled = None
        self.yScaled = None
        self.x_minMax = [0, 0]
        self.y_minMax = [0, 0]
        self.thetas = None
        self.X = None

    def initialize_data(self):
        self.data = getDataFile()
        self.mileages = convertNumpyArrayAndReshape(self.data, 'km')
        self.prices = convertNumpyArrayAndReshape(self.data, 'price')
        self.xScaled = normalizeData(self.mileages)
        self.yScaled = normalizeData(self.prices)
        self.x_minMax[0] = self.mileages.min()
        self.x_minMax[1] = self.mileages.max()
        self.y_minMax[0] = self.prices.min()
        self.y_minMax[1] = self.prices.max()
        self.X = np.hstack((self.xScaled["scaled_data"], np.ones_like(self.xScaled["scaled_data"])))
