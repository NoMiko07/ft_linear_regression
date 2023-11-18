import numpy as np
from dataSetClass import DataSet
from utils import estimatePrice, normalizeThisValue, denormalizeData, createThetasCSV
from graph import showLinearReg, showCostHistory

def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)


def grad(X, y, theta):
    m = len(y) - 1
    return 1/m * X.T.dot(model(X, theta) - y)

def grad_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta =  theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    return theta, cost_history

def coef_determination(y, prediction):
    """ To evaluate the precision of the linear regression model used """
    u = ((y - prediction)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - (u/v)

def	main():
    learning_rate = 0.1
    n_iterations = 1000
    newdata = DataSet()
    newdata.initialize_data()
    theta = np.random.randn(2, 1)
    newdata.thetas, cost_history = grad_descent(newdata.X, newdata.yScaled["scaled_data"], theta, learning_rate, n_iterations)
    createThetasCSV(newdata.thetas)
    prediction = model(newdata.X, newdata.thetas)
    prediction = denormalizeData(prediction, newdata.yScaled['scaler'])
    showLinearReg(newdata.mileages, newdata.prices, prediction)
    showCostHistory(cost_history)
    coef = coef_determination(newdata.prices, prediction) * 100
    print(f"The precision of my algorithm is {coef:.2f}%")
       

if __name__ == "__main__":
	main()