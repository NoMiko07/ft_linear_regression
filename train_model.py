import numpy as np
from dataSetClass import DataSet
from utils import denormalizeData, createThetasCSV
from graph import showLinearReg, showCostHistory

def model(X, theta):
    """
    Predict the output using a linear regression model.

    Parameters:
    - X (numpy array): Feature matrix.
    - theta (numpy array): Coefficient matrix.

    Return:
    - numpy array: Predicted output.
    """
    return X.dot(theta)


def cost_function(X, y, theta):
    """
    Calculate the cost function for linear regression.

    Parameters:
    - X (numpy array): Feature matrix.
    - y (numpy array): Actual output values.
    - theta (numpy array): Coefficient matrix.

    Return:
    - float: Cost of the linear regression model.
    """
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)


def grad(X, y, theta):
    """
    Compute the gradient of the cost function for linear regression.

    Parameters:
    - X (numpy array): Input features matrix.
    - y (numpy array): Target variable vector.
    - theta (numpy array): Coefficient vector.

    Return:
    - numpy array: Gradient vector.
    """
    m = len(y) - 1
    return 1/m * X.T.dot(model(X, theta) - y)

def grad_descent(X, y, theta, learning_rate, n_iterations):
    """
    Perform gradient descent optimization to minimize the cost function for linear regression.

    Parameters:
    - X (numpy array): Input features matrix.
    - y (numpy array): Target variable vector.
    - theta (numpy array): Initial coefficient vector.
    - learning_rate (float): Step size for gradient descent.
    - n_iterations (int): Number of iterations for gradient descent.

    Returns:
    - tuple: Final coefficient vector and an array containing the cost history.
    """
    cost_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta =  theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    return theta, cost_history

def coef_determination(y, prediction):
    """
    Calculate the coefficient of determination (R^2) for a regression model.

    Parameters:
    - y (numpy array): Actual output values.
    - prediction (numpy array): Predicted values.

    Return:
    - float: Coefficient of determination (R^2).
    """
    u = ((y - prediction)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - (u/v)

def	main():
    learning_rate = 0.3
    n_iterations = 1000
    newdata = DataSet()
    newdata.initialize_data()
    newdata.thetas, cost_history = grad_descent(newdata.X, newdata.yScaled["scaled_data"], newdata.thetas, learning_rate, n_iterations)
    createThetasCSV(newdata.thetas)
    prediction = model(newdata.X, newdata.thetas)
    prediction = denormalizeData(prediction, newdata.yScaled['scaler'])
    showLinearReg(newdata.mileages, newdata.prices, prediction)
    showCostHistory(cost_history, n_iterations)
    coef = coef_determination(newdata.prices, prediction) * 100
    print(f"The precision of my algorithm is {coef:.2f}%")
       

if __name__ == "__main__":
	main()