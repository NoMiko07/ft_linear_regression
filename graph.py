import matplotlib.pyplot as plt
from dataSetClass import DataSet

def showLinearReg(x, y, prediction):
    """Display the scatter plot of actual data points and the regression line."""
    plt.figure(1)
    plt.scatter(x, y)
    plt.plot(x, prediction, c='r')
    plt.ylabel('prices')
    plt.xlabel('mileages')
    plt.title('Linear Regression: Actual and Predicted')
    plt.show()

def showCostHistory(cost_history, n_iterations):
    """Display the cost history plot during the iterations of gradient descent."""
    plt.figure(2)
    plt.plot(range(n_iterations), cost_history)
    plt.ylabel('cost')
    plt.xlabel('numbers of iterations')
    plt.title('Cost History during Gradient Descent')
    plt.show()
    
def showLinearRegProgression(DataSet , thetas_history):
    """Display the evolution of the linear regression. """
    from train_model import model    
    from utils import denormalizeData
    plt.figure(3, figsize=(8, 6))
    plt.scatter(DataSet.mileages , DataSet.prices)
    colors = ['r', 'g', 'b', 'c', 'm']
    k = 0
    prediction = model(DataSet.X, thetas_history[0])
    prediction = denormalizeData(prediction, DataSet.yScaled['scaler'])
    plt.plot(DataSet.mileages, prediction, color=colors[0])
    for i in range(len(thetas_history)):
        color_index = i % len(colors)
        if i % 50 == 0:
            k += 1
            if k > 4:
                k = 0
            prediction = model(DataSet.X, thetas_history[i])
            prediction = denormalizeData(prediction, DataSet.yScaled['scaler'])
            plt.plot(DataSet.mileages, prediction, color=colors[k])