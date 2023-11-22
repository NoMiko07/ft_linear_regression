import matplotlib.pyplot as plt

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