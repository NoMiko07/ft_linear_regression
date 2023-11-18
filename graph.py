import matplotlib.pyplot as plt

def showLinearReg(x, y, prediction):
    plt.figure(1)
    plt.scatter(x, y)
    plt.plot(x, prediction, c='r')
    plt.ylabel('prices')
    plt.xlabel('mileages')

def showCostHistory(cost_history):
    plt.figure(2)
    plt.plot(range(1000), cost_history)
    plt.ylabel('cost history')
    plt.xlabel('numbers of iterations')