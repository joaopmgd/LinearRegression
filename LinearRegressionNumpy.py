import numpy as np
import matplotlib.pyplot as plt

def load_data():
    points = np.genfromtxt('Profit_X_Population.txt', delimiter = ',')
    X = points[:,0]
    y = points[:,1]
    return X, y

def initialize_parameters():
    W = np.random.randn()
    b = np.random.randn()
    return W, b

def calculate_total_cost(X, y, W, b):
    m = len(X)
    predictions = predict(X, W, b)
    squaredErrors = (predictions - y)**2
    J = (1/(2*m)) * sum(squaredErrors)
    return J

def predict(X, W, b):
    return W * X + b

def gradient_descent(X, y, W, b, alpha = 0.01, num_iterations = 1500):
    m = len(X)
    cost_history = []
    for i in range (num_iterations):
        predictions = predict(X, W, b)
        cost = predictions - y
        W_gradient = alpha * (1/m) * sum (cost * X)
        b_gradient = alpha * (1/m) * sum (cost)
        W = W - W_gradient
        b = b - b_gradient
        cost_history.append(calculate_total_cost(X, y, W, b))
    return W, b, cost_history

def plot_cost_history(cost_history):
    plt.plot(cost_history, color = 'blue')
    plt.title('Cost Reduction by Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

def plot_data_with_prediction(X, y, W, b, title):
    training_data = plt.scatter(X, y, color = 'red', label="Traning Data")
    points = np.arange(4.5, 23.5, 0.1)
    predictions = predict(points, W, b)
    prediction_line, = plt.plot(points, predictions, color = 'blue', label="Prediction")
    plt.legend(handles=[training_data, prediction_line], loc = 4)
    plt.title(title)
    plt.xlabel('Population of City in $10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

def run():
    X, y = load_data()
    W, b = initialize_parameters()
    plot_data_with_prediction(X, y, W, b, "Training Data with Random Prediction")
    W, b, cost_history = gradient_descent(X, y, W, b)
    plot_data_with_prediction(X, y, W, b, "Training Data with Learned Prediction")
    plot_cost_history(cost_history)

if __name__ == '__main__':
    run ()