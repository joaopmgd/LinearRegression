# Linear Regression is a form of Machine Learning where a linear function can approximate the outputs given a certain input.
# The function can be drawn and called as Line of Best Fit, where the cost/error/loss is minimized given all the datapoints.
# The imports are Numpy for the math calculations an matplotlib for plotting the data and the found linear function.
import numpy as np
import matplotlib.pyplot as plt

# load_data reads the data from a txt file and divedes it into X (input) and y (output).
def load_data():
    points = np.genfromtxt('Profit_X_Population.txt', delimiter = ',')
    X = points[:,0]
    y = points[:,1]
    return X, y

# W (weights) and b (bias) are  initialized with rando numbers, could use np.zero() too.
def initialize_parameters():
    W = np.random.randn()
    b = np.random.randn()
    return W, b

# The cost | loss is defined as the squared error, and the total loss is the mean squared error of all exemples.
def calculate_total_cost(X, y, W, b):
    m = len(X)
    predictions = predict(X, W, b)
    squaredErrors = (predictions - y)**2
    J = (1/(2*m)) * sum(squaredErrors)
    return J

# The predict function returns the prediction value when given an example, the weights and the bias values.
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

        if i % 50 == 0:
            plot_data_with_prediction(X, y, W, b)

    return W, b, cost_history

# The cost is computed at each 50 iterations of the Gradient Descent, and its value is stored to show the decreased value plotted in a graph.
def plot_cost_history(cost_history):
    plt.figure(2)
    plt.plot(cost_history, color = 'blue')
    plt.title('Cost Reduction by Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

# The data is plotted in a grapha with the Line of Best Fit predicted.
# This function is called in the gradient descent loop, so we can see the learning happening and the line changing to a better position.
def plot_data_with_prediction(X, y, W, b):
    plt.clf()
    training_data = plt.scatter(X, y, color = 'red', label="Traning Data")
    points = np.arange(4.5, 23.5, 0.1)
    predictions = predict(points, W, b)
    prediction_line, = plt.plot(points, predictions, color = 'blue', label="Prediction")
    plt.legend(handles=[training_data, prediction_line], loc = 4)
    plt.title("Training Data with Learning Prediction Function")
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.draw()
    plt.pause(0.00001)

# All the Machine Learning functions constitutes another Function called the model, where it:
# Reads the data, initialize the parameters, apply the cost reduction function, and predict for new values.
def model():
    X, y = load_data()
    W, b = initialize_parameters()
    W, b, cost_history = gradient_descent(X, y, W, b)
    plot_cost_history(cost_history)

if __name__ == '__main__':
    model()