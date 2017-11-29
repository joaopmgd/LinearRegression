# Linear Regression using Numpy

This project is part of my will to understand and implement Machine Learning. My objective is to understand the key concepts firts and then start using Frameworks for it.

The first step in this journey is to implement a Linear Regression Model where given a certain dataset, I want to find the Line that Best Fits the data points.

### Linear Regression Basics

The objective of this algorithm is to find a line that best represents the scattered data. This exercise shows an example of some data plotted in a graph, and is clearly possible to see that there is no single linear line that can correlate every X point to an Y point, but it is possible to find a line that best represents this dataset and has a small error (Loss) associated with it, it is called the Line of Best Fit

The Line of Best Fit is the best representation of the data done by a Linear Function. The Feedforward, that is, the prediction for each example using a linear function, in the first Epoch starts with random values for the Weight and bias and the predicted y_hat will compare poorly with the expected output Y in the Loss Function(J).

The sum of all Mean Square Error (MSE) of all predictions y_ and expected y is defined as the Loss function, so the goal is to minimize it by using the Backpropagation  of the error The Backpropagation will use the Gradient Descent method and take the partial derivative of each variable, W and B (called gradients dw and db), and use them with the Learning Rate alpha to update the Weight and bias.

For each Epoch, that is, for each round that every example in the training set was predicted and evaluated, the Loss decreases, in this example it is possible to see the learning happening as the error decreases over the number o iterations in the last graph.

The Line of Best Fit represents a pattern hidden in the data, the final outcome may not represent the real world and its variances but it can be a really good approximation for the proposed task. This example is a very simple one, it only has one input feature (Population of the city) and logic learned depends entirely on it, so it can be demonstrated as a great learning example but not a good real world application.

### Prerequisites

To keep things simple, this project will use just two libraries in python, one to make math computations and other to plot the graph with the dataset and the Line of Best Fit.

So in yout Python3 enviroment just make sure that you have this dependencies installed.

```
pip install numpy
```
```
python -mpip install -U pip
python -mpip install -U matplotlib
```

### Running

To run this project just make sure that both, python file is in the same location as the Profit_X_Population.txt file

To run it just execute the following command line:
```
python3 LinearRegressionNumpy.py
```

## Results

The Hyperparameters are initialized randomly, so the final total cost may vary, but it should be close to 4.480...

The Line of Best Fit must be plotted dynamically with its change over time with the dataset behind it, so we can see the learning process happening.


## References

This project was heavily influenced by the Coursera Courses of Andrew NG, Machine Learning and Deep Learning Specialization. There are concepts applied here of both courses and the example and dataset is from the Machine Learning Course.

Machine Learning:
https://www.coursera.org/learn/machine-learning

Deep Learning Specialization:
https://www.coursera.org/specializations/deep-learning

Thanks ;D