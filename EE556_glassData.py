# coding=utf-8
#!/usr/bin/python
# Cody Dillinger - EE556 - Homework 4 - MultiLayer Perceptron and Backpropagation for XOR Problem
import numpy as np
import matplotlib.pyplot as plt
import csv, math
m = 10
############################################################################


def activation_out(x):
    return (10.0 * np.exp(x)) / (1.0 + np.exp(x))
############################################################################


def activation_derivative_out(x):
    return (10.0 * np.exp(x)) / (np.exp(2.0*x) + 2.0*np.exp(x) + 1.0)
############################################################################


def activation1(x):
    return (1.0 * np.exp(x)) / (1.0 + np.exp(x))
############################################################################


def activation_derivative1(x):
    return (1.0 * np.exp(x)) / (np.exp(2.0*x) + 2.0*np.exp(x) + 1.0)
############################################################################


def mean_square_error(x):
    sum_ = 0
    sqr_err = []
    nan_found = False
    for i in range(len(x)):
        if math.isnan(x[i][0]):
            nan_found = True
        sqr = ((x[i][0]) **2)
        sum_ += sqr                     # sum of squares
        sqr_err.append([sqr])           # keep matrix of squared error values
    if nan_found:
        print 'nan found in error before squaring'
    return np.array(sqr_err), sum_ / len(x)       # return squares and mean square
############################################################################


class MLP:
    def __init__(self, inputs):
        self.inputs = inputs
        self.length = m                        # num hidden layers
        self.length0 = len(self.inputs[0])      # num features in feature vector
        self.w_input = np.random.random((self.length0, self.length)) *1
        self.w_hidden = np.random.random((self.length, 1)) *1
        #print 'w in', self.w_input
        #print 'w hidden', self.w_hidden

    def predict(self, input):
        sum1 = np.dot(input, self.w_input)
        layer1_out = activation1(sum1)
        sum2 = np.dot(layer1_out, self.w_hidden)
        layer2_out = activation_out(sum2)
        return layer2_out

    def learn(self, inputs, outputs, alpha):
        learning = True
        iter = 0
        convergence_value = 1       # step size when back propagation learning is "done"
        costs = []
        while learning:
            print 'w in', self.w_input[0], self.w_input[5]
            print 'w hidden', self.w_hidden[0], self.w_hidden[5]
            sum1 = np.dot(inputs, self.w_input)
            print 'sum1', sum1[0], sum1[5], sum1[10]
            layer1_out = activation1(sum1)
            print 'layer1 out', layer1_out[0], layer1_out[10], layer1_out[18]
            sum2 = np.dot(layer1_out, self.w_hidden)
            print 'sum2', sum2[0], sum2[5], sum2[10]
            layer2_out = activation1(sum2)
            print 'layer2 out', layer2_out[0], layer2_out[10], layer2_out[18]
            layer2_error = outputs - layer2_out
            print 'iter', iter, ', layer2 error', layer2_error[0], layer2_error[50], layer2_error[78], layer2_error[85], layer2_error[90]
            #layer2_sq_error = np.array([[layer2_error[0][0]**2],[layer2_error[1][0]**2],[layer2_error[2][0]**2],[layer2_error[3][0]**2]])
            #print 'layer 2 error', layer2_error
            square_err, mean_square_err = mean_square_error(layer2_error)
            print 'iter', iter, ', layer2 square err', square_err[0], square_err[50], square_err[78], square_err[85], square_err[90]
            costs.append(mean_square_err)
            #layer2_delta = square_err * activation_derivative(layer2_out)
            layer2_delta = (1/m) * layer2_error * activation_derivative_out(layer2_out)
            #if iter == 0:
            #    print 'delta calculated', layer2_delta
            #layer1_error = np.dot(layer2_delta, self.w_hidden.T)          # .T is transpose with np array
            dJ_dW_hidden = np.dot(layer2_delta.T, layer1_out)
            #if iter == 0:
            #    print 'l1 error calculated', layer1_error
            #layer1_delta = (1/m) * layer1_error * activation_derivative1(layer1_out)
            layer1_delta = np.dot(layer2_delta, dJ_dW_hidden) * activation_derivative1(layer1_out)
            dJ_dW1 = np.dot(layer1_delta.T, inputs)
            #if iter == 0:
            #    print 'l1 delta calculated', layer1_delta
            #print 'w are', self.w_hidden, self.w_input
            self.w_hidden = self.w_hidden - alpha * dJ_dW_hidden #np.dot(layer1_out.T, layer2_delta)
            self.w_input = self.w_input - alpha * dJ_dW1 #np.dot(inputs.T, layer1_delta)
            print 'iter', iter, ', error', mean_square_err
            iter += 1
            if mean_square_err < convergence_value or iter > 10000:
                learning = False
        #print 'w in', self.w_input
        #print 'w hidden', self.w_hidden
        return costs
############################################################################


def get_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        features = []
        class_parameter = []
        for row in reader:
            features.append([float(i) for i in row[:10]])
            class_parameter.append([int(row[10])])
    print features
    print class_parameter
    features = np.array(features)
    class_parameter = np.array(class_parameter)
    return features, class_parameter
############################################################################


def main():
    glass_training_features, glass_training_class = get_data('glass1.data')        # half of data
    glass_features, glass_correct_class = get_data('glass2.data')                  # other half of data
    perceptron = MLP(glass_training_features)
    costs = perceptron.learn(glass_training_features, glass_training_class, 1)   # train the parameters
    print 'prediction for training set:', perceptron.predict(glass_training_features)
    print 'prediction for inputs2', perceptron.predict(glass_features)
    print 'costs', costs
    plt.plot(costs)
    plt.ylabel('Cost: Mean Square Error')
    plt.title('Codys Cost Function (MSE) Plot for HW 4 Glass Problem')
    plt.xlabel('Number of Batch Gradient Steps')
    plt.show()
    return

main()
