# coding=utf-8
#!/usr/bin/python
# Cody Dillinger - EE556 - Homework 4 - MultiLayer Perceptron and Backpropagation for XOR Problem
import numpy as np


############################################################################
def activation(x):
    return ((2.0 * np.exp(x)) / (1.0 + np.exp(x))) - 1


############################################################################
def activation_derivative(x):
    return (2.0 * np.exp(x)) / (np.exp(2.0*x) + 2.0*np.exp(x) + 1.0)


############################################################################
class MLP:
    def __init__(self, inputs):
        self.inputs = inputs
        self.length = 2                           # num first layer sums
        self.length0 = len(self.inputs[0])        # num inputs in 1 input vector, plus bias input
        self.w_input = np.random.random((self.length0, self.length))
        self.w_hidden = np.random.random((self.length + 1, 1))
        print 'w in', self.w_input
        print 'w hidden', self.w_hidden

    def predict(self, input):
        print 'length from definition of class', self.length
        print 'length0 from definition of class', self.length0
        sum1 = np.dot(input, self.w_input)
        layer1_out = activation(sum1)
        sum2 = np.dot(layer1_out, self.w_hidden)
        layer2_out = activation(sum2)
        return layer2_out

    def learn(self, inputs, outputs, alpha):
        learning = True
        iter = 0
        convergence_value = .00001       # squared error for when back propagation learning is "done"
        while learning:
            sum1 = np.dot(inputs, self.w_input)
            print 'sum1', sum1
            layer1_out = activation(sum1)
            print 'layer1 out', layer1_out
            sum2_nobias = np.dot(layer1_out, [self.w_hidden[1], self.w_hidden[2]])
            print 'sum 2 no bias', sum2_nobias
            sum2 = sum2_nobias + self.w_hidden[0]       # np arrays add to each element, regular py arrays would append element
            print 'sum 2', sum2
            layer2_out = activation(sum2)
            print 'layer 2 out', layer2_out
            layer2_error = outputs - layer2_out
            print 'layer 2 error', layer2_error
            #if iter == 0:
            #    print 'layer2 error first iteration', layer2_error
            layer2_sq_error = np.square(layer2_error)   # np square squares individual elements
            print 'layer2 error, after squaring', layer2_sq_error
            layer2_delta = layer2_error * activation_derivative(layer2_out)
            print 'layer 2 delta calculated', layer2_delta
            layer1_error = np.dot(layer2_delta, (np.array([self.w_hidden[1], self.w_hidden[2]])).T)          # .T is transpose
            print 'l1 error calculated', layer1_error
            layer1_delta = layer1_error * activation_derivative(layer1_out)
            print 'l1 delta calculated', layer1_delta
            #print 'w are', self.w_hidden, self.w_input
            self.w_hidden = np.array([self.w_hidden[0], np.array([self.w_hidden[1], self.w_hidden[2]]) + alpha * np.dot(layer1_out.T, layer2_delta)])
            self.w_input = self.w_input + alpha * np.dot(inputs.T, layer1_delta)
            iter += 1
            if layer2_sq_error[0][0] < convergence_value and layer2_sq_error[1][0] < convergence_value and layer2_sq_error[2][0] < convergence_value and layer2_sq_error[3][0] < convergence_value:
                learning = False
        print 'w in', self.w_input
        print 'w hidden', self.w_hidden


def main():
    inputs1 = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1]])
    outputs = np.array([[0], [1], [1], [0], [1]])
    perceptron = MLP(inputs1)
    perceptron.learn(inputs1, outputs, .3)    # train the parameters
    inputs2 = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1]])
    print 'prediction for inputs1', perceptron.predict(inputs1)
    print 'prediction for inputs2', perceptron.predict(inputs2)


main()