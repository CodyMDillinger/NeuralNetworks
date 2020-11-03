# coding=utf-8
#!/usr/bin/python
# Cody Dillinger - EE556 - Homework 4 - MultiLayer Perceptron and Backpropagation for XOR Problem
import numpy as np
import matplotlib.pyplot as plt

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
        self.length = 5#len(self.inputs)
        self.length0 = len(self.inputs[0])
        self.w_input = np.random.random((self.length0, self.length))
        self.w_hidden = np.random.random((self.length, 1))
        #print 'w in', self.w_input
        #print 'w hidden', self.w_hidden

    def predict(self, input):
        sum1 = np.dot(input, self.w_input)
        layer1_out = activation(sum1)
        sum2 = np.dot(layer1_out, self.w_hidden)
        layer2_out = activation(sum2)
        return layer2_out

    def learn(self, inputs, outputs, alpha):
        learning = True
        iter = 0
        convergence_value = .00001       # squared error for when back propagation learning is "done"
        costs = []
        while learning:
            sum1 = np.dot(inputs, self.w_input)
            layer1_out = activation(sum1)
            sum2 = np.dot(layer1_out, self.w_hidden)
            layer2_out = activation(sum2)
            layer2_error = outputs - layer2_out
            #if iter == 0:
            #    print 'layer2 error first iteration', layer2_error
            #layer2_sq_error = np.array([[layer2_error[0][0]**2],[layer2_error[1][0]**2],[layer2_error[2][0]**2],[layer2_error[3][0]**2]])
            mean_square = .25 * (layer2_error[0][0]**2 + layer2_error[1][0]**2 + layer2_error[2][0]**2 + layer2_error[3][0]**2)
            costs.append(mean_square)
            if iter == 0:
                print 'layer2 mean square error first iteration', mean_square
            layer2_delta = layer2_error * activation_derivative(layer2_out)
            #if iter == 0:
            #    print 'delta calculated', layer2_delta
            layer1_error = np.dot(layer2_delta, self.w_hidden.T)          # .T is transpose
            #if iter == 0:
            #    print 'l1 error calculated', layer1_error
            layer1_delta = layer1_error * activation_derivative(layer1_out)
            #if iter == 0:
            #    print 'l1 delta calculated', layer1_delta
            #print 'w are', self.w_hidden, self.w_input
            self.w_hidden = self.w_hidden + alpha * np.dot(layer1_out.T, layer2_delta)
            self.w_input = self.w_input + alpha * np.dot(inputs.T, layer1_delta)
            iter += 1
            if mean_square < convergence_value:
                learning = False
        print 'w in', self.w_input
        print 'w hidden', self.w_hidden
        return costs


def main():
    inputs1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])
    perceptron = MLP(inputs1)
    costs = perceptron.learn(inputs1, outputs, 1)    # train the parameters
    inputs2 = np.array([[1, 1], [0, 0], [1, 0], [0, 1], [0, 0], [1, 1], [0, 1]])
    print 'prediction for inputs1', perceptron.predict(inputs1)
    print 'prediction for inputs2', perceptron.predict(inputs2)
    #print 'costs', costs
    plt.plot(costs)
    plt.ylabel('Cost: Mean Square Error')
    plt.title('Codys Cost Function (MSE) Plot for HW 4 XOR Problem')
    plt.xlabel('Number of Batch Gradient Steps')
    plt.show()
    return

main()
