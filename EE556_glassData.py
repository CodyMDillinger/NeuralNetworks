# coding=utf-8
#!/usr/bin/python
# Cody Dillinger - EE556 - Homework 4 - MultiLayer Perceptron and Backpropagation for XOR Problem
import numpy as np
import matplotlib.pyplot as plt
import csv, math

############################################################################


def activation_out(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
    #e_x = np.exp(x - np.max(x))
    #return e_x / e_x.sum()
    #return (10.0 * np.exp(0.001*x)) / (1.0 + np.exp(0.001*x))
############################################################################


def activation_derivative_out(x):
    #jacob = []
    #print 'x', x
    #for i in range(len(x)):
    #    jacob.append([])
    #    for j in range(len(x)):
    #        jacob[i].append([])
    #        if i == j:
    #            jacob[i][j] = x[i] * (1 - x[i])
    #        else:
    #            jacob[i][j] = -x[i] * x[j]
    #print 'jacobian', jacob
    #return jacob
    deriv = (.01 * np.exp(0.001*x)) / ((np.exp(0.001*x) + 1.0)**2)
    #while deriv.any() < .1:
    #    deriv = deriv * 1.5
    return deriv
    #return (10.0 * np.exp(x)) / (np.exp(2.0*x) + 2.0*np.exp(x) + 1.0)
############################################################################


def activation1(x):
    return (2.0 * np.exp(x)) / (1.0 + np.exp(x)) - 1
############################################################################


def activation_derivative1(x):
    deriv = (2.0 * np.exp(x)) / (np.exp(2.0*x) + 2.0*np.exp(x) + 1.0)
    #while deriv.any() < .1:
    #    deriv = deriv * 1.5
    return deriv
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
        self.neuron_num = 10  #len(self.inputs)      # num neurons in hidden layers
        self.feature_num = len(self.inputs[0])       # num features in feature vector
        self.hidden_layer_num = 10                   # num hidden layers
        self.w_input = np.random.random((self.feature_num, self.neuron_num)) # *.01
        self.w_hidden = []
        for i in range(self.hidden_layer_num):
            if i < self.hidden_layer_num - 1:
                self.w_hidden.append(np.random.random((self.neuron_num, self.neuron_num)))
            else:
                self.w_hidden.append(np.random.random((self.neuron_num, 1)))
        #self.w_hidden1 = np.random.random((self.neuron_num, self.neuron_num)) *.01
        #self.w_hidden2 = np.random.random((self.neuron_num, 1)) *.01
        # print 'w in', self.w_input
        # print 'w hidden', self.w_hidden

    def predict(self, input):                       # returns final layer output prediction
        input_next = input
        weight_next = self.w_input
        for i in range(self.hidden_layer_num):
            sum = np.dot(input_next, weight_next)
            layer_out = activation1(sum)
            weight_next = self.w_hidden[i]
            input_next = layer_out
        sum = np.dot(input_next, weight_next)
        layer_out = activation_out(sum)
        return layer_out

    def get_outputs(self, inputs):                  # returns more data (all layer outputs) than predict() function
        input_next = inputs
        weight_next = self.w_input
        layer_outputs = []
        for i in range(self.hidden_layer_num):
            sum = np.dot(input_next, weight_next)
            print 'some of sum', i, 'are', sum[0], sum[5], sum[10]
            layer_out = activation1(sum)
            print 'some of layer', i, 'out are', layer_out[0], layer_out[10], layer_out[18]
            layer_outputs.append(layer_out)
            weight_next = self.w_hidden[i]
            input_next = layer_out
        sum = np.dot(input_next, weight_next)
        layer_out = activation_out(sum)
        layer_outputs.append(layer_out)
        return layer_outputs

    def learn(self, inputs, outputs, alpha):
        learning = True
        iter = 0
        convergence_value = 1       # step size when back propagation learning is "done"
        costs = []
        while learning:
            print 'w in', self.w_input[0], self.w_input[2]
            for i in range(self.hidden_layer_num):
                print 'w hidden', i, 'parts are', self.w_hidden[0][0], self.w_hidden[0][2]
            layer_outputs = self.get_outputs(inputs)
            final_layer_error = outputs - layer_outputs[self.hidden_layer_num]
            #final_layer_error = -np.dot(outputs.T, np.log(activation_out(layer_outputs[self.hidden_layer_num])))
            square_err, mean_square_err = mean_square_error(final_layer_error)
            costs.append(mean_square_err)
            layer_deltas = []       # list in reverse order
            dJ_dW = []              # list in reverse order
            print 'final layer error', final_layer_error
            derivative_out = activation_derivative_out(layer_outputs[len(layer_outputs)-1])
            print 'derivative out', derivative_out
            layer_delta_out = (1/(len(inputs))) * final_layer_error * derivative_out
            #layer_delta_out = final_layer_error * derivative_out
            print 'layer delta out', layer_delta_out
            layer_deltas.append(layer_delta_out)
            for i in range(self.hidden_layer_num - 1):
                dJdW = np.dot(layer_deltas[i].T, layer_outputs[len(layer_outputs)-i-1])
                print 'dJdW', dJdW
                dJ_dW.append(dJdW)
                layer_deltas.append(np.dot(layer_deltas[i], dJ_dW[i]) * activation_derivative1(layer_outputs[len(layer_outputs)-i-1]))
            dJ_dW.append(np.dot(inputs.T, layer_deltas[len(layer_deltas) - 1]))    # layer1_delta.T, inputs)
            print 'len input', len(inputs)
            # (1/(len(inputs)))
            # layer3_delta = (1/(len(inputs))) * layer3_error * activation_derivative_out(layer_outputs[2])
            # print 'layer 3 delta', layer3_delta
            # layer2_delta = layer2_error * activation_derivative_out(layer2_out)
            # layer1_error = np.dot(layer2_delta, self.w_hidden.T)          # .T is transpose with np array
            # dJ_dW_hidden2 = np.dot(layer3_delta.T, layer_outputs[1])
            # print 'dj dw hidden 2', dJ_dW_hidden2
            # dJ_dW_hidden = np.dot(layer2_delta.T, layer1_out)
            # layer1_delta = (1/m) * layer1_error * activation_derivative1(layer1_out)
            # layer2_delta = np.dot(layer3_delta, dJ_dW_hidden2) * activation_derivative1(layer_outputs[1])
            # layer1_delta = np.dot(layer2_delta, dJ_dW_hidden) * activation_derivative1(layer1_out)
            # dJ_dW_hidden1 = np.dot(layer2_delta.T, layer_outputs[0])
            # print 'dj dw hidden 1', dJ_dW_hidden1
            # layer1_delta = np.dot(layer2_delta, dJ_dW_hidden1) * activation_derivative1(layer_outputs[0])
            #dJ_dW1 = np.dot(inputs.T, layer_deltas[len(layer_deltas)-1])#layer1_delta.T, inputs)
            #print 'dj dw 1', dJ_dW1
            # dJ_dW1 = np.dot(layer1_delta.T, inputs)
            for i in range(self.hidden_layer_num):
                self.w_hidden[i] = self.w_hidden[i] - alpha * dJ_dW[len(dJ_dW)-i-1]
            #self.w_hidden1 = self.w_hidden1 - alpha * dJ_dW_hidden1
            #self.w_hidden2 = self.w_hidden2 - alpha * dJ_dW_hidden2    # np.dot(layer1_out.T, layer2_delta)
            self.w_input = self.w_input - alpha * dJ_dW[0]      # np.dot(inputs.T, layer1_delta)
            print 'iter', iter, ', error', mean_square_err
            iter += 1
            if mean_square_err < convergence_value or iter > 3000:
                learning = False
        # print 'w in', self.w_input
        # print 'w hidden', self.w_hidden
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
    costs = perceptron.learn(glass_training_features, glass_training_class, 10)     # train the parameters
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
