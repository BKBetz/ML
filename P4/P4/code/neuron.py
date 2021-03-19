import math


class Neuron:

    def __init__(self, weights: list, bias: float):
        self.weights = weights
        self.bias = bias
        self.error = 0
        self.delta_weights = []
        self.n_output = 0
        self.inputs = []

    # use the inputs given by the layers to give an output
    def sigmoid(self, total: float):
        return 1 / (1 + math.exp(-total))

    def output(self, inputs: list):
        sum = 0
        # lists should be same length
        if len(inputs) == len(self.weights):
            for x in range(0, len(inputs)):
                # add wn*xn to sum
                sum += inputs[x] * self.weights[x]

        # the formula is w1*x1 w2*x2...wn*xn + bias. The sum is the wn*xn so we only add the bias
        total = sum + self.bias
        self.n_output = self.sigmoid(total)
        # save the inputs for the backprop
        self.inputs = inputs
        return self.n_output

    def calc_hidden_error(self, errors: list):
        # errors is a list of errors from the previous layer
        sum = 0
        for error, weight in zip(errors, self.weights):
            sum += weight * error

        e = self.n_output * (1-self.n_output) * sum
        # change the neuron error and return the error as well because it is needed to calculate the hidden error of the next layer
        self.error = e
        return e

    def update(self, bias: float):
        # gets correct bias from the layer
        self.bias = self.bias - bias
        for x in range(0, len(self.weights)):
            self.weights[x] -= self.delta_weights[x]

        # print for tests to show values after backprop
        print("VALUES AFTER UPDATE: "
              "weights: {}, error: {}, bias: {}".format([round(i, 3) for i in self.weights], round(self.error, 3), round(self.bias, 3)))

    def calc_error(self, target: float):
        e = self.n_output * (1-self.n_output) * -(target - self.n_output)
        self.error = e
        return e

    def calc_gradient(self, output: float):
        gw = output * self.error
        return gw

    def calc_weight_delta(self, lr: float):
        # empty delta weights to remove old values
        self.delta_weights = []
        for x in range(len(self.weights)):
            dw = lr * self.error * self.inputs[x]
            self.delta_weights.append(dw)

    def calc_bias_delta(self, lr: float):
        bw = lr * self.error
        return bw

    def __str__(self):
        return "Weight: {} \nBias: {}".format(self.weights, self.bias)
