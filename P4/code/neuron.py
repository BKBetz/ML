import math


class Neuron:

    def __init__(self, weights: list, bias: float):
        self.weights = weights
        self.bias = bias
        self.error = 0
        self.delta_weights = []
        self.output = 0

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
        self.output = self.sigmoid(total)
        return self.output

    def calc_hidden_error(self, errors: list):
        sum = 0
        for error, weight in zip(errors, self.weights):
            sum += weight * error

        e = self.output * (1-self.output) * sum

        return e

    def update(self, error: float, bias: float):

        self.error = error
        self.bias = bias

        for x in range(0, len(self.weights)):
            self.weights[x] -= self.delta_weights[x]

    def calc_error(self, output: float, target: float):
        e = output * (1-output) * -(target - output)
        return e

    def calc_gradient(self, output: float):
        gw = output * self.error
        return gw

    def calc_weight_delta(self, input: list, lr: float):
        for x in range(len(self.weights)):
            dw = lr * x * self.error * input[x]
            self.delta_weights.append(dw)

    def calc_bias_delta(self, lr: float):
        bw = lr * self.error
        return bw

    def __str__(self):
        return "Weight: {} \nBias: {}".format(self.weights, self.bias)


n1 = Neuron([-0.5, 0.5], 1.5, 1)

print(n1.calc_hidden_error(0.249375, [0.131, -0.015]))
n1.update([0, 0], 0)

print(n1)