import math

class Neuron:

    def __init__(self, weights: list, bias: float):
        self.weights = weights
        self.bias = bias

    # use the inputs given by the layers to give an output
    def activationfunc(self, inputs: list):
        sum = 0
        # lists should be same length
        if len(inputs) == len(self.weights):
            for x in range(0, len(inputs)):
                if inputs[x] == 1:
                    # add wn*xn to sum
                    sum += inputs[x] * self.weights[x]

        # the formula is w1*x1 w2*x2...wn*xn + bias. The sum is the wn*xn so we only add the bias
        total = sum + self.bias
        return 1 / (1 + math.exp(-total))

    def __str__(self):
        return "Weight: {} \nBias: {}".format(self.weights, self.bias)


n1 = Neuron([0.5, 0.5], -1)
print(n1.activationfunc([0, 0]))

