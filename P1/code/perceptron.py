class Perceptron:

    """
        Class gets weights(list), bias (float)
        based on the inputs it gets from the layer class, the function 'activationfunc' adds the weight to the sum
        if the total (sum + bias) < 0 return 0
        else return 1

        Why I don't use a treshold:

        According to the reader, the bias = -treshold and it can be used as a treshold (in a way).
        If the treshold (t) is 1, we can replace this by using a bias (b) of -1
        the check goes from:
        0 if sum(wi * xi) < t
        1 if sum(wi * xi) >= t

        to:
        0 if sum(wi * xi) + b < 0
        1 if sum(wi * xi) + b >= 0
    """

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
        if sum + self.bias < 0:
            return 0
        else:
            return 1

    def __str__(self):
        return "Weight: {} \nBias: {}".format(self.weights, self.bias)
