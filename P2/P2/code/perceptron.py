class Perceptron:

    """
        P2 is gewoon P1 met extra dingen dus ik heb P1 gecopy paste


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

    def __init__(self, weights: list, bias: float, lr: float):
        self.weights = weights
        self.bias = bias
        self.error_list = []
        self.lr = lr
        # score for looping
        self.score = 0

    # use the inputs given by the layers to give an output
    def activationfunc(self, sum: float):
        # the formula is w1*x1 w2*x2...wn*xn + bias. The sum is the wn*xn so we only add the bias
        if sum < 0:
            return 0
        else:
            return 1

    def output(self, inputs: list):
        # lists should be same length
        sum = 0
        if len(inputs) == len(self.weights):
            for x in range(0, len(inputs)):
                # add wn*xn to sum
                sum += inputs[x] * self.weights[x]

        total = sum + self.bias
        output = self.activationfunc(total)

        return output

    def update(self, inputs: list, target: int):
        # if a update isn't necessary (prediction is already correct) add 1 to score
        if self.output(inputs) == target:
            self.score += 1
        else:
            output = self.output(inputs)
            # error = target - output
            e = target - output
            # MSE = de som van alle errors^2 delen door lengte van de lijst. we kwadrateren de error dus en zetten deze in de lijst
            self.error_list.append(e**2)
            # delta bias = learning rate * error
            db = self.lr * e
            # new bias = old bias + delta bias
            self.bias = self.bias + db
            for x in range(0, len(inputs)):
                # delta weight[x] = learning rate * error * input[x]
                dw = self.lr * e * inputs[x]
                # new weight[x] = old weight[x] * delta weight[x]
                self.weights[x] = self.weights[x] + dw

    def error(self, e: list):
        if len(e) > 0:
            return sum(e) / len(e)
        else:
            return 0

    def __str__(self):
        return "Weight: {} \nBias: {} \nError: {}".format(self.weights, self.bias, self.error(self.error_list))
