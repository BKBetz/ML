class PerceptronLayer:
    """
        Class gets an list of perceptrons (class Perceptron)
        layer_output() is used to to give the output of each perceptron.
        The outcome of all outputs of all perceptrons is the output of the layer and is used for the next layer in the network.
        layer_length() gives the length of the layer. the length stand for how many perceptrons are in the layer
    """

    def __init__(self, perceptrons: list):
        self.perceptrons = perceptrons

    def layer_output(self, inputs: list):
        outputs = []
        for x in range(0, len(self.perceptrons)):
            # get the output of each perceptron in the layer and save it in a list
            p_output = self.perceptrons[x].activationfunc(inputs[x])
            outputs.append(p_output)
        return outputs

    def get_layer_length(self):
        return len(self.perceptrons)

    def __str__(self):
        return "perceptrons: {} \nOutputs = {}".format(len(self.perceptrons), self.layer_output())
