class PerceptronLayer:

    def __init__(self, perceptrons: list):
        self.perceptrons = perceptrons

    def layer_output(self, inputs: list):
        outputs = []
        for x in range(0, len(self.perceptrons)):
            # get the output of each perceptron in the layer and save it in a list
            p_output = self.perceptrons[x].get_output(x)
            outputs.append(p_output)

        return outputs

    def __str__(self):
        return "perceptrons: {} \nOutputs = {}".format(len(self.perceptrons), self.layer_output())
