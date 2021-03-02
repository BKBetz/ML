class NeuronLayer:

    def __init__(self, neurons: list):
        self.neurons = neurons

    def layer_output(self, inputs: list):
        outputs = []
        for neuron in self.neurons:
            # get the output of each neuron in the layer and save it in a list
            n_output = neuron.activationfunc(inputs)
            outputs.append(n_output)
        return outputs

    def get_layer_length(self):
        return len(self.neurons)

    def __str__(self):
        return "perceptrons: {} \nOutputs = {}".format(len(self.neurons), self.layer_output())
