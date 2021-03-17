from P4.code.neuron import Neuron
from P4.code.neuronlayer import NeuronLayer


class NeuronNetwork:

    def __init__(self, layers: list, lr: float, target: int):
        self.layers = layers
        self.learning_rate = lr
        self.target = target
        self.output = 0
        self.outputerror = []
        self.outputbias = []

    def feed_forward(self, inputs: list, n=0):
        # if n is smaller than len(layers)-1. it means we are still not at the end of the network.
        if n < len(self.layers)-1:
            # function to get the output of an layer
            output = self.layers[n].layer_output(inputs, self.target)
            new_inputs = output
            # loop again
            return self.feed_forward(new_inputs, n+1)

        else:
            # if n is bigger or equal to len(layers)-1 it means we are at te last layer which should give the final out
            # so we only return the output of that "final" layer and do not calculate the next (since it doesn't exist).
            output = self.layers[n].layer_output(inputs)
            # get the values of the output layer for later use
            self.output = output
            self.outputerror = self.layers[n].layer_errors
            self.outputbias = self.layers[n].layer_biases

            return output

    def backward_prop(self, inputs: list, target: float, n=0):
        x = len(self.layers)
        if x-n == 1:
            # when at the last layer..update the entire network
            self.update_network()
        else:
            output = self.layers[x-1].layer_output(inputs, False, self.target)
            new_inputs = output
            return self.backward_prop(new_inputs, target, n+1)

    def update_network(self):
        for layer in self.layers:
            layer.update_layer()

    def __str__(self):
        return "Layers: {}".format(len(self.layers))


f = Neuron([0.2, -0.4], 0, 1)
g = Neuron([0.7, 0.1], 0, 1)
o = Neuron([0.6, 0.9], 0, 1)

hidden = NeuronLayer([f, g])
output = NeuronLayer([o])
netwerk = NeuronNetwork([hidden, output])

print("iteratie 1")
netwerk.updateNetwerk([1, 1], [0])

print("iteratie 2")
netwerk.updateNetwerk([0, 1], [1])
