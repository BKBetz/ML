class NeuronNetwork:

    def __init__(self, layers: list):
        self.layers = layers

    def feed_forward(self, inputs: list, n=0):
        # function to get the output of an layer
        output = self.layers[n].layer_output(inputs)
        # if n is smaller than len(layers)-1. it means we are still not at the end of the network.
        if n < len(self.layers)-1:
            new_inputs = output
            # loop again
            return self.feed_forward(new_inputs, n+1)

        else:
            # if n is bigger or equal to len(layers)-1 it means we are at te last layer which should give the final out
            # so we only return the output of that "final" layer and do not calculate the next (since it doesn't exist).
            return output

    def __str__(self):
        return "Layers: {} \nOutput: {}".format(len(self.layers), self.feed_forward())