class PerceptronNetwork:
    """
        Class gets two lists.
        A list of inputs used for the first layer. and a list of all layers (class PerceptronLayer) in the network
        feedforward() gets the output of a layer and uses that output for the next layer till its looped through all layers.
    """

    def __init__(self, layers: list):
        self.layers = layers

    def feed_forward(self, inputs: list, n=0):
        # function to get the output of an layer
        output = self.layers[n].layer_output(inputs)
        # if n is smaller than len(layers)-1. it means we are still not at the end of the network.
        if n < len(self.layers)-1:
            # """
            #     get the amount of perceptrons for the next layer.
            #     so we can send the output of the previous layer to the next layer.
            #
            #     example:
            #     if layer 1 has 4 perceptrons and the output of those 4 (and therefore the output of the layer) is [1,0,0,1]
            #     and layer 2 had 3 perceptrons
            #     the input for layer 2 should be [[1,0,0,1],[1,0,0,1],[1,0,0,1]]
            # """
            # next_layer = self.layers[n+1].get_layer_length()
            #
            # new_input = []
            #
            # for x in range(0, next_layer):
            #     new_input.append(output)

            # change the current input to the new input so it can be used for the next layer
            new_inputs = output

            # loop again
            return self.feed_forward(new_inputs, n+1)

        else:
            # if n is bigger or equal to len(layers)-1 it means we are at te last layer which should give the final out
            # so we only return the output of that "final" layer and do not calculate the next (since it doesn't exist).
            return output

    def __str__(self):
        return "Layers: {} \nOutput: {}".format(len(self.layers), self.feed_forward())
