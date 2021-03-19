class NeuronNetwork:

    def __init__(self, layers: list, lr: float):
        self.layers = layers
        self.learning_rate = lr

    def feed_forward(self, inputs: list, target, n=0):
        # function to get the output of an layer
        output = self.layers[n].layer_output(inputs)
        # if n is smaller than len(layers)-1. it means we are still not at the end of the network.
        if n < len(self.layers)-1:
            new_inputs = output
            # loop again
            return self.feed_forward(new_inputs, target, n+1)

        else:
            # if n is bigger or equal to len(layers)-1 it means we are at te last layer which should give the final out
            # so we only return the output of that "final" layer and do not calculate the next (since it doesn't exist).
            output = self.layers[n].layer_output(inputs)
            # get error en delta's for the output layer
            self.layers[n].layer_changes(True, [], target, self.learning_rate)
            return output

    def backward_prop(self, target, n=1):
        x = len(self.layers)
        if x-n == 0:
            # when at the last layer..update the entire network
            self.update_network()
        else:
            # errors of previous layer
            errors = self.layers[x-n].layer_errors
            # get values of hidden layer with errors of previous layer
            self.layers[x-(n+1)].layer_changes(False, errors, target, self.learning_rate)
            return self.backward_prop(target, n+1)

    def update_network(self):
        for layer in self.layers:
            # call the update layer function for each layer which gives each neuron the correct values to update itself
            layer.update_layer()

    def calc_total_loss(self, lst: list):
        sum = 0
        for x in lst:
            # x[1] == target(s)
            # if the output layer has more than one outputneuron. the target is an list instead of int or float
            # because of this we need to use indexes to use the correct target
            if type(x[1]) == list:
                for y in range(len(x[1])):
                    sum += (x[0] - x[1][y]) ** 2
            else:
                sum += (x[1] - x[0]) ** 2
        return sum / (2 * len(lst))

    def train(self, inputs: list, target: list, epochs: int):
        n = 0
        while epochs > n:
            for x in range(0, len(target)):
                # calculate total loss with the current weight and bias
                loss_list = []
                # we use a extra loop to calculate the output for each input with the current weights and biases
                for y in range(0, len(target)):
                    output = self.feed_forward(inputs[y], target[y])
                    # output[0] is the output...it just returns it in a list
                    loss_list.append([output[0], target[y]])
                loss = self.calc_total_loss(loss_list)
                # actual output with the given input and current values
                output = self.feed_forward(inputs[x], target[x])

                print("\nEpoch {}: returns {} with an input of {} and a loss of {}".format(n+1, [round(i, 3) for i in output], inputs[x], round(loss,3)))

                self.backward_prop(target[x])

            n += 1

    def __str__(self):
        return "Layers: {}".format(len(self.layers))
