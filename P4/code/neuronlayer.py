class NeuronLayer:

    def __init__(self, neurons: list):
        self.neurons = neurons
        self.layer_errors = []
        self.layer_biases = []

    def layer_output(self, inputs: list, output: bool, target: float, lr:float):
        outputs = []
        for neuron in self.neurons:
            # get the output of each neuron in the layer and save it in a list
            n_output = neuron.activationfunc(inputs)
            outputs.append(n_output)

        return outputs

    def layer_error(self, output: bool, errors: list):
        for neuron in self.neurons:
            if output is True:
                e = neuron.calc_error(target)
                self.layer_errors.append(e)

            else:


            neuron.calc_weight_delta(inputs, lr)
            db = neuron.calc_bias_delta(lr)

            self.layer_biases.append(db)


    def update_layer(self):
        for x in range(0, len(self.neurons)):
            self.neurons[x].update(self.layer_errors[x], self.layer_biases[x])

    def get_layer_length(self):
        return len(self.neurons)

    def __str__(self):
        return "perceptrons: {} \nOutputs = {}".format(len(self.neurons), self.layer_output())
