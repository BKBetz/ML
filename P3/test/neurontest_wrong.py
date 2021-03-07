import unittest
from P3.code.neuron import Neuron
from P3.code.neuronlayer import NeuronLayer
from P3.code.neuronnetwork import NeuronNetwork

"""
    De neuron werkt niet alleen met 0 en 1 outputs maar kan ook bijvoorbeeld 0.3 als output krijgen. 
    Dit komt doordat een neuron gebruikt maakt van de sigmoid functie
    dus de antwoorden zullen nooit hetzelfde zijn als bij een perceptron.
    De perceptron heeft een stepfunction wat betekent dat het alleen 0 of 1 als output kan hebben.
    Een neuron kan een output tussen de 0 en 1 hebben.
    
    Hieronder staan de tests met de perceptron waardes in een neuron
    
    p.s de code hieronder had ik netter kunnen maken met for loops maar ik kwam in tijdnood
"""


class NeuronTestWrong(unittest.TestCase):

    def testNOT(self):
        t1 = Neuron([-1], 0.5)

        layer_1 = NeuronLayer([t1])

        network = NeuronNetwork([layer_1])
        outputs = []
        # round the number that you get from feedforward...the function returns a list so we get the first item with [0]
        o1 = round(network.feed_forward([1])[0])
        o2 = round(network.feed_forward([0])[0])
        outputs.append(o1)
        outputs.append(o2)
        # De invert werkt gewoon met dezelfde waardes
        self.assertEqual([0, 1], outputs)

    def testAND(self):
        t1 = Neuron([0.5, 0.5], -1)

        layer_1 = NeuronLayer([t1])

        network = NeuronNetwork([layer_1])
        outputs = []

        for x in range(0, 2):
            for y in range(0, 2):
                # round the number that you get from feedforward...the function returns a list so we get the first item with [0]
                o = round(network.feed_forward([x, y])[0])
                outputs.append(o)

        self.assertNotEqual([0, 0, 0, 1], outputs)

    def testOR(self):
        t1 = Neuron([0.5, 0.5], -0.5)

        layer_1 = NeuronLayer([t1])

        network = NeuronNetwork([layer_1])
        outputs = []
        for x in range(0, 2):
            for y in range(0, 2):
                # round the number that you get from feedforward...the function returns a list so we get the first item with [0]
                o = round(network.feed_forward([x, y])[0])
                outputs.append(o)

        self.assertNotEqual([0, 1, 1, 1], outputs)

    def testNOR(self):
        t1 = Neuron([-1, -1, -1], 0)

        layer_1 = NeuronLayer([t1])

        network = NeuronNetwork([layer_1])
        outputs = []
        for x in range(0, 2):
            for y in range(0, 2):
                for z in range(0, 2):
                    # round the number that you get from feedforward...the function returns a list so we get the first item with [0]
                    o = round(network.feed_forward([x, y, z])[0])
                    outputs.append(o)

        self.assertNotEqual([1, 0, 0, 0, 0, 0, 0, 0], outputs)

    def testHalfAdder(self):
        t1 = Neuron([1, 1], -1)
        t2 = Neuron([-1, -1], 1.5)
        t3 = Neuron([1, 1], -2)
        # carry
        t4 = Neuron([0, 0, 1], -1)
        # sum
        t5 = Neuron([1, 1, 0], -2)

        layer_1 = NeuronLayer([t1, t2, t3])
        layer_2 = NeuronLayer([t4, t5])

        network = NeuronNetwork([layer_1, layer_2])
        outputs = []
        for x in range(0, 2):
            for y in range(0, 2):
                # round the number that you get from feedforward...the function returns a list.
                # We need to round both items in the list so we seperate it with [0] and [1]
                o = [round(network.feed_forward([0, 0])[0]), round(network.feed_forward([0, 0])[1])]
                outputs.append(o)

        self.assertNotEqual([[0, 0], [0, 1], [0, 1], [1, 0]], outputs)
