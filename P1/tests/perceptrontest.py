import unittest
from P1.code.perceptron import Perceptron
from P1.code.perceptronlayer import PerceptronLayer
from P1.code.perceptronnetwork import PerceptronNetwork

"""
    All tests that where given in the exercise are in this file.
    For all separate perceptrons I use a layer since this returns all outputs in a list
"""


class PerceptronTest(unittest.TestCase):

    def testNOT(self):
        t1 = Perceptron([-1], 0.5)
        t2 = Perceptron([-1], 0.5)

        layer_1 = PerceptronLayer([t1, t2])

        inputs = [[0], [1]]
        self.assertEqual([1, 0], layer_1.layer_output(inputs))

    def testAND(self):
        t1 = Perceptron([0.5, 0.5], -1)
        t2 = Perceptron([0.5, 0.5], -1)
        t3 = Perceptron([0.5, 0.5], -1)
        t4 = Perceptron([0.5, 0.5], -1)

        layer_1 = PerceptronLayer([t1, t2, t3, t4])

        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.assertEqual([0, 0, 0, 1], layer_1.layer_output(inputs))

    def testOR(self):
        t1 = Perceptron([0.5, 0.5], -0.5)
        t2 = Perceptron([0.5, 0.5], -0.5)
        t3 = Perceptron([0.5, 0.5], -0.5)
        t4 = Perceptron([0.5, 0.5], -0.5)

        layer_1 = PerceptronLayer([t1, t2, t3, t4])

        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.assertEqual([0, 1, 1, 1], layer_1.layer_output(inputs))

    def testNOR(self):
        t1 = Perceptron([-1, -1, -1], 0)
        t2 = Perceptron([-1, -1, -1], 0)
        t3 = Perceptron([-1, -1, -1], 0)
        t4 = Perceptron([-1, -1, -1], 0)
        t5 = Perceptron([-1, -1, -1], 0)
        t6 = Perceptron([-1, -1, -1], 0)
        t7 = Perceptron([-1, -1, -1], 0)
        t8 = Perceptron([-1, -1, -1], 0)

        layer_1 = PerceptronLayer([t1, t2, t3, t4, t5, t6, t7, t8])

        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        self.assertEqual([1, 0, 0, 0, 0, 0, 0, 0], layer_1.layer_output(inputs))

    def testPARTY(self):
        t1 = Perceptron([0.6, 0.3, 0.2], -0.4)
        t2 = Perceptron([0.6, 0.3, 0.2], -0.4)
        t3 = Perceptron([0.6, 0.3, 0.2], -0.4)
        t4 = Perceptron([0.6, 0.3, 0.2], -0.4)
        t5 = Perceptron([0.6, 0.3, 0.2], -0.4)
        t6 = Perceptron([0.6, 0.3, 0.2], -0.4)
        t7 = Perceptron([0.6, 0.3, 0.2], -0.4)
        t8 = Perceptron([0.6, 0.3, 0.2], -0.4)

        layer_1 = PerceptronLayer([t1, t2, t3, t4, t5, t6, t7, t8])

        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        self.assertEqual([0, 0, 0, 1, 1, 1, 1, 1], layer_1.layer_output(inputs))

    def testXOR(self):
        t1 = Perceptron([1, 1], -0.5)
        t2 = Perceptron([-1, -1], 1.5)

        layer_1 = PerceptronLayer([t1, t2])

        t3 = Perceptron([1, 1], -2)

        layer_2 = PerceptronLayer([t3])

        network = PerceptronNetwork([[0, 0], [0, 1], [1, 0], [1, 1]], [layer_1, layer_2])

        self.assertEqual([0, 1, 1, 0], network.feed_forward())
