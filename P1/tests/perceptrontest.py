import unittest
from P1.code.perceptron import Perceptron
from P1.code.perceptronlayer import PerceptronLayer
from P1.code.perceptronnetwork import PerceptronNetwork

"""
    All tests that where given in the exercise are in this file.
"""


class PerceptronTest(unittest.TestCase):

    def testNOT(self):
        t1 = Perceptron([-1], 0.5)

        layer_1 = PerceptronLayer([t1])

        network = PerceptronNetwork([layer_1])

        self.assertEqual([0], network.feed_forward([1]))
        self.assertEqual([1], network.feed_forward([0]))

    def testAND(self):
        t1 = Perceptron([0.5, 0.5], -1)

        layer_1 = PerceptronLayer([t1])

        network = PerceptronNetwork([layer_1])

        self.assertEqual([0], network.feed_forward([0, 0]))
        self.assertEqual([0], network.feed_forward([0, 1]))
        self.assertEqual([0], network.feed_forward([1, 0]))
        self.assertEqual([1], network.feed_forward([1, 1]))

    def testOR(self):
        t1 = Perceptron([0.5, 0.5], -0.5)

        layer_1 = PerceptronLayer([t1])

        network = PerceptronNetwork([layer_1])

        self.assertEqual([0], network.feed_forward([0, 0]))
        self.assertEqual([1], network.feed_forward([0, 1]))
        self.assertEqual([1], network.feed_forward([1, 0]))
        self.assertEqual([1], network.feed_forward([1, 1]))

    def testNOR(self):
        t1 = Perceptron([-1, -1, -1], 0)

        layer_1 = PerceptronLayer([t1])

        network = PerceptronNetwork([layer_1])

        self.assertEqual([1], network.feed_forward([0, 0, 0]))
        self.assertEqual([0], network.feed_forward([0, 0, 1]))
        self.assertEqual([0], network.feed_forward([0, 1, 0]))
        self.assertEqual([0], network.feed_forward([0, 1, 1]))
        self.assertEqual([0], network.feed_forward([1, 0, 0]))
        self.assertEqual([0], network.feed_forward([1, 0, 1]))
        self.assertEqual([0], network.feed_forward([1, 1, 0]))
        self.assertEqual([0], network.feed_forward([1, 1, 1]))

    def testPARTY(self):
        t1 = Perceptron([0.6, 0.3, 0.2], -0.4)

        layer_1 = PerceptronLayer([t1])

        network = PerceptronNetwork([layer_1])

        self.assertEqual([0], network.feed_forward([0, 0, 0]))
        self.assertEqual([0], network.feed_forward([0, 0, 1]))
        self.assertEqual([0], network.feed_forward([0, 1, 0]))
        self.assertEqual([1], network.feed_forward([0, 1, 1]))
        self.assertEqual([1], network.feed_forward([1, 0, 0]))
        self.assertEqual([1], network.feed_forward([1, 0, 1]))
        self.assertEqual([1], network.feed_forward([1, 1, 0]))
        self.assertEqual([1], network.feed_forward([1, 1, 1]))

    def testXOR(self):
        t1 = Perceptron([1, 1], -0.5)
        t2 = Perceptron([-1, -1], 1.5)

        layer_1 = PerceptronLayer([t1, t2])

        t3 = Perceptron([1, 1], -2)

        layer_2 = PerceptronLayer([t3])

        network = PerceptronNetwork([layer_1, layer_2])

        self.assertEqual([0], network.feed_forward([0, 0]))
        self.assertEqual([1], network.feed_forward([0, 1]))
        self.assertEqual([1], network.feed_forward([1, 0]))
        self.assertEqual([0], network.feed_forward([1, 1]))

    def testHalfAdder(self):
        t1 = Perceptron([1, 1], -1)
        t2 = Perceptron([-1, -1], 1.5)
        t3 = Perceptron([1, 1], -2)
        # carry
        t4 = Perceptron([0, 0, 1], -1)
        # sum
        t5 = Perceptron([1, 1, 0], -2)

        layer_1 = PerceptronLayer([t1, t2, t3])
        layer_2 = PerceptronLayer([t4, t5])

        network = PerceptronNetwork([layer_1, layer_2])

        self.assertEqual([0, 0], network.feed_forward([0, 0]))
        self.assertEqual([0, 1], network.feed_forward([0, 1]))
        self.assertEqual([0, 1], network.feed_forward([1, 0]))
        self.assertEqual([1, 0], network.feed_forward([1, 1]))
