from P1.code.perceptron import Perceptron
from P1.code.perceptronlayer import PerceptronLayer
from P1.code.perceptronnetwork import PerceptronNetwork

x1 = Perceptron([0.5, 0.5], 0.5)
x2 = Perceptron([0.4, 0.5], -1)

lx = PerceptronLayer([x1, x2])
ly = PerceptronLayer([x1, x2])

pn = PerceptronNetwork([[1, 1], [1, 0]], [lx, ly])
# should give [1,0]. if you change the 0.4 to 0.5 it should give [1,1]
print(pn.feed_forward())