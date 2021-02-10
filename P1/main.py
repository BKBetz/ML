from P1.perceptron import Perceptron
from P1.perceptronlayer import PerceptronLayer

"""
    The __str__ of the preceptron class returned many values which I thought were important
    but for faster testing I made a second function that only returns the output of the perceptron
    
    The code below shows all possibilities for all gates we needed to test
"""

# INVERT (NOT) TEST
t1 = Perceptron([0], [-1], 0.5)
t2 = Perceptron([1], [-1], 0.5)
layer_1 = PerceptronLayer([t1, t2])
# this is the complete print with all the information of each perceptor uncomment for more info
# print("{} \n{}".format(t1, t2))
# this is the print of only the outputs
print(layer_1)


# AND TEST
t3 = Perceptron([0, 0], [0.5, 0.5], -1)
t4 = Perceptron([0, 1], [0.5, 0.5], -1)
t5 = Perceptron([1, 0], [0.5, 0.5], -1)
t6 = Perceptron([1, 1], [0.5, 0.5], -1)
# this is the complete print with all the information of each perceptor uncomment for more info
# print("{} \n{} \n{} \n{}".format(t3, t4, t5, t6))
# this is the print of only the outputs
print("AND results: {},{},{},{}".format(t3.get_output(), t4.get_output(), t5.get_output(), t6.get_output()))


# OR TEST
t7 = Perceptron([0, 0], [0.5, 0.5], -0.5)
t8 = Perceptron([0, 1], [0.5, 0.5], -0.5)
t9 = Perceptron([1, 0], [0.5, 0.5], -0.5)
t10 = Perceptron([1, 1], [0.5, 0.5], -0.5)
# this is the complete print with all the information of each perceptor uncomment for more info
# print("{} \n{} \n{} \n{}".format(t7, t8, t9, t10))
# this is the print of only the outputs
print("OR results: {},{},{},{}".format(t7.get_output(), t8.get_output(), t9.get_output(), t10.get_output()))


# NOR TEST
t11 = Perceptron([0, 0, 0], [-1, -1, -1], 0)
t12 = Perceptron([0, 0, 1], [-1, -1, -1], 0)
t13 = Perceptron([0, 1, 0], [-1, -1, -1], 0)
t14 = Perceptron([0, 1, 1], [-1, -1, -1], 0)
t15 = Perceptron([1, 0, 0], [-1, -1, -1], 0)
t16 = Perceptron([1, 0, 1], [-1, -1, -1], 0)
t17 = Perceptron([1, 1, 0], [-1, -1, -1], 0)
t18 = Perceptron([1, 1, 1], [-1, -1, -1], 0)
# this is the complete print with all the information of each perceptor uncomment for more info
# print("{} \n{} \n{} \n{} \n{} \n{} \n{} \n {}".format(t11, t12, t13, t14, t15, t16, t17, 18))
# this is the print of only the outputs
print("NOR results: {},{},{},{},{},{},{},{}".format(t11.get_output(), t12.get_output(), t13.get_output(), t14.get_output(),
                                                    t15.get_output(), t16.get_output(), t17.get_output(), t18.get_output()))

# PARTY TEST
t19 = Perceptron([0, 0, 0], [0.6, 0.3, 0.2], -0.4)
t20 = Perceptron([0, 0, 1], [0.6, 0.3, 0.2], -0.4)
t21 = Perceptron([0, 1, 0], [0.6, 0.3, 0.2], -0.4)
t22 = Perceptron([0, 1, 1], [0.6, 0.3, 0.2], -0.4)
t23 = Perceptron([1, 0, 0], [0.6, 0.3, 0.2], -0.4)
t24 = Perceptron([1, 0, 1], [0.6, 0.3, 0.2], -0.4)
t25 = Perceptron([1, 1, 0], [0.6, 0.3, 0.2], -0.4)
t26 = Perceptron([1, 1, 1], [0.6, 0.3, 0.2], -0.4)
# this is the complete print with all the information of each perceptor uncomment for more info
# print("{} \n{} \n{} \n{} \n{} \n{} \n{} \n {}".format(t19, t20, t21, t22, t23, t24, t25, 26))
# this is the print of only the outputs
print("PARTY results: {},{},{},{},{},{},{},{}".format(t19.get_output(), t20.get_output(), t21.get_output(), t22.get_output(),
                                                    t23.get_output(), t24.get_output(), t25.get_output(), t26.get_output()))

