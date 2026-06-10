import numpy as np
import math
import matplotlib.pyplot as plt
def f(x):
    return 3*x**2 - 4*x + 5
x = 3
h = 0.0001
lr = 0.01
for i in range(170):
    slope = (f(x+h) - f(x)) / h ##gradient
    x = x - (lr*slope)

print(x)
print(slope)
