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

a, b, c = 2.0, -3.0, 10.0
h = 0.0001
lr = 0.01

def f(a, b, c):
    return a*b + c   # this is the expression Karpathy uses

for i in range(100):
    # numerical gradient for each input
    da = (f(a+h, b, c) - f(a, b, c)) / h
    db = (f(a, b+h, c) - f(a, b, c)) / h
    dc = (f(a, b, c+h) - f(a, b, c)) / h

    # update each input
    a = a - lr * da
    b = b - lr * db
    c = c - lr * dc
