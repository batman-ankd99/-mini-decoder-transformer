import numpy as np
import math
import matplotlib.pyplot as plt
def f(x):
    return 3*x**2 - 4*x + 5


xs = np.arange(-5, 5 , 0.25)
ys = f(xs)
ys

plt.plot(xs, ys)
