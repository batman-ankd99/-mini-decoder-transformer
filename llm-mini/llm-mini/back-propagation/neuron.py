x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

#weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

#bias of NN
b = Value(6.8813735870195432, label='b')

x1w1 =  x1 * w1
x1w1.label = 'x1w1'

x2w2 = x2 * w2
x2w2.label = 'x2w2'

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = 'x1w1x2w2'

n = x1w1x2w2 + b
n.label = 'n'

o = n.tanh(); o.label = 'o'

o.grad = 1.0
#o = tanh(n)**2
#do/dn = 1 - tanh(n)**2
#do/dn = 1 - o.data**2
n.grad = 1 - o.data**2

# backpropagting through + sign, just transfer gradient equally to back nodes whereever plus sign comes
x1w1x2w2.grad = 0.5
b.grad = 0.5

#plus again so distribute gradien equally again to back nodes of b and x1w1x2w2

x1w1.grad = 0.5
x2w2.grad = 0.5
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

draw_dot(o)
