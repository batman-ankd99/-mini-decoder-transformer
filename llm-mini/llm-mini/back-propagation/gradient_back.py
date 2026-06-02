def lol():

    h = 0.0001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0) ; f.label='f'
    L = d*f; L.label= 'L'
    L1 = L

    a = Value(2.0+h, label='a')
    b = Value(-3.0, label='b')
    c = Value(10, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0) ; f.label='f'
    L = d*f; L.label= 'L'
    L2 = L

    grad_a = (L2 - L1)/h
    print(grad_a)
