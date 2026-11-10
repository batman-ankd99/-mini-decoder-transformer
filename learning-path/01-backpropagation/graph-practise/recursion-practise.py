def fact(n):
    if (n == 0 or n ==1):
        return 1
    else:
        fac = n * fact(n-1)
        return fac

fact(4)
