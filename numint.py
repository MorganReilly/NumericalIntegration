#!/usr/bin/env python
# coding: utf-8

# In[15]:


"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Skeleton/solution for Assignment 1: Numerical Integration

By writing my name below and submitting this file, I/we declare that
all additions to the provided skeleton file are my/our own work, and that
I/we have not seen any work on this assignment by another student/group.

Student name(s): Yashitha Agarwal, Morgan Reilly, Akanksha
Student ID(s): 20230091, 20235398, 20231242

"""

import numpy as np
import sympy
import itertools
import math

def numint_py(f, a, b, n):
    """Numerical integration. For a function f, calculate the definite
    integral of f from a to b by approximating with n "slices" and the
    "lb" scheme. This function must use pure Python, no Numpy.

    >>> round(numint_py(math.sin, 0, 1, 100), 5)
    0.45549
    >>> round(numint_py(lambda x: 1, 0, 1, 100), 5)
    1.0
    >>> round(numint_py(math.exp, 1, 2, 100), 5)
    4.64746

    """
    A = 0
    w = (b - a) / n # width of one slice
    
    import decimal

    def Finput(a, b, w): # function to get list of input points to be passed to the function 
        while a < b: 
            yield float(a)
            a += decimal.Decimal(w)
            
    x = list(Finput(a, b, str(w))) # create input list

    
    for i in x:
        A += f(i) * w # # using lb scheme: calculate f(x) * dx for each value in input list and increment A 
        
    return A

def numint(f, a, b, n, scheme='mp'):
    """Numerical integration. For a function f, calculate the definite
    integral of f from a to b by approximating with n "slices" and the
    given scheme. This function should use Numpy, and eg np.linspace()
    will be useful.
    
    >>> round(numint(np.sin, 0, 1, 100, 'lb'), 5)
    0.45549
    >>> round(numint(lambda x: np.ones_like(x), 0, 1, 100), 5)
    1.0
    >>> round(numint(np.exp, 1, 2, 100, 'lb'), 5)
    4.64746
    >>> round(numint(np.exp, 1, 2, 100, 'mp'), 5)
    4.67075
    >>> round(numint(np.exp, 1, 2, 100, 'ub'), 5)
    4.69417

    """
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    A = 0
    w = (b - a) / n # width of one slice
    x = np.arange(a,b,w) # create list of input points to be passed to the function 

    for i in x:
        if scheme == 'mp':            
            A += f((i + (i + w)) / 2) * w # using mp scheme: calculate f((x + (x+1) / 2)) * dx for each value in input list and increment A
            
        elif scheme == 'lb':
            A += f(i) * w # using lb scheme: calculate f(x) * dx for each value in input list and increment A 
            
        elif scheme == 'ub':
            A += f(i + w) * w # using ub scheme: calculate f(x) * dx for each value in input list and increment A 
            
    return A    

def true_integral(fstr, a, b):
    """Using Sympy, calculate the definite integral of f from a to b and
    return as a float. Here fstr is an expression in x, as a str. It
    should use eg "np.sin" for the sin function.

    This function is quite tricky, so you are not expected to
    understand it or change it! However, you should understand how to
    use it. See the doctest examples.

    >>> true_integral("np.sin(x)", 0, 2 * np.pi)
    0.0
    >>> true_integral("x**2", 0, 1)
    0.3333333333333333

    STUDENTS SHOULD NOT ALTER THIS FUNCTION.

    """
    x = sympy.symbols("x")
    # make fsym, a Sympy expression in x, now using eg "sympy.sin"
    fsym = eval(fstr.replace("np", "sympy")) 
    A = sympy.integrate(fsym, (x, a, b)) # definite integral
    A = float(A.evalf()) # convert to float
    return A

def numint_err(fstr, a, b, n, scheme):
    """For a given function fstr and bounds a, b, evaluate the error
    achieved by numerical integration on n points with the given
    scheme. Return the true value, absolute error, and relative error
    as a tuple.

    Notice that the relative error will be infinity when the true
    value is zero. None of the examples in our assignment will have a
    true value of zero.

    >>> print("%.4f %.4f %.4f" % numint_err("x**2", 0, 1, 10, 'lb'))
    0.3333 0.0483 0.1450

    """
    f = eval("lambda x: " + fstr) # f is a Python function
    A = true_integral(fstr, a, b)
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    
    T = numint(f, a, b, n, scheme)
    AE = abs(A - T)
    RE = abs(AE / A)
    
    return (A, AE, RE)

def make_table(f_ab_s, ns, schemes):
    """For each function f with associated bounds (a, b), and each value
    of n and each scheme, calculate the absolute and relative error of
    numerical integration and print out one line of a table. This
    function doesn't need to return anything, just print. Each
    function and bounds will be a tuple (f, a, b), so the argument
    f_ab_s is a list of tuples.

    Hint 1: use print() with the format string
    "%s,%.2f,%.2f,%d,%s,%.4g,%.4g,%.4g", or an equivalent f-string approach.
    
    Hint 2: consider itertools.

    >>> make_table([("x**2", 0, 1), ("np.sin(x)", 0, 1)], [10, 100], ['lb', 'mp'])
    x**2,0.00,1.00,10,lb,0.3333,0.04833,0.145
    x**2,0.00,1.00,10,mp,0.3333,0.0008333,0.0025
    x**2,0.00,1.00,100,lb,0.3333,0.004983,0.01495
    x**2,0.00,1.00,100,mp,0.3333,8.333e-06,2.5e-05
    np.sin(x),0.00,1.00,10,lb,0.4597,0.04246,0.09236
    np.sin(x),0.00,1.00,10,mp,0.4597,0.0001916,0.0004168
    np.sin(x),0.00,1.00,100,lb,0.4597,0.004211,0.009161
    np.sin(x),0.00,1.00,100,mp,0.4597,1.915e-06,4.167e-06

    """   
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    for f_ab_s, ns, schemes in itertools.product(f_ab_s, ns, schemes):
        res = numint_err(f_ab_s[0], f_ab_s[1], f_ab_s[2], ns, schemes)
        print("%s,%.2f,%.2f,%d,%s,%.4g,%.4g,%.4g" % (f_ab_s[0], f_ab_s[1], f_ab_s[2], ns, schemes, res[0], res[1], res[2]))
    
def main():
    """Call make_table() as specified in the pdf."""
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    make_table([("np.cos(x)", 0, math.pi/2), ("np.sin(2*x)", 0, 1), ("np.exp(x)", 0, 1)],  [10, 100, 1000], ['lb', 'mp'])
    
    # uncomment to run the numint_nd function
    # print(round(numint_nd([np.sin, np.cos],[0, 1],[1, 2],[100, 100]), 5))
    # print(round(numint_nd([np.exp, np.exp, np.exp],[0, 1, 0],[1, 2, 3],[1000, 1000, 1000]), 5))

"""
On comparing the results of each function with the 'lb' and 'mp' schemes, 'mp' scheme has better results than 'lb' scheme. 
We can also infer that as the number of slices 'n' increases, the Relative error decreases for the same scheme.
For example, in the function, "sin(2x)" when 'n = 10', the difference between the relative error of the 'lb' and 'mp' scheme 
is 0.06587. Whereas, when 'n = 1000', the difference between the relative error of the 'lb' and 'mp' scheme is 0.000642.
Therefore, the results of the 'mp' scheme with higher value of 'n' is closest to the value of the True Integral. 
"""

def numint_nd(f, a, b, n):
    """numint in any number of dimensions.

    f: a function of m arguments
    a: a tuple of m values indicating the lower bound per dimension
    b: a tuple of m values indicating the upper bound per dimension
    n: a tuple of m values indicating the number of steps per dimension

    STUDENTS ADD DOCTESTS
    >>> round(numint_nd([np.sin, np.cos],[0, 1],[1, 2],[100, 100]), 5)
    0.03113
    
    >>> round(numint_nd([np.exp, np.exp, np.exp],[0, 1, 0],[1, 2, 3],[1000, 1000, 1000]), 5)
    153.30086

    """

    # My implementation uses Numpy and the mid-point scheme, but you
    # are free to use pure Python and/or any other scheme if you prefer.
    
    # Hint: calculate w, the step-size, per dimension
    w = [(bi - ai) / ni for (ai, bi, ni) in zip(a, b, n)]

    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    # create a list of input points for each dimension
    points = [np.linspace(ai, bi, ni) for (ai, bi, ni) in zip(a, b, n)]

    integrals = []

    for p, f, w in zip(points, f, w):
        R = 0
        for x in p:        
            R += (f(x)) * w # Calculate the f(x) * dx for each dimension
        integrals.append(R) # List of all f(x) * dx values

    Final = np.prod(integrals) # Product of all f(x) * dx values. Eg: For double integral: f(x) * dx * f(y) * dy
    return Final


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()


# In[ ]:




