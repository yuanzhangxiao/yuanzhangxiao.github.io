# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cvxpy as cvx


m = 30
n = 100
A_real = np.random.randn(m, n)
A_imag = np.random.randn(m, n)
b_real = np.random.randn(m, 1)
b_imag = np.random.randn(m, 1)

A_temp1 = np.hstack( ( A_real, -A_imag ) )
A_temp2 = np.hstack( ( A_imag, A_real ) )

A_tilde = np.vstack( ( A_temp1, A_temp2 ) )
b_tilde = np.vstack( ( b_real, b_imag ) )


z = cvx.Variable(2*n,1)
t = cvx.Variable(1,1)


constraints = [
                A_tilde * z == b_tilde
              ]

for i in range(0, n-1, 1):
    ci = np.zeros( (2, 2*n) )
    ci[0][i] = 1
    ci[1][n+i] = 1
    soc_constraint = (cvx.norm( ci*z ) <= t)
    constraints += [ soc_constraint ]

obj = Minimize( t )


prob = Problem(obj, constraints)

result = prob.solve()

print(result)