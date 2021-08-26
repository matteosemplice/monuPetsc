#!/bin/ipython

from sympy import diff

from sympy.abc import x, y, z

#s = x*x + 2*y*y -3*z*z
s=z
c = 1

gamma = 1
sigma = 0

grad = [diff(s,x),diff(s,y),diff(s,z)]
gammaGrad = gamma*grad

lapl = diff(gammaGrad[0],x)+diff(gammaGrad[1],y)+diff(gammaGrad[2],z)

f = sigma*s - lapl

print("f:")
print(f)

from sympy.utilities.codegen import codegen

codegen((('f_',f),('s_',s),('c_',c),('Gamma_',gamma),('Sigma_',sigma)), "C99", "test", header=False, empty=True,argument_sequence=(x,y,z),to_files=True,project="")
