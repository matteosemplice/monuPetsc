#!/bin/ipython3

#from sympy import diff
from sympy import * #serve per usare le varie funzioni...

from sympy.abc import x, y, z

s = x*x + 2*y*y +3*z*z - cos(2*pi*z)*exp(-x*y)
#s=z
c = 1.0

gamma = exp(x-y+z)
sigma = sin(x-y+2*z)

grad = [diff(s,x),diff(s,y),diff(s,z)]
gammaGrad = [gamma*grad[0],gamma*grad[1],gamma*grad[2]]

lapl = diff(gammaGrad[0],x)+diff(gammaGrad[1],y)+diff(gammaGrad[2],z)

f = sigma*s - lapl

print("f:")
print(f)

from sympy.utilities.codegen import codegen

codegen((('f_',f),('s_',s),('c_',c),('Gamma_',gamma),('Sigma_',sigma)), "C99", "test", header=False, empty=True,argument_sequence=(x,y,z),to_files=True,project="")
