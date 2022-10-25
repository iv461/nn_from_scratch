from sympy import * 
from sympy.abc import x, y, z 

f = x * y * z 

diff(f, x) 
# >>> y * z 

diff(f, y) 
# >>> x * z 



