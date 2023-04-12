from sympy import * 
from sympy.abc import x, y, z 

f = x * y * z 
print(f"diff(f, x) {diff(f, x) }")
# >>> y * z 
print(f"diff(f, y) {diff(f, y)}")
print("g = x / y")
g = x / y 

print(f"diff(f, y) {diff(g, y)}")
# >>> x * z 






