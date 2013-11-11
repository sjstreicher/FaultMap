import numpy
import matplotlib.pyplot as plt

N = 1000
h = 5
 
y = numpy.random.randn(N)
x = numpy.zeros_like(y)
 
for i in range(h, len(y)):
    x[i] = x[i-1] + y[i - h]
    
plt.plot(range(0, len(x)), x)