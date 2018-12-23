# -*- coding: utf-8 -*-
import numpy as np
a = np.array([1,2,3,4,5])
print(a)


# vectorization
a = np.random.rand(1000000)
b = np.random.rand(1000000)


import time
tic = time.time()
c = np.dot(a, b)
toc = time.time()

print('c=', c)
print('vectorization ={} {} '.format(str((toc-tic)*1000),'ms'))

# For loop
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print('c=', c)
print('Forloop ={} {} '.format(str((toc-tic)*1000),'ms'))
