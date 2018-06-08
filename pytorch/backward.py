# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:23:43 2018

@author: omf
"""
import torch as t
from torch.autograd import Variable as v
# compute jacobian matrix
x = t.FloatTensor([2, 1]).view(1, 2)
x = v(x, requires_grad=True)
y = v(t.FloatTensor([[1, 2], [3, 4]]))t

z = t.mm(x, y)
jacobian = t.zeros((2, 2))
z.backward(t.FloatTensor([[1, 0]]), retain_graph=True)  # dz1/dx1, dz1/dx2
jacobian[:, 0] = x.grad.data
x.grad.data.zero_()
z.backward(t.FloatTensor([[0, 1]]))  # dz2/dx1, dz2/dx2
jacobian[:, 1] = x.grad.data
print('=========jacobian========')
print('x')
print(x.data)
print('y')
print(y.data)
print('compute result')
print(z.data)
print('jacobian matrix is')
print(jacobian)