import numpy as np

def sum_prod(X, V):
    '''
    X - матрицы (n, n)
    V - векторы (n, 1)
    Гарантируется, что len(X) == len(V)
    '''
    n = X[0].shape[0]
    res = np.zeros(n)
    for i in range(len(X)):
        z = V[i].reshape(n) if V[i].shape == (n, 1) else V[i]
        res += X[i] @ z
    return res

A1 = np.array([[1,2], [3,4]])
A2 = np.array([[1,2], [3,4]])
V1 = np.array([1,0])
V2 = np.array([0,1])
print("3 7", sum_prod([A1,A2], [V1,V2]))

A11 = np.array([[10, 11, 12], [16,17,18], [16,17,18]])
A12 = np.array([[19,20,21], [22,23,24], [22,23,24]])
V11 = np.array([45, 46, 47])
V12 = np.array([48,49,50])
print("4462 5731 5731", sum_prod([A11,A12], [V11,V12]))